

import os
import logging
import time
from pathlib import Path

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from pdb import set_trace as st

from .diff_mask import *

from .operators import *

def init_implicit_prompts(length,dim,patch_size=16):
    prompt_embeddings = nn.Parameter(torch.zeros(
                1, length, dim))
        # xavier_uniform initialization
    val = math.sqrt(6. / float(3 * reduce(mul, _pair(patch_size), 1) + dim))
    nn.init.uniform_(prompt_embeddings.data, -val, val)

    return prompt_embeddings


class ImgPromptEncoder(nn.Module):
    def __init__(self,img_c,img_h,img_w,hidden_dim,scale=4):
        super(ImgPromptEncoder,self).__int__()

        self.resize=transforms.Resize((int(img_h/scale),int(img_h/scale)))

        self.encoder=nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=2, stride=2),
            LayerNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, hidden_dim, kernel_size=1),
        )

    def forward(self,x):
        scale_x = self.resize(x)
        y=self.encoder(scale_x).flatten(2).transpose(1,2)
    
        return y
    

class DensityEstimationLayer(nn.Module):
    def __init__(self,in_channels,out_dim,out_h,out_w):
        super(DensityEstimationLayer, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AdaptiveAvgPool2d((out_h,out_w))
        
        # Define the fully connected layers
        self.fc1 = nn.Conv2d(64, 512,1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(512, out_dim,1)
        
    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Pass the output through the fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        x = x.reshape(x.shape[0],x.shape[1],-1).transpose(1,2)
        # Return the estimated density
        return x
    

class PromptProj(nn.Module):
    def __init__(self,embed_dim, proj_input_dim, proj_hidd_dim, promptL, promptLayers=12):
        super(PromptProj, self).__init__()

        # projector to generate cooeficients
        self.proj_input_dim = proj_input_dim
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.prompt_len = promptL
        self.prompt_layers = promptLayers

     
        self.lin1 = nn.Linear(self.proj_input_dim, self.proj_hidd_dim)
        self.lin2 = nn.Linear(self.proj_hidd_dim, self.prompt_len*self.prompt_layers*self.proj_output_dim)
        self.act1 = nn.GELU()
        self.act2 = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.act1(self.lin1(x))
        prompts = self.lin2(x).view(-1,self.prompt_layers,self.prompt_len,self.proj_output_dim)
        return prompts
    

    
class PromptEncoderDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, promptL, promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(PromptEncoderDecoder, self).__init__()

        # projector to generate cooeficients
        self.layers=layers
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.prompt_len_per_layer = promptL
        self.prompt_layers = promptLayers

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.prompt_embeddings = nn.Parameter(torch.empty(1,
            self.prompt_len_per_layer*self.prompt_layers,
            self.proj_hidd_dim
        ))
        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        torch.nn.init.normal_(self.prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.props_pe,std=0.02)

        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)

        
    def forward(self, x):

        x = x + self.props_pe
        x = self.encoder(x) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
        # fetch cls token for prediction
        prompts = self.decoder(self.prompt_embeddings.expand(x.shape[0],-1,-1),x)
        prompts = self.proj(prompts)
        if self.prompt_layers>1:
            return prompts.view(-1,self.prompt_layers,self.prompt_len_per_layer,self.embed_dim)
        return prompts

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projections
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out(attn_output)

        return output, attn_weights

class PosNegCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, norm=0, residual=0):
        super(PosNegCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm = norm
        self.residual = residual

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projections
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # softmax on all visual attributes
        pos_attn_weights = nn.functional.softmax(scores, dim=-1) 
        # how to derive negative attention weights
        # Step 1: Initialize w_i^- as the complement of w_i^+
        neg_attn_weights = (1 - pos_attn_weights)/(1-pos_attn_weights).sum(dim=-1,keepdim=True)
        # Step 2: Apply mutual exclusivity constraint
        neg_attn_weights = torch.min(neg_attn_weights, 1e-6/(pos_attn_weights+1e-10))
        # Step 3: Normalize the negative attention weights to sum to 1 along the last dimension
        neg_attn_weights = neg_attn_weights / neg_attn_weights.sum(dim=-1, keepdim=True)

        pos_attn_output = torch.matmul(pos_attn_weights, V)
        neg_attn_output = torch.matmul(neg_attn_weights, V)

        # Concatenate heads and put through final linear layer
        pos_attn_output = pos_attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        neg_attn_output = neg_attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        pos_output = self.out(pos_attn_output)
        neg_output = self.out(neg_attn_output)
        if self.norm:
            pos_output=self.layer_norm(pos_output)
            neg_output=self.layer_norm(neg_output)

        return pos_output, neg_output, pos_attn_weights, neg_attn_weights
    

# cross attention prompt generator
class QKVPromptG(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, promptL, promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(QKVPromptG, self).__init__()

        # projector to generate cooeficients
        self.layers=layers
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.prompt_len_per_layer = promptL
        self.prompt_layers = promptLayers

        self.cross_attention = CrossAttention(self.proj_hidd_dim, num_heads=nhead)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.prompt_embeddings = nn.Parameter(torch.empty(1,
            self.prompt_len_per_layer*self.prompt_layers,
            self.proj_hidd_dim
        ))


        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))

        torch.nn.init.normal_(self.prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.props_pe,std=0.02)

        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)
  
    def forward(self, x):

        x = x + self.props_pe

        # Generate positive prompts： cross_attention(Q,K,V)
        generated_prompts = self.cross_attention(self.prompt_embeddings.expand(x.shape[0],-1,-1), x, x)
        prompts=self.proj(generated_prompts)

        if self.prompt_layers>1:
            return prompts.view(-1,self.prompt_layers,self.prompt_len_per_layer,self.embed_dim)
        
        return prompts


# cross attention-based positive/negative prompt generator
# initialize two sets of visual prompts refering to pos and neg
# before injecting visual attributes into visual prompts, we further apply orthogonalization on visual prompts.
class QKVPosNegPromptG(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(QKVPosNegPromptG, self).__init__()
        # pos_promptLen: length of positive prompts per layer
        # neg_promptLen: length of negative prompts per layer
        # pos_promptLayers: number of layers for positive prompts
        # neg_promptLayers: number of layers for negative prompts
        # projector to generate cooeficients
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.pos_prompt_len_per_layer = pos_promptLen
        self.neg_prompt_len_per_layer = neg_promptLen
        self.pos_prompt_layers = pos_promptLayers
        self.neg_prompt_layers = neg_promptLayers

        self.pos_cross_attention = CrossAttention(self.proj_hidd_dim, num_heads=nhead)
        self.neg_cross_attention = CrossAttention(self.proj_hidd_dim, num_heads=nhead)
        self.cross_attention = CrossAttention(self.proj_hidd_dim, num_heads=nhead)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.neg_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.neg_prompt_len_per_layer*self.neg_prompt_layers,
            self.proj_hidd_dim
        ))


        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))

        torch.nn.init.normal_(self.pos_prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.neg_prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.props_pe,std=0.02)

        self.orthogonalize_prompts()

        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)

    def orthogonalize_prompts(self):
        with torch.no_grad():
            # Reshape prompts to 2D for easier computation
            pos_2d = self.pos_prompt_embeddings.view(-1, self.pos_prompt_embeddings.size(-1))
            neg_2d = self.neg_prompt_embeddings.view(-1, self.neg_prompt_embeddings.size(-1))
            
            # Compute the dot product
            dot_product = torch.sum(pos_2d * neg_2d, dim=1, keepdim=True)
            
            # Compute the squared norm of the positive prompt
            pos_norm_sq = torch.sum(pos_2d ** 2, dim=1, keepdim=True)
            
            # Compute the projection of negative prompt onto positive prompt
            projection = (dot_product / pos_norm_sq) * pos_2d
            
            # Subtract the projection to make negative prompt orthogonal to positive prompt
            neg_2d -= projection
            
            # Optionally, normalize the prompts
            pos_2d = F.normalize(pos_2d, dim=-1)
            neg_2d = F.normalize(neg_2d, dim=-1)
            
            # Reshape back to original dimensions
            self.pos_prompt_embeddings.data = pos_2d.view(self.pos_prompt_embeddings.shape)
            self.neg_prompt_embeddings.data = neg_2d.view(self.neg_prompt_embeddings.shape)
  
    def forward(self, x):

        x = x + self.props_pe

        # Generate positive prompts： cross_attention(Q,K,V)
        generated_pos_prompts, self.pos_atten = self.cross_attention(self.pos_prompt_embeddings.expand(x.shape[0],-1,-1), x, x)
        # mark:
        # option 1: do nothing on negative prompts
        # option 2: generate negative prompts by cross_attention(Q,K,V)
        generated_neg_prompts, self.neg_atten = self.cross_attention(self.neg_prompt_embeddings.expand(x.shape[0],-1,-1), x, x)

        generated_posneg_prompts=torch.cat([generated_pos_prompts,generated_neg_prompts],dim=1)
        
        posneg_prompts=self.proj(generated_posneg_prompts)
        
        pos_prompts,neg_prompts=torch.split(posneg_prompts,[self.pos_prompt_len_per_layer*self.pos_prompt_layers,self.neg_prompt_len_per_layer*self.neg_prompt_layers],dim=1)   
        

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)
        
        return pos_prompts, neg_prompts


class QKVPosNegNormPromptG(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(QKVPosNegNormPromptG, self).__init__()
        # pos_promptLen: length of positive prompts per layer
        # neg_promptLen: length of negative prompts per layer
        # pos_promptLayers: number of layers for positive prompts
        # neg_promptLayers: number of layers for negative prompts
        # projector to generate cooeficients
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.pos_prompt_len_per_layer = pos_promptLen
        self.neg_prompt_len_per_layer = neg_promptLen
        self.pos_prompt_layers = pos_promptLayers
        self.neg_prompt_layers = neg_promptLayers

        # self.pos_cross_attention = CrossAttention(self.proj_hidd_dim, num_heads=nhead)
        # self.neg_cross_attention = CrossAttention(self.proj_hidd_dim, num_heads=nhead)
        self.cross_attention = CrossAttention(self.proj_hidd_dim, num_heads=nhead)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.neg_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.neg_prompt_len_per_layer*self.neg_prompt_layers,
            self.proj_hidd_dim
        ))


        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))

        torch.nn.init.normal_(self.pos_prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.neg_prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.props_pe,std=0.02)

        self.orthogonalize_prompts()

        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)

    def orthogonalize_prompts(self):
        with torch.no_grad():
            # Reshape prompts to 2D for easier computation
            pos_2d = self.pos_prompt_embeddings.view(-1, self.pos_prompt_embeddings.size(-1))
            neg_2d = self.neg_prompt_embeddings.view(-1, self.neg_prompt_embeddings.size(-1))
            
            # Compute the dot product
            dot_product = torch.sum(pos_2d * neg_2d, dim=1, keepdim=True)
            
            # Compute the squared norm of the positive prompt
            pos_norm_sq = torch.sum(pos_2d ** 2, dim=1, keepdim=True)
            
            # Compute the projection of negative prompt onto positive prompt
            projection = (dot_product / pos_norm_sq) * pos_2d
            
            # Subtract the projection to make negative prompt orthogonal to positive prompt
            neg_2d -= projection
            
            # Optionally, normalize the prompts
            pos_2d = F.normalize(pos_2d, dim=-1)
            neg_2d = F.normalize(neg_2d, dim=-1)
            
            # Reshape back to original dimensions
            self.pos_prompt_embeddings.data = pos_2d.view(self.pos_prompt_embeddings.shape)
            self.neg_prompt_embeddings.data = neg_2d.view(self.neg_prompt_embeddings.shape)
  
    def forward(self, x):

        x = x + self.props_pe

        # Generate positive prompts： cross_attention(Q,K,V)
        generated_pos_prompts, self.pos_atten = self.cross_attention(self.pos_prompt_embeddings.expand(x.shape[0],-1,-1), x, x)
        # mark:
        # option 1: do nothing on negative prompts
        # option 2: generate negative prompts by cross_attention(Q,K,V)
        generated_neg_prompts, self.neg_atten = self.cross_attention(self.neg_prompt_embeddings.expand(x.shape[0],-1,-1), x, x)

        pos_prompts=self.proj(generated_pos_prompts)
        neg_prompts=self.proj(generated_neg_prompts)

        pos_prompts=F.normalize(pos_prompts,dim=-1)
        neg_prompts=F.normalize(neg_prompts,dim=-1) 

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)
        
        return pos_prompts, neg_prompts
    

# cross attention-based positive/negative prompt generator
# initialize two sets of visual prompts refering to pos and neg
# after injecting visual attributes into visual prompts, we further apply orthogonalization on visual prompts.
class QKVPosNegOrthAfterPromptG(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(QKVPosNegOrthAfterPromptG, self).__init__()
        # pos_promptLen: length of positive prompts per layer
        # neg_promptLen: length of negative prompts per layer
        # pos_promptLayers: number of layers for positive prompts
        # neg_promptLayers: number of layers for negative prompts
        # projector to generate cooeficients
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.pos_prompt_len_per_layer = pos_promptLen
        self.neg_prompt_len_per_layer = neg_promptLen
        self.pos_prompt_layers = pos_promptLayers
        self.neg_prompt_layers = neg_promptLayers

        self.pos_cross_attention = CrossAttention(self.proj_hidd_dim, num_heads=nhead)
        self.neg_cross_attention = CrossAttention(self.proj_hidd_dim, num_heads=nhead)
        self.cross_attention = CrossAttention(self.proj_hidd_dim, num_heads=nhead)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.neg_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.neg_prompt_len_per_layer*self.neg_prompt_layers,
            self.proj_hidd_dim
        ))


        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))

        torch.nn.init.normal_(self.pos_prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.neg_prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.props_pe,std=0.02)

        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)

    def householder_orthogonalize(self, pos, neg):
        # Reshape for easier manipulation
        _, sequence_length, feat_dim = pos.size()

        # Compute the sum vector
        v = pos + neg
        
        # Compute the Householder vector
        v_norm = torch.norm(v, dim=2, keepdim=True)
        u = v / (v_norm + 1e-8)  # Add small epsilon for numerical stability
        
        # Compute the Householder matrix
        H_matrix = torch.eye(feat_dim, device=pos.device).unsqueeze(0).unsqueeze(0) - 2 * u.unsqueeze(3) * u.unsqueeze(2)
        # H_matrix shape: (B, S, D, D)

        # Apply the Householder transformation
        pos_ortho = torch.matmul(H_matrix, pos.unsqueeze(3)).squeeze(3)
        neg_ortho = torch.matmul(H_matrix, neg.unsqueeze(3)).squeeze(3)
        
        return pos_ortho, neg_ortho
  
    def forward(self, x):

        x = x + self.props_pe

        # Generate positive prompts： cross_attention(Q,K,V)
        generated_pos_prompts = self.cross_attention(self.pos_prompt_embeddings.expand(x.shape[0],-1,-1), x, x)
        # mark:
        # option 1: do nothing on negative prompts
        # option 2: generate negative prompts by cross_attention(Q,K,V)
        generated_neg_prompts = self.cross_attention(self.neg_prompt_embeddings.expand(x.shape[0],-1,-1), x, x)

        pos_prompts=self.proj(generated_pos_prompts)
        neg_prompts=self.proj(generated_neg_prompts)

        pos_prompts, neg_prompts = self.householder_orthogonalize(pos_prompts, neg_prompts)


        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)
        
        return pos_prompts, neg_prompts  

# cross attention-based positive/negative prompt generator
# inilize pos visual embeddings, and derive neg visual emebddings
class QKVOrthoPosNegPromptG(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu',ortho_prompt_after=0):
        super(QKVOrthoPosNegPromptG, self).__init__()
        # pos_promptLen: length of positive prompts per layer
        # neg_promptLen: length of negative prompts per layer
        # pos_promptLayers: number of layers for positive prompts
        # neg_promptLayers: number of layers for negative prompts
        # projector to generate cooeficients
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.pos_prompt_len_per_layer = pos_promptLen
        self.neg_prompt_len_per_layer = neg_promptLen
        self.pos_prompt_layers = pos_promptLayers
        self.neg_prompt_layers = neg_promptLayers
        self.ortho_prompt_after = ortho_prompt_after

        self.cross_attention = PosNegCrossAttention(self.proj_hidd_dim, num_heads=nhead)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))

        torch.nn.init.normal_(self.pos_prompt_embeddings, std=0.02)
    
        torch.nn.init.normal_(self.props_pe,std=0.02)

        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)

    def householder_orthogonalize(self, x, y, eps=1e-8):
        """
        Orthogonalize tensor y with respect to tensor x.

        Parameters:
        - x: Tensor of shape (batch_size, sequence_length, dim)
        - y: Tensor of shape (batch_size, sequence_length, dim)
        - eps: Small epsilon value to prevent division by zero

        Returns:
        - x_orth: Tensor x (unchanged)
        - y_orth: Tensor y orthogonalized with respect to x
        """
        # Compute the dot product between x and y along the last dimension
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)  # Shape: (batch_size, sequence_length, 1)
    
        # Compute the dot product of x with itself
        xx_dot = torch.sum(x * x, dim=-1, keepdim=True)  # Shape: (batch_size, sequence_length, 1)
    
        # Compute the projection scalar
        proj_scalar = xy_dot / (xx_dot + eps)  # Shape: (batch_size, sequence_length, 1)
    
        # Orthogonalize y with respect to x
        y_orth = y - proj_scalar * x
    
        return x, y_orth


    def forward(self, x):

        x = x + self.props_pe

        # Generate positive prompts： cross_attention(Q,K,V)
        generated_pos_prompts, generated_neg_prompts, self.pos_atten, self.neg_atten = self.cross_attention(self.pos_prompt_embeddings.expand(x.shape[0],-1,-1), x, x)
        # based on the pos_atten, generate negative prompts

        pos_prompts=self.proj(generated_pos_prompts)
        neg_prompts=self.proj(generated_neg_prompts)

        if self.ortho_prompt_after:
            pos_prompts, neg_prompts = self.householder_orthogonalize(pos_prompts, neg_prompts)


        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)
        
        return pos_prompts, neg_prompts  


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    


class ImagePromptEncoderDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, promptL, promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(ImagePromptEncoderDecoder, self).__init__()

        # projector to generate cooeficients
        self.layers=layers
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.prompt_len_per_layer = promptL
        self.prompt_layers = promptLayers

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.prompt_embeddings = nn.Parameter(torch.empty(1,
            self.prompt_len_per_layer*self.prompt_layers,
            self.proj_hidd_dim
        ))
   
        torch.nn.init.normal_(self.prompt_embeddings, std=0.02)


        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)

        self.patch_embed = PatchEmbed(img_size=224, patch_size=16,in_chans=6,embed_dim=self.proj_hidd_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.proj_hidd_dim))
        self.pos_drop = nn.Dropout(p=0.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

        
    def forward(self, x):
        tokenS = self.prepare_tokens(x)
        x = self.encoder(tokenS) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
        # fetch cls token for prediction
        prompts = self.decoder(self.prompt_embeddings.expand(x.shape[0],-1,-1),x)
        prompts = self.proj(prompts)
        if self.prompt_layers>1:
            return prompts.view(-1,self.prompt_layers,self.prompt_len_per_layer,self.embed_dim)
        return prompts



class GeneralPromptEncoderDecoder(nn.Module):
    def __init__(self, embed_dim, proj_hidd_dim, promptL, promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(GeneralPromptEncoderDecoder, self).__init__()

        # projector to generate cooeficients
        self.layers=layers
        self.embed_dim = embed_dim
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.prompt_len_per_layer = promptL
        self.prompt_layers = promptLayers

        self.prompt_embeddings = nn.Parameter(torch.empty(1,
            self.prompt_len_per_layer*self.prompt_layers,
            self.proj_hidd_dim
        ))
        
        torch.nn.init.normal_(self.prompt_embeddings, std=0.02)


        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)

        
    def forward(self, x):

        x = self.encoder(x) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
        # fetch cls token for prediction
        prompts = self.decoder(self.prompt_embeddings.expand(x.shape[0],-1,-1),x)
        prompts = self.proj(prompts)
        if self.prompt_layers>1:
            return prompts.view(-1,self.prompt_layers,self.prompt_len_per_layer,self.embed_dim)
        return prompts
    


class ImageEmbedding(nn.Module):
    def __init__(self,proj_hidd_dim,img_size=224, patch_size=16,in_chans=6,):
        super(ImageEmbedding, self).__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,in_chans=in_chans,embed_dim=proj_hidd_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, proj_hidd_dim))
        self.pos_drop = nn.Dropout(p=0.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)
    
    def forward(self,x):

        tokens = self.prepare_tokens(x)

        return tokens


class AutoWeakAugOrtho(nn.Module):
    def __init__(self, embedding_dim, pos_orth_weights=-1, neg_orth_weights=-1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_orth_weights = pos_orth_weights
        self.neg_orth_weights = neg_orth_weights
        if self.pos_orth_weights==-1:
            self.alpha = nn.Parameter(torch.tensor(0.1))  # Amplification strength
        else:
            self.alpha = pos_orth_weights
            
        if self.neg_orth_weights==-1:
            self.beta = nn.Parameter(torch.tensor(0.05))
        else:
            self.beta = neg_orth_weights

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, Feat, B, C):
        # F: (batch_size, sequence_length, embedding_dim)
        # B, C: (batch_size, sequence_length, embedding_dim)
        
        # Extract the first token from F
        F_first = Feat[:, 0:1, :]  # (batch_size, 1, embedding_dim)

        # Determine alpha and beta values
        if self.pos_orth_weights==-1:
            alpha = F.softplus(self.alpha)
        else:
            alpha = self.alpha
        if self.neg_orth_weights==-1:
            beta = F.softplus(self.beta)
        else:
            beta = self.beta

        # Compute cosine similarity
        F_norm = F_first / (F_first.norm(dim=-1, keepdim=True) + 1e-8)
        B_norm = B / (B.norm(dim=-1, keepdim=True) + 1e-8)
        C_norm = C / (C.norm(dim=-1, keepdim=True) + 1e-8)

        sim_FB = torch.bmm(F_norm, B_norm.transpose(1, 2))  # (batch_size, 1, sequence_length)
        sim_FC = torch.bmm(F_norm, C_norm.transpose(1, 2))  # (batch_size, 1, sequence_length)

        # Compute weights
        weights_B = F.softmax(sim_FB, dim=-1)
        weights_C = F.softmax(sim_FC, dim=-1)

        # Compute amplification and weakening terms
        amp = torch.bmm(weights_B, B)  # (batch_size, 1, embedding_dim)
        weak = torch.bmm(weights_C, C)  # (batch_size, 1, embedding_dim)

        # Update F_first
        F_new_first = F_first + alpha * amp - beta * weak

        # Apply layer normalization
        F_new_first = self.layer_norm(F_new_first)

        # Combine the enhanced first token with the rest of the sequence
        F_new = torch.cat([F_new_first, Feat[:, 1:, :]], dim=1)

        return F_new  

class AutoWeakAugOrthoV2(nn.Module):
    def __init__(self, embedding_dim, pos_orth_weights=-1, neg_orth_weights=-1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_orth_weights = pos_orth_weights
        self.neg_orth_weights = neg_orth_weights
        if self.pos_orth_weights==-1:
            self.alpha = nn.Parameter(torch.empty(12,))  
            torch.nn.init.normal_(self.alpha, std=0.02)
            # Amplification strength
        else:
            self.alpha = pos_orth_weights
            
        if self.neg_orth_weights==-1:
            self.beta = nn.Parameter(torch.empty(12,))
            torch.nn.init.normal_(self.beta, std=0.02)
        else:
            self.beta = neg_orth_weights
    def forward(self, A_sequence, B, C,layer_index):
        """
        Amplify A with B and weaken A with C based on their orthogonal components.
    
        Args:
        A_sequence (torch.Tensor): Shape (batch_size, sequence_length, feat_dim)
        B (torch.Tensor): Shape (batch_size, sequence_length, feat_dim)
        C (torch.Tensor): Shape (batch_size, sequence_length, feat_dim)
        amplify_factor (float): Factor to amplify A with B
        weaken_factor (float): Factor to weaken A with C
    
        Returns:
        torch.Tensor: Modified A_sequence with shape (batch_size, 1, feat_dim)
        """
        # Determine alpha and beta values
        if self.pos_orth_weights==-1:
            amplify_factor = F.softplus(self.alpha[layer_index])
        else:
            amplify_factor = self.alpha

        if self.neg_orth_weights==-1:
            weaken_factor = F.softplus(self.beta[layer_index])
        else:
            weaken_factor = self.beta

        batch_size, seq_len, feat_dim = B.shape
    
        # Reshape A to (batch_size, feat_dim) for easier computation
        A= A_sequence[:, 0:1, :]
        A_reshaped = A.squeeze(1)
    
        # Normalize A
        A_norm = torch.nn.functional.normalize(A_reshaped, dim=1)
    
        # Process B
        B_mean = B.mean(dim=1)  # Average over sequence_length
        B_norm = torch.nn.functional.normalize(B_mean, dim=1)
    
        # Compute cosine similarity between A and B
        cos_AB = torch.sum(A_norm * B_norm, dim=1, keepdim=True)
    
        # Compute rejection features (orthogonal component)
        rejection_AB = cos_AB * B_norm
    
        # Compute orthogonal component
        orthogonal_AB = A_norm - rejection_AB
    
        # Normalize orthogonal component
        orthogonal_AB_norm = torch.nn.functional.normalize(orthogonal_AB, dim=1)
    
        # Amplify A with B
        A_amplified = A_norm + amplify_factor * orthogonal_AB_norm
    
        # Process C
        C_mean = C.mean(dim=1)  # Average over sequence_length
        C_norm = torch.nn.functional.normalize(C_mean, dim=1)
    
        # Compute cosine similarity between A and C
        cos_AC = torch.sum(A_amplified * C_norm, dim=1, keepdim=True)
    
        # Compute rejection features (orthogonal component)
        rejection_AC = cos_AC * C_norm
    
        # Compute orthogonal component
        orthogonal_AC = A_amplified - rejection_AC
    
        # Normalize orthogonal component
        orthogonal_AC_norm = torch.nn.functional.normalize(orthogonal_AC, dim=1)
    
        # Weaken A with C
        A_final = A_amplified - weaken_factor * orthogonal_AC_norm
    
        # Normalize the final result
        A_final_norm = torch.nn.functional.normalize(A_final, dim=1)
    
        # Reshape back to (batch_size, 1, feat_dim) 
        A_new = torch.cat([A_final_norm.unsqueeze(1), A_sequence[:, 1:, :]], dim=1)
        
        return A_new



class PosNegPromptEncoderDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu',ortho_prompt_after=0):
        super(PosNegPromptEncoderDecoder, self).__init__()

        # pos_promptLen: length of positive prompts per layer
        # neg_promptLen: length of negative prompts per layer
        # pos_promptLayers: number of layers for positive prompts
        # neg_promptLayers: number of layers for negative prompts
        # projector to generate cooeficients
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.pos_prompt_len_per_layer = pos_promptLen
        self.neg_prompt_len_per_layer = neg_promptLen
        self.pos_prompt_layers = pos_promptLayers
        self.neg_prompt_layers = neg_promptLayers
        self.ortho_prompt_after = ortho_prompt_after
        self.nhead = nhead

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        torch.nn.init.normal_(self.prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.props_pe,std=0.02)

        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)
        
        self.maskG = MutuallyExclusiveGatedAttentionMask(self.proj_hidd_dim)

    def householder_orthogonalize(self, x, y, eps=1e-8):
        """
        Orthogonalize tensor y with respect to tensor x.

        Parameters:
        - x: Tensor of shape (batch_size, sequence_length, dim)
        - y: Tensor of shape (batch_size, sequence_length, dim)
        - eps: Small epsilon value to prevent division by zero

        Returns:
        - x_orth: Tensor x (unchanged)
        - y_orth: Tensor y orthogonalized with respect to x
        """
        # Compute the dot product between x and y along the last dimension
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)  # Shape: (batch_size, sequence_length, 1)
    
        # Compute the dot product of x with itself
        xx_dot = torch.sum(x * x, dim=-1, keepdim=True)  # Shape: (batch_size, sequence_length, 1)
    
        # Compute the projection scalar
        proj_scalar = xy_dot / (xx_dot + eps)  # Shape: (batch_size, sequence_length, 1)
    
        # Orthogonalize y with respect to x
        y_orth = y - proj_scalar * x
    
        return x, y_orth

    def hard_mask(self, y_soft):
        """
        Convert a soft mask to a hard mask.

        Parameters:
        - mask: Soft mask of shape (batch_size, sequence_length)

        Returns:
        - hard_mask: Hard mask of shape (batch_size, sequence_length)
        """
        # Straight-through estimator
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return_hard = y_hard - y_soft.detach() + y_soft

        return return_hard.bool().unbind(-1)
    
    def forward(self, x):

        x = x + self.props_pe
        x = self.encoder(x) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
       
        self.pos_mask, self.neg_mask = self.maskG(x)
        pos_mask_hard, neg_mask_hard = self.hard_mask(torch.stack((self.pos_mask,self.neg_mask),dim=-1))
        self.pos_mask_hard = pos_mask_hard.detach()
        self.neg_mask_hard = neg_mask_hard.detach()

        pos_x = torch.mul(x,self.pos_mask.unsqueeze(-1))
        neg_x = torch.mul(x,self.neg_mask.unsqueeze(-1))

        # Attention:
        # memory_mask and memory_key_padding_mask: it's a boolean tensor where True values indicate positions to be masked out (ignored in attention computation).
        # So, we use neg_mask to mask out negative prompts in positive prompts generation, and use pos_mask to mask out positive prompts in negative prompts generation.
        # memeory_mask and key_padding mask must be bool, otherwise, it will be added to attention weights.
        # memory_mask=self.neg_mask_hard.unsqueeze(1).unsqueeze(1).expand(-1,self.nhead,self.pos_prompt_len_per_layer*self.pos_prompt_layers,-1).reshape(-1,self.pos_prompt_len_per_layer*self.pos_prompt_layers,self.pos_mask.shape[-1]),
        pos_prompts = self.decoder(
                    self.prompt_embeddings.expand(x.shape[0],-1,-1),
                                   pos_x)
        self.pos_atten = self.decoder.layers[0]._mha_block_attn_weights

        neg_prompts = self.decoder(
                    self.prompt_embeddings.expand(x.shape[0],-1,-1),
                                   neg_x)
        
        self.neg_atten = self.decoder.layers[0]._mha_block_attn_weights

        pos_prompts = self.proj(pos_prompts)
        neg_prompts = self.proj(neg_prompts)

        if self.ortho_prompt_after:
            pos_prompts, neg_prompts = self.householder_orthogonalize(pos_prompts, neg_prompts)

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)

        return pos_prompts, neg_prompts 


class GlobalPosNegPromptEncoderDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu',ortho_prompt_after=0):
        super(GlobalPosNegPromptEncoderDecoder, self).__init__()

        # pos_promptLen: length of positive prompts per layer
        # neg_promptLen: length of negative prompts per layer
        # pos_promptLayers: number of layers for positive prompts
        # neg_promptLayers: number of layers for negative prompts
        # projector to generate cooeficients
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.pos_prompt_len_per_layer = pos_promptLen
        self.neg_prompt_len_per_layer = neg_promptLen
        self.pos_prompt_layers = pos_promptLayers
        self.neg_prompt_layers = neg_promptLayers
        self.ortho_prompt_after = ortho_prompt_after
        self.nhead = nhead

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        torch.nn.init.normal_(self.prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.props_pe,std=0.02)

        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)
        
        self.maskG = MutuallyExclusiveGatedAttentionGlobalMask(self.proj_hidd_dim,self.prompt_num_s + self.prompt_num_m)

    def householder_orthogonalize(self, x, y, eps=1e-8):
        """
        Orthogonalize tensor y with respect to tensor x.

        Parameters:
        - x: Tensor of shape (batch_size, sequence_length, dim)
        - y: Tensor of shape (batch_size, sequence_length, dim)
        - eps: Small epsilon value to prevent division by zero

        Returns:
        - x_orth: Tensor x (unchanged)
        - y_orth: Tensor y orthogonalized with respect to x
        """
        # Compute the dot product between x and y along the last dimension
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)  # Shape: (batch_size, sequence_length, 1)
    
        # Compute the dot product of x with itself
        xx_dot = torch.sum(x * x, dim=-1, keepdim=True)  # Shape: (batch_size, sequence_length, 1)
    
        # Compute the projection scalar
        proj_scalar = xy_dot / (xx_dot + eps)  # Shape: (batch_size, sequence_length, 1)
    
        # Orthogonalize y with respect to x
        y_orth = y - proj_scalar * x
    
        return x, y_orth

    def hard_mask(self, y_soft):
        """
        Convert a soft mask to a hard mask.

        Parameters:
        - mask: Soft mask of shape (batch_size, sequence_length)

        Returns:
        - hard_mask: Hard mask of shape (batch_size, sequence_length)
        """
        # Straight-through estimator
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return_hard = y_hard - y_soft.detach() + y_soft

        return return_hard.bool().unbind(-1)
    
    def forward(self, x):

        x = x + self.props_pe
        x = self.encoder(x) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
       
        self.pos_mask, self.neg_mask = self.maskG(x)
        pos_mask_hard, neg_mask_hard = self.hard_mask(torch.stack((self.pos_mask,self.neg_mask),dim=-1))
        self.pos_mask_hard = pos_mask_hard.detach()
        self.neg_mask_hard = neg_mask_hard.detach()

        pos_x = torch.mul(x,self.pos_mask.unsqueeze(-1).unsqueeze(0))
        neg_x = torch.mul(x,self.neg_mask.unsqueeze(-1).unsqueeze(0))

        # Attention:
        # memory_mask and memory_key_padding_mask: it's a boolean tensor where True values indicate positions to be masked out (ignored in attention computation).
        # So, we use neg_mask to mask out negative prompts in positive prompts generation, and use pos_mask to mask out positive prompts in negative prompts generation.
        # memeory_mask and key_padding mask must be bool, otherwise, it will be added to attention weights.
        # memory_mask=self.neg_mask_hard.unsqueeze(1).unsqueeze(1).expand(-1,self.nhead,self.pos_prompt_len_per_layer*self.pos_prompt_layers,-1).reshape(-1,self.pos_prompt_len_per_layer*self.pos_prompt_layers,self.pos_mask.shape[-1]),
        pos_prompts = self.decoder(
                    self.prompt_embeddings.expand(x.shape[0],-1,-1),
                                   pos_x)
        self.pos_atten = self.decoder.layers[0]._mha_block_attn_weights

        neg_prompts = self.decoder(
                    self.prompt_embeddings.expand(x.shape[0],-1,-1),
                                   neg_x)
        
        self.neg_atten = self.decoder.layers[0]._mha_block_attn_weights

        pos_prompts = self.proj(pos_prompts)
        neg_prompts = self.proj(neg_prompts)

        if self.ortho_prompt_after:
            pos_prompts, neg_prompts = self.householder_orthogonalize(pos_prompts, neg_prompts)

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)

        return pos_prompts, neg_prompts 


class GlobalBalancePosNegPromptEncoderDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu',ortho_prompt_after=0):
        super(GlobalBalancePosNegPromptEncoderDecoder, self).__init__()

        # pos_promptLen: length of positive prompts per layer
        # neg_promptLen: length of negative prompts per layer
        # pos_promptLayers: number of layers for positive prompts
        # neg_promptLayers: number of layers for negative prompts
        # projector to generate cooeficients
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.pos_prompt_len_per_layer = pos_promptLen
        self.neg_prompt_len_per_layer = neg_promptLen
        self.pos_prompt_layers = pos_promptLayers
        self.neg_prompt_layers = neg_promptLayers
        self.ortho_prompt_after = ortho_prompt_after
        self.nhead = nhead

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        torch.nn.init.normal_(self.prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.props_pe,std=0.02)

        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)
        
        self.maskG = MutuallyExclusiveGatedAttentionGlobalBalanceMask(self.proj_hidd_dim,self.prompt_num_s + self.prompt_num_m)

    def householder_orthogonalize(self, x, y, eps=1e-8):
        """
        Orthogonalize tensor y with respect to tensor x.

        Parameters:
        - x: Tensor of shape (batch_size, sequence_length, dim)
        - y: Tensor of shape (batch_size, sequence_length, dim)
        - eps: Small epsilon value to prevent division by zero

        Returns:
        - x_orth: Tensor x (unchanged)
        - y_orth: Tensor y orthogonalized with respect to x
        """
        # Compute the dot product between x and y along the last dimension
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)  # Shape: (batch_size, sequence_length, 1)
    
        # Compute the dot product of x with itself
        xx_dot = torch.sum(x * x, dim=-1, keepdim=True)  # Shape: (batch_size, sequence_length, 1)
    
        # Compute the projection scalar
        proj_scalar = xy_dot / (xx_dot + eps)  # Shape: (batch_size, sequence_length, 1)
    
        # Orthogonalize y with respect to x
        y_orth = y - proj_scalar * x
    
        return x, y_orth

    def hard_mask(self, y_soft):
        """
        Convert a soft mask to a hard mask.

        Parameters:
        - mask: Soft mask of shape (batch_size, sequence_length)

        Returns:
        - hard_mask: Hard mask of shape (batch_size, sequence_length)
        """
        # Straight-through estimator
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return_hard = y_hard - y_soft.detach() + y_soft

        return return_hard.bool().unbind(-1)
    
    def forward(self, x):

        x = x + self.props_pe
        x = self.encoder(x) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
       
        self.pos_mask, self.neg_mask = self.maskG(x)
        pos_mask_hard, neg_mask_hard = self.hard_mask(torch.stack((self.pos_mask,self.neg_mask),dim=-1))
        self.pos_mask_hard = pos_mask_hard.detach()
        self.neg_mask_hard = neg_mask_hard.detach()

        # print(torch.stack((self.pos_mask,self.neg_mask),dim=-1))
        pos_x = torch.mul(x,self.pos_mask.unsqueeze(-1).unsqueeze(0))
        neg_x = torch.mul(x,self.neg_mask.unsqueeze(-1).unsqueeze(0))

        # Attention:
        # memory_mask and memory_key_padding_mask: it's a boolean tensor where True values indicate positions to be masked out (ignored in attention computation).
        # So, we use neg_mask to mask out negative prompts in positive prompts generation, and use pos_mask to mask out positive prompts in negative prompts generation.
        # memeory_mask and key_padding mask must be bool, otherwise, it will be added to attention weights.
        # memory_mask=self.neg_mask_hard.unsqueeze(1).unsqueeze(1).expand(-1,self.nhead,self.pos_prompt_len_per_layer*self.pos_prompt_layers,-1).reshape(-1,self.pos_prompt_len_per_layer*self.pos_prompt_layers,self.pos_mask.shape[-1]),
        pos_prompts = self.decoder(
                    self.prompt_embeddings.expand(x.shape[0],-1,-1),
                                   pos_x,
                                   memory_key_padding_mask=self.neg_mask_hard.unsqueeze(0).expand(x.shape[0],-1))
        self.pos_atten = self.decoder.layers[0]._mha_block_attn_weights

        neg_prompts = self.decoder(
                    self.prompt_embeddings.expand(x.shape[0],-1,-1),
                                   neg_x,
                                   memory_key_padding_mask=self.pos_mask_hard.unsqueeze(0).expand(x.shape[0],-1))
        
        self.neg_atten = self.decoder.layers[0]._mha_block_attn_weights

        pos_prompts = self.proj(pos_prompts)
        neg_prompts = self.proj(neg_prompts)

        if self.ortho_prompt_after:
            pos_prompts, neg_prompts = self.householder_orthogonalize(pos_prompts, neg_prompts)

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)

        return pos_prompts, neg_prompts 