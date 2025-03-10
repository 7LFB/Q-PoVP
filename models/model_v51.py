from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout

from .vision_transformer import *
from .operators import *
from .promptEncoders import *

from pdb import set_trace as st

import timm



class XPrompt(nn.Module):
  
    def __init__(self,args):
        super(XPrompt, self).__init__()

        self.args = args
        self.vit = vit_small(patch_size=args.patch_size,pretrained_weights=args.pretrained_weights)
        self.classifier= nn.Linear(args.hidden_size, args.num_classes)

        self.proj = nn.Linear(self.args.sample_number,self.args.proj_hidd_dim)
        self.promptG=QKVPosNegPromptG(self.args.embed_num,self.args.hidden_size,self.args.prompt_s_num,self.args.prompt_m_num,self.args.proj_hidd_dim,pos_promptLen=args.pos_prompt_token_length,neg_promptLen=args.neg_prompt_token_length,nhead=args.nhead,layers=args.layers,activation=args.activation)
       
        self.prompt_dropout = Dropout(args.prompt_dropout)
        self.last_selfattn_weights=None

        # if project the prompt embeddings
        if self.args.prompt_project > -1:
            # only for prepend / add
            prompt_dim = self.args.prompt_project
            self.prompt_proj = nn.Linear(
                prompt_dim, args.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = args.hidden_size
            self.prompt_proj = nn.Identity()

    def orthogonalize(self, x, visual_prompt, operator='add', weights=1.0):
        # x shape: (batch_size, sequence_length, embedding_dim)
        # visual_prompt shape: (batch_size, sequence_length, embedding_dim)
        
        # Extract the first token from x
        x_first_token = x[:, 0, :].unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)
        
        # Compute the dot product between x_first_token and visual_prompt
        # Result shape: (batch_size, 1, sequence_length)
        dot_product = torch.bmm(x_first_token, visual_prompt.transpose(1, 2))
        
        # Compute the norm of visual_prompt
        # Result shape: (batch_size, sequence_length, 1)
        negative_norm_sq = torch.sum(visual_prompt ** 2, dim=-1, keepdim=True)
        
        # Compute the projection coefficients
        # Result shape: (batch_size, 1, sequence_length)
        proj_coeff = dot_product / negative_norm_sq.transpose(1, 2)
        
        # Compute the projection and subtract from x_first_token
        # Result shape: (batch_size, 1, embedding_dim)
        projection = torch.bmm(proj_coeff, visual_prompt)

        if operator == 'add':
            x_first_token_orthogonalized = x_first_token + weights*projection
        elif operator == 'minus':
            x_first_token_orthogonalized = x_first_token - weights*projection
        
        # Replace the first token in x with the orthogonalized version
        x_filtered = torch.cat([x_first_token_orthogonalized, x[:, 1:, :]], dim=1)
        
        return x_filtered
       
    
    def forward_deep_prompt(self, x, pos_prompts=None, neg_prompts=None):
        hidden_states = None
        pos_deep_prompt_embeddings = pos_prompts
        neg_deep_prompt_embeddings = neg_prompts
        num_layers = self.args.vit_num_layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.vit.blocks[i](x)
            else:
                pos_deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                    pos_deep_prompt_embeddings[:,i]))
                neg_deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                    neg_deep_prompt_embeddings[:,i]))

                # filter negative prompt from cls token
                hidden_states=self.orthogonalize(hidden_states,neg_deep_prompt_emb,operator='minus',weights=self.args.neg_ortho_weights)
                # enhance potive prompt to cls token
                hidden_states=self.orthogonalize(hidden_states,pos_deep_prompt_emb,operator='add',weights=self.args.pos_ortho_weights)
               
                hidden_states, self.last_selfattn_weights = self.vit.blocks[i](hidden_states,return_attention=True)

        return hidden_states
    
    def forward(self, x, prompt1=None, prompt2=None, prompt3=None):
        nTiles,c,h,w=x.shape
        image = x[:,:3,:,:]
        self.emb_Q=self.proj(prompt1)
        self.pos_deep_prompt_embeddings, self.neg_deep_prompt_embeddings = self.promptG(self.emb_Q)
        
        tokenS_ = self.vit.prepare_tokens(image)

        # layer-0's prompt processing
        tokenS = self.orthogonalize(tokenS_,self.neg_deep_prompt_embeddings[:,0],operator='minus',weights=self.args.neg_ortho_weights)
        tokenS = self.orthogonalize(tokenS,self.pos_deep_prompt_embeddings[:,0],operator='add',weights=self.args.pos_ortho_weights)

        # layer>=1's prompt processing
        if self.args.prompt_deep:
            tokenS=self.forward_deep_prompt(tokenS,self.pos_deep_prompt_embeddings,self.neg_deep_prompt_embeddings)
        else:
            tokenS = self.vit.encoder(tokenS)

        tokenS = self.vit.norm(tokenS)
        hidden = tokenS[:,0]
        logits = self.classifier(hidden)

        return logits
    




      



