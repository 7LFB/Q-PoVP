import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
from pdb import set_trace as st


# ----- Orthogonal Prompt Generator -----
# distinguish by two cross-attention modules

class OrthogonalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, bias=True):
        super().__init__()
        self.dim=dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Orthogonal projections for Q, K, V
        self.q_proj = P.orthogonal(nn.Linear(dim, dim, bias=bias))
        self.k_proj = P.orthogonal(nn.Linear(dim, dim, bias=bias))
        self.v_proj = P.orthogonal(nn.Linear(dim, dim, bias=bias))
        
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = P.orthogonal(nn.Linear(dim, dim, bias=bias))

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size = query.size(0)
        
        # Orthogonal projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Compute output
        out = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.dim)
        out = self.out_proj(out)
        
        return out, attn


class PosNegOrthoPromptG(nn.Module):
    def __init__(
        self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu',ortho_prompt_after=0,dropout=0.1
    ):
        super().__init__()
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
        self.dropout=dropout
        
        # Initialize learnable prompt embeddings
        self.base_prompts = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))
        torch.nn.init.normal_(self.base_prompts, std=0.02)
        
        # Orthogonal projections for attributes
        self.attr_proj = nn.Linear(self.proj_hidd_dim, self.proj_hidd_dim)
        
        # Apply orthogonal constraints
        # self.attr_proj = P.orthogonal(self.attr_proj)
        
        # Orthogonal attention layers
        self.pos_attention = CrossAttention(
            self.proj_hidd_dim, self.nhead
        )
        self.neg_attention = CrossAttention(
            self.proj_hidd_dim, self.nhead
        )
       
        
        # Output projections with orthogonal constraints
        self.pos_output_proj = nn.Linear(self.proj_hidd_dim, self.proj_output_dim)
        self.neg_output_proj = nn.Linear(self.proj_hidd_dim, self.proj_output_dim)
        # self.pos_output_proj = P.orthogonal(self.pos_output_proj)
        # self.neg_output_proj = P.orthogonal(self.neg_output_proj)

    def forward(self, visual_attributes, attribute_masks=None):
        batch_size = visual_attributes.size(0)
        
        # Expand base prompts for batch processing
        prompts = self.base_prompts.expand(batch_size, -1, -1)
        
        # Project attributes with orthogonal constraints
        attr = self.attr_proj(visual_attributes)
        
        # Generate positive and negative prompts through attention
        pos_prompts, self.pos_atten = self.pos_attention(
            prompts, attr, attr
        )
        neg_prompts, self.neg_atten = self.neg_attention(
            prompts, attr, attr
        )
        
        # Final orthogonal projections
        pos_prompts = self.pos_output_proj(pos_prompts)
        neg_prompts = self.neg_output_proj(neg_prompts)

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)
        

        return pos_prompts, neg_prompts


#
# distinguish by two base prompts
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * \
            num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projections
        Q = self.query(query).view(
            batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads,
                               self.head_dim).transpose(1, 2)
        V = self.value(value).view(
            batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(
            1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out(attn_output)

        return output, attn_weights


class RawQKVPosNegPromptG(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(RawQKVPosNegPromptG, self).__init__()
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

        # self.orthogonalize_prompts()

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


        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)
        
        return pos_prompts, neg_prompts



# cross attention-based positive/negative prompt generator
# initialize two sets of visual prompts refering to pos and neg
# before injecting visual attributes into visual prompts, we further apply orthogonalization on visual prompts.
class QKVPosNegPromptG(nn.Module):
    def __init__(self, embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1, neg_promptLen=1, pos_promptLayers=12, neg_promptLayers=12, nhead=8, layers=1, activation='relu', ortho_prompt_after=0):
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


        self.cross_attention = CrossAttention(
            self.proj_hidd_dim, num_heads=nhead)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,self.pos_prompt_len_per_layer*self.pos_prompt_layers, self.proj_hidd_dim))
        torch.nn.init.orthogonal_(self.pos_prompt_embeddings)
        self.neg_prompt_embeddings = nn.Parameter(torch.empty(1,self.neg_prompt_len_per_layer*self.neg_prompt_layers, self.proj_hidd_dim))
        torch.nn.init.orthogonal_(self.neg_prompt_embeddings)

    
        self.props_pe = nn.Parameter(torch.empty(1,
                                                 self.prompt_num_s + self.prompt_num_m,
                                                 self.proj_hidd_dim
                                                 ))

        torch.nn.init.normal_(self.props_pe, std=0.02)

        self.output_proj = nn.Linear(
            self.proj_hidd_dim, self.embed_dim)


    def forward(self, x):

        x = x + self.props_pe

        # Generate positive prompts： cross_attention(Q,K,V)
        generated_pos_prompts, self.pos_atten = self.cross_attention(
            self.pos_prompt_embeddings.expand(x.shape[0], -1, -1), x, x)
        # mark:
        # option 1: do nothing on negative prompts
        # option 2: generate negative prompts by cross_attention(Q,K,V)
        generated_neg_prompts, self.neg_atten = self.cross_attention(
            self.neg_prompt_embeddings.expand(x.shape[0], -1, -1), x, x)

        pos_prompts = self.output_proj(generated_pos_prompts)
        neg_prompts = self.output_proj(generated_neg_prompts)

        if self.pos_prompt_layers > 1:
            pos_prompts = pos_prompts.view(-1, self.pos_prompt_layers,
                                           self.pos_prompt_len_per_layer, self.embed_dim)
        if self.neg_prompt_layers > 1:
            neg_prompts = neg_prompts.view(-1, self.neg_prompt_layers,
                                           self.neg_prompt_len_per_layer, self.embed_dim)

        return pos_prompts, neg_prompts



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

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))
        self.neg_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        # torch.nn.init.normal_(self.pos_prompt_embeddings, std=0.02)
        # torch.nn.init.normal_(self.neg_prompt_embeddings, std=0.02)
        torch.nn.init.orthogonal_(self.pos_prompt_embeddings)
        torch.nn.init.orthogonal_(self.neg_prompt_embeddings)
        torch.nn.init.normal_(self.props_pe,std=0.02)
        self.orthogonalize_prompts()

        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

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
        x = self.encoder(x) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
       
        # Attention:
        # memory_mask and memory_key_padding_mask: it's a boolean tensor where True values indicate positions to be masked out (ignored in attention computation).
        # So, we use neg_mask to mask out negative prompts in positive prompts generation, and use pos_mask to mask out positive prompts in negative prompts generation.
        # memeory_mask and key_padding mask must be bool, otherwise, it will be added to attention weights.
        # memory_mask=self.neg_mask_hard.unsqueeze(1).unsqueeze(1).expand(-1,self.nhead,self.pos_prompt_len_per_layer*self.pos_prompt_layers,-1).reshape(-1,self.pos_prompt_len_per_layer*self.pos_prompt_layers,self.pos_mask.shape[-1]),
        pos_prompts = self.decoder(
                    self.pos_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.pos_atten = self.decoder.layers[0]._mha_block_attn_weights


        neg_prompts = self.decoder(
                    self.neg_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.neg_atten = self.decoder.layers[0]._mha_block_attn_weights


        pos_prompts = self.proj(pos_prompts)
        neg_prompts = self.proj(neg_prompts)

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)

        return pos_prompts, neg_prompts 
    
    
class PosNegPromptDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu',ortho_prompt_after=0):
        super(PosNegPromptDecoder, self).__init__()

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

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))
        self.neg_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        # torch.nn.init.normal_(self.pos_prompt_embeddings, std=0.02)
        # torch.nn.init.normal_(self.neg_prompt_embeddings, std=0.02)
        torch.nn.init.orthogonal_(self.pos_prompt_embeddings)
        torch.nn.init.orthogonal_(self.neg_prompt_embeddings)
        torch.nn.init.normal_(self.props_pe,std=0.02)
        self.orthogonalize_prompts()

       
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

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
       
        # Attention:
        # memory_mask and memory_key_padding_mask: it's a boolean tensor where True values indicate positions to be masked out (ignored in attention computation).
        # So, we use neg_mask to mask out negative prompts in positive prompts generation, and use pos_mask to mask out positive prompts in negative prompts generation.
        # memeory_mask and key_padding mask must be bool, otherwise, it will be added to attention weights.
        # memory_mask=self.neg_mask_hard.unsqueeze(1).unsqueeze(1).expand(-1,self.nhead,self.pos_prompt_len_per_layer*self.pos_prompt_layers,-1).reshape(-1,self.pos_prompt_len_per_layer*self.pos_prompt_layers,self.pos_mask.shape[-1]),
        pos_prompts = self.decoder(
                    self.pos_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.pos_atten = self.decoder.layers[0]._mha_block_attn_weights


        neg_prompts = self.decoder(
                    self.neg_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.neg_atten = self.decoder.layers[0]._mha_block_attn_weights


        pos_prompts = self.proj(pos_prompts)
        neg_prompts = self.proj(neg_prompts)

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)

        return pos_prompts, neg_prompts 
    
class PosNegOrthoInitPromptDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu',ortho_prompt_after=0):
        super(PosNegOrthoInitPromptDecoder, self).__init__()

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

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))
        self.neg_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        # torch.nn.init.normal_(self.pos_prompt_embeddings, std=0.02)
        # torch.nn.init.normal_(self.neg_prompt_embeddings, std=0.02)
        torch.nn.init.orthogonal_(self.pos_prompt_embeddings)
        torch.nn.init.orthogonal_(self.neg_prompt_embeddings)
        torch.nn.init.normal_(self.props_pe,std=0.02)
        self.orthogonalize_prompts()

       
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)
        self.proj = P.orthogonal(self.proj)
        
 
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
       
        # Attention:
        # memory_mask and memory_key_padding_mask: it's a boolean tensor where True values indicate positions to be masked out (ignored in attention computation).
        # So, we use neg_mask to mask out negative prompts in positive prompts generation, and use pos_mask to mask out positive prompts in negative prompts generation.
        # memeory_mask and key_padding mask must be bool, otherwise, it will be added to attention weights.
        # memory_mask=self.neg_mask_hard.unsqueeze(1).unsqueeze(1).expand(-1,self.nhead,self.pos_prompt_len_per_layer*self.pos_prompt_layers,-1).reshape(-1,self.pos_prompt_len_per_layer*self.pos_prompt_layers,self.pos_mask.shape[-1]),
        pos_prompts = self.decoder(
                    self.pos_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.pos_atten = self.decoder.layers[0]._mha_block_attn_weights


        neg_prompts = self.decoder(
                    self.neg_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.neg_atten = self.decoder.layers[0]._mha_block_attn_weights


        pos_prompts = self.proj(pos_prompts)
        neg_prompts = self.proj(neg_prompts)

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)

        return pos_prompts, neg_prompts 
    

class PosNegNormalInitPromptDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu',ortho_prompt_after=0):
        super(PosNegNormalInitPromptDecoder, self).__init__()

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

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))
        self.neg_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        torch.nn.init.normal_(self.pos_prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.neg_prompt_embeddings, std=0.02)
        # torch.nn.init.orthogonal_(self.pos_prompt_embeddings)
        # torch.nn.init.orthogonal_(self.neg_prompt_embeddings)
        torch.nn.init.normal_(self.props_pe,std=0.02)
        self.orthogonalize_prompts()

       
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

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
       
        # Attention:
        # memory_mask and memory_key_padding_mask: it's a boolean tensor where True values indicate positions to be masked out (ignored in attention computation).
        # So, we use neg_mask to mask out negative prompts in positive prompts generation, and use pos_mask to mask out positive prompts in negative prompts generation.
        # memeory_mask and key_padding mask must be bool, otherwise, it will be added to attention weights.
        # memory_mask=self.neg_mask_hard.unsqueeze(1).unsqueeze(1).expand(-1,self.nhead,self.pos_prompt_len_per_layer*self.pos_prompt_layers,-1).reshape(-1,self.pos_prompt_len_per_layer*self.pos_prompt_layers,self.pos_mask.shape[-1]),
        pos_prompts = self.decoder(
                    self.pos_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.pos_atten = self.decoder.layers[0]._mha_block_attn_weights


        neg_prompts = self.decoder(
                    self.neg_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.neg_atten = self.decoder.layers[0]._mha_block_attn_weights


        pos_prompts = self.proj(pos_prompts)
        neg_prompts = self.proj(neg_prompts)

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)

        return pos_prompts, neg_prompts     

class PosNegPromptEncoderDualDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu',ortho_prompt_after=0):
        super(PosNegPromptEncoderDualDecoder, self).__init__()

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

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))
        self.neg_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        # torch.nn.init.normal_(self.pos_prompt_embeddings, std=0.02)
        # torch.nn.init.normal_(self.neg_prompt_embeddings, std=0.02)
        torch.nn.init.orthogonal_(self.pos_prompt_embeddings)
        torch.nn.init.orthogonal_(self.neg_prompt_embeddings)
        torch.nn.init.normal_(self.props_pe,std=0.02)
        self.orthogonalize_prompts()

        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

        self.pos_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        
        self.neg_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

        self.pos_proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)
        self.neg_proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)
        
 
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
        x = self.encoder(x) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
       
        # Attention:
        # memory_mask and memory_key_padding_mask: it's a boolean tensor where True values indicate positions to be masked out (ignored in attention computation).
        # So, we use neg_mask to mask out negative prompts in positive prompts generation, and use pos_mask to mask out positive prompts in negative prompts generation.
        # memeory_mask and key_padding mask must be bool, otherwise, it will be added to attention weights.
        # memory_mask=self.neg_mask_hard.unsqueeze(1).unsqueeze(1).expand(-1,self.nhead,self.pos_prompt_len_per_layer*self.pos_prompt_layers,-1).reshape(-1,self.pos_prompt_len_per_layer*self.pos_prompt_layers,self.pos_mask.shape[-1]),
        pos_prompts = self.pos_decoder(
                    self.pos_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.pos_atten = self.pos_decoder.layers[0]._mha_block_attn_weights


        neg_prompts = self.neg_decoder(
                    self.neg_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.neg_atten = self.neg_decoder.layers[0]._mha_block_attn_weights


        pos_prompts = self.pos_proj(pos_prompts)
        neg_prompts = self.neg_proj(neg_prompts)

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)

        return pos_prompts, neg_prompts 
    

class PosNegPromptDualDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, pos_promptLen=1,neg_promptLen=1, pos_promptLayers=12,neg_promptLayers=12,nhead=8,layers=1,activation='relu',ortho_prompt_after=0):
        super(PosNegPromptDualDecoder, self).__init__()

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

        self.pos_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))
        self.neg_prompt_embeddings = nn.Parameter(torch.empty(1,
            self.pos_prompt_len_per_layer*self.pos_prompt_layers,
            self.proj_hidd_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        # torch.nn.init.normal_(self.pos_prompt_embeddings, std=0.02)
        # torch.nn.init.normal_(self.neg_prompt_embeddings, std=0.02)
        torch.nn.init.orthogonal_(self.pos_prompt_embeddings)
        torch.nn.init.orthogonal_(self.neg_prompt_embeddings)
        torch.nn.init.normal_(self.props_pe,std=0.02)
        self.orthogonalize_prompts()

        # Attention: batch_first=True

        self.pos_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        
        self.neg_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)

        self.pos_proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)
        self.neg_proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)
        
 
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
        # Attention:
        # memory_mask and memory_key_padding_mask: it's a boolean tensor where True values indicate positions to be masked out (ignored in attention computation).
        # So, we use neg_mask to mask out negative prompts in positive prompts generation, and use pos_mask to mask out positive prompts in negative prompts generation.
        # memeory_mask and key_padding mask must be bool, otherwise, it will be added to attention weights.
        # memory_mask=self.neg_mask_hard.unsqueeze(1).unsqueeze(1).expand(-1,self.nhead,self.pos_prompt_len_per_layer*self.pos_prompt_layers,-1).reshape(-1,self.pos_prompt_len_per_layer*self.pos_prompt_layers,self.pos_mask.shape[-1]),
        pos_prompts = self.pos_decoder(
                    self.pos_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.pos_atten = self.pos_decoder.layers[0]._mha_block_attn_weights


        neg_prompts = self.neg_decoder(
                    self.neg_prompt_embeddings.expand(x.shape[0],-1,-1),
                                   x)
        self.neg_atten = self.neg_decoder.layers[0]._mha_block_attn_weights


        pos_prompts = self.pos_proj(pos_prompts)
        neg_prompts = self.neg_proj(neg_prompts)

        if self.pos_prompt_layers>1:
            pos_prompts=pos_prompts.view(-1,self.pos_prompt_layers,self.pos_prompt_len_per_layer,self.embed_dim)
        if self.neg_prompt_layers>1:
            neg_prompts=neg_prompts.view(-1,self.neg_prompt_layers,self.neg_prompt_len_per_layer,self.embed_dim)

        return pos_prompts, neg_prompts 