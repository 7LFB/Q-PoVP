import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
from pdb import set_trace as st

class EfficientBatchProcessor(nn.Module):
    def __init__(self, embedding_dim, preserve_ratio=0.7):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.preserve_ratio = preserve_ratio
        self.direction_weight = nn.Parameter(torch.ones(1))
        
    def batch_gram_schmidt(self, v1, v2):
        """
        Batch-wise Gram-Schmidt orthogonalization using tensor operations
        Args:
            v1, v2: [batch_size, embedding_dim]
        """
        # Compute projections for entire batch at once
        v1_norm = F.normalize(v1, p=2, dim=-1)
        dot_products = torch.sum(v1_norm * v2, dim=-1, keepdim=True)  # [batch_size, 1]
        v2_orthogonal = v2 - dot_products * v1_norm
        v2_orthogonal = F.normalize(v2_orthogonal, p=2, dim=-1)
        return v2_orthogonal
    
    def forward(self, x, pos_prompt, neg_prompt):
        """
        Efficient batch processing using tensor operations
        Args:
            x: [batch_size, seq_len, embedding_dim]
            pos_prompt: [batch_size, 1, embedding_dim]
            neg_prompt: [batch_size, 1, embedding_dim]
        """
        batch_size = x.shape[0]
        
        # Extract first tokens and prompts
        first_tokens = x[:, 0]  # [batch_size, embedding_dim]
        pos_prompt = pos_prompt.squeeze(1)  # [batch_size, embedding_dim]
        neg_prompt = neg_prompt.squeeze(1)  # [batch_size, embedding_dim]
        
        # 1. Compute token magnitudes for scale preservation
        token_magnitudes = torch.norm(first_tokens, dim=-1, keepdim=True)  # [batch_size, 1]
        
        # 2. Normalize all inputs simultaneously
        tokens_norm = F.normalize(first_tokens, p=2, dim=-1)
        pos_norm = F.normalize(pos_prompt, p=2, dim=-1)
        neg_norm = F.normalize(neg_prompt, p=2, dim=-1)
        
        # 3. Create orthogonal directions for entire batch
        pos_direction = self.batch_gram_schmidt(tokens_norm, pos_norm)
        neg_direction = self.batch_gram_schmidt(tokens_norm, neg_norm)
        
        # 4. Compute projections for entire batch
        def batch_scaled_projection(v, directions):
            """Compute scaled projections for entire batch"""
            cos_sims = torch.sum(v * directions, dim=-1, keepdim=True)  # [batch_size, 1]
            return cos_sims * directions
        
        # 5. Compute modifications
        pos_components = batch_scaled_projection(tokens_norm, pos_direction)
        neg_components = batch_scaled_projection(tokens_norm, neg_direction)
        
        # 6. Apply modifications with scale preservation
        modifications = self.direction_weight * (pos_components - neg_components)
        modified = first_tokens + token_magnitudes * modifications
        
        # 7. Scale-preserving interpolation using batch operations
        cosine_sims = torch.sum(first_tokens * modified, dim=-1, keepdim=True) / \
                     (token_magnitudes * torch.norm(modified, dim=-1, keepdim=True))
        alpha = torch.sigmoid(cosine_sims)
        
        # 8. Final combination with scale preservation
        final_tokens = alpha * modified + (1 - alpha) * first_tokens
        final_tokens = final_tokens * (token_magnitudes / torch.norm(final_tokens, dim=-1, keepdim=True))
        
        # 9. Combine with rest of sequence
        output = torch.cat([
            final_tokens.unsqueeze(1),  # [batch_size, 1, embedding_dim]
            x[:, 1:, :]  # Rest of sequence
        ], dim=1)
        
        return output


class ContrastiveTokenProcessor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Learnable temperature for softmax scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        # Direction importance weights
        self.pos_weight = nn.Parameter(torch.ones(1))
        self.neg_weight = nn.Parameter(torch.ones(1))
        
    def compute_info_preserving_attention(self, query, key, value):
        """
        Compute attention scores while preserving mutual information
        Args:
            query: [batch_size, embedding_dim]
            key: [batch_size, embedding_dim]
            value: [batch_size, embedding_dim]
        """
        # Scaled dot-product attention with learned temperature
        attn = torch.sum(query * key, dim=-1, keepdim=True) / self.temperature
        attn_weights = torch.sigmoid(attn)
        
        return attn_weights * value
    
    def forward(self, x, pos_prompt, neg_prompt):
        """
        Process first token to amplify positive and exclude negative while preserving information
        Args:
            x: [batch_size, seq_len, embedding_dim]
            pos_prompt: [batch_size, 1, embedding_dim]
            neg_prompt: [batch_size, 1, embedding_dim]
        """
        # Extract first tokens
        first_tokens = x[:, 0]  # [batch_size, embedding_dim]
        pos_prompt = pos_prompt.squeeze(1)  # [batch_size, embedding_dim]
        neg_prompt = neg_prompt.squeeze(1)  # [batch_size, embedding_dim]
        
        # Preserve original magnitudes
        token_norms = torch.norm(first_tokens, dim=-1, keepdim=True)
        
        # Normalize inputs for stable computation
        first_tokens_norm = F.normalize(first_tokens, p=2, dim=-1)
        pos_prompt_norm = F.normalize(pos_prompt, p=2, dim=-1)
        neg_prompt_norm = F.normalize(neg_prompt, p=2, dim=-1)
        
        # 1. Compute orthogonal basis for positive and negative spaces
        def get_orthogonal_space(v, reference):
            # Project out the reference direction
            proj = torch.sum(v * reference, dim=-1, keepdim=True) * reference
            orthogonal = v - proj
            return F.normalize(orthogonal, p=2, dim=-1)
        
        pos_orthogonal = get_orthogonal_space(pos_prompt_norm, neg_prompt_norm)
        
        # 2. Information-preserving positive enhancement
        pos_enhancement = self.compute_info_preserving_attention(
            first_tokens_norm, pos_orthogonal, pos_orthogonal
        )
        
        # 3. Negative removal through contrast
        neg_similarity = torch.sum(first_tokens_norm * neg_prompt_norm, dim=-1, keepdim=True)
        neg_mask = (1.0 - torch.sigmoid(neg_similarity * 5.0))  # Soft mask for negative removal
        
        # 4. Combine modifications with adaptive weighting
        modified = first_tokens_norm + \
                  self.pos_weight * pos_enhancement - \
                  self.neg_weight * neg_mask * neg_prompt_norm
        
        # 5. Preserve mutual information through residual connection
        info_gate = torch.sigmoid(
            torch.sum(first_tokens_norm * modified, dim=-1, keepdim=True)
        )
        
        # 6. Final combination with scale preservation
        final_tokens = info_gate * modified + (1 - info_gate) * first_tokens_norm
        final_tokens = final_tokens * token_norms
        
        # Combine with rest of sequence
        output = torch.cat([
            final_tokens.unsqueeze(1),
            x[:, 1:, :]
        ], dim=1)
        
        return output


class ContrastiveInfoMaxProcessor(nn.Module):
    def __init__(self, embedding_dim, temperature=0.07):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
        # Projection networks
        self.token_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.prompt_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def info_nce_loss(self, query, positive, negative):
        """
        Compute InfoNCE-style contrastive loss
        """
        # Project features to unit sphere
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        
        # Compute logits
        pos_logits = torch.sum(query * positive, dim=-1, keepdim=True) / self.temperature
        neg_logits = torch.sum(query * negative, dim=-1, keepdim=True) / self.temperature
        
        # Compute probabilities
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        probs = torch.softmax(logits, dim=-1)
        
        return probs[:, 0:1]  # Return positive probability
    
    def forward(self, x, pos_prompt, neg_prompt):
        """
        Process tokens using contrastive info maximization
        """
        # Extract first tokens
        first_tokens = x[:, 0]
        pos_prompt = pos_prompt.squeeze(1)
        neg_prompt = neg_prompt.squeeze(1)
        
        # Preserve original magnitudes
        token_norms = torch.norm(first_tokens, dim=-1, keepdim=True)
        
        # Project features
        tokens_proj = self.token_proj(first_tokens)
        pos_proj = self.prompt_proj(pos_prompt)
        neg_proj = self.prompt_proj(neg_prompt)
        
        # Compute contrastive weights
        pos_weights = self.info_nce_loss(tokens_proj, pos_proj, neg_proj)
        
        # Modify tokens
        modified = first_tokens + \
                  pos_weights * pos_prompt - \
                  (1 - pos_weights) * neg_prompt
        
        # Scale preservation
        modified = modified * (token_norms / torch.norm(modified, dim=-1, keepdim=True))
        
        # Combine with sequence
        output = torch.cat([
            modified.unsqueeze(1),
            x[:, 1:, :]
        ], dim=1)
        
        return output



class AmplifyWeaken:
    def __init__(self, embedding_dim, hidden_dim=128,num_heads=8):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = Autoencoder(embedding_dim, hidden_dim).to(self.device)

    def amplify_and_weaken_autoencoder(self, x, pos_visual_prompts, neg_visual_prompts, pos_weights=1.0, neg_weights=1.0):
        # x shape: (batch_size, sequence_length, embedding_dim)
        # visual_prompts shape: (batch_size, sequence_length, embedding_dim)
        
        x_first_token = x[:, 0, :].unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)
        
        # Encode and decode the first token
        encoded_pos = self.autoencoder.encoder(pos_visual_prompts)
        encoded_neg = self.autoencoder.encoder(neg_visual_prompts)
        
        # Amplify and weaken in the latent space
        encoded_x = self.autoencoder.encoder(x_first_token)
        amplified = encoded_x + pos_weights * encoded_pos.mean(dim=1, keepdim=True)
        weakened = encoded_x - neg_weights * encoded_neg.mean(dim=1, keepdim=True)
        
        # Decode back to original space
        x_first_token_orthogonalized = self.autoencoder.decoder(amplified - weakened)

        # Replace the first token in x with the modified one
        x_filtered = torch.cat([x_first_token_orthogonalized, x[:, 1:, :]], dim=1)
        
        return x_filtered

class Autoencoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(embedding_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

  