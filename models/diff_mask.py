import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as st

class GumbelSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, tau=1.0, training=False, dim=-1, eps=1e-10):
        ctx.tau = tau
        ctx.training = training
        ctx.dim = dim
        ctx.eps = eps
        
        if training:
            # Sample from Gumbel(0, 1) during training
            u = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
            
            # Gumbel-Softmax trick
            y_soft = F.softmax((logits + gumbel_noise) / tau, dim=dim)
        else:
            # During testing, just use regular softmax
            y_soft = F.softmax(logits, dim=dim)
            
        
        # Straight-through estimator
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return_hard = y_hard - y_soft.detach() + y_soft
        return_soft = y_soft
        
        if training:
            ret = return_soft
        else:
            ret = return_hard
        
        ctx.save_for_backward(ret)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * y * (1 - y) / ctx.tau
        
        return grad_input, None, None, None, None, None



class MutuallyExclusiveGatedAttentionMask(nn.Module):
    def __init__(self, d_model,sigma=0.5):
        super(MutuallyExclusiveGatedAttentionMask, self).__init__()
        self.gate = nn.Linear(d_model, 2, bias=False)  # Output logits for two masks without bias
        self.gumbel_softmax = GumbelSoftmax.apply
        self.temperature = sigma

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        logits = self.gate(x)  # (batch_size, seq_len, 2)
        gate_scores = self.gumbel_softmax(logits, self.temperature, self.training)
        return gate_scores.unbind(-1)  # Returns a tuple of two tensors


class MutuallyExclusiveGatedAttentionGlobalMask(nn.Module):
    def __init__(self, d_model, seq_len, sigma=0.5):
        super(MutuallyExclusiveGatedAttentionGlobalMask, self).__init__()
        self.gate = nn.Linear(d_model, 2, bias=False)  # Output logits for two masks without bias
        self.global_gate_score = nn.Parameter(torch.randn(seq_len, 2))  # Learnable parameter for global gate score
        self.smoothing_factor = nn.Parameter(torch.tensor(0.9))  # Learnable smoothing factor
        self.gumbel_softmax = GumbelSoftmax.apply
        self.temperature = sigma

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        logits = self.gate(x)  # (batch_size, seq_len, 2)
        
        if self.training:
            # Update global gate score using batch samples with EMA
            batch_gate_score = logits.mean(dim=0)  # (seq_len, 2)
            self.global_gate_score.data = self.smoothing_factor * self.global_gate_score.data + (1 - self.smoothing_factor) * batch_gate_score.data

        # Apply Gumbel-Softmax
        gate_scores = self.gumbel_softmax(self.global_gate_score, self.temperature, self.training)

        
        return gate_scores.unbind(-1)  # Returns a tuple of two tensors


class MutuallyExclusiveGatedAttentionGlobalBalanceMask(nn.Module):
    def __init__(self, d_model, seq_len, sigma=0.5, smoothing_factor=0.9):
        super(MutuallyExclusiveGatedAttentionGlobalBalanceMask, self).__init__()
        self.gate = nn.Linear(d_model, 2, bias=False)
        # Initialize global gate score to be balanced (close to 0.5, 0.5)
        self.register_buffer('global_gate_score', torch.zeros(seq_len, 2))
        self.global_gate_score[:, 0] = 0.5
        self.global_gate_score[:, 1] = 0.5
        self.smoothing_factor = smoothing_factor
        self.gumbel_softmax = GumbelSoftmax.apply
        self.temperature = sigma

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        logits = self.gate(x)  # (batch_size, seq_len, 2)
        
        if self.training:
            # Update global gate score using batch samples with EMA
            with torch.no_grad():
                batch_gate_score = torch.softmax(logits, dim=-1).mean(dim=0)  # (seq_len, 2)
                self.global_gate_score = (
                    self.smoothing_factor * self.global_gate_score + 
                    (1 - self.smoothing_factor) * batch_gate_score
                )
        
        print('global_gate_score', self.global_gate_score)
        # Apply Gumbel-Softmax
        gate_scores = self.gumbel_softmax(self.global_gate_score, self.temperature, self.training)
        print('gate_scores', gate_scores)
        print('---'*10)
        
        # Check for consistent bias
        diff = gate_scores[:, 0] - gate_scores[:, 1]
        if torch.all(diff > 0) or torch.all(diff < 0):
            # Randomly select one position to swap
            pos = torch.randint(0, gate_scores.size(0), (1,))
            gate_scores[pos] = torch.flip(gate_scores[pos], [-1])
            # Also swap the global gate score at the same position to prevent bias
            with torch.no_grad():
                self.global_gate_score[pos] = torch.flip(self.global_gate_score[pos], [-1])

        return gate_scores.unbind(-1)  # Returns a tuple of two tensors


