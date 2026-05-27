import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Queries (B, H, N, d_k)
            k: Keys (B, H, M, d_k)
            v: Values (B, H, M, d_v)
            mask: Optional mask (B, 1, N, M)
        """
        d_k = q.size(-1)
        # scores: (B, H, N, M)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Mask should be boolean, where True means 'keep' and False means 'mask out'
            scores = scores.masked_fill(~mask, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # output: (B, H, N, d_v)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        B = query.size(0)
        
        # Linear projections & reshape to (B, H, SeqLen, d_k)
        q = self.w_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attn_weights = self.attention(q, k, v, mask=mask)
        
        # Concatenate heads & project
        # x: (B, SeqLen, H * d_k) -> (B, SeqLen, d_model)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        output = self.w_o(x)
        
        if return_attn:
            return output, attn_weights
        return output
