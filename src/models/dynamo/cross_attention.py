import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        # PyTorch native MHA handles W_Q, W_K, W_V internally
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # FFN sublayer
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, H_star: torch.Tensor, H_geom: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_star: Pooled dynamic ensemble features (B, N, 256) -> Acts as Key, Value
            H_geom: Membrane geometry features (B, N, 256) -> Acts as Query
        """
        # Q = H_geom, K = H_star, V = H_star
        attn_out, _ = self.mha(query=H_geom, key=H_star, value=H_star)
        
        # Residual + LayerNorm
        H_fused = self.layer_norm(attn_out + H_geom)
        
        # FFN with residual
        H_fused = H_fused + self.ffn(H_fused)
        
        return H_fused
