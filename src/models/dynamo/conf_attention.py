import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # RMSF projection and temperature scaling
        self.rmsf_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus() # tau > 0
        )
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, H_t: torch.Tensor, H_ref: torch.Tensor, rmsf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_t: Snapshot representations (B, T, N, 256)
            H_ref: Reference representation (B, N, 256)
            rmsf: RMSF per residue (B, N, 1)
        Returns:
            H_star: Pooled ensemble representation (B, N, 256)
        """
        # Embed RMSF and compute temperature
        rmsf_emb = self.rmsf_mlp(rmsf) # (B, N, 256)
        tau = self.tau_mlp(rmsf)       # (B, N, 256)
        
        # Query construction: q_r = W_Q * [H_ref || rmsf_emb]
        q_r = self.W_q(torch.cat([H_ref, rmsf_emb], dim=-1)) # (B, N, 256)
        
        # Key and Value from snapshots
        k_t = self.W_k(H_t) # (B, T, N, 256)
        v_t = self.W_v(H_t) # (B, T, N, 256)
        
        # Attention weight calculation
        q_r_exp = q_r.unsqueeze(1) # (B, 1, N, 256)
        tau_exp = tau.unsqueeze(1) # (B, 1, N, 256)
        
        # e_t = (q_r * k_t) / tau_r
        e_t = (q_r_exp * k_t).sum(dim=-1, keepdim=True) / (tau_exp + 1e-6) # (B, T, N, 1)
        a_t = F.softmax(e_t, dim=1) # Softmax over T dimension
        
        # Ensemble pool: H* = Sum_t(a_t * v_t)
        H_pool = (a_t * v_t).sum(dim=1) # (B, N, 256)
        
        # Residual and LayerNorm
        H_star = self.layer_norm(H_pool + H_ref)
        
        return H_star
