import torch
import torch.nn as nn
import math

class InvariantPointAttention(nn.Module):
    """
    Simplified Invariant Point Attention (IPA).
    Updates scalar features (c) based on spatial proximity of frames (R, t).
    """
    def __init__(self, c_dim: int, num_heads: int, num_scalar_qk: int, num_point_qk: int, num_point_v: int):
        super().__init__()
        self.c_dim = c_dim
        self.num_heads = num_heads
        self.h_dim = c_dim // num_heads
        
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        
        # Scalar Queries, Keys, Values
        self.w_q = nn.Linear(c_dim, num_heads * self.h_dim)
        self.w_k = nn.Linear(c_dim, num_heads * self.h_dim)
        self.w_v = nn.Linear(c_dim, num_heads * self.h_dim)
        
        # Point Queries, Keys, Values (generated in local frame)
        self.q_pts = nn.Linear(c_dim, num_heads * num_point_qk * 3)
        self.k_pts = nn.Linear(c_dim, num_heads * num_point_qk * 3)
        self.v_pts = nn.Linear(c_dim, num_heads * num_point_v * 3)
        
        # Attention spatial weight scale (learned per head)
        self.gamma = nn.Parameter(torch.ones(num_heads))
        self.softplus = nn.Softplus()
        
        # Final output projection
        self.out_proj = nn.Linear(
            num_heads * self.h_dim + num_heads * num_point_v * 3, 
            c_dim
        )

    def _transform_points_global(self, pts_local, R, t):
        """ Transforms points from local frame to global frame. """
        # pts_local: (B, N, H, P, 3)
        # R: (B, N, 3, 3), t: (B, N, 3)
        R_expanded = R.view(R.size(0), R.size(1), 1, 1, 3, 3)
        t_expanded = t.view(t.size(0), t.size(1), 1, 1, 3)
        pts_global = torch.einsum('bnhpij,bnhpj->bnhpi', R_expanded, pts_local) + t_expanded
        return pts_global

    def _transform_points_local(self, pts_global, R, t):
        """ Transforms points from global frame back to local frame. """
        R_expanded = R.view(R.size(0), R.size(1), 1, 1, 3, 3)
        t_expanded = t.view(t.size(0), t.size(1), 1, 1, 3)
        pts_centered = pts_global - t_expanded
        # Inverse of rotation matrix is its transpose
        pts_local = torch.einsum('bnhpji,bnhpj->bnhpi', R_expanded, pts_centered)
        return pts_local

    def forward(self, c, R, t):
        """
        Args:
            c: Scalar condition/features (B, N, c_dim)
            R: Rotation matrices (B, N, 3, 3)
            t: Translation vectors (B, N, 3)
        """
        B, N, _ = c.shape
        H = self.num_heads
        
        # 1. Scalar Q, K, V
        q = self.w_q(c).view(B, N, H, self.h_dim)
        k = self.w_k(c).view(B, N, H, self.h_dim)
        v = self.w_v(c).view(B, N, H, self.h_dim)
        
        # 2. Point Q, K, V (in local frame)
        qp_local = self.q_pts(c).view(B, N, H, self.num_point_qk, 3)
        kp_local = self.k_pts(c).view(B, N, H, self.num_point_qk, 3)
        vp_local = self.v_pts(c).view(B, N, H, self.num_point_v, 3)
        
        # Transform Point Q, K to global frame for distance computation
        qp_global = self._transform_points_global(qp_local, R, t)
        kp_global = self._transform_points_global(kp_local, R, t)
        
        # 3. Compute Attention Weights
        # Scalar attention (B, H, N, N)
        attn_scalar = torch.einsum('bnhd,bmhd->bhnm', q, k) / math.sqrt(self.h_dim)
        
        # Spatial attention (B, H, N, N) based on squared distances of global points
        # ||q_i - k_j||^2
        diffs = qp_global.unsqueeze(2) - kp_global.unsqueeze(1) # (B, N, N, H, P, 3)
        sq_dists = torch.sum(diffs ** 2, dim=-1) # (B, N, N, H, P)
        attn_spatial = torch.sum(sq_dists, dim=-1).permute(0, 3, 1, 2) # (B, H, N, N)
        
        # Apply learned spatial scale (gamma)
        gamma = self.softplus(self.gamma).view(1, H, 1, 1)
        
        # Combined attention weights
        attn_logits = attn_scalar - (gamma * attn_spatial / 2.0) * math.sqrt(2.0 / 9.0)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        
        # 4. Aggregate Values
        # Scalar aggregation
        out_scalar = torch.einsum('bhnm,bmhd->bnhd', attn_weights, v)
        out_scalar = out_scalar.reshape(B, N, H * self.h_dim)
        
        # Point aggregation (in global frame)
        vp_global = self._transform_points_global(vp_local, R, t)
        out_pts_global = torch.einsum('bhnm,bmhpj->bnhpj', attn_weights, vp_global)
        
        # Transform aggregated points back to local frame
        out_pts_local = self._transform_points_local(out_pts_global, R, t)
        out_pts_local = out_pts_local.reshape(B, N, H * self.num_point_v * 3)
        
        # 5. Final Projection
        out = self.out_proj(torch.cat([out_scalar, out_pts_local], dim=-1))
        
        return out
