# src/models/pmpgen/ipa_denoiser.py
import torch
import torch.nn as nn

class IPAGVPDenoiser(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.t_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Mocking the 6-layer Invariant Point Attention + GVP blocks
        # In practice, this takes SE(3) frames, time embeddings, and conditioning context
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim * 2) # Placeholder for IPA/GVP layer
            for _ in range(num_layers)
        ])
        
        # Output projections for translation (R^3) and rotation (SO(3) tangent space)
        self.v_trans = nn.Linear(hidden_dim * 2, 3)
        self.v_rot = nn.Linear(hidden_dim * 2, 3) 

    def forward(self, x_t, t, c, anchor_mask):
        """
        x_t: Current SE(3) frames (B, N, 4, 4) or split into (coords, rotations)
        t: Time step (B, 1)
        c: Conditioning context (B, N, 256)
        """
        t_embed = self.t_emb(t).unsqueeze(1).expand(-1, c.size(1), -1) # (B, N, 256)
        
        # Conditioning Injection: c concat to node features each step
        h = torch.cat([c, t_embed], dim=-1) # (B, N, 512)
        
        for block in self.blocks:
            h = block(h) # In reality, IPA attends over x_t frames here
            
        # Predict vector field v_\theta
        v_t_trans = self.v_trans(h) # Translation velocities
        v_t_rot = self.v_rot(h)     # Rotation velocities
        
        # Zero-grad on anchors: Anchors do not move
        v_t_trans = v_t_trans.masked_fill(anchor_mask.expand_as(v_t_trans), 0.0)
        v_t_rot = v_t_rot.masked_fill(anchor_mask.expand_as(v_t_rot), 0.0)
        
        return v_t_trans, v_t_rot
