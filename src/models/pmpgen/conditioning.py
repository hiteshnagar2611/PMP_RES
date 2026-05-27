# src/models/pmpgen/conditioning.py
import torch
import torch.nn as nn
from src.models.dynamo.gvp_encoder import GVP_GNN_Encoder # Reusing Phase 1 encoder

class GeometryDynamicsEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=256):
        # in_dim = 3 (normal) + 1 (depth) + 1 (RMSF) = 5
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, geom_dynamics_features):
        # Maps to c_mem \in R^256
        return self.mlp(geom_dynamics_features)

class ConditioningFusion(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8):
        super().__init__()
        self.scaffold_encoder = GVP_GNN_Encoder(out_dim=hidden_dim)
        self.geom_dyn_encoder = GeometryDynamicsEncoder(hidden_dim=hidden_dim)
        
        # Cross Attention: c_struct queries c_mem
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, scaffold_graph, geom_features, anchor_mask):
        """
        anchor_mask: (B, N, 1) boolean mask from DynaMo where 1 = binding patch
        """
        # 1. Encode structure
        # c_struct \in R^{N x 256}
        c_struct = self.scaffold_encoder(scaffold_graph.node_x, scaffold_graph.edge_idx, scaffold_graph.edge_attr)
        
        # 2. Encode geometry + dynamics
        # c_mem \in R^{N x 256} (assuming per-residue features like depth/RMSF)
        c_mem = self.geom_dyn_encoder(geom_features)
        
        # 3. Cross Attention Fusion
        attn_out, _ = self.cross_attn(query=c_struct, key=c_mem, value=c_mem)
        c_fused = self.norm(c_struct + attn_out)
        
        # 4. Hard constraint on anchor residues (Binding patch mask applied)
        # Anchors retain pure structural encoding, others get fused context
        c = torch.where(anchor_mask, c_struct, c_fused)
        
        return c # (B, N, 256) - conditioning context
