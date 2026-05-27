import torch
import torch.nn as nn
from typing import Tuple

from src.models.dynamo.gvp_encoder import GVP_GNN_Encoder
from src.models.dynamo.conf_attention import ConfAttentionPool
from src.models.dynamo.geometry_path import MembraneGeometryEncoder
from src.models.dynamo.cross_attention import CrossModalAttention
from src.models.dynamo.phys_gate import PhysicochemicalGate
from src.models.dynamo.classifier import DynamoClassifier

class DynaMo(nn.Module):
    """
    DynaMo — Dynamic Membrane Oracle (Full Architecture)
    """
    def __init__(self, 
                 hidden_dim: int = 256,
                 geom_feature_dim: int = 4,
                 phys_feature_dim: int = 4):
        super().__init__()
        
        # 1. Shared Encoder (imported from previously built module)
        self.encoder = GVP_GNN_Encoder(out_dim=hidden_dim)
        
        # 2. Novel Core Branches
        self.attention_pool = ConfAttentionPool(hidden_dim=hidden_dim)
        self.geometry_path = MembraneGeometryEncoder(
            geom_feature_dim=geom_feature_dim, 
            hidden_dim=hidden_dim
        )
        
        # 3. Fusion & Gating
        self.fusion = CrossModalAttention(hidden_dim=hidden_dim)
        self.gate = PhysicochemicalGate(
            phys_feature_dim=phys_feature_dim, 
            hidden_dim=hidden_dim
        )
        
        # 4. Output
        self.classifier = DynamoClassifier(hidden_dim=hidden_dim)

    def forward(self, 
                node_x_t: Tuple[torch.Tensor, torch.Tensor],
                edge_idx_t: torch.Tensor,
                edge_attr_t: Tuple[torch.Tensor, torch.Tensor],
                node_x_ref: Tuple[torch.Tensor, torch.Tensor],
                edge_idx_ref: torch.Tensor,
                edge_attr_ref: Tuple[torch.Tensor, torch.Tensor],
                rmsf: torch.Tensor,
                geom_features: torch.Tensor,
                phys_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Note: The `_t` inputs should be batched such that the time dimension T 
        is folded into the batch dimension for the GVP encoder, then reshaped.
        
        Returns:
            logits: (B, N) raw predictions
            H_out: (B, N, 256) final representation (used for contrastive loss)
        """
        # 1. Encoding
        # In a real PyG forward pass, T snapshots are processed sequentially or via batching
        H_t = self.encoder(node_x_t, edge_idx_t, edge_attr_t)       # Reshape to (B, T, N, 256)
        H_ref = self.encoder(node_x_ref, edge_idx_ref, edge_attr_ref) # (B, N, 256)
        
        # 2. Attention Pooling
        H_star = self.attention_pool(H_t, H_ref, rmsf)
        
        # 3. Membrane Geometry
        H_geom = self.geometry_path(geom_features)
        
        # 4. Fusion
        H_fused = self.fusion(H_star, H_geom)
        
        # 5. Gating
        H_out = self.gate(H_fused, phys_features)
        
        # 6. Classification
        logits = self.classifier(H_out)
        
        return logits, H_out
