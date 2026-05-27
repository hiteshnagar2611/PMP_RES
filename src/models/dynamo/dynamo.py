import torch
import torch.nn as nn
from typing import Tuple

from src.models.dynamo.fusion import FeatureFusion
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
                 plm_dim: int = 1280,
                 plm_proj_dim: int = 128,
                 hidden_dim: int = 256,
                 geom_feature_dim: int = 4,
                 phys_feature_dim: int = 4):
        super().__init__()
        
        # 0. Feature Projection & Fusion
        self.feature_fusion = FeatureFusion(plm_dim=plm_dim, plm_proj_dim=plm_proj_dim)
        
        # 1. Shared Encoder 
        # Note: GVP_GNN_Encoder defaults to node_in_dims=(175, 6)
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
                plm_emb_t: torch.Tensor,
                geom_scalars_t: torch.Tensor,
                v_node_t: torch.Tensor,
                edge_idx_t: torch.Tensor,
                edge_attr_t: Tuple[torch.Tensor, torch.Tensor],
                plm_emb_ref: torch.Tensor,
                geom_scalars_ref: torch.Tensor,
                v_node_ref: torch.Tensor,
                edge_idx_ref: torch.Tensor,
                edge_attr_ref: Tuple[torch.Tensor, torch.Tensor],
                rmsf: torch.Tensor,
                mem_geom_features: torch.Tensor,
                phys_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            plm_emb_*: ESM-2 embeddings (B, N, 1280) or batched over T (B, T, N, 1280)
            geom_scalars_*: Structural scalars like SASA, depth (B, N, 47)
            v_node_*: SE(3)-equivariant vector features (B, N, 6, 3)
            rmsf: Conformational flexibility (B, N, 1)
            mem_geom_features: Membrane geometry scalars (B, N, 4)
            phys_features: Physicochemical scalars for gating (B, N, 4)
            
        Returns:
            logits: (B, N) raw binding predictions
            H_out: (B, N, 256) final representation (used for contrastive loss)
        """
        # 1. Feature Fusion (Project PLM and concat with geometric scalars)
        s_node_t = self.feature_fusion(plm_emb_t, geom_scalars_t)
        s_node_ref = self.feature_fusion(plm_emb_ref, geom_scalars_ref)
        
        # Group into (scalar, vector) tuples for the GVP
        node_x_t = (s_node_t, v_node_t)
        node_x_ref = (s_node_ref, v_node_ref)
        
        # 2. Encoding
        H_t = self.encoder(node_x_t, edge_idx_t, edge_attr_t)         # Reshape externally to (B, T, N, 256)
        H_ref = self.encoder(node_x_ref, edge_idx_ref, edge_attr_ref) # (B, N, 256)
        
        # 3. Attention Pooling
        H_star = self.attention_pool(H_t, H_ref, rmsf)
        
        # 4. Membrane Geometry
        H_geom = self.geometry_path(mem_geom_features)
        
        # 5. Cross-Modal Fusion
        H_fused = self.fusion(H_star, H_geom)
        
        # 6. Gating
        H_out = self.gate(H_fused, phys_features)
        
        # 7. Classification
        logits = self.classifier(H_out)
        
        return logits, H_out
