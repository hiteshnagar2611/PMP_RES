import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """
    Fuses Pre-trained Language Model (PLM) embeddings with structural node features.
    Projects high-dimensional language representations and concatenates them 
    with structural scalars to form the initial s_node input for the GVP-GNN.
    """
    def __init__(self, plm_dim: int = 1280, plm_proj_dim: int = 128):
        super().__init__()
        # Linear(1280 -> 128)
        self.plm_proj = nn.Linear(plm_dim, plm_proj_dim)

    def forward(self, plm_embeddings: torch.Tensor, geom_scalars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            plm_embeddings: Precomputed ESM-2 features. Shape: (..., 1280)
            geom_scalars: Structural scalars (dihedrals, SASA, HBE, depth). Shape: (..., 47)
                          
        Returns:
            s_node: Fused scalar node features. Shape: (..., 175)
        """
        # 1. Project PLM embeddings
        plm_proj = self.plm_proj(plm_embeddings) # (..., 128)
        
        # 2. Concatenate with geometric/structural scalars
        # 128 (PLM) + 47 (Geom) = 175 
        s_node = torch.cat([plm_proj, geom_scalars], dim=-1) 
        
        return s_node
