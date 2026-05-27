import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ProteinFeatureExtractor(nn.Module):
    """
    Constructs scalar and vector node/edge features for the GVP-GNN.
    Ensures rigorous separation of SO(3)-invariant (scalar) and 
    SO(3)-equivariant (vector) features.
    """
    def __init__(self, plm_dim: int = 1280, plm_proj_dim: int = 128):
        super().__init__()
        # PLM projection (1280 -> 128) as per the Phase 1 architecture
        self.plm_proj = nn.Linear(plm_dim, plm_proj_dim)
        
    def _construct_node_vectors(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Constructs 6 vector features per node (N, 6, 3).
        Assumes coords shape is (N, 3, 3) representing [N, C-alpha, C] atoms.
        If only C-alpha (N, 3) is provided, it falls back to sequential vectors.
        """
        if coords.dim() == 3 and coords.size(1) == 3:
            # Full backbone coordinates provided: (N, 3, 3) -> N, CA, C
            N_coords = coords[:, 0, :]
            CA_coords = coords[:, 1, :]
            C_coords = coords[:, 2, :]
            
            # 1 & 2: Local bond vectors
            v1 = N_coords - CA_coords
            v2 = C_coords - CA_coords
            
            # 3: Cross product (normal to backbone plane)
            v3 = torch.cross(v1, v2, dim=-1)
            
            # 4 & 5: Sequential chain vectors (forward and backward)
            v4 = torch.zeros_like(CA_coords)
            v4[:-1] = CA_coords[1:] - CA_coords[:-1]
            
            v5 = torch.zeros_like(CA_coords)
            v5[1:] = CA_coords[:-1] - CA_coords[1:]
            
            # 6: Sequential cross product
            v6 = torch.cross(v4, v5, dim=-1)
            
        else:
            # Fallback to just C-alpha trace
            CA_coords = coords
            v1 = torch.zeros_like(CA_coords)
            v1[:-1] = CA_coords[1:] - CA_coords[:-1]
            v2 = torch.zeros_like(CA_coords)
            v2[1:] = CA_coords[:-1] - CA_coords[1:]
            v3 = torch.cross(v1, v2, dim=-1)
            v4, v5, v6 = -v1, -v2, -v3 # Placeholders to fill the 6x3 requirement
            
        # Normalize all vectors to ensure numerical stability in GVP
        vectors = torch.stack([v1, v2, v3, v4, v5, v6], dim=1) # (N, 6, 3)
        vectors = F.normalize(vectors, dim=-1)
        return vectors

    def forward(self, 
                coords: torch.Tensor, 
                plm_embeddings: torch.Tensor, 
                geom_scalars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: Backbone coordinates (N, 3) or (N, 3, 3).
            plm_embeddings: Precomputed ESM-2 features (N, 1280).
            geom_scalars: Dihedrals, SASA, HBE, Depth concatenated (N, 47).
                          (128 PLM + 47 Geom = 175 total scalar features).
                          
        Returns:
            s_node: Scalar node features (N, 175) -> SO(3) invariant
            V_node: Vector node features (N, 6, 3) -> SO(3) equivariant
        """
        # 1. Scalar Features (Invariant)
        plm_proj = self.plm_proj(plm_embeddings) # (N, 128)
        s_node = torch.cat([plm_proj, geom_scalars], dim=-1) # (N, 175)
        
        # 2. Vector Features (Equivariant)
        V_node = self._construct_node_vectors(coords) # (N, 6, 3)
        
        return s_node, V_node
