import torch
import torch.nn as nn

class MembraneGeometryEncoder(nn.Module):
    def __init__(self, geom_feature_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        # Input features: depth, tilt, membrane-facing SASA, amphipathic score
        self.proj = nn.Sequential(
            nn.Linear(geom_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, geom_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geom_features: Scalar membrane geometry features (B, N, 4)
        Returns:
            H_geom: SE(3)-invariant geometry representation (B, N, 256)
        """
        return self.proj(geom_features)
