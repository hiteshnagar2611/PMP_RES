import torch
import torch.nn as nn

class PhysicochemicalGate(nn.Module):
    def __init__(self, phys_feature_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        # Input: KD_score, net_charge, rel_SASA, amph_score
        self.W_gate = nn.Linear(phys_feature_dim, hidden_dim)

    def forward(self, H_fused: torch.Tensor, psi_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_fused: Fused representations from cross-attention (B, N, 256)
            psi_features: Physicochemical features (B, N, 4)
        """
        # g_i = sigmoid(W_gate * psi_i + b)
        gate = torch.sigmoid(self.W_gate(psi_features))
        
        # Element-wise modulation
        H_out = gate * H_fused
        
        return H_out
