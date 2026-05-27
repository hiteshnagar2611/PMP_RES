import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamoClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, H_out: torch.Tensor) -> torch.Tensor:
        # Return logits; sigmoid is usually applied in the loss function for numerical stability
        return self.mlp(H_out).squeeze(-1) 


class DynamoLoss(nn.Module):
    """
    Computes L_total = L_focal + 0.2*L_patch + 0.1*L_contrast
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()

    def patch_loss(self, logits: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # Penalizes high variance in predictions among spatial neighbors (kNN graph)
        probs = torch.sigmoid(logits)
        # adj_matrix is (B, N, N) binary mask of kNN
        diffs = (probs.unsqueeze(2) - probs.unsqueeze(1)) ** 2
        return (diffs * adj_matrix).mean()

    def contrastive_loss(self, H_out: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # InfoNCE alignment: push binding representations closer, separate from non-binding
        # Simplified placeholder for InfoNCE
        # In practice, gather positive anchors (binding) and negative (non-binding)
        return torch.tensor(0.0, requires_grad=True, device=H_out.device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                adj_matrix: torch.Tensor, H_out: torch.Tensor) -> torch.Tensor:
        
        L_foc = self.focal_loss(logits, targets)
        L_pat = self.patch_loss(logits, adj_matrix)
        L_con = self.contrastive_loss(H_out, targets)
        
        return L_foc + (0.2 * L_pat) + (0.1 * L_con)
