import torch
import torch.nn as nn

class MembraneGuidance(nn.Module):
    """
    Provides gradient-based guidance during inference to satisfy physical priors.
    """
    def __init__(self, lambda_guidance: float = 1.0):
        super().__init__()
        self.lambda_guidance = lambda_guidance

    def compute_energy(self, x_t_trans, depth_target, normal_vector):
        """
        Calculates E_mem = ||depth_pred - depth_target||^2
        """
        # Calculate predicted depth as the dot product of coords with the membrane normal
        # normal_vector shape: (3,)
        depth_pred = torch.einsum('bni,i->bn', x_t_trans, normal_vector)
        
        # E_mem is the squared error
        E_mem = torch.nn.functional.mse_loss(depth_pred, depth_target, reduction='sum')
        return E_mem

    def get_gradient(self, x_t_trans, depth_target, normal_vector):
        """
        Returns \nabla E_mem(x_t). Requires x_t_trans to have requires_grad=True.
        """
        with torch.enable_grad():
            x_t_trans_guided = x_t_trans.detach().requires_grad_(True)
            
            energy = self.compute_energy(x_t_trans_guided, depth_target, normal_vector)
            
            # Compute gradient of energy w.r.t spatial coordinates
            grad_x = torch.autograd.grad(energy, x_t_trans_guided)[0]
            
        return grad_x
