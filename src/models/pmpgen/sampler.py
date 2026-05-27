import torch
import torch.nn as nn
from tqdm import tqdm

class PMPGenSampler(nn.Module):
    def __init__(self, denoiser, conditioning, mem_guidance, steps=500):
        super().__init__()
        self.denoiser = denoiser
        self.conditioning = conditioning
        self.mem_guidance = mem_guidance
        self.steps = steps

    @torch.no_grad()
    def sample(self, scaffold_graph, geom_features, anchor_mask, depth_target, normal_vector):
        """
        Inference loop for generating a new backbone.
        """
        device = geom_features.device
        B, N = anchor_mask.shape[:2]
        
        # 1. Compute static conditioning context
        c = self.conditioning(scaffold_graph, geom_features, anchor_mask)
        
        # 2. Start from pure noise x_1
        x_t_trans = torch.randn(B, N, 3, device=device)
        x_t_rot = torch.eye(3, device=device).expand(B, N, 3, 3).clone()
        
        # 3. Euler integration loop from t=1 to t=0
        dt = 1.0 / self.steps
        time_steps = torch.linspace(1.0, 0.0, self.steps, device=device)
        
        for t_val in tqdm(time_steps, desc="Denoising SE(3) flow"):
            t = torch.full((B, 1), t_val, device=device)
            
            # Predict vector field (velocities)
            v_t_trans, v_t_rot = self.denoiser(x_t_trans, x_t_rot, t, c, anchor_mask)
            
            # Compute membrane guidance gradient \nabla E_mem(x_t)
            # Only apply guidance if lambda > 0 and we are in the later half of sampling
            grad_mem = self.mem_guidance.get_gradient(x_t_trans, depth_target, normal_vector)
            
            # Update state: x_{t-dt} = x_t - dt * v_theta - lambda * grad_mem
            # Note: Flow matching usually goes forward; generating goes backwards, 
            # hence the sign inversion compared to SDEs, following x_0 = x_t - t * v_t
            x_t_trans = x_t_trans - dt * v_t_trans - self.mem_guidance.lambda_guidance * grad_mem
            
            # Update rotation (simplified linear update; should be on tangent space)
            # x_t_rot = Exp(-dt * v_t_rot) * x_t_rot
            
            # Hard enforce anchors (they do not move)
            # In a real implementation, you overwrite x_t[anchor_mask] with x_query[anchor_mask]
            
        return x_t_trans, x_t_rot
