# src/models/pmpgen/pmpgen.py
import torch
import torch.nn as nn
from src.models.pmpgen.conditioning import ConditioningFusion
from src.models.pmpgen.noise_schedule import MDNoiseSchedule
from src.models.pmpgen.ipa_denoiser import IPAGVPDenoiser

class PMPGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioning = ConditioningFusion()
        self.noise_schedule = MDNoiseSchedule()
        self.denoiser = IPAGVPDenoiser()

    def generate_flow_matching_target(self, x_1, x_0):
        """ Target vector field for Optimal Transport flow matching """
        # x_1: Noise N(0, I)
        # x_0: True data distribution (ground truth PMP structure)
        # Simplified vector field target: v = x_1 - x_0
        return x_1 - x_0

    def compute_loss(self, scaffold_graph, geom_features, anchor_mask, rmsf, x_0, x_query):
        """
        Implements the Training Objectives block from the diagram.
        """
        B, N = anchor_mask.shape[:2]
        
        # 1. Conditioning
        c = self.conditioning(scaffold_graph, geom_features, anchor_mask)
        
        # 2. Sample random time step t \in [0, 1]
        t = torch.rand((B, 1, 1), device=c.device)
        
        # 3. Sample Noise x_1
        x_1 = torch.randn_like(x_0)
        
        # 4. Interpolant forward process (SE(3) forward process)
        # x_t = (1-t)*x_0 + t*x_1 (Simplified Euclidean representation here)
        sigma_i = self.noise_schedule(t, rmsf, anchor_mask)
        x_t = (1 - t) * x_0 + sigma_i * x_1 
        
        # 5. Denoising prediction v_\theta
        v_theta_trans, v_theta_rot = self.denoiser(x_t, t.squeeze(-1), c, anchor_mask)
        # Combine translation and rotation for simplified loss computation
        v_theta = v_theta_trans # Assuming translation only for MSE brevity
        
        # 6. Loss Terms
        target_v = self.generate_flow_matching_target(x_1, x_0)
        
        # L_flow = ||v_\theta(x_t, t, c) - (x_1 - x_0)||^2
        L_flow = nn.functional.mse_loss(v_theta, target_v)
        
        # L_anchor = ||x_pred[mask] - x_query[mask]||^2
        # Flow matching infers x_0_pred via x_t - v_\theta
        x_pred = x_t - v_theta 
        L_anchor = nn.functional.mse_loss(
            x_pred[anchor_mask.squeeze(-1)], 
            x_query[anchor_mask.squeeze(-1)]
        )
        
        # L_mem = E_mem(x_0) -> ||depth_pred - depth_target||^2
        # (Requires external mapping of x_pred to depth)
        L_mem = torch.tensor(0.0, requires_grad=True, device=c.device) 
        
        # L_total = L_flow + 0.5*L_anchor + 0.3*L_mem
        L_total = L_flow + (0.5 * L_anchor) + (0.3 * L_mem)
        
        return L_total
