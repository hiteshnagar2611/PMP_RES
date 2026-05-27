# src/models/pmpgen/noise_schedule.py
import torch
import torch.nn as nn

class MDNoiseSchedule(nn.Module):
    def __init__(self):
        super().__init__()
        # Maps RMSF scalar to a noise multiplier f(RMSF_i)
        self.rmsf_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Softplus() # Ensure strictly positive multiplier
        )

    def forward(self, t, rmsf, anchor_mask):
        """
        t: Time step \in [0, 1]
        rmsf: (B, N, 1) Root Mean Square Fluctuation
        anchor_mask: (B, N, 1) True if residue is part of the fixed binding patch
        """
        # Base schedule (linear for standard OT-flow)
        sigma_base = t 
        
        # f(RMSF_i): High RMSF -> more noise, Low RMSF -> less noise
        rmsf_factor = self.rmsf_mlp(rmsf) 
        
        # sigma_i(t) = sigma_base(t) * f(RMSF_i)
        sigma_i = sigma_base * rmsf_factor
        
        # \sigma_anchor = 0 (fixed, never noised)
        sigma_i = sigma_i.masked_fill(anchor_mask, 0.0)
        
        return sigma_i
