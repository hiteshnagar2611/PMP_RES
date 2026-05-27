import torch

def so3_log_map(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the logarithmic map on SO(3), converting a rotation matrix 
    into an axis-angle vector (Lie algebra).
    Args:
        R: Rotation matrix (..., 3, 3)
    Returns:
        Axis-angle vector (..., 3)
    """
    # 1. Compute the trace to find the angle theta
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    # Clamp to avoid NaN in acos due to numerical instability
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)
    
    # 2. Compute the axis (skew-symmetric part of R)
    # v = 1/(2*sin(theta)) * [R_32 - R_23, R_13 - R_31, R_21 - R_12]
    v = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1)
    
    sin_theta = torch.sin(theta).unsqueeze(-1)
    
    # Taylor expansion guard for small angles (theta -> 0)
    # lim_{theta->0} (theta / (2*sin(theta))) = 0.5
    factor = torch.where(
        theta.unsqueeze(-1) < 1e-4,
        0.5 * torch.ones_like(sin_theta),
        theta.unsqueeze(-1) / (2.0 * sin_theta + eps)
    )
    
    return factor * v

def so3_exp_map(omega: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Computes the exponential map on SO(3) using Rodrigues' rotation formula.
    Maps an axis-angle vector to a 3x3 rotation matrix.
    Args:
        omega: Axis-angle vector (..., 3)
    Returns:
        R: Rotation matrix (..., 3, 3)
    """
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    axis = omega / (theta + 1e-8)
    
    # Skew-symmetric matrix K from axis
    K = torch.zeros(*omega.shape[:-1], 3, 3, device=omega.device, dtype=omega.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    I = I.expand_as(K)
    
    # Rodrigues' formula: R = I + sin(theta)K + (1-cos(theta))K^2
    theta = theta.unsqueeze(-1) # Match K shape for broadcasting
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    R = I + sin_theta * K + (1.0 - cos_theta) * torch.matmul(K, K)
    
    # Taylor expansion guard for small angles
    R_small = I + torch.matmul(K, K) * (theta ** 2) / 2.0
    
    return torch.where((theta < eps).expand_as(R), R_small, R)

def sample_igso3(shape, sigma, device):
    """
    Placeholder for sampling from the Isotropic Gaussian on SO(3).
    A true exact sample requires an infinite series approximation. 
    For engineering purposes, injecting scaled axis-angle noise and mapping 
    via so3_exp_map is often used as a fast approximation.
    """
    # Approximate IGSO3 by sampling normal Lie algebra and mapping to SO(3)
    omega = torch.randn(*shape, 3, device=device) * sigma
    return so3_exp_map(omega)
