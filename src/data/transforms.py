import torch
import math
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class RandomSO3Rotation(BaseTransform):
    """
    Applies a random 3D rotation to the graph's coordinates (pos) and 
    equivariant vector features (edge_vector, v_node) if they exist.
    """
    def __call__(self, data: Data) -> Data:
        # Generate random rotation matrix
        R = self._generate_random_rotation_matrix().to(data.pos.device)
        
        # Rotate coordinates: X_rot = X @ R^T
        data.pos = torch.matmul(data.pos, R.T)
        
        # Rotate equivariant edge vectors (E, 3)
        if hasattr(data, 'edge_vector') and data.edge_vector is not None:
            data.edge_vector = torch.matmul(data.edge_vector, R.T)
            
        # Rotate equivariant node vectors (N, 6, 3)
        if hasattr(data, 'v_node') and data.v_node is not None:
            # Einsum handles the batching over the 6 vector channels nicely
            data.v_node = torch.einsum('n v c, d c -> n v d', data.v_node, R)
            
        return data

    def _generate_random_rotation_matrix(self):
        random_matrix = torch.randn(3, 3)
        q, r = torch.linalg.qr(random_matrix)
        d = torch.diag(r)
        ph = d.sign()
        q *= ph
        if torch.det(q) < 0:
            q[:, 0] *= -1
        return q


class AddGaussianNoise(BaseTransform):
    """
    Adds small structural noise to node coordinates to prevent overfitting.
    Used during Phase 1 training to make the GVP robust to minor conformational variations.
    """
    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, data: Data) -> Data:
        noise = torch.randn_like(data.pos) * self.std
        data.pos = data.pos + noise
        
        # Note: If pos changes, technically edge_vector and distances should be recomputed.
        # In practice, for small noise, this can be skipped, or placed BEFORE the graph builder
        # in the dataset pipeline.
        return data
