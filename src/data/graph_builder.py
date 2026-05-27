import torch
from torch_cluster import knn_graph
from torch_geometric.data import Data

class ProteinGraphBuilder:
    """
    Constructs k-Nearest Neighbor (kNN) graphs from 3D protein coordinates.
    Designed to integrate with PyTorch Geometric (PyG) dataloaders.
    """
    def __init__(self, k: int = 16):
        """
        Args:
            k (int): Number of nearest neighbors for graph construction. 
                     Defaults to 16 as per the DynaMo architecture.
        """
        self.k = k

    def build_graph(self, coords: torch.Tensor, node_features: torch.Tensor = None) -> Data:
        """
        Builds a spatial kNN graph from coordinate data.

        Args:
            coords (torch.Tensor): Shape (N, 3). Typically C-alpha coordinates.
            node_features (torch.Tensor, optional): Shape (N, D). Scalar node features 
                                                    (e.g., PLM embeddings).

        Returns:
            torch_geometric.data.Data: A PyG Data object containing topology and basic geometry.
        """
        if coords.dim() != 2 or coords.size(1) != 3:
            raise ValueError(f"Expected coords of shape (N, 3), got {coords.shape}")

        # Ensure coordinates are standard float32 for stable distance computation
        coords = coords.contiguous().float()
        
        # 1. Compute kNN topology
        # knn_graph returns edge_index of shape (2, N * k)
        # loop=False prevents a node from being its own neighbor
        edge_index = knn_graph(coords, k=self.k, loop=False)
        
        # 2. Extract basic geometric edge features
        row, col = edge_index
        # Vector pointing from neighbor (col) to target node (row)
        edge_vectors = coords[row] - coords[col] 
        
        # Scalar distance (L2 norm)
        edge_distances = torch.linalg.norm(edge_vectors, dim=-1, keepdim=True)
        
        # Normalized directional vectors (crucial for Geometric Vector Perceptrons)
        edge_vectors_normalized = edge_vectors / (edge_distances + 1e-8)
        
        # 3. Package into PyG Data object
        data = Data(
            pos=coords,
            edge_index=edge_index,
            edge_attr=edge_distances,         # Scalar edge features \in R^1
            edge_vector=edge_vectors_normalized # Vector edge features \in R^3
        )
        
        if node_features is not None:
            data.x = node_features
            
        return data

    def __call__(self, coords: torch.Tensor, node_features: torch.Tensor = None) -> Data:
        return self.build_graph(coords, node_features)

# --- Helper functions for complex batching/chain logic (Optional) ---

def remove_cross_chain_edges(edge_index: torch.Tensor, chain_ids: torch.Tensor) -> torch.Tensor:
    """
    Filters out edges that connect different protein chains if simulating 
    multi-chain complexes where spatial proximity shouldn't imply a physical bond.
    """
    row, col = edge_index
    same_chain_mask = chain_ids[row] == chain_ids[col]
    return edge_index[:, same_chain_mask]
