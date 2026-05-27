import torch
import torch.nn as nn
from typing import Tuple
from torch_geometric.nn import MessagePassing
from src.models.shared.gvp_primitives import GVP_MLP, GVPResidualBlock, GVP

class GVPGraphConv(MessagePassing):
    """
    A single layer of GVP-GNN Message Passing.
    Computes interactions between connected nodes using their scalar and vector features,
    alongside the geometric edge features (distance and direction).
    """
    def __init__(self, node_dims: Tuple[int, int], edge_dims: Tuple[int, int], 
                 hidden_dims: Tuple[int, int], drop_rate: float = 0.1):
        # We use 'mean' aggregation to normalize varying neighborhood sizes (though k is fixed at 16)
        super().__init__(aggr='mean')
        
        self.node_dims = node_dims
        
        # Message function: takes [source_node, target_node, edge_attr]
        in_scalar = node_dims[0] * 2 + edge_dims[0]
        in_vector = node_dims[1] * 2 + edge_dims[1]
        
        self.message_mlp = GVP_MLP(
            in_dims=(in_scalar, in_vector),
            out_dims=hidden_dims,
            hidden_dims=hidden_dims,
            num_layers=2,
            drop_rate=drop_rate
        )
        
        # Node update function: takes [current_node, aggregated_messages]
        self.update_residual = GVPResidualBlock(node_dims, drop_rate=drop_rate)
        self.msg_proj = GVP(hidden_dims, node_dims, activations=(None, None))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], 
                edge_index: torch.Tensor, 
                edge_attr: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tuple of node scalars and vectors (s, V)
            edge_index: Graph connectivity (2, num_edges)
            edge_attr: Tuple of edge scalars and vectors (e_s, e_V)
        """
        # PyG propagate implicitly calls message() and then aggregates
        aggr_msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Project aggregated messages back to node dimensions
        msg_s, msg_v = self.msg_proj(aggr_msg)
        
        # Add to current state and apply residual update block
        s_in = x[0] + msg_s
        v_in = x[1] + msg_v
        
        return self.update_residual((s_in, v_in))

    def message(self, x_i: Tuple[torch.Tensor, torch.Tensor], 
                x_j: Tuple[torch.Tensor, torch.Tensor], 
                edge_attr: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_i represents the target node, x_j represents the source node
        s_i, v_i = x_i
        s_j, v_j = x_j
        e_s, e_v = edge_attr
        
        # Concatenate scalars and vectors
        s_cat = torch.cat([s_i, s_j, e_s], dim=-1)
        v_cat = torch.cat([v_i, v_j, e_v], dim=-2) # Concat along the vector channel dimension
        
        return self.message_mlp((s_cat, v_cat))


class GVPReadout(nn.Module):
    """
    Final projection from SO(3)-equivariant (s, V) space to purely invariant H \in R^{N x 256}.
    Implements the diagram's logic: 
    V' = sigma(gate) * W_V * V
    s' = MLP([s, ||V'||]) 
    """
    def __init__(self, node_dims: Tuple[int, int], out_dim: int = 256):
        super().__init__()
        s_dim, v_dim = node_dims
        
        # Standard GVP without vector outputs essentially does this operation natively,
        # but we build it explicitly to match the diagram's notation.
        self.W_v = nn.Linear(v_dim, v_dim, bias=False)
        self.v_gate = nn.Sequential(
            nn.Linear(s_dim, v_dim),
            nn.Sigmoid()
        )
        
        # The final MLP maps the original scalars + the norms of the transformed vectors
        self.mlp = nn.Sequential(
            nn.Linear(s_dim + v_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        s, V = x
        
        # 1. Vector transformation: W_v * V
        # V is (N, v_dim, 3). W_v mixes the vector channels.
        V_trans = self.W_v(V.transpose(1, 2)).transpose(1, 2)
        
        # 2. Gating: V' = sigma(gate) * V_trans
        gate = self.v_gate(s).unsqueeze(-1) # (N, v_dim, 1)
        V_prime = gate * V_trans
        
        # 3. Vector norms: ||V'||
        v_norms = torch.linalg.norm(V_prime, dim=-1) # (N, v_dim)
        
        # 4. Scalar output: s' = MLP([s, ||V'||])
        s_cat = torch.cat([s, v_norms], dim=-1) # (N, s_dim + v_dim)
        H = self.mlp(s_cat) # (N, 256)
        
        return H


class GVP_GNN_Encoder(nn.Module):
    """
    The full 3-layer GVP-GNN backbone used by both DynaMo and PMPGen.
    Shared weights across snapshots are achieved by passing batched/looped 
    snapshots through this same instantiated module.
    """
    def __init__(self, 
                 node_in_dims: Tuple[int, int] = (175, 6),
                 edge_in_dims: Tuple[int, int] = (1, 1),
                 hidden_dims: Tuple[int, int] = (256, 16),
                 out_dim: int = 256,
                 num_layers: int = 3,
                 drop_rate: float = 0.1):
        super().__init__()
        
        # Initial embedding to hidden dimensions
        self.node_embed = GVP(node_in_dims, hidden_dims, activations=(None, None))
        self.edge_embed = GVP(edge_in_dims, hidden_dims, activations=(None, None))
        
        # 3x Message Passing Layers
        self.layers = nn.ModuleList([
            GVPGraphConv(hidden_dims, hidden_dims, hidden_dims, drop_rate)
            for _ in range(num_layers)
        ])
        
        # Final projection to H \in R^{N x 256}
        self.readout = GVPReadout(hidden_dims, out_dim=out_dim)

    def forward(self, 
                node_x: Tuple[torch.Tensor, torch.Tensor], 
                edge_index: torch.Tensor, 
                edge_attr: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            node_x: Tuple of (s_node \in R^{N x 175}, V_node \in R^{N x 6 x 3})
            edge_index: (2, num_edges)
            edge_attr: (e_dist \in R^{E x 1}, e_vec \in R^{E x 1 x 3})
            
        Returns:
            H: Encoded representation (N, 256)
        """
        # Embed initial features
        x = self.node_embed(node_x)
        e = self.edge_embed(edge_attr)
        
        # Apply GVP-GNN layers
        for layer in self.layers:
            x = layer(x, edge_index, e)
            
        # Final readout to scalar representation
        H_out = self.readout(x)
        
        return H_out
