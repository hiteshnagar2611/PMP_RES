import torch
import torch.nn as nn
from typing import Tuple

# Assuming standard installation: pip install gvp-pytorch
# We import the core primitives to wrap them in project-specific blocks
from gvp import GVP, LayerNorm, Dropout

class GVP_MLP(nn.Module):
    """
    A multi-layer perceptron built out of Geometric Vector Perceptrons.
    Maintains the (scalar, vector) tuple routing automatically.
    """
    def __init__(self, in_dims: Tuple[int, int], out_dims: Tuple[int, int], 
                 hidden_dims: Tuple[int, int], num_layers: int = 3, 
                 drop_rate: float = 0.1, activations=(torch.relu, torch.sigmoid)):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            GVP(in_dims, hidden_dims, activations=activations)
        )
        self.layers.append(LayerNorm(hidden_dims))
        self.layers.append(Dropout(drop_rate))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GVP(hidden_dims, hidden_dims, activations=activations))
            self.layers.append(LayerNorm(hidden_dims))
            self.layers.append(Dropout(drop_rate))
            
        # Output layer (usually linear, so no activations on the final step)
        self.layers.append(GVP(hidden_dims, out_dims, activations=(None, None)))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tuple of (s, V) where s \in R^{N x d_s} and V \in R^{N x d_v x 3}
        Returns:
            Tuple of transformed (s, V)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class GVPResidualBlock(nn.Module):
    """
    A single GVP layer with a pre-norm residual connection.
    Standardized for both Phase 1 (DynaMo) and Phase 2 (PMPGen) encoders.
    """
    def __init__(self, dims: Tuple[int, int], drop_rate: float = 0.1, 
                 activations=(torch.relu, torch.sigmoid)):
        super().__init__()
        self.norm = LayerNorm(dims)
        self.gvp = GVP(dims, dims, activations=activations)
        self.dropout = Dropout(drop_rate)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        s, v = x
        
        # Pre-norm
        s_norm, v_norm = self.norm(x)
        
        # GVP + Dropout
        s_out, v_out = self.gvp((s_norm, v_norm))
        s_out, v_out = self.dropout((s_out, v_out))
        
        # Residual connection
        return s + s_out, v + v_out


class GVPConvBlock(nn.Module):
    """
    Wraps message passing using GVPs. 
    Assumes message compilation happens externally (e.g., via PyG MessagePassing) 
    and this block acts as the node-update function.
    """
    def __init__(self, node_dims: Tuple[int, int], edge_dims: Tuple[int, int], 
                 hidden_dims: Tuple[int, int], drop_rate: float = 0.1):
        super().__init__()
        
        # Maps concatenated [node_i, node_j, edge_ij] down to hidden representation
        in_scalar = node_dims[0] * 2 + edge_dims[0]
        in_vector = node_dims[1] * 2 + edge_dims[1]
        
        self.message_func = GVP_MLP(
            in_dims=(in_scalar, in_vector),
            out_dims=hidden_dims,
            hidden_dims=hidden_dims,
            num_layers=2,
            drop_rate=drop_rate
        )
        
        # Updates node state based on aggregated messages
        self.update_func = GVPResidualBlock(node_dims, drop_rate=drop_rate)
        
        # Projects aggregated message to match node dims for the residual update
        self.msg_proj = GVP(hidden_dims, node_dims, activations=(None, None))

    def forward(self, node_x: Tuple[torch.Tensor, torch.Tensor], 
                aggr_messages: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_x: Current node features (s, V)
            aggr_messages: Pooled incoming messages for each node (s_msg, V_msg)
        """
        # Project messages to node dimensions
        msg_proj = self.msg_proj(aggr_messages)
        
        # Add to current node state (acting as input to the update residual block)
        s_in = node_x[0] + msg_proj[0]
        v_in = node_x[1] + msg_proj[1]
        
        # Apply GVP update with residual
        return self.update_func((s_in, v_in))
