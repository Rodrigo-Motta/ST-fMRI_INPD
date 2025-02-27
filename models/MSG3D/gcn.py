import torch
import numpy as np
import torch.nn as nn

from utils.graph_ops import normalize_adjacency_matrix
from utils.layers import MLP

class GraphConv(nn.Module):
    """
    A single-scale graph convolution layer (GCN-like) that:
      1. Builds one adjacency matrix with self-loops.
      2. Optionally applies a learned mask to the adjacency.
      3. Multiplies input features by A (via einsum).
      4. Applies an MLP to the output features.

    Args:
        in_channels (int):   Number of input channels.
        out_channels (int):  Number of output channels.
        A_binary (ndarray):  Base adjacency matrix (V x V).
        use_mask (bool):     Whether to learn an additive mask for A.
        dropout (float):     Dropout rate for MLP.
        activation (str):    Activation function for MLP (e.g., 'relu').

    Input shape:
        x of shape (N, C, T, V)
          - N: Batch size
          - C: Channels (in_channels)
          - T: Temporal dimension
          - V: Number of nodes

    Output shape:
        (N, out_channels, T, V)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 use_mask=True,
                 dropout=0.0,
                 activation='relu'):
        super().__init__()

        # Create single adjacency with self-loops.
        # Then normalize it.
        A = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        A = normalize_adjacency_matrix(A)
        self.A = torch.tensor(A, dtype=torch.float32)

        self.use_mask = use_mask
        if self.use_mask:
            # Small random initialization for the adjacency mask.
            self.A_res = nn.Parameter(
                nn.init.uniform_(torch.Tensor(*self.A.shape), -1e-6, 1e-6)
            )

        # MLP: input dimension = in_channels, output dimension = out_channels
        # We apply dropout + activation as configured.
        self.mlp = MLP(
            in_channels, 
            [out_channels], 
            dropout=dropout, 
            activation=activation
        )

    def forward(self, x):
        """
        x shape: (N, C, T, V)
        """
        # Move adjacency to the same device/dtype as x
        A = self.A.to(x.device, x.dtype)
        if self.use_mask:
            A = A + self.A_res.to(x.device, x.dtype)

        # Multiply features by adjacency on node dimension: 'vu,nctu->nctv'
        #  v -> node out dimension
        #  u -> node in dimension
        #  n -> batch index
        #  c -> channel index
        #  t -> time
        support = torch.einsum('vu,nctu->nctv', A, x)  # (N, C, T, V)

        # Apply MLP on the channel dimension
        out = self.mlp(support)  # -> (N, out_channels, T, V)

        return out