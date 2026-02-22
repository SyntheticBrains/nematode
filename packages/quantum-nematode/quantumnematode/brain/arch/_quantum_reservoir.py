"""
Shared Quantum Reservoir Utilities.

Common components for quantum reservoir-based brain architectures (QRC, QRH).
Extracted from qrc.py to enable reuse without duplication.

Functions
---------
- build_readout_network: Build a classical readout MLP or linear layer with
  orthogonal weight initialization.
"""

from __future__ import annotations

import numpy as np
from torch import nn


def build_readout_network(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    readout_type: str = "mlp",
    num_layers: int = 1,
) -> nn.Module:
    """Build a classical readout network with orthogonal initialization.

    Constructs either an MLP or linear readout layer for mapping quantum
    reservoir features to action logits or value estimates.

    Parameters
    ----------
    input_dim : int
        Dimension of input features (e.g., 2^N for QRC, N+N(N-1)/2 for QRH).
    hidden_dim : int
        Number of hidden units per layer (used only for MLP readout).
    output_dim : int
        Number of output units (e.g., num_actions or 1 for value).
    readout_type : str
        Type of readout: "mlp" or "linear" (default "mlp").
    num_layers : int
        Number of hidden layers in MLP readout (default 1).

    Returns
    -------
    nn.Module
        The readout network with orthogonal weight initialization.
    """
    if readout_type == "mlp":
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        network: nn.Module = nn.Sequential(*layers)
    else:
        network = nn.Linear(input_dim, output_dim)

    # Initialize weights for better gradient flow
    for module in network.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return network
