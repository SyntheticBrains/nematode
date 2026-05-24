"""Topology Protocol for brain architectures.

A ``BrainTopology`` exposes the structural and forward-pass interface of a
brain's network, factored out from learning-rule concerns (optimisers,
replay buffers, value heads). The same topology can be paired with different
learning rules; the same learning rule can drive different topologies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class BrainTopology(Protocol):
    """Structural and forward-pass interface of a brain's network.

    Implementations carry weight tensors as state. The forward pass is free
    of optimiser, replay-buffer, or value-head side effects — those belong
    to the paired ``LearningRule``.
    """

    n_inputs: int
    n_outputs: int
    n_hidden: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass through the topology.

        Returns a tensor of shape compatible with ``n_outputs``. The call
        SHALL NOT mutate optimiser state, replay-buffer state, or
        value-head state — those mutations belong on the learning rule.
        """
        ...

    def apply_weight_mask(self, weights: torch.Tensor) -> torch.Tensor:
        """Project a candidate weight tensor onto the topology's allowed manifold.

        For dense topologies the default is the identity function. For
        sparse/strict-mask topologies (e.g. connectome-constrained), this
        zeros out weights along non-existent edges. Called by the paired
        learning rule after every optimiser step on the topology's
        masked-weight tensor.
        """
        ...
