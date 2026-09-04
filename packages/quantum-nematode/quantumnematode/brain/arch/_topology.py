"""Topology Protocol for brain architectures.

A ``BrainTopology`` exposes the structural seam a learning rule needs — the
weight-mask projector and the learnable parameters — factored out from
learning-rule concerns (optimisers, replay buffers, value heads). The same
topology can be paired with different learning rules; the same learning rule
can drive different topologies.

Forward-pass signatures are deliberately NOT part of the Protocol: they are
topology-specific (``ConnectomeTopology`` takes multi-channel sensor
features, not a single ``x``). A rule that needs to re-forward experience
under current weights — as PPO does once per minibatch per epoch — calls its
concrete topology's own methods; that surface is beyond the Protocol, which
carries only the seam every rule shares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from torch import nn


@runtime_checkable
class BrainTopology(Protocol):
    """Structural seam between a brain's network and its learning rule.

    Implementations carry weight tensors as state and expose the two things
    a rule genuinely touches: the parameters it may update and the projector
    that keeps updated weights on the topology's allowed manifold. Forward
    passes stay free of optimiser, replay-buffer, or value-head side
    effects — those belong to the paired ``LearningRule``.
    """

    @property
    def learnable_parameters(self) -> list[nn.Parameter]:
        """Parameters a learning rule may update.

        Reflects the topology's enabled optional blocks (e.g. predator /
        thermotaxis projections, continuous ``log_std``): disabled blocks
        contribute nothing, so optimisers see byte-identical parameter
        sets across builds that differ only in disabled options.
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
