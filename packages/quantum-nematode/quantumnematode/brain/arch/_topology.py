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


@runtime_checkable
class PlasticTopology(Protocol):
    """What a local plasticity rule touches, and nothing else.

    A three-factor rule needs the weights it may change, the eligibility
    trace accumulated for each of them, and the edge mask each lives on.
    Exposing exactly that lets one rule drive substrates that look nothing
    alike -- a single recurrent 302x302 matrix over a sparse anatomical
    edge set, or a stack of dense feedforward layers -- without the rule
    naming either. "Substrate-generic" then describes the code, not just
    the equation.

    The lists are **aligned**: entry ``i`` of each refers to the same
    plastic tensor, and traces and masks have that tensor's shape. They are
    lists from the outset so a substrate with one plastic tensor and one
    with one per layer share a code path. A dense substrate exposes an
    all-true mask rather than none, so mask-dependent telemetry (the
    saturated fraction) means the same thing everywhere.
    """

    @property
    def enable_activity_traces(self) -> bool:
        """Whether eligibility traces are allocated and accumulating."""
        ...

    @property
    def plastic_weights(self) -> list[torch.Tensor]:
        """The weight tensors a plasticity rule may update, in a fixed order."""
        ...

    @property
    def eligibility_traces(self) -> list[torch.Tensor]:
        """One trace per plastic weight, aligned and shape-matched."""
        ...

    @property
    def plastic_masks(self) -> list[torch.Tensor]:
        """One boolean edge mask per plastic weight, aligned and shape-matched."""
        ...

    @property
    def plastic_fan_in_axes(self) -> list[int]:
        """Per plastic weight, the axis to reduce over for one unit's incoming weights."""
        ...

    @property
    def plastic_perturbations(self) -> list[torch.Tensor]:
        """Per plastic weight, the perturbation its post-synaptic units acted on last step.

        Empty unless the topology is perturbing. Indexed like
        ``plastic_post_activities``, along the axis complementary to the fan-in
        axis. The perturbation is added to the unit's PRE-activation, so the
        unit acts on it through its own nonlinearity -- an eligibility built
        from a perturbation the unit did not act on describes a counterfactual
        the network never took, and the resulting estimator is biased.
        """
        ...

    @property
    def plastic_post_activities(self) -> list[torch.Tensor]:
        """Per plastic weight, the activity of its post-synaptic units at the last trace step.

        Indexed along the axis complementary to that weight's fan-in axis, and
        the same vector the trace's post-synaptic factor was built from, so a
        rule term reading it and the Hebbian term agree on what the unit did.
        Read only by terms that need it; a substrate pays nothing for exposing
        a view over state it already keeps.
        """
        ...

    def reset_traces(self) -> None:
        """Zero every eligibility trace; a documented no-op when traces are off."""
        ...

    def advance_schedule(self) -> None:
        """Advance the perturbation schedule by one step; a no-op without one.

        Called where a unit of training begins -- an episode for a brain, a trial for a
        harness driving the topology directly. Deliberately separate from
        ``reset_traces``: the traces are also reset when a policy is loaded, and a load
        must restart the schedule rather than advance it.
        """
        ...

    def reset_schedule(self) -> None:
        """Return the perturbation schedule to its initial scale; a no-op without one.

        Called where a policy is loaded: a warm start explores the loaded weights from the
        initial scale rather than from wherever a previous run's schedule had reached.
        """
        ...
