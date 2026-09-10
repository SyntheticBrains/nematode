"""The perturbation scale's decay over episodes, shared by every plastic topology.

The scale has two incompatible jobs. As a probe it must be large enough to move behaviour,
or the eligibility it produces is indistinguishable from noise. As jitter on a competent
policy it is damage, and a scale large enough to learn with can take such a policy apart
before the rule acts. Nothing in the estimator requires one value to serve both, so the
scale may decay: explore at the scale that learns, and run at the scale that can be run.

The path is geometric rather than linear because the scale's effect on both jobs is closer
to multiplicative than additive, and because a geometric path spends more of its length near
the small values, where retention is decided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeNoiseSchedule:
    """A geometric decay from ``initial`` to ``final`` over ``episodes`` steps.

    A step is whatever the caller counts as the start of one unit of training: an episode
    for a brain, a trial for a harness driving a topology directly. The schedule reaches
    ``final`` exactly at ``episodes`` and is constant after, so an arm's endpoint is a stated
    scale rather than wherever the decay happened to reach.
    """

    initial: float
    final: float
    episodes: int

    def __post_init__(self) -> None:
        """Refuse a schedule that is not a decay to a positive floor.

        The configuration refuses these too, but a schedule can be built directly by a
        harness that never loads a config, and a silently inverted or zero-floored one would
        anneal the wrong way or restore the Hebbian eligibility.
        """
        if not math.isfinite(self.initial) or not math.isfinite(self.final):
            msg = (
                f"schedule bounds must be finite, got initial={self.initial}, "
                f"final={self.final}: a non-finite bound makes every scale it produces "
                "not-a-number, which perturbs nothing and silently voids the arm."
            )
            raise ValueError(msg)
        if self.episodes <= 0:
            msg = f"anneal length must be positive, got {self.episodes}"
            raise ValueError(msg)
        if self.final <= 0.0:
            msg = (
                f"final scale must be positive, got {self.final}: a scale of zero does not "
                "silence the eligibility, it returns it to pre-synaptic times post-synaptic "
                "activity."
            )
            raise ValueError(msg)
        if self.final >= self.initial:
            msg = f"final scale {self.final} must be below initial scale {self.initial}"
            raise ValueError(msg)

    def scale_at(self, step: int) -> float:
        """Return the scale for a step index, counting from zero."""
        if step >= self.episodes:
            return self.final
        if step <= 0:
            return self.initial
        return self.initial * (self.final / self.initial) ** (step / self.episodes)
