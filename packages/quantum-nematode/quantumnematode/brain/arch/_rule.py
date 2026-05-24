"""Learning-rule Protocol for brain architectures.

A ``LearningRule`` encapsulates how a topology's weights are updated from
experience. It owns the optimiser, value head (if any), replay buffer (if
any), advantage estimator (if any), and gradient clipper (if any). The
paired ``BrainTopology`` is pure structure; the rule owns the mechanism
that mutates the topology's learnable weights from collected experience.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from quantumnematode.brain.arch._topology import BrainTopology


class RuleStepReport(BaseModel):
    """Summary of a single learning-rule step.

    Fields are optional because not every rule produces every component
    (e.g. REINFORCE has no value loss; DQN has no entropy term).
    """

    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    total_loss: float | None = None
    grad_norm: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class LearningRule(Protocol):
    """How a topology's weights are updated from experience.

    The rule owns optimiser state, replay buffers, value heads, advantage
    estimators, and gradient clippers. Per-episode lifecycle is handled
    via ``reset_episode``.
    """

    def step(self, topology: BrainTopology, batch: Any) -> RuleStepReport:  # noqa: ANN401 — rule-specific experience shape
        """Apply one update to the topology's learnable weights.

        Computes gradients with respect to ``topology`` parameters, runs
        the optimiser update, and projects any updated weights through
        ``topology.apply_weight_mask`` to honour the topology's edge mask
        (a no-op for dense topologies).
        """
        ...

    def reset_episode(self) -> None:
        """Clear per-episode rule state at episode start.

        Clears advantage-estimator buffers and any rule-owned recurrent
        caches; SHALL NOT clear the optimiser state or the persistent
        replay buffer.
        """
        ...
