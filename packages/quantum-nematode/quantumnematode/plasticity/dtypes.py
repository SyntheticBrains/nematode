"""Data structures for plasticity evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Metrics from a single evaluation block."""

    objective_name: str
    transition_point: str
    mean_success_rate: float
    mean_reward: float
    mean_steps: float


@dataclass
class PhaseTrainingResult:
    """Per-episode training metrics for a single phase."""

    phase_name: str
    episode_successes: list[bool] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)


@dataclass
class SeedResult:
    """All results for a single seed run."""

    seed: int
    training_results: list[PhaseTrainingResult] = field(default_factory=list)
    eval_results: list[EvalResult] = field(default_factory=list)
    backward_forgetting: float | None = None
    forward_transfer: float | None = None
    plasticity_retention: float | None = None
