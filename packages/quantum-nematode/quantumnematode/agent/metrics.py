"""Episode metrics tracking for the quantum nematode agent."""

from __future__ import annotations

from quantumnematode.report.dtypes import PerformanceMetrics, TerminationReason


class MetricsTracker:
    """Tracks performance metrics across episodes.

    The metrics tracker accumulates statistics during episode execution,
    including success rate, step counts, rewards, and environment-specific
    metrics like food collection efficiency.

    Attributes
    ----------
    success_count : int
        Number of successful episodes (goal reached).
    total_steps : int
        Total steps taken across all episodes.
    total_rewards : float
        Cumulative rewards across all episodes.
    foods_collected : int
        Total number of foods collected (for dynamic environments).
    distance_efficiencies : list[float]
        Distance efficiency for each food collected (for dynamic environments).
    """

    def __init__(self) -> None:
        """Initialize the metrics tracker with zero counters."""
        self.success_count = 0
        self.total_steps = 0
        self.total_rewards = 0.0
        self.foods_collected = 0
        self.distance_efficiencies: list[float] = []
        # Predator tracking
        self.total_predator_encounters = 0
        self.total_successful_evasions = 0
        self.total_predator_deaths = 0
        self.total_starved = 0
        self.total_max_steps = 0
        self.total_interrupted = 0

    def track_episode_completion(  # noqa: PLR0913 - comprehensive episode tracking requires many parameters
        self,
        success: bool,  # noqa: FBT001 - simple boolean flag is clearest API
        steps: int,
        reward: float = 0.0,
        foods_collected: int = 0,
        distance_efficiencies: list[float] | None = None,
        predator_encounters: int = 0,
        successful_evasions: int = 0,
        termination_reason: TerminationReason | None = None,
    ) -> None:
        """Track the completion of an episode.

        Parameters
        ----------
        success : bool
            Whether the episode ended successfully (goal reached).
        steps : int
            Number of steps taken in the episode.
        reward : float
            Total reward received for the episode.
        foods_collected : int, optional
            Number of foods collected during the episode (dynamic environments only).
        distance_efficiencies : list[float] | None, optional
            For dynamic environments, list of distance efficiencies for foods
            collected during the episode. None for static environments.
        predator_encounters : int, optional
            Number of predator encounters in this episode.
        successful_evasions : int, optional
            Number of successful evasions in this episode.
        termination_reason : TerminationReason | None, optional
            Reason the episode terminated.
        """
        if success:
            self.success_count += 1
        self.total_steps += steps
        self.total_rewards += reward
        self.foods_collected += foods_collected
        if distance_efficiencies is not None:
            self.distance_efficiencies.extend(distance_efficiencies)

        # Track predator-related metrics
        self.total_predator_encounters += predator_encounters
        self.total_successful_evasions += successful_evasions

        # Track termination reasons
        if termination_reason:
            if termination_reason == TerminationReason.PREDATOR:
                self.total_predator_deaths += 1
            elif termination_reason == TerminationReason.STARVED:
                self.total_starved += 1
            elif termination_reason == TerminationReason.MAX_STEPS:
                self.total_max_steps += 1
            elif termination_reason == TerminationReason.INTERRUPTED:
                self.total_interrupted += 1

    def calculate_metrics(
        self,
        total_runs: int,
        predators_enabled: bool = False,  # noqa: FBT001, FBT002 - flag needed to distinguish predator/non-predator envs
    ) -> PerformanceMetrics:
        """Calculate final performance metrics.

        Parameters
        ----------
        total_runs : int
            Total number of episodes/runs executed.
        predators_enabled : bool, optional
            Whether predators are enabled in the environment (default: False).
            When True, predator metrics will be 0.0 for zero encounters.
            When False, predator metrics will be None (non-predator environment).

        Returns
        -------
        PerformanceMetrics
            Calculated performance metrics including success rate, average steps,
            average reward, and foraging efficiency.
        """
        success_rate = self.success_count / total_runs if total_runs > 0 else 0.0
        average_steps = self.total_steps / total_runs if total_runs > 0 else 0.0
        average_reward = self.total_rewards / total_runs if total_runs > 0 else 0.0

        # Calculate foraging efficiency (foods per run)
        # Note: This differs from foods_per_step which is tracked separately
        foraging_efficiency = self.foods_collected / total_runs if total_runs > 0 else 0.0

        # Calculate average distance efficiency
        average_distance_efficiency = None
        if self.distance_efficiencies:
            average_distance_efficiency = sum(self.distance_efficiencies) / len(
                self.distance_efficiencies,
            )

        # Calculate average foods collected per run (only if in dynamic environment)
        average_foods_collected = None
        if self.foods_collected > 0 and total_runs > 0:
            average_foods_collected = self.foods_collected / total_runs

        # Calculate predator metrics
        # Distinguish between predator-enabled environments (0.0) and non-predator (None)
        average_predator_encounters = None
        average_successful_evasions = None
        if predators_enabled and total_runs > 0:
            # Predator-enabled environment: use 0.0 for zero encounters
            average_predator_encounters = self.total_predator_encounters / total_runs
            average_successful_evasions = self.total_successful_evasions / total_runs
        # else: Non-predator environment: keep as None

        return PerformanceMetrics(
            success_rate=success_rate,
            average_steps=average_steps,
            average_reward=average_reward,
            foraging_efficiency=foraging_efficiency,
            average_distance_efficiency=average_distance_efficiency,
            average_foods_collected=average_foods_collected,
            total_successes=self.success_count,
            total_starved=self.total_starved,
            total_predator_deaths=self.total_predator_deaths,
            total_max_steps=self.total_max_steps,
            total_interrupted=self.total_interrupted,
            average_predator_encounters=average_predator_encounters,
            average_successful_evasions=average_successful_evasions,
        )

    def reset(self) -> None:
        """Reset all metrics except success count to zero."""
        self.total_steps = 0
        self.total_rewards = 0.0
        self.foods_collected = 0
        self.distance_efficiencies = []
        self.total_predator_encounters = 0
        self.total_successful_evasions = 0
        self.total_predator_deaths = 0
        self.total_starved = 0
        self.total_max_steps = 0
        self.total_interrupted = 0
