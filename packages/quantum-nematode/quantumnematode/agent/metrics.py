"""Episode metrics tracking for the quantum nematode agent."""

from __future__ import annotations

from quantumnematode.report.dtypes import PerformanceMetrics


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

    def track_episode_completion(
        self,
        success: bool,  # noqa: FBT001 - simple boolean flag is clearest API
        reward: float = 0.0,
    ) -> None:
        """Track the completion of an episode.

        Parameters
        ----------
        success : bool
            Whether the episode ended successfully (goal reached).
        reward : float
            Reward received for the last step.
        """
        if success:
            self.success_count += 1
        self.total_rewards += reward

    def track_food_collection(self, distance_efficiency: float | None = None) -> None:
        """Track food collection event.

        Parameters
        ----------
        distance_efficiency : float | None, optional
            For dynamic environments, the ratio of optimal distance to actual
            distance traveled. None for static environments.
        """
        self.foods_collected += 1
        if distance_efficiency is not None:
            self.distance_efficiencies.append(distance_efficiency)

    def track_reward(self, reward: float) -> None:
        """Track a single reward.

        Parameters
        ----------
        reward : float
            Reward received for this instance.
        """
        self.total_rewards += reward

    def track_step(self, reward: float = 0.0) -> None:
        """Track a single step.

        Parameters
        ----------
        reward : float
            Reward received for this step.
        """
        self.total_steps += 1
        self.total_rewards += reward

    def calculate_metrics(self, total_runs: int) -> PerformanceMetrics:
        """Calculate final performance metrics.

        Parameters
        ----------
        total_runs : int
            Total number of episodes/runs executed.

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

        return PerformanceMetrics(
            success_rate=success_rate,
            average_steps=average_steps,
            average_reward=average_reward,
            foraging_efficiency=foraging_efficiency,
            average_distance_efficiency=average_distance_efficiency,
            average_foods_collected=average_foods_collected,
        )

    def reset(self) -> None:
        """Reset all metrics except success count to zero."""
        self.total_steps = 0
        self.total_rewards = 0.0
        self.foods_collected = 0
        self.distance_efficiencies = []
