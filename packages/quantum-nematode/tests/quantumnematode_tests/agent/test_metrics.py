"""Tests for the metrics tracking system."""

import pytest
from quantumnematode.agent.metrics import MetricsTracker
from quantumnematode.report.dtypes import TerminationReason


class TestMetricsTrackerInitialization:
    """Test metrics tracker initialization."""

    def test_initialize_with_zeros(self):
        """Test that tracker initializes with zero counters."""
        tracker = MetricsTracker()

        assert tracker.success_count == 0
        assert tracker.total_steps == 0
        assert tracker.total_rewards == 0.0
        assert tracker.foods_collected == 0
        assert tracker.distance_efficiencies == []
        # Verify predator tracking counters
        assert tracker.total_predator_encounters == 0
        assert tracker.total_successful_evasions == 0
        # Verify termination reason counters
        assert tracker.total_predator_deaths == 0
        assert tracker.total_starved == 0
        assert tracker.total_max_steps == 0
        assert tracker.total_interrupted == 0


class TestEpisodeCompletion:
    """Test episode completion tracking."""

    def test_track_successful_episode(self):
        """Test tracking a successful episode."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=True, steps=5, reward=10.5)

        assert tracker.success_count == 1
        assert tracker.total_steps == 5
        assert tracker.total_rewards == pytest.approx(10.5)

    def test_track_failed_episode(self):
        """Test tracking a failed episode."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=False, steps=3, reward=-5.0)

        assert tracker.success_count == 0
        assert tracker.total_steps == 3
        assert tracker.total_rewards == pytest.approx(-5.0)

    def test_track_multiple_episodes(self):
        """Test tracking multiple episodes accumulates correctly."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=True, steps=5, reward=10.0)
        tracker.track_episode_completion(success=False, steps=3, reward=5.0)
        tracker.track_episode_completion(success=True, steps=4, reward=15.0)

        assert tracker.success_count == 2
        assert tracker.total_steps == 12
        assert tracker.total_rewards == pytest.approx(30.0)


class TestMetricsCalculation:
    """Test final metrics calculation."""

    def test_calculate_metrics_with_data(self):
        """Test metrics calculation with tracked data."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=True, steps=5, reward=10.0)
        tracker.track_episode_completion(success=False, steps=10, reward=5.0)
        tracker.track_episode_completion(success=True, steps=15, reward=15.0)

        tracker.foods_collected = 2

        metrics = tracker.calculate_metrics(total_runs=3)

        assert metrics.success_rate == pytest.approx(2 / 3)
        assert metrics.average_reward == pytest.approx(30.0 / 3)
        assert metrics.average_steps == pytest.approx(30.0 / 3)
        assert metrics.foraging_efficiency == pytest.approx(2 / 3)

    def test_calculate_metrics_no_data(self):
        """Test metrics calculation with no tracked data."""
        tracker = MetricsTracker()

        metrics = tracker.calculate_metrics(total_runs=5)

        assert metrics.success_rate == 0.0
        assert metrics.average_steps == 0.0
        assert metrics.average_reward == 0.0
        assert metrics.foraging_efficiency == 0.0

    def test_calculate_metrics_zero_runs(self):
        """Test metrics calculation with zero runs."""
        tracker = MetricsTracker()
        tracker.track_episode_completion(success=True, steps=5, reward=10.0)

        metrics = tracker.calculate_metrics(total_runs=0)

        assert metrics.success_rate == 0.0
        assert metrics.average_steps == 0.0
        assert metrics.average_reward == 0.0
        assert metrics.foraging_efficiency == 0.0

    def test_calculate_metrics_all_successful(self):
        """Test metrics with all successful episodes."""
        tracker = MetricsTracker()

        for _ in range(10):
            tracker.track_episode_completion(success=True, steps=5, reward=5.5)

        metrics = tracker.calculate_metrics(total_runs=10)

        assert metrics.success_rate == 1.0
        assert metrics.average_steps == pytest.approx(5.0)
        assert metrics.average_reward == pytest.approx(5.5)

    def test_calculate_metrics_all_failed(self):
        """Test metrics with all failed episodes."""
        tracker = MetricsTracker()

        for _ in range(5):
            tracker.track_episode_completion(success=False, steps=10, reward=-2.0)

        metrics = tracker.calculate_metrics(total_runs=5)

        assert metrics.success_rate == 0.0
        assert metrics.average_steps == pytest.approx(10.0)
        assert metrics.average_reward == pytest.approx(-2.0)


class TestMetricsReset:
    """Test metrics reset functionality."""

    def test_reset_clears_all_data(self):
        """Test that reset clears all tracked data."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=True, steps=5, reward=10.0)

        tracker.reset()

        assert tracker.success_count == 1  # Success count is not reset
        assert tracker.total_steps == 0
        assert tracker.total_rewards == 0.0
        assert tracker.foods_collected == 0
        assert tracker.distance_efficiencies == []
        # Verify predator tracking counters are reset
        assert tracker.total_predator_encounters == 0
        assert tracker.total_successful_evasions == 0
        # Verify termination reason counters are reset
        assert tracker.total_predator_deaths == 0
        assert tracker.total_starved == 0
        assert tracker.total_max_steps == 0
        assert tracker.total_interrupted == 0

    def test_reset_allows_fresh_tracking(self):
        """Test that tracking works correctly after reset."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=True, steps=5, reward=10.0)
        tracker.reset()
        tracker.track_episode_completion(success=False, steps=3, reward=5.0)

        assert tracker.success_count == 1  # Success count is not reset
        assert tracker.total_steps == 3
        assert tracker.total_rewards == pytest.approx(5.0)


class TestMetricsForagingEfficiency:
    """Test foraging efficiency calculations."""

    def test_foraging_efficiency_with_foods(self):
        """Test foraging efficiency when foods are collected."""
        tracker = MetricsTracker()

        for _ in range(15):
            tracker.foods_collected += 1

        metrics = tracker.calculate_metrics(total_runs=5)

        assert metrics.foraging_efficiency == pytest.approx(3.0)  # 15 foods / 5 runs

    def test_foraging_efficiency_without_foods(self):
        """Test foraging efficiency when no foods collected."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=False, steps=3, reward=0.0)

        metrics = tracker.calculate_metrics(total_runs=1)

        assert metrics.foraging_efficiency == 0.0

    def test_foraging_efficiency_fractional(self):
        """Test foraging efficiency with fractional results."""
        tracker = MetricsTracker()

        tracker.foods_collected = 2

        metrics = tracker.calculate_metrics(total_runs=3)

        assert metrics.foraging_efficiency == pytest.approx(2 / 3)


class TestPredatorMetrics:
    """Test predator-related metrics calculation."""

    def test_predator_metrics_non_predator_environment(self):
        """Test that predator metrics are None for non-predator environments."""
        tracker = MetricsTracker()

        # Track episodes without predator encounters
        tracker.track_episode_completion(success=True, steps=5, reward=10.0)
        tracker.track_episode_completion(success=True, steps=6, reward=12.0)

        metrics = tracker.calculate_metrics(total_runs=2, predators_enabled=False)

        # Non-predator environment: metrics should be None
        assert metrics.average_predator_encounters is None
        assert metrics.average_successful_evasions is None

    def test_predator_metrics_enabled_zero_encounters(self):
        """Test that predator metrics are 0.0 for predator environments with zero encounters."""
        tracker = MetricsTracker()

        # Track episodes in predator environment but no encounters
        tracker.track_episode_completion(success=True, steps=5, reward=10.0)
        tracker.track_episode_completion(success=True, steps=6, reward=12.0)

        metrics = tracker.calculate_metrics(total_runs=2, predators_enabled=True)

        # Predator-enabled environment with zero encounters: metrics should be 0.0
        assert metrics.average_predator_encounters == 0.0
        assert metrics.average_successful_evasions == 0.0

    def test_predator_metrics_enabled_with_encounters(self):
        """Test predator metrics calculation with encounters."""
        tracker = MetricsTracker()

        # Track episodes with predator encounters
        tracker.track_episode_completion(
            success=True,
            steps=10,
            reward=5.0,
            predator_encounters=3,
            successful_evasions=2,
        )
        tracker.track_episode_completion(
            success=True,
            steps=8,
            reward=8.0,
            predator_encounters=1,
            successful_evasions=1,
        )

        metrics = tracker.calculate_metrics(total_runs=2, predators_enabled=True)

        # Average: (3+1)/2 = 2.0 encounters, (2+1)/2 = 1.5 evasions
        assert metrics.average_predator_encounters == pytest.approx(2.0)
        assert metrics.average_successful_evasions == pytest.approx(1.5)

    def test_predator_metrics_distinction(self):
        """Test that we can distinguish predator-enabled from non-predator environments."""
        tracker1 = MetricsTracker()
        tracker2 = MetricsTracker()

        # Both track same episodes (no encounters)
        for tracker in [tracker1, tracker2]:
            tracker.track_episode_completion(success=True, steps=5, reward=10.0)

        metrics_no_predators = tracker1.calculate_metrics(
            total_runs=1,
            predators_enabled=False,
        )
        metrics_with_predators = tracker2.calculate_metrics(
            total_runs=1,
            predators_enabled=True,
        )

        # Non-predator: None, Predator-enabled: 0.0
        assert metrics_no_predators.average_predator_encounters is None
        assert metrics_with_predators.average_predator_encounters == 0.0
        assert metrics_no_predators.average_successful_evasions is None
        assert metrics_with_predators.average_successful_evasions == 0.0


class TestTerminationReasonTracking:
    """Test termination reason tracking and counters."""

    def test_termination_reason_predator(self):
        """Test that predator deaths are tracked correctly."""
        tracker = MetricsTracker()

        # Track episodes with predator deaths
        tracker.track_episode_completion(
            success=False,
            steps=10,
            reward=-50.0,
            termination_reason=TerminationReason.PREDATOR,
        )
        tracker.track_episode_completion(
            success=False,
            steps=8,
            reward=-50.0,
            termination_reason=TerminationReason.PREDATOR,
        )

        assert tracker.total_predator_deaths == 2
        assert tracker.total_starved == 0
        assert tracker.total_max_steps == 0
        assert tracker.total_interrupted == 0

        metrics = tracker.calculate_metrics(total_runs=2)
        assert metrics.total_predator_deaths == 2

    def test_termination_reason_starved(self):
        """Test that starvation deaths are tracked correctly."""
        tracker = MetricsTracker()

        # Track episodes with starvation
        tracker.track_episode_completion(
            success=False,
            steps=15,
            reward=10.0,
            termination_reason=TerminationReason.STARVED,
        )
        tracker.track_episode_completion(
            success=False,
            steps=12,
            reward=8.0,
            termination_reason=TerminationReason.STARVED,
        )
        tracker.track_episode_completion(
            success=False,
            steps=18,
            reward=12.0,
            termination_reason=TerminationReason.STARVED,
        )

        assert tracker.total_starved == 3
        assert tracker.total_predator_deaths == 0
        assert tracker.total_max_steps == 0
        assert tracker.total_interrupted == 0

        metrics = tracker.calculate_metrics(total_runs=3)
        assert metrics.total_starved == 3

    def test_termination_reason_max_steps(self):
        """Test that max steps terminations are tracked correctly."""
        tracker = MetricsTracker()

        # Track episodes that hit max steps
        tracker.track_episode_completion(
            success=False,
            steps=1000,
            reward=20.0,
            termination_reason=TerminationReason.MAX_STEPS,
        )

        assert tracker.total_max_steps == 1
        assert tracker.total_predator_deaths == 0
        assert tracker.total_starved == 0
        assert tracker.total_interrupted == 0

        metrics = tracker.calculate_metrics(total_runs=1)
        assert metrics.total_max_steps == 1

    def test_termination_reason_interrupted(self):
        """Test that interrupted episodes are tracked correctly."""
        tracker = MetricsTracker()

        # Track interrupted episodes
        tracker.track_episode_completion(
            success=False,
            steps=50,
            reward=5.0,
            termination_reason=TerminationReason.INTERRUPTED,
        )
        tracker.track_episode_completion(
            success=False,
            steps=75,
            reward=8.0,
            termination_reason=TerminationReason.INTERRUPTED,
        )

        assert tracker.total_interrupted == 2
        assert tracker.total_predator_deaths == 0
        assert tracker.total_starved == 0
        assert tracker.total_max_steps == 0

        metrics = tracker.calculate_metrics(total_runs=2)
        assert metrics.total_interrupted == 2

    def test_termination_reason_goal_reached(self):
        """Test that goal reached doesn't increment failure counters."""
        tracker = MetricsTracker()

        # Track successful episodes (goal reached)
        tracker.track_episode_completion(
            success=True,
            steps=25,
            reward=100.0,
            termination_reason=TerminationReason.GOAL_REACHED,
        )

        # No failure counters should be incremented
        assert tracker.total_predator_deaths == 0
        assert tracker.total_starved == 0
        assert tracker.total_max_steps == 0
        assert tracker.total_interrupted == 0

        metrics = tracker.calculate_metrics(total_runs=1)
        assert metrics.total_successes == 1
        assert metrics.total_predator_deaths == 0
        assert metrics.total_starved == 0
        assert metrics.total_max_steps == 0
        assert metrics.total_interrupted == 0

    def test_termination_reason_completed_all_food(self):
        """Test that completing all food doesn't increment failure counters."""
        tracker = MetricsTracker()

        # Track successful episodes (all food collected)
        tracker.track_episode_completion(
            success=True,
            steps=150,
            reward=250.0,
            foods_collected=10,
            termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
        )

        # No failure counters should be incremented
        assert tracker.total_predator_deaths == 0
        assert tracker.total_starved == 0
        assert tracker.total_max_steps == 0
        assert tracker.total_interrupted == 0

        metrics = tracker.calculate_metrics(total_runs=1)
        assert metrics.total_successes == 1
        assert metrics.total_predator_deaths == 0

    def test_termination_reason_mixed_scenarios(self):
        """Test tracking multiple different termination reasons."""
        tracker = MetricsTracker()

        # Mix of different termination reasons
        tracker.track_episode_completion(
            success=True,
            steps=20,
            reward=100.0,
            termination_reason=TerminationReason.GOAL_REACHED,
        )
        tracker.track_episode_completion(
            success=False,
            steps=10,
            reward=-50.0,
            termination_reason=TerminationReason.PREDATOR,
        )
        tracker.track_episode_completion(
            success=False,
            steps=15,
            reward=5.0,
            termination_reason=TerminationReason.STARVED,
        )
        tracker.track_episode_completion(
            success=False,
            steps=1000,
            reward=30.0,
            termination_reason=TerminationReason.MAX_STEPS,
        )
        tracker.track_episode_completion(
            success=False,
            steps=50,
            reward=10.0,
            termination_reason=TerminationReason.INTERRUPTED,
        )
        tracker.track_episode_completion(
            success=False,
            steps=12,
            reward=-50.0,
            termination_reason=TerminationReason.PREDATOR,
        )

        assert tracker.success_count == 1
        assert tracker.total_predator_deaths == 2
        assert tracker.total_starved == 1
        assert tracker.total_max_steps == 1
        assert tracker.total_interrupted == 1

        metrics = tracker.calculate_metrics(total_runs=6)
        assert metrics.success_rate == pytest.approx(1 / 6)
        assert metrics.total_successes == 1
        assert metrics.total_predator_deaths == 2
        assert metrics.total_starved == 1
        assert metrics.total_max_steps == 1
        assert metrics.total_interrupted == 1

    def test_termination_reason_none_doesnt_increment_counters(self):
        """Test that None termination reason doesn't increment any counters."""
        tracker = MetricsTracker()

        # Track episodes without termination reason
        tracker.track_episode_completion(
            success=True,
            steps=20,
            reward=50.0,
            termination_reason=None,
        )
        tracker.track_episode_completion(
            success=False,
            steps=10,
            reward=-10.0,
            termination_reason=None,
        )

        # No termination counters should be incremented
        assert tracker.total_predator_deaths == 0
        assert tracker.total_starved == 0
        assert tracker.total_max_steps == 0
        assert tracker.total_interrupted == 0

        metrics = tracker.calculate_metrics(total_runs=2)
        assert metrics.total_predator_deaths == 0
        assert metrics.total_starved == 0
        assert metrics.total_max_steps == 0
        assert metrics.total_interrupted == 0

    def test_termination_counters_reset_behavior(self):
        """Test that reset clears termination counters."""
        tracker = MetricsTracker()

        # Track episodes with different termination reasons
        tracker.track_episode_completion(
            success=False,
            steps=10,
            reward=-50.0,
            termination_reason=TerminationReason.PREDATOR,
        )
        tracker.track_episode_completion(
            success=False,
            steps=15,
            reward=5.0,
            termination_reason=TerminationReason.STARVED,
        )

        assert tracker.total_predator_deaths == 1
        assert tracker.total_starved == 1

        # Reset should clear termination counters
        tracker.reset()

        assert tracker.total_predator_deaths == 0
        assert tracker.total_starved == 0
        assert tracker.total_max_steps == 0
        assert tracker.total_interrupted == 0
