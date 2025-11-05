"""Tests for the metrics tracking system."""

import pytest
from quantumnematode.metrics import MetricsTracker


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


class TestEpisodeCompletion:
    """Test episode completion tracking."""

    def test_track_successful_episode(self):
        """Test tracking a successful episode."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=True, reward=10.5)

        assert tracker.success_count == 1
        assert tracker.total_rewards == pytest.approx(10.5)

    def test_track_failed_episode(self):
        """Test tracking a failed episode."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=False, reward=-5.0)

        assert tracker.success_count == 0
        assert tracker.total_rewards == pytest.approx(-5.0)

    def test_track_multiple_episodes(self):
        """Test tracking multiple episodes accumulates correctly."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=True, reward=10.0)
        tracker.track_episode_completion(success=False, reward=5.0)
        tracker.track_episode_completion(success=True, reward=15.0)

        assert tracker.success_count == 2
        assert tracker.total_rewards == pytest.approx(30.0)


class TestFoodCollection:
    """Test food collection tracking."""

    def test_track_food_without_efficiency(self):
        """Test tracking food collection without distance efficiency."""
        tracker = MetricsTracker()

        tracker.track_food_collection(distance_efficiency=None)

        assert tracker.foods_collected == 1
        assert tracker.distance_efficiencies == []

    def test_track_food_with_efficiency(self):
        """Test tracking food collection with distance efficiency."""
        tracker = MetricsTracker()

        tracker.track_food_collection(distance_efficiency=0.85)

        assert tracker.foods_collected == 1
        assert tracker.distance_efficiencies == [0.85]

    def test_track_multiple_foods(self):
        """Test tracking multiple food collections."""
        tracker = MetricsTracker()

        tracker.track_food_collection(distance_efficiency=0.80)
        tracker.track_food_collection(distance_efficiency=0.90)
        tracker.track_food_collection(distance_efficiency=None)
        tracker.track_food_collection(distance_efficiency=0.75)

        assert tracker.foods_collected == 4
        assert tracker.distance_efficiencies == [0.80, 0.90, 0.75]


class TestStepTracking:
    """Test individual step tracking."""

    def test_track_single_step(self):
        """Test tracking a single step."""
        tracker = MetricsTracker()

        tracker.track_step(reward=0.1)

        assert tracker.total_steps == 1
        assert tracker.total_rewards == pytest.approx(0.1)

    def test_track_multiple_steps(self):
        """Test tracking multiple steps."""
        tracker = MetricsTracker()

        tracker.track_step(reward=0.1)
        tracker.track_step(reward=0.2)
        tracker.track_step(reward=-0.05)
        tracker.track_step(reward=0.15)

        assert tracker.total_steps == 4
        assert tracker.total_rewards == pytest.approx(0.4)

    def test_track_step_with_negative_reward(self):
        """Test tracking steps with negative rewards."""
        tracker = MetricsTracker()

        tracker.track_step(reward=-1.0)

        assert tracker.total_steps == 1
        assert tracker.total_rewards == pytest.approx(-1.0)


class TestMetricsCalculation:
    """Test final metrics calculation."""

    def test_calculate_metrics_with_data(self):
        """Test metrics calculation with tracked data."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=True, reward=10.0)
        tracker.track_episode_completion(success=False, reward=5.0)
        tracker.track_episode_completion(success=True, reward=15.0)
        tracker.track_food_collection(distance_efficiency=0.85)
        tracker.track_food_collection(distance_efficiency=0.90)

        metrics = tracker.calculate_metrics(total_runs=3)

        assert metrics.success_rate == pytest.approx(2 / 3)
        assert metrics.average_reward == pytest.approx(30.0 / 3)
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
        tracker.track_episode_completion(success=True, reward=10.0)

        metrics = tracker.calculate_metrics(total_runs=0)

        assert metrics.success_rate == 0.0
        assert metrics.average_steps == 0.0
        assert metrics.average_reward == 0.0
        assert metrics.foraging_efficiency == 0.0

    def test_calculate_metrics_all_successful(self):
        """Test metrics with all successful episodes."""
        tracker = MetricsTracker()

        for _ in range(10):
            tracker.track_episode_completion(success=True, reward=5.5)

        metrics = tracker.calculate_metrics(total_runs=10)

        assert metrics.success_rate == 1.0
        assert metrics.average_reward == pytest.approx(5.5)

    def test_calculate_metrics_all_failed(self):
        """Test metrics with all failed episodes."""
        tracker = MetricsTracker()

        for _ in range(5):
            tracker.track_episode_completion(success=False, reward=-2.0)

        metrics = tracker.calculate_metrics(total_runs=5)

        assert metrics.success_rate == 0.0
        assert metrics.average_reward == pytest.approx(-2.0)


class TestMetricsReset:
    """Test metrics reset functionality."""

    def test_reset_clears_all_data(self):
        """Test that reset clears all tracked data."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=True, reward=10.0)
        tracker.track_food_collection(distance_efficiency=0.85)
        tracker.track_step(reward=0.5)

        tracker.reset()

        assert tracker.success_count == 1  # Success count is not reset
        assert tracker.total_steps == 0
        assert tracker.total_rewards == 0.0
        assert tracker.foods_collected == 0
        assert tracker.distance_efficiencies == []

    def test_reset_allows_fresh_tracking(self):
        """Test that tracking works correctly after reset."""
        tracker = MetricsTracker()

        tracker.track_step()
        tracker.track_episode_completion(success=True, reward=10.0)
        tracker.reset()
        tracker.track_step()
        tracker.track_step()
        tracker.track_step()
        tracker.track_episode_completion(success=False, reward=5.0)

        assert tracker.success_count == 1  # Success count is not reset
        assert tracker.total_steps == 3
        assert tracker.total_rewards == pytest.approx(5.0)


class TestMetricsForagingEfficiency:
    """Test foraging efficiency calculations."""

    def test_foraging_efficiency_with_foods(self):
        """Test foraging efficiency when foods are collected."""
        tracker = MetricsTracker()

        for _ in range(15):
            tracker.track_food_collection()

        metrics = tracker.calculate_metrics(total_runs=5)

        assert metrics.foraging_efficiency == pytest.approx(3.0)  # 15 foods / 5 runs

    def test_foraging_efficiency_without_foods(self):
        """Test foraging efficiency when no foods collected."""
        tracker = MetricsTracker()

        tracker.track_episode_completion(success=False, reward=0.0)

        metrics = tracker.calculate_metrics(total_runs=1)

        assert metrics.foraging_efficiency == 0.0

    def test_foraging_efficiency_fractional(self):
        """Test foraging efficiency with fractional results."""
        tracker = MetricsTracker()

        tracker.track_food_collection()
        tracker.track_food_collection()

        metrics = tracker.calculate_metrics(total_runs=3)

        assert metrics.foraging_efficiency == pytest.approx(2 / 3)
