import pytest
from quantumnematode.agent.tracker import EpisodeTracker


class TestEpisodeTrackerInitialization:
    """Test episode tracker initialization."""

    def test_initialization(self):
        """Test that the EpisodeTracker initializes with correct default values."""
        tracker = EpisodeTracker()

        assert tracker.steps == 0
        assert tracker.rewards == 0.0
        assert tracker.foods_collected == 0
        assert tracker.distance_efficiencies == []


class TestFoodCollection:
    """Test food collection tracking."""

    def test_track_food_without_efficiency(self):
        """Test tracking food collection without distance efficiency."""
        tracker = EpisodeTracker()

        tracker.track_food_collection(distance_efficiency=None)

        assert tracker.foods_collected == 1
        assert tracker.distance_efficiencies == []

    def test_track_food_with_efficiency(self):
        """Test tracking food collection with distance efficiency."""
        tracker = EpisodeTracker()

        tracker.track_food_collection(distance_efficiency=0.85)

        assert tracker.foods_collected == 1
        assert tracker.distance_efficiencies == [0.85]

    def test_track_multiple_foods(self):
        """Test tracking multiple food collections."""
        tracker = EpisodeTracker()

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
        tracker = EpisodeTracker()

        tracker.track_step(reward=0.1)

        assert tracker.steps == 1
        assert tracker.rewards == pytest.approx(0.1)

    def test_track_multiple_steps(self):
        """Test tracking multiple steps."""
        tracker = EpisodeTracker()

        tracker.track_step(reward=0.1)
        tracker.track_step(reward=0.2)
        tracker.track_step(reward=-0.05)
        tracker.track_step(reward=0.15)

        assert tracker.steps == 4
        assert tracker.rewards == pytest.approx(0.4)

    def test_track_step_with_negative_reward(self):
        """Test tracking steps with negative rewards."""
        tracker = EpisodeTracker()

        tracker.track_step(reward=-1.0)

        assert tracker.steps == 1
        assert tracker.rewards == pytest.approx(-1.0)


class TestRewardTracking:
    """Test reward tracking."""

    def test_track_single_reward(self):
        """Test tracking a single reward."""
        tracker = EpisodeTracker()

        tracker.track_reward(reward=5.0)

        assert tracker.rewards == pytest.approx(5.0)

    def test_track_multiple_rewards(self):
        """Test tracking multiple rewards."""
        tracker = EpisodeTracker()

        tracker.track_reward(reward=3.0)
        tracker.track_reward(reward=-1.0)
        tracker.track_reward(reward=2.5)

        assert tracker.rewards == pytest.approx(4.5)


class TestForagingEfficiency:
    """Test foraging efficiency calculations."""

    def test_foraging_efficiency_with_foods(self):
        """Test foraging efficiency when foods are collected."""
        tracker = EpisodeTracker()

        for _ in range(15):
            tracker.track_food_collection(distance_efficiency=0.8)

        assert tracker.foods_collected == 15
        assert len(tracker.distance_efficiencies) == 15
        assert all(de == 0.8 for de in tracker.distance_efficiencies)

    def test_foraging_efficiency_no_foods(self):
        """Test foraging efficiency when no foods collected."""
        tracker = EpisodeTracker()

        assert tracker.foods_collected == 0
        assert len(tracker.distance_efficiencies) == 0


class TestEpisodeReset:
    """Test metrics reset functionality."""

    def test_reset_clears_all_data(self):
        """Test that reset clears all tracked data."""
        tracker = EpisodeTracker()

        tracker.track_food_collection(distance_efficiency=0.85)
        tracker.track_step(reward=0.5)

        tracker.reset()

        assert tracker.foods_collected == 0
        assert tracker.distance_efficiencies == []
        assert tracker.steps == 0
        assert tracker.rewards == 0.0
        assert tracker.foods_collected == 0
        assert tracker.distance_efficiencies == []

    def test_reset_allows_fresh_tracking(self):
        """Test that tracking works correctly after reset."""
        tracker = EpisodeTracker()

        tracker.track_step()
        tracker.reset()
        tracker.track_step()
        tracker.track_step()
        tracker.track_step()
        tracker.track_reward(reward=5.0)

        assert tracker.foods_collected == 0
        assert tracker.distance_efficiencies == []
        assert tracker.steps == 3
        assert tracker.rewards == pytest.approx(5.0)
