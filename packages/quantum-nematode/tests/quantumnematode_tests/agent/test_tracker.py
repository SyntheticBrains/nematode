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


class TestPredatorMetricsTracking:
    """Test predator-specific metrics tracking."""

    def test_initialization_includes_predator_metrics(self):
        """Test that the EpisodeTracker initializes with predator metrics."""
        tracker = EpisodeTracker()

        assert tracker.predator_encounters == 0
        assert tracker.successful_evasions == 0
        assert tracker.in_danger is False

    def test_track_predator_encounter(self):
        """Test tracking predator encounters."""
        tracker = EpisodeTracker()

        tracker.predator_encounters = 1
        assert tracker.predator_encounters == 1

        tracker.predator_encounters = 5
        assert tracker.predator_encounters == 5

    def test_track_successful_evasion(self):
        """Test tracking successful predator evasions."""
        tracker = EpisodeTracker()

        tracker.successful_evasions = 1
        assert tracker.successful_evasions == 1

        tracker.successful_evasions = 3
        assert tracker.successful_evasions == 3

    def test_track_danger_status(self):
        """Test tracking whether agent is in danger."""
        tracker = EpisodeTracker()

        # Initially not in danger
        assert tracker.in_danger is False

        # Enter danger
        tracker.in_danger = True
        assert tracker.in_danger is True

        # Exit danger
        tracker.in_danger = False
        assert tracker.in_danger is False

    def test_multiple_encounters_and_evasions(self):
        """Test tracking multiple encounters and evasions."""
        tracker = EpisodeTracker()

        # Simulate episode with predator interactions
        tracker.predator_encounters = 10
        tracker.successful_evasions = 7

        assert tracker.predator_encounters == 10
        assert tracker.successful_evasions == 7

        # Calculate evasion rate
        evasion_rate = tracker.successful_evasions / tracker.predator_encounters
        assert evasion_rate == pytest.approx(0.7)

    def test_reset_clears_predator_metrics(self):
        """Test that reset clears predator metrics."""
        tracker = EpisodeTracker()

        # Set predator metrics
        tracker.predator_encounters = 5
        tracker.successful_evasions = 3
        tracker.in_danger = True

        # Reset
        tracker.reset()

        # Verify all predator metrics are cleared
        assert tracker.predator_encounters == 0
        assert tracker.successful_evasions == 0
        assert tracker.in_danger is False

    def test_predator_metrics_with_food_tracking(self):
        """Test that predator metrics work alongside food tracking."""
        tracker = EpisodeTracker()

        # Track food and predator metrics together
        tracker.track_food_collection(distance_efficiency=0.85)
        tracker.predator_encounters = 2
        tracker.successful_evasions = 1
        tracker.track_step(reward=0.5)

        # Verify both types of metrics are tracked
        assert tracker.foods_collected == 1
        assert tracker.distance_efficiencies == [0.85]
        assert tracker.predator_encounters == 2
        assert tracker.successful_evasions == 1
        assert tracker.steps == 1
        assert tracker.rewards == pytest.approx(0.5)

    def test_satiety_history_tracking(self):
        """Test satiety history tracking."""
        tracker = EpisodeTracker()

        # Satiety history should be empty initially
        assert tracker.satiety_history == []

        # This is tested via integration, but the property should exist
        assert hasattr(tracker, "satiety_history")
