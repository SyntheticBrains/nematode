"""Tests for report data types."""

import pytest
from pydantic import ValidationError
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.report.dtypes import (
    BrainDataSnapshot,
    EpisodeTrackingData,
    PerformanceMetrics,
    SimulationResult,
    TerminationReason,
    TrackingData,
)


class TestSimulationResult:
    """Test SimulationResult data model."""

    def test_create_simulation_result(self):
        """Test creating a valid SimulationResult."""
        result = SimulationResult(
            run=1,
            steps=100,
            path=[(0, 0), (1, 1), (2, 2)],
            total_reward=50.0,
            last_total_reward=45.0,
            termination_reason=TerminationReason.GOAL_REACHED,
            success=True,
        )

        assert result.run == 1
        assert result.steps == 100
        assert len(result.path) == 3
        assert result.total_reward == 50.0
        assert result.last_total_reward == 45.0
        assert result.termination_reason == TerminationReason.GOAL_REACHED
        assert result.success is True
        assert result.foods_collected is None

    def test_simulation_result_with_empty_path(self):
        """Test SimulationResult with an empty path."""
        result = SimulationResult(
            run=1,
            steps=0,
            path=[],
            total_reward=0.0,
            last_total_reward=0.0,
            termination_reason=TerminationReason.MAX_STEPS,
            success=False,
        )

        assert len(result.path) == 0
        assert result.steps == 0
        assert result.success is False

    def test_simulation_result_negative_values(self):
        """Test SimulationResult can handle negative rewards."""
        result = SimulationResult(
            run=1,
            steps=50,
            path=[(0, 0)],
            total_reward=-10.5,
            last_total_reward=-15.0,
            termination_reason=TerminationReason.STARVED,
            success=False,
        )

        assert result.total_reward == -10.5
        assert result.last_total_reward == -15.0
        assert result.termination_reason == TerminationReason.STARVED

    def test_simulation_result_missing_required_field(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            SimulationResult(  # type: ignore[call-arg]
                run=1,
                steps=100,
                # Missing required fields: path, total_reward, etc.
            )

    def test_simulation_result_serialization(self):
        """Test that SimulationResult can be serialized to dict."""
        result = SimulationResult(
            run=1,
            steps=100,
            path=[(0, 0), (1, 1)],
            total_reward=50.0,
            last_total_reward=45.0,
            termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
            success=True,
            foods_collected=5,
        )

        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["run"] == 1
        assert data["steps"] == 100
        assert len(data["path"]) == 2
        assert data["termination_reason"] == "completed_all_food"
        assert data["success"] is True
        assert data["foods_collected"] == 5


class TestEpisodeTrackingData:
    """Test EpisodeTrackingData data model."""

    def test_create_episode_tracking_data(self):
        """Test creating EpisodeTrackingData with all fields."""
        episode_data = EpisodeTrackingData(
            satiety_history=[100.0, 95.0, 90.0, 85.0],
            foods_collected=2,
            distance_efficiencies=[0.95, 0.88],
        )

        assert len(episode_data.satiety_history) == 4
        assert episode_data.satiety_history[0] == 100.0
        assert episode_data.foods_collected == 2
        assert len(episode_data.distance_efficiencies) == 2
        assert episode_data.distance_efficiencies[0] == 0.95

    def test_create_empty_episode_tracking_data(self):
        """Test creating EpisodeTrackingData with default values."""
        episode_data = EpisodeTrackingData()

        assert episode_data.satiety_history == []
        assert episode_data.foods_collected == 0
        assert episode_data.distance_efficiencies == []

    def test_episode_tracking_data_partial(self):
        """Test creating EpisodeTrackingData with only some fields."""
        episode_data = EpisodeTrackingData(foods_collected=5)

        assert episode_data.foods_collected == 5
        assert episode_data.satiety_history == []
        assert episode_data.distance_efficiencies == []

    def test_episode_tracking_data_serialization(self):
        """Test serializing EpisodeTrackingData to dict."""
        episode_data = EpisodeTrackingData(
            satiety_history=[100.0, 90.0],
            foods_collected=3,
            distance_efficiencies=[0.9, 0.85, 0.92],
        )

        data = episode_data.model_dump()

        assert isinstance(data, dict)
        assert data["satiety_history"] == [100.0, 90.0]
        assert data["foods_collected"] == 3
        assert data["distance_efficiencies"] == [0.9, 0.85, 0.92]

    def test_episode_tracking_data_modification(self):
        """Test modifying EpisodeTrackingData after creation."""
        episode_data = EpisodeTrackingData()

        episode_data.satiety_history.append(100.0)
        episode_data.satiety_history.append(95.0)
        episode_data.foods_collected = 1
        episode_data.distance_efficiencies.append(0.88)

        assert len(episode_data.satiety_history) == 2
        assert episode_data.foods_collected == 1
        assert len(episode_data.distance_efficiencies) == 1


class TestTrackingData:
    """Test TrackingData data model."""

    def test_create_empty_tracking_data(self):
        """Test creating TrackingData with default empty dicts."""
        tracking = TrackingData()

        assert isinstance(tracking.brain_data, dict)
        assert len(tracking.brain_data) == 0
        assert isinstance(tracking.episode_data, dict)
        assert len(tracking.episode_data) == 0

    def test_create_tracking_data_with_data(self):
        """Test creating TrackingData with initial data."""
        brain_history = BrainHistoryData()
        brain_history.rewards.append(10.0)

        tracking = TrackingData(brain_data={0: brain_history})

        assert len(tracking.brain_data) == 1
        assert 0 in tracking.brain_data
        brain_data_0 = tracking.brain_data[0]
        assert isinstance(brain_data_0, BrainHistoryData)
        assert brain_data_0.rewards == [10.0]

    def test_add_data_to_tracking(self):
        """Test adding data to TrackingData after creation."""
        tracking = TrackingData()

        brain_history1 = BrainHistoryData()
        brain_history1.rewards.append(10.0)

        brain_history2 = BrainHistoryData()
        brain_history2.rewards.append(15.0)

        tracking.brain_data[0] = brain_history1
        tracking.brain_data[1] = brain_history2

        assert len(tracking.brain_data) == 2
        bd0 = tracking.brain_data[0]
        bd1 = tracking.brain_data[1]
        assert isinstance(bd0, BrainHistoryData)
        assert isinstance(bd1, BrainHistoryData)
        assert bd0.rewards == [10.0]
        assert bd1.rewards == [15.0]

    def test_tracking_data_type_annotation(self):
        """Test that tracking data properly validates types."""
        # This should work - valid BrainHistoryData
        tracking = TrackingData(brain_data={0: BrainHistoryData()})
        assert len(tracking.brain_data) == 1

    def test_tracking_data_serialization(self):
        """Test that TrackingData can be serialized."""
        brain_history = BrainHistoryData()
        brain_history.rewards.append(10.0)

        tracking = TrackingData(brain_data={0: brain_history})
        data = tracking.model_dump()

        assert isinstance(data, dict)
        assert "brain_data" in data
        assert "episode_data" in data
        assert 0 in data["brain_data"]

    def test_tracking_data_with_episode_data(self):
        """Test creating TrackingData with both brain and episode data."""
        brain_history = BrainHistoryData()
        brain_history.rewards.append(10.0)

        episode_data = EpisodeTrackingData(
            satiety_history=[100.0, 95.0],
            foods_collected=1,
            distance_efficiencies=[0.9],
        )

        tracking = TrackingData(
            brain_data={0: brain_history},
            episode_data={0: episode_data},
        )

        assert len(tracking.brain_data) == 1
        assert len(tracking.episode_data) == 1
        brain_data_0 = tracking.brain_data[0]
        assert isinstance(brain_data_0, BrainHistoryData)
        assert brain_data_0.rewards == [10.0]
        assert tracking.episode_data[0].foods_collected == 1
        assert len(tracking.episode_data[0].satiety_history) == 2

    def test_add_episode_data_to_tracking(self):
        """Test adding episode data to TrackingData after creation."""
        tracking = TrackingData()

        # Add brain data for run 0
        brain_history = BrainHistoryData()
        brain_history.rewards.append(10.0)
        tracking.brain_data[0] = brain_history

        # Add episode data for run 0
        episode_data = EpisodeTrackingData(
            satiety_history=[100.0, 90.0, 80.0],
            foods_collected=2,
            distance_efficiencies=[0.95, 0.88],
        )
        tracking.episode_data[0] = episode_data

        assert len(tracking.brain_data) == 1
        assert len(tracking.episode_data) == 1
        assert tracking.episode_data[0].foods_collected == 2
        assert len(tracking.episode_data[0].distance_efficiencies) == 2

    def test_tracking_data_multiple_runs_with_episodes(self):
        """Test TrackingData with multiple runs including episode data."""
        tracking = TrackingData()

        # Run 0
        brain_history_0 = BrainHistoryData()
        brain_history_0.rewards.append(10.0)
        episode_data_0 = EpisodeTrackingData(foods_collected=3)

        tracking.brain_data[0] = brain_history_0
        tracking.episode_data[0] = episode_data_0

        # Run 1
        brain_history_1 = BrainHistoryData()
        brain_history_1.rewards.append(15.0)
        episode_data_1 = EpisodeTrackingData(foods_collected=5)

        tracking.brain_data[1] = brain_history_1
        tracking.episode_data[1] = episode_data_1

        assert len(tracking.brain_data) == 2
        assert len(tracking.episode_data) == 2
        assert tracking.episode_data[0].foods_collected == 3
        assert tracking.episode_data[1].foods_collected == 5

    def test_tracking_data_accepts_brain_data_snapshot(self):
        """Test TrackingData with BrainDataSnapshot in brain_data."""
        snapshot = BrainDataSnapshot(last_values={"rewards": 10.0})
        tracking = TrackingData(brain_data={0: snapshot})

        assert len(tracking.brain_data) == 1
        brain_data_0 = tracking.brain_data[0]
        assert isinstance(brain_data_0, BrainDataSnapshot)
        assert brain_data_0.last_values["rewards"] == 10.0

    def test_tracking_data_mixed_brain_data_types(self):
        """Test TrackingData with both BrainHistoryData and BrainDataSnapshot."""
        brain_history = BrainHistoryData()
        brain_history.rewards.append(10.0)
        snapshot = BrainDataSnapshot(last_values={"rewards": 15.0})

        tracking = TrackingData(brain_data={0: brain_history, 1: snapshot})

        assert len(tracking.brain_data) == 2
        assert isinstance(tracking.brain_data[0], BrainHistoryData)
        assert isinstance(tracking.brain_data[1], BrainDataSnapshot)
        assert tracking.brain_data[0].rewards == [10.0]
        assert tracking.brain_data[1].last_values["rewards"] == 15.0

    def test_tracking_data_partial_episode_data(self):
        """Test TrackingData where only some runs have episode data (mixed environments)."""
        tracking = TrackingData()

        # Run 0 - both brain and episode data (foraging environment)
        brain_history_0 = BrainHistoryData()
        episode_data_0 = EpisodeTrackingData(foods_collected=2)
        tracking.brain_data[0] = brain_history_0
        tracking.episode_data[0] = episode_data_0

        # Run 1 - only brain data (maze environment)
        brain_history_1 = BrainHistoryData()
        tracking.brain_data[1] = brain_history_1

        assert len(tracking.brain_data) == 2
        assert len(tracking.episode_data) == 1
        assert 0 in tracking.episode_data
        assert 1 not in tracking.episode_data


class TestPerformanceMetrics:
    """Test PerformanceMetrics data model."""

    def test_create_dynamic_foraging_metrics(self):
        """Test creating metrics for dynamic foraging environment."""
        metrics = PerformanceMetrics(
            success_rate=0.80,
            average_steps=75.0,
            average_reward=150.0,
            foraging_efficiency=0.15,
            average_distance_efficiency=0.92,
            average_foods_collected=8.5,
        )

        assert metrics.success_rate == 0.80
        assert metrics.average_steps == 75.0
        assert metrics.average_reward == 150.0
        assert metrics.foraging_efficiency == 0.15
        assert metrics.average_distance_efficiency == 0.92
        assert metrics.average_foods_collected == 8.5

    def test_performance_metrics_zero_success_rate(self):
        """Test metrics with zero success rate."""
        metrics = PerformanceMetrics(
            success_rate=0.0,
            average_steps=100.0,
            average_reward=-50.0,
        )

        assert metrics.success_rate == 0.0
        assert metrics.average_reward < 0

    def test_performance_metrics_perfect_success_rate(self):
        """Test metrics with perfect success rate."""
        metrics = PerformanceMetrics(
            success_rate=1.0,
            average_steps=25.0,
            average_reward=200.0,
        )

        assert metrics.success_rate == 1.0

    def test_performance_metrics_partial_foraging_fields(self):
        """Test that foraging fields can be set independently."""
        metrics = PerformanceMetrics(
            success_rate=0.5,
            average_steps=50.0,
            average_reward=75.0,
            foraging_efficiency=0.10,
            # average_distance_efficiency and average_foods_collected left as None
        )

        assert metrics.foraging_efficiency == 0.10
        assert metrics.average_distance_efficiency is None
        assert metrics.average_foods_collected is None

    def test_performance_metrics_serialization(self):
        """Test that PerformanceMetrics can be serialized."""
        metrics = PerformanceMetrics(
            success_rate=0.75,
            average_steps=50.5,
            average_reward=100.0,
            foraging_efficiency=0.15,
            average_distance_efficiency=0.92,
            average_foods_collected=8.5,
        )

        data = metrics.model_dump()
        assert isinstance(data, dict)
        assert data["success_rate"] == 0.75
        assert data["average_steps"] == 50.5
        assert data["foraging_efficiency"] == 0.15

    def test_performance_metrics_missing_required_field(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            PerformanceMetrics(  # type: ignore[call-arg]
                success_rate=0.75,
                # Missing required fields: average_steps, average_reward
            )

    def test_performance_metrics_round_trip(self):
        """Test serialization and deserialization round trip."""
        original = PerformanceMetrics(
            success_rate=0.85,
            average_steps=42.0,
            average_reward=120.5,
            foraging_efficiency=0.12,
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back to object
        restored = PerformanceMetrics(**data)

        assert restored.success_rate == original.success_rate
        assert restored.average_steps == original.average_steps
        assert restored.average_reward == original.average_reward
        assert restored.foraging_efficiency == original.foraging_efficiency
