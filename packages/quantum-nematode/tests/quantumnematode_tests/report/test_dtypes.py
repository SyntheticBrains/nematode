"""Tests for report data types."""

import pytest
from pydantic import ValidationError
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.report.dtypes import PerformanceMetrics, SimulationResult, TrackingData


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
            efficiency_score=0.85,
        )

        assert result.run == 1
        assert result.steps == 100
        assert len(result.path) == 3
        assert result.total_reward == 50.0
        assert result.last_total_reward == 45.0
        assert result.efficiency_score == 0.85

    def test_simulation_result_with_empty_path(self):
        """Test SimulationResult with an empty path."""
        result = SimulationResult(
            run=1,
            steps=0,
            path=[],
            total_reward=0.0,
            last_total_reward=0.0,
            efficiency_score=0.0,
        )

        assert len(result.path) == 0
        assert result.steps == 0

    def test_simulation_result_negative_values(self):
        """Test SimulationResult can handle negative rewards."""
        result = SimulationResult(
            run=1,
            steps=50,
            path=[(0, 0)],
            total_reward=-10.5,
            last_total_reward=-15.0,
            efficiency_score=0.2,
        )

        assert result.total_reward == -10.5
        assert result.last_total_reward == -15.0

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
            efficiency_score=0.85,
        )

        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["run"] == 1
        assert data["steps"] == 100
        assert len(data["path"]) == 2


class TestTrackingData:
    """Test TrackingData data model."""

    def test_create_empty_tracking_data(self):
        """Test creating TrackingData with default empty dict."""
        tracking = TrackingData()

        assert isinstance(tracking.data, dict)
        assert len(tracking.data) == 0

    def test_create_tracking_data_with_data(self):
        """Test creating TrackingData with initial data."""
        brain_history = BrainHistoryData()
        brain_history.rewards.append(10.0)

        tracking = TrackingData(data={0: brain_history})

        assert len(tracking.data) == 1
        assert 0 in tracking.data
        assert tracking.data[0].rewards == [10.0]

    def test_add_data_to_tracking(self):
        """Test adding data to TrackingData after creation."""
        tracking = TrackingData()

        brain_history1 = BrainHistoryData()
        brain_history1.rewards.append(10.0)

        brain_history2 = BrainHistoryData()
        brain_history2.rewards.append(15.0)

        tracking.data[0] = brain_history1
        tracking.data[1] = brain_history2

        assert len(tracking.data) == 2
        assert tracking.data[0].rewards == [10.0]
        assert tracking.data[1].rewards == [15.0]

    def test_tracking_data_type_annotation(self):
        """Test that tracking data properly validates types."""
        # This should work - valid BrainHistoryData
        tracking = TrackingData(data={0: BrainHistoryData()})
        assert len(tracking.data) == 1

    def test_tracking_data_serialization(self):
        """Test that TrackingData can be serialized."""
        brain_history = BrainHistoryData()
        brain_history.rewards.append(10.0)

        tracking = TrackingData(data={0: brain_history})
        data = tracking.model_dump()

        assert isinstance(data, dict)
        assert "data" in data
        assert 0 in data["data"]


class TestPerformanceMetrics:
    """Test PerformanceMetrics data model."""

    def test_create_static_maze_metrics(self):
        """Test creating metrics for static maze environment (no foraging fields)."""
        metrics = PerformanceMetrics(
            success_rate=0.75,
            average_steps=50.5,
            average_reward=100.0,
        )

        assert metrics.success_rate == 0.75
        assert metrics.average_steps == 50.5
        assert metrics.average_reward == 100.0
        # Foraging fields should be None for static environments
        assert metrics.foraging_efficiency is None
        assert metrics.average_distance_efficiency is None
        assert metrics.average_foods_collected is None

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
