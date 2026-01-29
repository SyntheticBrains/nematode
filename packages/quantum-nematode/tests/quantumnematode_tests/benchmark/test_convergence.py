"""Tests for convergence detection and analysis."""

import pytest
from quantumnematode.benchmark.convergence import (
    analyze_convergence,
    calculate_composite_score,
    calculate_post_convergence_metrics,
    detect_convergence,
)
from quantumnematode.report.dtypes import SimulationResult, TerminationReason


class TestConvergenceDetection:
    """Test convergence detection algorithm."""

    def test_detect_convergence_stable_success(self):
        """Test convergence detection with stable 100% success rate."""
        # 50 runs with 100% success after warmup (need min 30 for detection)
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0,
                last_total_reward=10.0,
                termination_reason=TerminationReason.GOAL_REACHED,
                success=i >= 10,  # First 10 fails, then succeeds
            )
            for i in range(50)
        ]

        convergence_run = detect_convergence(results)

        # Should detect convergence at the first stable window (run 11, 1-indexed)
        # Window covers runs 11-20 (1-indexed), which are indices 10-19 (0-indexed)
        assert convergence_run is not None
        assert convergence_run == 11  # First run where 10-run window is stable (1-indexed)
        assert convergence_run < 50  # Before end

    def test_detect_convergence_never_converges(self):
        """Test that oscillating performance doesn't converge."""
        # Alternating success/fail - high variance
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0 if i % 2 == 0 else 0.0,
                last_total_reward=10.0 if i % 2 == 0 else 0.0,
                termination_reason=TerminationReason.GOAL_REACHED,
                success=i % 2 == 0,
            )
            for i in range(50)
        ]

        convergence_run = detect_convergence(results)

        # Should NOT converge due to high variance
        assert convergence_run is None

    def test_detect_convergence_insufficient_runs(self):
        """Test that too few runs returns None."""
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0,
                last_total_reward=10.0,
                termination_reason=TerminationReason.GOAL_REACHED,
                success=True,
            )
            for i in range(25)  # Less than min_total_runs (30)
        ]

        convergence_run = detect_convergence(results, min_total_runs=30)

        # Should return None due to insufficient total runs
        assert convergence_run is None

    def test_detect_convergence_gradual_improvement(self):
        """Test convergence with gradual improvement to stability."""
        # Linearly improving success rate
        results = []
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0 if i >= 25 else 0.0,
                last_total_reward=10.0 if i >= 25 else 0.0,
                termination_reason=TerminationReason.GOAL_REACHED,
                success=i >= 25,  # Stable after run 25
            )
            for i in range(50)
        ]

        convergence_run = detect_convergence(results)

        # Should detect convergence at run 26
        # (1-indexed, first run where window 26-35 is 100% stable)
        assert convergence_run is not None
        assert convergence_run == 26  # First stable window (1-indexed)
        assert convergence_run <= 36  # Not too late

    def test_detect_convergence_early_success(self):
        """Test that convergence is detected at the earliest stable successful window."""
        # 50 runs with 100% success from index 4 onward (i.e., run 5, 1-indexed)
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0,
                last_total_reward=10.0,
                termination_reason=TerminationReason.GOAL_REACHED,
                success=i >= 4,  # First 4 fails (indices 0-3), then succeeds
            )
            for i in range(50)
        ]

        convergence_run = detect_convergence(results)

        # Should detect convergence at run 5 (1-indexed, first stable successful window)
        # Window from runs 5-14 (1-indexed) = indices 4-13 (0-indexed) has 100% success
        assert convergence_run is not None
        assert convergence_run == 5  # Early detection at actual convergence point (1-indexed)


class TestPostConvergenceMetrics:
    """Test post-convergence metrics calculation."""

    def test_post_convergence_with_convergence_point(self):
        """Test metrics calculated after convergence point."""
        results = [
            SimulationResult(
                run=i,
                steps=100 if i < 20 else 50,  # Better after convergence
                path=[],
                total_reward=5.0 if i < 20 else 10.0,
                last_total_reward=5.0 if i < 20 else 10.0,
                termination_reason=TerminationReason.GOAL_REACHED,
                success=i >= 15,  # Success rate improves
                foods_collected=5 if i >= 20 else 2,
            )
            for i in range(50)
        ]

        # convergence_run=21 (1-indexed) means starting from index 20 (0-indexed)
        metrics = calculate_post_convergence_metrics(results, convergence_run=21)

        # Should only consider runs from run 21 onward (index 20+)
        assert metrics["success_rate"] == 1.0  # 100% after run 21
        assert metrics["avg_steps"] == 50.0  # Faster after convergence
        assert metrics["avg_foods"] == 5.0  # More food after convergence
        assert metrics["variance"] == 0.0  # Perfect stability

    def test_post_convergence_fallback_to_last_n(self):
        """Test fallback to last 10 runs when no convergence."""
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0,
                last_total_reward=10.0,
                termination_reason=TerminationReason.GOAL_REACHED,
                success=i >= 40,  # Only last 10 succeed
            )
            for i in range(50)
        ]

        metrics = calculate_post_convergence_metrics(
            results,
            convergence_run=None,
            fallback_window=10,
        )

        # Should use last 10 runs
        assert metrics["success_rate"] == 1.0  # Last 10 all succeed
        assert metrics["variance"] == 0.0

    def test_post_convergence_distance_efficiency(self):
        """Test distance efficiency calculation."""
        # Need to pass full result set including convergence point
        results = [
            SimulationResult(
                run=i,
                steps=100,
                path=[],
                total_reward=10.0,
                last_total_reward=10.0,
                termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
                success=True,
                foods_collected=10,
                average_distance_efficiency=0.34,  # 34% efficiency
            )
            for i in range(30)
        ]

        # convergence_run=21 (1-indexed) means starting from index 20 (0-indexed)
        metrics = calculate_post_convergence_metrics(results, convergence_run=21)

        # Distance efficiency = average of post-convergence runs
        assert metrics["distance_efficiency"] is not None
        assert metrics["distance_efficiency"] == pytest.approx(0.34, abs=0.001)


class TestCompositeScore:
    """Test composite score calculation."""

    def test_composite_score_perfect_performance(self):
        """Test composite score with perfect metrics."""
        score = calculate_composite_score(
            success_rate=1.0,  # Perfect success
            distance_efficiency=1.0,  # Perfect navigation efficiency
            runs_to_convergence=10,  # Fast convergence (10/50 = 0.8 speed score)
            variance=0.0,  # Perfect stability
            total_runs=50,
        )

        # Score = 0.4*1.0 + 0.3*1.0 + 0.2*0.8 + 0.1*1.0 = 0.96
        assert score == pytest.approx(0.96, abs=0.01)

    def test_composite_score_poor_performance(self):
        """Test composite score with poor metrics."""
        score = calculate_composite_score(
            success_rate=0.0,  # No success
            distance_efficiency=0.0,  # No efficiency
            runs_to_convergence=None,  # Never converged
            variance=0.5,  # High variance
            total_runs=50,
        )

        # Score = 0.4*0.0 + 0.3*0.0 + 0.2*0.0 + 0.1*0.0 = 0.0
        # (Variance over max gets clamped to 0)
        assert score == 0.0

    def test_composite_score_component_weights(self):
        """Test that component weights sum to 1.0."""
        # Components: success (0.4), efficiency (0.3), speed (0.2), stability (0.1)
        weights_sum = 0.4 + 0.3 + 0.2 + 0.1
        assert weights_sum == pytest.approx(1.0)


class TestAnalyzeConvergence:
    """Test full convergence analysis workflow."""

    def test_analyze_convergence_complete_workflow(self):
        """Test end-to-end convergence analysis."""
        # Simulate learning curve: poor start, then stable good performance
        results = []
        for i in range(50):
            if i < 15:
                # Early learning - poor performance
                success = False
                steps = 200
                foods = 2
            else:
                # Converged - good performance
                success = True
                steps = 80
                foods = 10

            results.append(
                SimulationResult(
                    run=i,
                    steps=steps,
                    path=[],
                    total_reward=10.0 if success else 0.0,
                    last_total_reward=10.0 if success else 0.0,
                    termination_reason=TerminationReason.COMPLETED_ALL_FOOD
                    if success
                    else TerminationReason.STARVED,
                    success=success,
                    foods_collected=foods,
                    average_distance_efficiency=0.34 if success else None,  # 34% efficiency
                ),
            )

        metrics = analyze_convergence(results, total_runs=50)

        # Verify convergence detected
        assert metrics.converged is True
        assert metrics.convergence_run is not None
        # Convergence at run 16 (1-indexed, first stable window with 100% success from 16-25)
        assert metrics.convergence_run == 16

        # Verify post-convergence metrics are strong
        assert metrics.post_convergence_success_rate == 1.0
        assert metrics.post_convergence_avg_steps is not None
        assert metrics.post_convergence_avg_steps < 100  # Better than early runs
        assert metrics.distance_efficiency is not None
        assert metrics.distance_efficiency == pytest.approx(0.34, abs=0.001)  # 34% efficiency

        # Verify composite score is reasonable
        assert metrics.composite_score > 0.5  # Should be decent
        assert metrics.composite_score <= 1.0  # Within valid range

    def test_analyze_convergence_no_convergence(self):
        """Test analysis when learning never converges."""
        # Random performance - no pattern
        import random

        random.seed(42)
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=random.random() * 10,  # noqa: S311
                last_total_reward=random.random() * 10,  # noqa: S311
                termination_reason=TerminationReason.MAX_STEPS,
                success=random.random() > 0.5,  # noqa: S311
            )
            for i in range(50)
        ]

        metrics = analyze_convergence(results, total_runs=50)

        # Should not converge
        assert metrics.converged is False
        assert metrics.convergence_run is None
        assert metrics.runs_to_convergence is None

        # Should still calculate fallback metrics
        assert metrics.post_convergence_success_rate is not None
        assert metrics.composite_score is not None
