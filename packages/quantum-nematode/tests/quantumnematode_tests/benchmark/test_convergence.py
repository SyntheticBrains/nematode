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

        # Level-agnostic onset = first window reaching within `band` of the converged
        # level (1.0 here). The 9-success/1-fail boundary window [indices 9..18] is
        # within band of 1.0, so onset is run 10 (1-indexed) — one earlier than the
        # legacy "first fully-homogeneous window" (11). The averaged plateau metric is
        # unchanged within sampling noise (see test_high_band_metric_regression).
        assert convergence_run is not None
        assert convergence_run == 10
        assert convergence_run < 50  # Before end

    def test_detect_convergence_stationary_intermediate_plateau(self):
        """A stable INTERMEDIATE plateau (~45%) is converged (level-agnostic).

        Regression for the T7 sub-saturation band: the legacy variance-gate
        returned None here (a ~45% plateau never produces a near-homogeneous
        window), silently degrading the ranked metric to a noisy last-N fallback.
        """

        # Deterministic, evenly-spread 45% plateau (no warm-up): (i*9 % 20) < 9 is a
        # scrambled permutation of residues, so every window holds ~45% (no blocky
        # phase artifact vs the no-trend gate's comparison blocks).
        def _hit(i: int) -> bool:
            return (i * 9) % 20 < 9

        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0 if _hit(i) else 0.0,
                last_total_reward=10.0 if _hit(i) else 0.0,
                termination_reason=TerminationReason.GOAL_REACHED
                if _hit(i)
                else TerminationReason.MAX_STEPS,
                success=_hit(i),
            )
            for i in range(200)
        ]

        convergence_run = detect_convergence(results)

        assert convergence_run is not None  # legacy detector returned None here
        metrics = calculate_post_convergence_metrics(results, convergence_run)
        assert metrics["success_rate"] == pytest.approx(0.45, abs=0.02)

    def test_detect_convergence_still_trending_returns_none(self):
        """A run still climbing at its budget is NOT converged (flag, don't mis-rank)."""
        # Monotonic ramp that never flattens: success probability rises across the run
        # so the final block still exceeds the preceding block by more than `band`.
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0 if (i % 10) < (i // 20) else 0.0,
                last_total_reward=10.0 if (i % 10) < (i // 20) else 0.0,
                termination_reason=TerminationReason.GOAL_REACHED
                if (i % 10) < (i // 20)
                else TerminationReason.MAX_STEPS,
                success=(i % 10) < (i // 20),
            )
            for i in range(200)
        ]

        convergence_run = detect_convergence(results)

        # Still trending up at the end (no-trend gate fails) -> not converged.
        assert convergence_run is None

    def test_detect_convergence_flat_zero_warmup_not_a_plateau(self):
        """A long all-fail prefix before ignition is NOT reported as the plateau."""
        # Fail for the first 120 runs, then a stable ~80% plateau.
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0 if (i >= 120 and (i % 5) < 4) else 0.0,
                last_total_reward=10.0 if (i >= 120 and (i % 5) < 4) else 0.0,
                termination_reason=TerminationReason.GOAL_REACHED
                if (i >= 120 and (i % 5) < 4)
                else TerminationReason.MAX_STEPS,
                success=(i >= 120 and (i % 5) < 4),
            )
            for i in range(280)
        ]

        convergence_run = detect_convergence(results)

        assert convergence_run is not None
        # Onset is after ignition (~run 120), NOT in the flat-zero prefix.
        assert convergence_run > 100
        metrics = calculate_post_convergence_metrics(results, convergence_run)
        assert metrics["success_rate"] == pytest.approx(0.8, abs=0.05)

    def test_high_band_metric_regression(self):
        """High-band plateau: the averaged metric is unchanged within sampling noise.

        The onset can shift by <=1 run vs the legacy detector, but the load-bearing
        quantity — the post-convergence success rate — stays ~1.0.
        """
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0,
                last_total_reward=10.0,
                termination_reason=TerminationReason.GOAL_REACHED,
                success=i >= 10,
            )
            for i in range(50)
        ]

        convergence_run = detect_convergence(results)
        metrics = calculate_post_convergence_metrics(results, convergence_run)

        assert metrics["success_rate"] == pytest.approx(1.0, abs=0.03)

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

        # Level-agnostic onset: the boundary window reaching within `band` of the
        # converged level (1.0) is run 25 (1-indexed) — one earlier than the legacy
        # first-fully-homogeneous window (26).
        assert convergence_run is not None
        assert convergence_run == 25
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

        # Level-agnostic onset: the boundary window reaching within `band` of the
        # converged level (1.0) is run 4 (1-indexed) — one earlier than the legacy
        # first-fully-homogeneous window (5). Early detection is preserved.
        assert convergence_run is not None
        assert convergence_run == 4


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
        # Level-agnostic onset: the boundary window reaching within `band` of the
        # converged level (1.0) is run 15 (1-indexed) — one earlier than the legacy
        # first-fully-homogeneous window (16); the averaged metric stays ~1.0.
        assert metrics.convergence_run == 15

        # Verify post-convergence metrics are strong (one boundary failure included).
        assert metrics.post_convergence_success_rate == pytest.approx(1.0, abs=0.03)
        assert metrics.post_convergence_avg_steps is not None
        assert metrics.post_convergence_avg_steps < 100  # Better than early runs
        assert metrics.distance_efficiency is not None
        assert metrics.distance_efficiency == pytest.approx(0.34, abs=0.001)  # 34% efficiency

        # Verify composite score is reasonable
        assert metrics.composite_score > 0.5  # Should be decent
        assert metrics.composite_score <= 1.0  # Within valid range

    def test_analyze_convergence_no_convergence(self):
        """Test analysis when learning never converges (still trending at the budget).

        Under level-agnostic semantics, "never converges" means the run is still
        improving at its budget — NOT a stationary intermediate plateau (a stable
        ~50% IS a converged plateau and is detected; see
        test_detect_convergence_stationary_intermediate_plateau). The fallback
        last-N metric still populates so downstream reporting never crashes.
        """
        # Monotonic ramp that never flattens by the end.
        results = [
            SimulationResult(
                run=i,
                steps=50,
                path=[],
                total_reward=10.0 if (i % 10) < (i // 10) else 0.0,
                last_total_reward=10.0 if (i % 10) < (i // 10) else 0.0,
                termination_reason=TerminationReason.GOAL_REACHED
                if (i % 10) < (i // 10)
                else TerminationReason.MAX_STEPS,
                success=(i % 10) < (i // 10),
            )
            for i in range(100)
        ]

        metrics = analyze_convergence(results, total_runs=100)

        # Still trending -> not converged.
        assert metrics.converged is False
        assert metrics.convergence_run is None
        assert metrics.runs_to_convergence is None

        # Should still calculate fallback metrics (so reporting never crashes).
        assert metrics.post_convergence_success_rate is not None
        assert metrics.composite_score is not None
