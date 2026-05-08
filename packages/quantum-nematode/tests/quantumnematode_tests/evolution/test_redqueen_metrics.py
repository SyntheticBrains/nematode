"""Tests for :mod:`quantumnematode.evolution.redqueen_metrics`.

Synthetic series with known answers — sine for cycling, monotone ramp
for escalation, anti-phase pair for fitness_lag, identical-deltas for
coupled_rate, uniform-improvement matrix for generality. Plus negative
cases (flat, zero-variance, random noise) so the metrics fail closed
rather than producing spurious detections.

Coverage:
- Phenotypic cycling → TestPhenotypicCycling.
- Trait escalation → TestTraitEscalation.
- Fitness lag → TestFitnessLag.
- Coupled rate → TestCoupledRate.
- Generality → TestGenerality.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantumnematode.evolution.redqueen_metrics import (
    coupled_rate,
    fitness_lag,
    generality,
    phenotypic_cycling,
    trait_escalation,
)

# ---------------------------------------------------------------------------
# Phenotypic cycling
# ---------------------------------------------------------------------------


class TestPhenotypicCycling:
    """Cycling: pure sine + linear trend rejection + statistical significance."""

    def test_detects_pure_sine_at_period_8(self) -> None:
        """Sine wave at period 8 SHALL be detected with dominant_period within +/-1."""
        t = np.arange(40)
        series = np.sin(2.0 * np.pi * t / 8.0)
        out = phenotypic_cycling(series, lag_range=(3, 15))
        assert out["cycling_detected"] is True
        assert out["dominant_period"] is not None
        assert abs(out["dominant_period"] - 8) <= 1
        assert out["p_value"] < 0.05

    def test_rejects_pure_linear_ramp(self) -> None:
        """Pure linear trend SHALL NOT be flagged as cycling."""
        series = np.arange(40, dtype=np.float64)
        out = phenotypic_cycling(series)
        assert out["cycling_detected"] is False
        # Per the implementation: pure-linear residuals fall below the
        # rel-variance gate, so dominant_period is None and p_value is NaN.
        assert out["dominant_period"] is None
        assert math.isnan(out["p_value"])

    def test_rejects_random_noise(self) -> None:
        """Random Gaussian noise SHALL NOT be flagged as cycling (high p_value)."""
        rng = np.random.default_rng(seed=42)
        series = rng.normal(0.0, 1.0, size=50)
        out = phenotypic_cycling(series)
        assert out["cycling_detected"] is False
        # White noise has a peak somewhere by chance; the p-value is
        # what protects against false positives.
        assert out["p_value"] >= 0.05

    def test_below_minimum_length_returns_no_signal(self) -> None:
        """Series shorter than the lag range upper bound SHALL return no-signal."""
        out = phenotypic_cycling([1.0, 2.0, 3.0], lag_range=(3, 15))
        assert out["cycling_detected"] is False
        assert out["dominant_period"] is None
        assert math.isnan(out["p_value"])

    def test_invalid_lag_range_raises(self) -> None:
        """Invalid `lag_range` (low >= high or low < 1) SHALL raise."""
        series = np.arange(40, dtype=np.float64)
        with pytest.raises(ValueError, match="lag_range"):
            phenotypic_cycling(series, lag_range=(0, 10))
        with pytest.raises(ValueError, match="lag_range"):
            phenotypic_cycling(series, lag_range=(10, 5))


# ---------------------------------------------------------------------------
# Trait escalation
# ---------------------------------------------------------------------------


class TestTraitEscalation:
    """Escalation: monotone increase + flat-series rejection + windowing."""

    def test_detects_monotone_increase(self) -> None:
        """Linear ramp + small noise over generations 0..34 SHALL fire with positive slope."""
        rng = np.random.default_rng(seed=42)
        # Ramp slope 1.0 + small Gaussian noise.
        series = np.arange(35, dtype=np.float64) + rng.normal(0.0, 0.3, size=35)
        out = trait_escalation(series)
        assert out["escalation_detected"] is True
        assert out["slope_sign"] == 1
        assert out["p_value"] < 0.05
        # Slope estimate should be near 1.0 within tolerance.
        assert abs(out["slope"] - 1.0) < 0.2

    def test_rejects_flat_series(self) -> None:
        """Zero-mean noise around a constant SHALL NOT fire."""
        rng = np.random.default_rng(seed=42)
        series = np.ones(35, dtype=np.float64) + rng.normal(0.0, 0.1, size=35)
        out = trait_escalation(series)
        assert out["escalation_detected"] is False

    def test_window_skips_bootstrap_noise(self) -> None:
        """Series with noisy first 5 gens + clean trend after SHALL fit only the trend.

        Constructs a series where gens 0..4 are large noise centered at
        zero (with no trend) and gens 5..29 are a clean monotone ramp.
        With the default window `(5, 30)` the regression sees only the
        ramp and reports a confident positive slope.
        """
        rng = np.random.default_rng(seed=42)
        bootstrap = rng.normal(0.0, 100.0, size=5)  # huge noise, no trend
        trend = np.arange(25, dtype=np.float64) * 2.0  # clean slope 2.0
        series = np.concatenate([bootstrap, trend])
        out = trait_escalation(series, gen_window=(5, 30))
        assert out["escalation_detected"] is True
        assert out["slope_sign"] == 1
        # Slope should be near 2.0 (the trend slope) — without windowing
        # it would be diluted by the bootstrap noise.
        assert abs(out["slope"] - 2.0) < 0.5

    def test_short_series_returns_nan(self) -> None:
        """Series shorter than `gen_window[0]` SHALL return all-nan no-signal."""
        out = trait_escalation([1.0, 2.0, 3.0], gen_window=(5, 30))
        assert out["escalation_detected"] is False
        assert math.isnan(out["slope"])
        assert math.isnan(out["p_value"])


# ---------------------------------------------------------------------------
# Fitness lag
# ---------------------------------------------------------------------------


class TestFitnessLag:
    """Spec scenarios "Detects Anti-Phase Coupling" + "Returns NaN For Trivial Series"."""

    def test_detects_constructed_lag(self) -> None:
        """Two sine series with phase offset SHALL recover the lag within +/-1.

        Construction: `a = sin(g)` and `b = sin(g - shift)` so `b`
        trails `a` by `shift` generations. `fitness_lag` tracks peak
        positive correlation, so the expected best lag is +shift
        (where the two series align in phase).
        """
        n = 80
        constructed_shift = 4
        t = np.arange(n, dtype=np.float64)
        a = np.sin(2.0 * np.pi * t / 12.0)
        b = np.sin(2.0 * np.pi * (t - constructed_shift) / 12.0)
        out = fitness_lag(a, b, max_lag=15)
        assert abs(out - constructed_shift) <= 1.0

    def test_zero_variance_series_returns_nan(self) -> None:
        """Two constant series SHALL return NaN (cross-correlation undefined)."""
        out = fitness_lag(np.ones(20), np.ones(20))
        assert math.isnan(out)

    def test_max_lag_exceeds_length_returns_nan(self) -> None:
        """`max_lag` >= series length SHALL return NaN rather than out-of-bounds."""
        a = np.arange(10, dtype=np.float64)
        b = np.arange(10, dtype=np.float64)
        out = fitness_lag(a, b, max_lag=20)
        assert math.isnan(out)


# ---------------------------------------------------------------------------
# Coupled rate
# ---------------------------------------------------------------------------


class TestCoupledRate:
    """Spec scenarios "High Coupling Detected" + "Independent Series"."""

    def test_perfect_coupling_returns_one(self) -> None:
        """Identical deltas (lock-step change) SHALL return ~+1."""
        rng = np.random.default_rng(seed=42)
        a = np.cumsum(rng.normal(0.0, 1.0, size=30))
        b = a + 100.0  # constant offset → identical deltas
        coef = coupled_rate(a, b)
        assert coef == pytest.approx(1.0)

    def test_anti_coupling_returns_negative_one(self) -> None:
        """Negated deltas (anti-phase change) SHALL return ~-1."""
        rng = np.random.default_rng(seed=42)
        a = np.cumsum(rng.normal(0.0, 1.0, size=30))
        b = -a + 50.0  # negated deltas
        coef = coupled_rate(a, b)
        assert coef == pytest.approx(-1.0)

    def test_independent_series_returns_near_zero(self) -> None:
        """Independent random walks SHALL return |coef| << 1."""
        rng = np.random.default_rng(seed=42)
        a = np.cumsum(rng.normal(0.0, 1.0, size=200))
        b = np.cumsum(rng.normal(0.0, 1.0, size=200))
        coef = coupled_rate(a, b)
        assert abs(coef) < 0.2

    def test_zero_variance_returns_nan(self) -> None:
        """Constant series SHALL return NaN (delta variance is zero)."""
        coef = coupled_rate(np.ones(20), np.arange(20))
        assert math.isnan(coef)


# ---------------------------------------------------------------------------
# Generality
# ---------------------------------------------------------------------------


class TestGenerality:
    """Spec scenarios "High Generality" + "Self-Play Overfitting"."""

    def test_uniform_improvement_returns_near_plus_one(self) -> None:
        """Uniform improvement across all opponents SHALL score ~+1."""
        # Each column is the same ramp + small noise → all opponents
        # improve at the same rate → mean slope == max |slope| ⇒ +1.
        rng = np.random.default_rng(seed=42)
        ramp = np.arange(10, dtype=np.float64).reshape(-1, 1)
        mat = ramp + rng.normal(0.0, 0.1, size=(10, 4))
        scalar = generality(mat)
        assert scalar > 0.9

    def test_self_play_overfitting_returns_near_zero_or_negative(self) -> None:
        """Mixed improving + declining opponents SHALL score near 0 (mean slope cancels)."""
        # Two opponents improve, two decline at the same magnitude →
        # mean slope = 0 → scalar = 0.
        n_probes = 10
        improvers = np.tile(
            np.arange(n_probes, dtype=np.float64).reshape(-1, 1),
            (1, 2),
        )
        decliners = -improvers
        mat = np.concatenate([improvers, decliners], axis=1)
        scalar = generality(mat)
        assert abs(scalar) < 0.1

    def test_too_few_probes_returns_nan(self) -> None:
        """Probe matrix with fewer than 2 rows SHALL return NaN (slope undefined)."""
        mat = np.ones((1, 4), dtype=np.float64)
        assert math.isnan(generality(mat))

    def test_zero_movement_returns_zero(self) -> None:
        """All-flat probe matrix (no slope) SHALL return 0.0."""
        mat = np.ones((10, 4), dtype=np.float64) * 0.5
        scalar = generality(mat)
        assert scalar == 0.0


# ---------------------------------------------------------------------------
# Cross-cutting input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Shape mismatch / wrong dimensionality across the metrics."""

    def test_phenotypic_cycling_rejects_2d_input(self) -> None:
        """2-D input to `phenotypic_cycling` SHALL raise."""
        with pytest.raises(ValueError, match="1-D"):
            phenotypic_cycling(np.zeros((10, 2)))

    def test_fitness_lag_rejects_shape_mismatch(self) -> None:
        """Mismatched shapes SHALL raise."""
        with pytest.raises(ValueError, match="shapes must match"):
            fitness_lag(np.arange(10), np.arange(20))

    def test_generality_rejects_1d_input(self) -> None:
        """1-D input to `generality` SHALL raise (probe matrix must be 2-D)."""
        with pytest.raises(ValueError, match="2-D"):
            generality(np.arange(10))
