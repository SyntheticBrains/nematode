"""Tests for the klinotaxis behavioural bias-curve metrics (§2)."""

import math

from quantumnematode.report.dtypes import BehaviourStep
from quantumnematode.validation import behavioural_curves as bc


def _steps(headings, dc_dts, grad_dirs=None, grad_strengths=None) -> list[BehaviourStep]:
    """Build a synthetic trajectory; positions advance one unit along each heading."""
    n = len(headings)
    grad_dirs = grad_dirs if grad_dirs is not None else [0.0] * n
    grad_strengths = grad_strengths if grad_strengths is not None else [1.0] * n
    out, x, y = [], 0.0, 0.0
    for i in range(n):
        x += math.cos(headings[i])
        y += math.sin(headings[i])
        out.append(
            BehaviourStep(
                step=i,
                x=x,
                y=y,
                heading_rad=headings[i],
                concentration=0.5,
                dc_dt=dc_dts[i],
                grad_dir=grad_dirs[i],
                grad_strength=grad_strengths[i],
            ),
        )
    return out


def _klinokinesis_worm() -> list[BehaviourStep]:
    """Sharp turns only when heading down-gradient (dc_dt < 0)."""
    headings, dc_dts, h = [0.0], [], 0.0
    for t in range(20):
        turn = t % 2 == 0
        dc_dts.append(-1.0 if turn else 1.0)  # dc_dt at the transition's start step
        h += 2.0 if turn else 0.05
        headings.append(h)
    dc_dts.append(0.0)  # final step has no outgoing transition
    return _steps(headings, dc_dts)


def _weathervane_worm() -> list[BehaviourStep]:
    """Heading relaxes toward grad_dir=0 -> gradual curving toward the gradient."""
    headings, h = [], 1.5
    for _ in range(30):
        headings.append(h)
        h *= 0.9  # gradual turn toward the gradient (grad_dir = 0)
    return _steps(headings, [0.0] * len(headings), grad_dirs=[0.0] * len(headings))


def test_klinokinesis_ratio_above_one_when_turns_track_down_gradient():
    """Down-gradient turns -> klinokinesis ratio > 1, and curve A higher at dc_dt < 0."""
    kin = bc.kinematics(_klinokinesis_worm(), theta_sharp=1.0)
    ratio = bc.klinokinesis_ratio(kin)
    assert ratio is not None
    assert ratio > 1.0
    curve = bc.turn_rate_vs_dcdt(kin)
    down = [
        v
        for c, v in zip(curve.bin_centers, curve.values, strict=True)
        if c < 0 and math.isfinite(v)
    ]
    up = [
        v
        for c, v in zip(curve.bin_centers, curve.values, strict=True)
        if c > 0 and math.isfinite(v)
    ]
    assert max(down) > max(up)  # more reorientations heading down-gradient


def test_weathervane_slope_positive_when_curving_toward_gradient():
    """Curving toward the gradient -> positive weathervane slope + toward-sign curving bins."""
    kin = bc.kinematics(_weathervane_worm(), theta_sharp=1.0)
    slope = bc.weathervane_slope(kin)
    assert slope is not None
    assert slope > 0.0
    curve = bc.curving_rate_vs_bearing(kin)
    # bearing is negative here (grad at 0, heading > 0); curving should be negative (toward 0).
    neg = [
        v
        for c, v in zip(curve.bin_centers, curve.values, strict=True)
        if c < 0 and math.isfinite(v)
    ]
    assert neg
    assert all(v < 0 for v in neg)


def test_wrap_around_heading_change():
    """A heading step across +/-pi wraps to a small change, not ~2pi."""
    kin = bc.kinematics(_steps([math.pi - 0.1, -(math.pi - 0.1)], [0.0, 0.0]), theta_sharp=1.0)
    assert len(kin) == 1
    assert abs(kin[0].dtheta) < 0.5  # ~0.2, not ~2*pi - 0.2
    assert not kin[0].is_turn


def test_threshold_free_magnitude_ratio_above_one_down_gradient():
    """Larger turns heading down-gradient -> threshold-free magnitude ratio > 1 (no theta_sharp)."""
    kin = bc.kinematics(_klinokinesis_worm(), theta_sharp=1.0)
    ratio = bc.klinokinesis_magnitude_ratio(kin)
    assert ratio is not None
    assert ratio > 1.0


def test_threshold_free_weathervane_slope_positive_toward_gradient():
    """Curving toward the gradient -> positive all-step weathervane slope (theta_sharp-free)."""
    kin = bc.kinematics(_weathervane_worm(), theta_sharp=1.0)
    slope = bc.weathervane_slope_all(kin)
    assert slope is not None
    assert slope > 0.0


def test_min_path_len_floor_excludes_creep_outliers():
    """A tiny-displacement creep step (dtheta/path_len explodes) is dropped once floored."""
    # Three clean strides toward grad_dir=0 plus one near-stationary creep with a big turn.
    headings = [0.5, 0.4, 0.3, 0.2]
    steps = _steps(headings, [0.0] * len(headings), grad_dirs=[0.0] * len(headings))
    # Append a near-stationary step: displacement ~0 but a large heading change.
    last = steps[-1]
    steps.append(
        BehaviourStep(
            step=len(steps),
            x=last.x + 1e-3,
            y=last.y + 1e-3,
            heading_rad=last.heading_rad + 0.4,
            concentration=0.5,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=1.0,
        ),
    )
    kin = bc.kinematics(steps, theta_sharp=1.0)
    floor = bc.suggest_min_path_len(kin)
    assert floor > 0.0
    floored = bc.weathervane_slope_all(kin, min_path_len=floor)
    unfloored = bc.weathervane_slope_all(kin, min_path_len=0.0)
    assert floored is not None
    assert abs(floored) < abs(unfloored)  # the creep outlier no longer dominates the slope


def test_suggest_min_path_len_empty_is_zero():
    """No usable strides -> a zero floor (no-op)."""
    assert bc.suggest_min_path_len([]) == 0.0


def test_threshold_free_metrics_are_theta_sharp_invariant():
    """The threshold-free statistics do not depend on theta_sharp (unlike the thresholded ones)."""
    worm = _klinokinesis_worm()
    lo = bc.kinematics(worm, theta_sharp=0.1)
    hi = bc.kinematics(worm, theta_sharp=3.0)
    assert bc.klinokinesis_magnitude_ratio(lo) == bc.klinokinesis_magnitude_ratio(hi)
    assert bc.weathervane_slope_all(lo) == bc.weathervane_slope_all(hi)


def test_empty_and_single_step_safe():
    """No trajectory / one step -> no kinematics, and the reducers return empty/None."""
    assert bc.kinematics([], theta_sharp=1.0) == []
    assert bc.kinematics(_steps([0.0], [0.0]), theta_sharp=1.0) == []
    assert bc.turn_rate_vs_dcdt([]).values == []
    assert bc.curving_rate_vs_bearing([]).values == []
    assert bc.klinokinesis_ratio([]) is None
    assert bc.weathervane_slope([]) is None
    assert bc.klinokinesis_magnitude_ratio([]) is None
    assert bc.weathervane_slope_all([]) is None


def test_suggest_theta_sharp_returns_a_high_percentile():
    """The calibrated threshold sits above the gradual-turn bulk of the |dtheta| distribution."""
    worm = _klinokinesis_worm()
    theta = bc.suggest_theta_sharp(worm, percentile=75.0)
    assert 0.05 < theta <= 2.0  # between the gradual (0.05) and sharp (2.0) modes
