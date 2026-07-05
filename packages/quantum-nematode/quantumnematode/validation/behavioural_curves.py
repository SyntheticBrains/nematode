"""Behavioural klinotaxis bias curves for real-worm chemotaxis validation.

Pure functions over a captured behavioural trajectory (``list[BehaviourStep]``) that reconstruct the
two documented *C. elegans* gradient-navigation strategies from the RL worm's own output:

- **klinokinesis** (biased random walk): TURN-RATE vs dC/dt — sharp reorientations (pirouettes) are
  more frequent heading down-gradient (Pierce-Shimomura et al. 1999).
- **klinotaxis** (weathervane): CURVING-RATE vs bearing-to-gradient — the worm gradually curves
  toward the gradient (Iino & Yoshida 2009).

Each transition's heading change splits into a sharp reorientation (``|dtheta| > theta_sharp``) and
a gradual curving; the two bias curves + their reduced statistics (down/up turn-rate ratio;
weathervane slope) are computed here as pure functions, testable in isolation.

Sign conventions (heading + gradient are world-frame radians, CCW-positive): the bearing is
``wrap(grad_dir - heading)``; a worm curving *toward* the gradient produces a heading change with
the same sign as the bearing, so the weathervane slope is **positive**, and the klinokinesis ratio
(down/up-gradient turn-rate) is **> 1**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantumnematode.report.dtypes import BehaviourStep

_TWO_PI = 2.0 * math.pi
_EPS = 1e-9
_MIN_SLOPE_POINTS = 2


def _wrap(theta: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (theta + math.pi) % _TWO_PI - math.pi


@dataclass(slots=True)
class StepKinematics:
    """Per-transition kinematics between two consecutive ``BehaviourStep`` records."""

    dtheta: float  # wrapped heading change (rad)
    is_turn: bool  # sharp reorientation: |dtheta| > theta_sharp
    path_len: float  # step displacement (mm)
    dc_dt: float  # dC/dt at the earlier step (the decision covariate)
    bearing: float  # wrap(grad_dir - heading) at the earlier step (rad)
    grad_strength: float  # gradient magnitude at the earlier step


@dataclass(slots=True)
class BiasCurve:
    """A binned rate-vs-covariate curve (a bin value is NaN when the bin is empty)."""

    bin_centers: list[float]
    values: list[float]
    counts: list[int]


def kinematics(steps: Sequence[BehaviourStep], theta_sharp: float) -> list[StepKinematics]:
    """Per-transition kinematics (``len(steps) - 1`` records); empty for fewer than two steps."""
    out: list[StepKinematics] = []
    for a, b in pairwise(steps):
        dtheta = _wrap(b.heading_rad - a.heading_rad)
        out.append(
            StepKinematics(
                dtheta=dtheta,
                is_turn=abs(dtheta) > theta_sharp,
                path_len=math.hypot(b.x - a.x, b.y - a.y),
                dc_dt=a.dc_dt,
                bearing=_wrap(a.grad_dir - a.heading_rad),
                grad_strength=a.grad_strength,
            ),
        )
    return out


def rate_vs_binned_covariate(
    covariate: np.ndarray,
    value: np.ndarray,
    bin_edges: np.ndarray,
) -> BiasCurve:
    """Mean ``value`` per ``covariate`` bin (a rate when ``value`` is a 0/1 indicator)."""
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    idx = np.clip(np.digitize(covariate, bin_edges) - 1, 0, len(centers) - 1)
    vals: list[float] = []
    counts: list[int] = []
    for k in range(len(centers)):
        mask = idx == k
        n = int(mask.sum())
        counts.append(n)
        vals.append(float(value[mask].mean()) if n else float("nan"))
    return BiasCurve(bin_centers=[float(c) for c in centers], values=vals, counts=counts)


def _usable(
    kin: Sequence[StepKinematics],
    min_grad_strength: float,
) -> list[StepKinematics]:
    """Transitions with a usable gradient + a real displacement (bearing/curving are defined)."""
    return [k for k in kin if k.grad_strength > min_grad_strength and k.path_len > _EPS]


def _gradual(
    kin: Sequence[StepKinematics],
    min_grad_strength: float,
) -> list[StepKinematics]:
    """Non-reorientation usable transitions (gradual curving for the thresholded weathervane)."""
    return [k for k in _usable(kin, min_grad_strength) if not k.is_turn]


def _slope(usable: Sequence[StepKinematics]) -> float | None:
    """Least-squares slope of signed curving-rate (rad/mm) vs bearing; None if degenerate."""
    if len(usable) < _MIN_SLOPE_POINTS:
        return None
    bearing = np.array([k.bearing for k in usable])
    curving = np.array([k.dtheta / k.path_len for k in usable])
    if float(bearing.var()) < _EPS:
        return None
    return float(np.polyfit(bearing, curving, 1)[0])


def turn_rate_vs_dcdt(kin: Sequence[StepKinematics], n_bins: int = 7) -> BiasCurve:
    """Curve A: reorientation rate binned by dC/dt (klinokinesis)."""
    if not kin:
        return BiasCurve([], [], [])
    dcdt = np.array([k.dc_dt for k in kin])
    is_turn = np.array([1.0 if k.is_turn else 0.0 for k in kin])
    lo, hi = float(dcdt.min()), float(dcdt.max())
    if hi - lo < _EPS:
        hi = lo + 1.0
    return rate_vs_binned_covariate(dcdt, is_turn, np.linspace(lo, hi, n_bins + 1))


def curving_rate_vs_bearing(
    kin: Sequence[StepKinematics],
    n_bins: int = 8,
    min_grad_strength: float = 0.0,
) -> BiasCurve:
    """Curve B: mean signed gradual curving-rate (rad/mm) binned by bearing (weathervane)."""
    grad = _gradual(kin, min_grad_strength)
    if not grad:
        return BiasCurve([], [], [])
    bearing = np.array([k.bearing for k in grad])
    curving = np.array([k.dtheta / k.path_len for k in grad])  # rad/mm, signed
    return rate_vs_binned_covariate(bearing, curving, np.linspace(-math.pi, math.pi, n_bins + 1))


def klinokinesis_ratio(kin: Sequence[StepKinematics]) -> float | None:
    """Down/up-gradient turn-rate ratio (> 1 = biased random walk); None if either side is empty."""
    down = [1.0 if k.is_turn else 0.0 for k in kin if k.dc_dt < 0]
    up = [1.0 if k.is_turn else 0.0 for k in kin if k.dc_dt > 0]
    if not down or not up:
        return None
    up_rate = float(np.mean(up))
    return float("inf") if up_rate < _EPS else float(np.mean(down)) / up_rate


def weathervane_slope(
    kin: Sequence[StepKinematics],
    min_grad_strength: float = 0.0,
) -> float | None:
    """Thresholded weathervane: gradual (non-turn) curving-rate-vs-bearing slope (> 0 = toward)."""
    return _slope(_gradual(kin, min_grad_strength))


def klinokinesis_magnitude_ratio(kin: Sequence[StepKinematics]) -> float | None:
    """Threshold-free klinokinesis: mean |dtheta| down- vs up-gradient (> 1 = larger turns down).

    Avoids the sharp/gradual threshold entirely by comparing turn MAGNITUDE (not a thresholded
    rate) heading down- vs up-gradient; robust when the |dtheta| distribution has no natural cut.
    None if either side is empty; inf if the up-gradient mean is ~0.
    """
    down = [abs(k.dtheta) for k in kin if k.dc_dt < 0]
    up = [abs(k.dtheta) for k in kin if k.dc_dt > 0]
    if not down or not up:
        return None
    up_mean = float(np.mean(up))
    return float("inf") if up_mean < _EPS else float(np.mean(down)) / up_mean


def weathervane_slope_all(
    kin: Sequence[StepKinematics],
    min_grad_strength: float = 0.0,
) -> float | None:
    """Threshold-free weathervane: the curving-rate-vs-bearing slope over ALL usable steps.

    The threshold-free companion to :func:`weathervane_slope` - it does not exclude sharp
    reorientations, so it does not depend on ``theta_sharp``. > 0 = curves toward the gradient.
    """
    return _slope(_usable(kin, min_grad_strength))


def suggest_theta_sharp(steps: Sequence[BehaviourStep], percentile: float = 90.0) -> float:
    """Calibrate the sharp-turn threshold from the |Δθ| distribution (a high-percentile cut)."""
    if len(steps) < _MIN_SLOPE_POINTS:
        return math.pi / 4
    dth = np.array([abs(_wrap(b.heading_rad - a.heading_rad)) for a, b in pairwise(steps)])
    return float(np.percentile(dth, percentile))
