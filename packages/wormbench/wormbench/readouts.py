"""Trajectory readouts the benchmark grades, read alike from real and simulated tracks.

Every readout takes positions sampled each second, never a primitive label, posture or a source's
own flag:

- :func:`reversal_onsets` finds reorientations that begin with a reversal, off food. Validated
  against real and simulated off-food tracks, it misses most reversals on a lawn, where they are
  short and slow (recall 0.43 against Scheer's on-food reversal flags), so lawn statistics are not
  built from it. An onset is a step that turns back against the track's recent heading, moves at
  least a minimum distance and, with ``retrace`` on, heads back toward where the worm was. The
  minimum distance is relative to the track's own median step, with an absolute floor where the
  units are mm, so pixels and millimetres read alike.
- :func:`reference_onsets` turns a source's per-frame reverse label into onset times, by one bout
  rule.
- :func:`match_onsets` pairs detected and reference onsets within a tolerance.
- :func:`local_search_curve` and :func:`fit_local_search` give onsets per 2-min bin after removal
  from food and their decay, ``alpha * exp(-gamma * (t - t0)) + beta``.
- :func:`quintile_ratio` is the onset rate in the worm's lowest dC/dt quintile over its highest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

FLOOR_MM = 0.01
# How closely a reversal's step must point back toward where the worm was W seconds earlier.
RETRACE_DEG = 45.0
# A reversal bout longer than this is cut, so a worm that never resumes cannot silence the detector.
MAX_BOUT_S = 30
# The decay rate's search range, per minute: a fit on either bound found no decay in the range.
GAMMA_GRID_MIN_PER_MIN = 1e-3
GAMMA_GRID_MAX_PER_MIN = 3.0
_GAMMA_GRID_POINTS = 400
# A reversal bout: the reverse label's runs after gaps of at most a third of a second are merged,
# kept if they last at least two thirds of a second (two frames at 3 frames/s).
MERGE_GAP_S = 1.0 / 3.0
MIN_RUN_S = 2.0 / 3.0
_EPS = 1e-9


@dataclass(frozen=True)
class DetectorParams:
    """The reversal-onset detector's parameters."""

    theta_deg: float = 120.0
    window_s: int = 3
    kappa: float = 0.25
    refractory_s: int = 3
    retrace: bool = True


# The parameters the detector was tuned to on real off-food tracks and validated with; the
# benchmark's off-food readouts use these.
OFF_FOOD_DETECTOR = DetectorParams(
    theta_deg=135.0,
    window_s=3,
    kappa=0.25,
    refractory_s=5,
    retrace=False,
)


def reversal_onsets(
    x: np.ndarray,
    y: np.ndarray,
    params: DetectorParams = DetectorParams(),  # noqa: B008 - an immutable default
    *,
    floor: float | None = FLOOR_MM,
) -> np.ndarray:
    """Return the seconds at which reversal-initiated reorientations begin.

    ``x`` and ``y`` are positions each second, NaN where missing. ``floor`` is the absolute minimum
    step in the positions' units (``FLOOR_MM`` for mm, ``None`` for pixels). Step ``t`` runs from
    sample ``t`` to ``t + 1``; an onset at ``t`` means that step turned back.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:  # noqa: PLR2004 - a step needs two samples
        return np.zeros(0, dtype=int)
    dx, dy = np.diff(x), np.diff(y)
    length = np.hypot(dx, dy)
    valid = np.isfinite(length)
    above = valid & (length > (floor if floor is not None else 0.0))
    if not above.any():
        return np.zeros(0, dtype=int)
    d_min = max(params.kappa * float(np.median(length[above])), floor or 0.0)
    usable = valid & (length >= d_min)
    # The recent heading: the sum of the usable steps in the previous W seconds.
    ux, uy = np.where(usable, dx, 0.0), np.where(usable, dy, 0.0)
    cx, cy = np.concatenate(([0.0], np.cumsum(ux))), np.concatenate(([0.0], np.cumsum(uy)))
    w = params.window_s
    t = np.arange(len(dx))
    start = np.maximum(t - w, 0)
    hx, hy = cx[t] - cx[start], cy[t] - cy[start]
    h_norm = np.hypot(hx, hy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (dx * hx + dy * hy) / (length * h_norm)
    angle = np.degrees(np.arccos(np.clip(np.nan_to_num(cosine, nan=1.0), -1.0, 1.0)))
    candidate = usable & (h_norm > _EPS) & (t >= 1) & (angle > params.theta_deg)
    if params.retrace:
        # The step must head back along the incoming path: toward where the worm was W seconds ago,
        # within RETRACE_DEG. A reversal does; a sharp forward turn runs off to one side.
        back = t - w
        ok = back >= 0
        bx = np.where(ok, x[np.maximum(back, 0)], np.nan) - x[:-1]
        by = np.where(ok, y[np.maximum(back, 0)], np.nan) - y[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            toward = (dx * bx + dy * by) / (length * np.hypot(bx, by))
            candidate &= np.isfinite(toward) & (toward > math.cos(math.radians(RETRACE_DEG)))
    onsets: list[int] = []
    allowed = 0
    for step in np.flatnonzero(candidate):
        if step < allowed:
            continue
        onsets.append(int(step))
        # The bout lasts while the worm keeps moving against the heading it had; the refractory
        # runs from its end, so resuming that heading is not a second onset.
        heading = np.array([hx[step], hy[step]])
        end = step + 1
        while end < len(dx) and end - step < MAX_BOUT_S:
            if not valid[end]:
                break
            if usable[end] and dx[end] * heading[0] + dy[end] * heading[1] > 0:
                break
            end += 1
        allowed = end + params.refractory_s
    return np.asarray(onsets, dtype=int)


def valid_seconds(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return which one-second steps have both ends present."""
    return np.isfinite(np.diff(np.asarray(x, dtype=float))) & np.isfinite(
        np.diff(np.asarray(y, dtype=float)),
    )


def reference_onsets(label: np.ndarray, frames_per_second: float) -> np.ndarray:
    """Return the onset times (s) of a per-frame reverse label's bouts, by the one bout rule."""
    on = np.asarray(label, dtype=bool).copy()
    gap = math.floor(MERGE_GAP_S * frames_per_second + _EPS)
    shortest = math.ceil(MIN_RUN_S * frames_per_second - _EPS)
    edges = np.flatnonzero(np.diff(np.concatenate(([True], on, [True])).astype(np.int8)))
    for begin, end in zip(edges[::2], edges[1::2], strict=True):
        if begin > 0 and end < len(on) and end - begin <= gap:
            on[begin:end] = True
    edges = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(np.int8), [0]))))
    starts = [b for b, e in zip(edges[::2], edges[1::2], strict=True) if e - b >= shortest]
    return np.asarray(starts, dtype=float) / frames_per_second


def match_onsets(detected: np.ndarray, reference: np.ndarray, tolerance: float = 1.0) -> int:
    """Return how many detected onsets match a reference onset within ``tolerance``, one to one."""
    taken = np.zeros(len(reference), dtype=bool)
    matched = 0
    for d in np.sort(np.asarray(detected, dtype=float)):
        close = np.flatnonzero(~taken & (np.abs(reference - d) <= tolerance))
        if close.size:
            taken[close[np.argmin(np.abs(reference[close] - d))]] = True
            matched += 1
    return matched


@dataclass(frozen=True)
class Curve:
    """Onsets per bin: each bin's start (s), mean, SEM and the animals counted."""

    starts_s: np.ndarray
    mean: np.ndarray
    sem: np.ndarray
    n: np.ndarray


def local_search_curve(  # noqa: PLR0913 - the onsets, their presence and the bins
    onsets: Sequence[np.ndarray],
    valid: Sequence[np.ndarray],
    *,
    bin_s: int = 120,
    start_s: int = 0,
    end_s: int | None = None,
    min_valid: float = 0.9,
) -> Curve:
    """Return onsets per bin over animals, counting a bin only where an animal is present enough.

    ``onsets`` are each animal's onset seconds since removal, ``valid`` its per-second presence.
    """
    stop = end_s if end_s is not None else min(len(v) for v in valid)
    starts = np.arange(start_s, stop - bin_s + 1, bin_s)
    counts: list[list[float]] = [[] for _ in starts]
    for animal_onsets, present in zip(onsets, valid, strict=True):
        for i, b in enumerate(starts):
            window = present[b : b + bin_s]
            if len(window) == bin_s and window.mean() >= min_valid:
                counts[i].append(float(np.sum((animal_onsets >= b) & (animal_onsets < b + bin_s))))
    mean = np.array([np.mean(c) if c else np.nan for c in counts])
    sem = np.array(
        [np.std(c, ddof=1) / math.sqrt(len(c)) if len(c) > 1 else np.nan for c in counts],
    )
    return Curve(starts_s=starts, mean=mean, sem=sem, n=np.array([len(c) for c in counts]))


@dataclass(frozen=True)
class DecayFit:
    """``alpha * exp(-gamma * (t - t0)) + beta``, gamma per minute, t in minutes."""

    alpha: float
    gamma: float
    beta: float
    t0: float
    residual: float


def _linear(t: np.ndarray, values: np.ndarray, gamma: float) -> tuple[float, float, float]:
    design = np.column_stack([np.exp(-gamma * t), np.ones_like(t)])
    (alpha, beta), *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = float(np.sum((design @ np.array([alpha, beta]) - values) ** 2))
    return float(alpha), float(beta), residual


def fit_local_search(t_min: np.ndarray, values: np.ndarray, t0: float) -> DecayFit:
    """Fit ``alpha * exp(-gamma * (t - t0)) + beta`` by least squares over bin centres (minutes).

    For each gamma the best alpha and beta are linear; gamma is found on a log grid from
    ``GAMMA_GRID_MIN_PER_MIN`` to ``GAMMA_GRID_MAX_PER_MIN``, then refined by golden-section search
    between the grid's neighbours.
    """
    t = np.asarray(t_min, dtype=float) - t0
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    t, v = t[keep], v[keep]
    grid = np.geomspace(GAMMA_GRID_MIN_PER_MIN, GAMMA_GRID_MAX_PER_MIN, _GAMMA_GRID_POINTS)
    errors = [_linear(t, v, g)[2] for g in grid]
    best = int(np.argmin(errors))
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        a = high - ratio * (high - low)
        b = low + ratio * (high - low)
        if _linear(t, v, a)[2] < _linear(t, v, b)[2]:
            high = b
        else:
            low = a
    gamma = (low + high) / 2
    alpha, beta, residual = _linear(t, v, gamma)
    return DecayFit(alpha=alpha, gamma=gamma, beta=beta, t0=t0, residual=residual)


def quintile_ratio(
    onsets: Sequence[np.ndarray],
    dcdt: Sequence[np.ndarray],
    valid: Sequence[np.ndarray],
) -> float:
    """Return the onset rate in the worm's lowest dC/dt quintile over the rate in its highest.

    Quintiles are taken within each worm over its valid seconds; onsets and seconds are pooled over
    worms per quintile. Any monotone rescaling of dC/dt leaves the result unchanged.
    """
    events = np.zeros(5)
    seconds = np.zeros(5)
    for animal_onsets, derivative, present in zip(onsets, dcdt, valid, strict=True):
        d = np.asarray(derivative, dtype=float)[: len(present)]
        use = present[: len(d)] & np.isfinite(d)
        if use.sum() < 5:  # noqa: PLR2004 - five quintiles need five seconds
            continue
        ranks = np.empty(use.sum(), dtype=int)
        ranks[np.argsort(d[use], kind="stable")] = np.arange(use.sum())
        quintile = np.full(len(d), -1)
        quintile[use] = (ranks * 5) // use.sum()
        for q in range(5):
            seconds[q] += np.sum(quintile == q)
        for onset in animal_onsets:
            if 0 <= onset < len(quintile) and quintile[onset] >= 0:
                events[quintile[onset]] += 1
    rates = np.divide(events, seconds, out=np.full(5, np.nan), where=seconds > 0)
    return float(rates[0] / rates[4]) if rates[4] > 0 else float("nan")
