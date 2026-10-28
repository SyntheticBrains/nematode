"""Chemotaxis statistics on a measured odour field, from positions and the field alone.

A population is a list of track fragments, each a run of 1-s positions in the field's frame, and a
callable giving the field's concentration there. Real and simulated fragments are read the same
way:

- **up-gradient drift** (mm/s): the mean, over valid seconds, of the step's velocity along the
  field's gradient direction at the step's midpoint;
- **the quintile ratio:** reorientation onsets per second (the off-food detector) in the lowest
  quintile of dC/dt, against the highest, the quintiles the population's own (Pierce-Shimomura
  1999's pirouette analysis, unit-free), graded as its logarithm since it is positively skewed;
- **the weathervane slope** (degrees per second): the least-squares slope, over run seconds more
  than :data:`RUN_MARGIN_S` from an onset, of the heading's turning rate on the sine of the bearing
  to the gradient. Positive means turning toward higher concentration.

The bands, from a handful of recordings, are wide enough that a worm ignoring the gradient can sit
inside them. Chemotaxis must therefore also be **present**: drift and the log quintile ratio each
above zero at the population's own lower bound (:func:`chemotaxis_presence`). The weathervane slope
takes no part, since a real recording's own bound on it can fall below zero.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from quantumnematode.benchmark.readouts import OFF_FOOD_DETECTOR, reversal_onsets

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

CHEMOTAXIS_STATISTICS = ("drift_mm_s", "log_quintile_ratio", "weathervane_deg_s")
RUN_MARGIN_S = 2
HEADING_FLOOR_MM = 0.02
GRADIENT_STEP_MM = 0.15
QUINTILES = 5
# The fewest samples a fragment needs for its steps and the detector; a regression's fewest points.
MIN_FRAGMENT_SAMPLES = 3
MIN_REGRESSION_POINTS = 3
PRESENCE_STATISTICS = ("drift_mm_s", "log_quintile_ratio")
PRESENCE_RESAMPLES = 2000
PRESENCE_QUANTILE = 0.025
PRESENCE_SEED = 2041


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def gradient_direction(
    concentration: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Return the field's gradient direction at positions, by central differences."""
    h = GRADIENT_STEP_MM
    gx = concentration(x + h, y) - concentration(x - h, y)
    gy = concentration(x, y + h) - concentration(x, y - h)
    return np.arctan2(gy, gx)


def fragment_measures(
    x: np.ndarray,
    y: np.ndarray,
    concentration: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    """Return one fragment's per-step measures: drift, dC/dt, onsets, turning and bearing."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dx, dy = np.diff(x), np.diff(y)
    valid = np.isfinite(dx) & np.isfinite(dy)
    mx, my = x[:-1] + dx / 2, y[:-1] + dy / 2
    direction = np.full(len(dx), np.nan)
    if valid.any():
        direction[valid] = gradient_direction(concentration, mx[valid], my[valid])
    drift = dx * np.cos(direction) + dy * np.sin(direction)
    c = np.full(len(x), np.nan)
    present = np.isfinite(x) & np.isfinite(y)
    if present.any():
        c[present] = concentration(x[present], y[present])
    dcdt = np.diff(c)
    onset = np.zeros(len(dx), dtype=bool)
    onsets = (
        reversal_onsets(x, y, OFF_FOOD_DETECTOR)
        if present.sum() >= MIN_FRAGMENT_SAMPLES
        else np.array([])
    )
    # The detector's onset at step o is the step that turns back. It is flagged on the step before,
    # so the quintile ratio pairs each reversal with the dC/dt the worm met before it turned, not
    # with the change its own turning back produced; an onset at step 0 has no step before it.
    for o in np.asarray(onsets, dtype=int):
        if 0 < o <= len(dx):
            onset[o - 1] = True
    near = np.zeros(len(dx), dtype=bool)
    for o in np.flatnonzero(onset):
        near[max(0, o - RUN_MARGIN_S) : o + RUN_MARGIN_S + 1] = True
    heading = np.where(np.hypot(dx, dy) >= HEADING_FLOOR_MM, np.arctan2(dy, dx), np.nan)
    turning = np.degrees(_wrap(np.diff(heading)))
    bearing = _wrap(direction[:-1] - heading[:-1])
    run = ~near[:-1] & ~near[1:] & np.isfinite(turning) & np.isfinite(bearing)
    return {
        "valid": valid,
        "drift": drift,
        "dcdt": dcdt,
        "onset": onset,
        "turning": turning[run],
        "sin_bearing": np.sin(bearing[run]),
    }


def fragment_samples(
    fragments: Sequence[tuple[np.ndarray, np.ndarray]],
    concentration: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> list[dict[str, np.ndarray]]:
    """Return each usable fragment's samples the statistics pool: computed once, pooled later."""
    out = []
    for x, y in fragments:
        if len(x) < MIN_FRAGMENT_SAMPLES:
            continue
        m = fragment_measures(x, y, concentration)
        keep = m["valid"] & np.isfinite(m["dcdt"])
        out.append(
            {
                "drift": m["drift"][m["valid"]],
                "dcdt": m["dcdt"][keep],
                "onset": m["onset"][keep],
                "turning": m["turning"],
                "sin_bearing": m["sin_bearing"],
            },
        )
    return out


def pooled_statistics(samples: Sequence[dict[str, np.ndarray]]) -> dict[str, float]:
    """Return the chemotaxis statistics of fragments' pooled samples; NaN where undefined."""
    if not samples:
        return dict.fromkeys(CHEMOTAXIS_STATISTICS, math.nan)
    d = np.concatenate([s["drift"] for s in samples])
    rate = np.concatenate([s["onset"] for s in samples]).astype(float)
    change = np.concatenate([s["dcdt"] for s in samples])
    ratio = math.nan
    if len(change):
        edges = np.quantile(change, np.linspace(0, 1, QUINTILES + 1))
        low = rate[change <= edges[1]]
        high = rate[change >= edges[-2]]
        if len(low) and len(high) and high.mean() > 0:
            ratio = float(low.mean() / high.mean())
    w = np.concatenate([s["turning"] for s in samples])
    b = np.concatenate([s["sin_bearing"] for s in samples])
    if len(w) >= MIN_REGRESSION_POINTS and np.var(b) > 0:
        slope = float(np.cov(b, w, ddof=0)[0, 1] / np.var(b))
    else:
        slope = math.nan
    return {
        "drift_mm_s": float(d.mean()) if len(d) else math.nan,
        "log_quintile_ratio": math.log(ratio) if ratio > 0 else math.nan,
        "weathervane_deg_s": slope,
    }


def chemotaxis_statistics(
    fragments: Sequence[tuple[np.ndarray, np.ndarray]],
    concentration: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, float]:
    """Return a population's chemotaxis statistics from its fragments; NaN where undefined."""
    return pooled_statistics(fragment_samples(fragments, concentration))


def chemotaxis_presence(
    groups: Sequence[Sequence[dict[str, np.ndarray]]],
    seed: int = PRESENCE_SEED,
) -> dict[str, Any]:
    """Return whether chemotaxis is present: drift and the log ratio above 0 at their lower bounds.

    ``groups`` are the resampled units' samples (:func:`fragment_samples`): one per animal for a
    simulated population, one per fragment for a real recording, whose fragments cannot be joined
    into animals. Units are resampled with replacement; each statistic's lower bound is the
    :data:`PRESENCE_QUANTILE` quantile of its resampled values, and a resample where it is undefined
    counts as no chemotaxis.
    """
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(PRESENCE_RESAMPLES, len(groups))) if groups else []
    values: dict[str, list[float]] = {k: [] for k in PRESENCE_STATISTICS}
    for pick in draws:
        stats = pooled_statistics([sample for i in pick for sample in groups[i]])
        for k in PRESENCE_STATISTICS:
            values[k].append(stats[k] if math.isfinite(stats[k]) else -math.inf)
    lower = {
        k: float(np.quantile(v, PRESENCE_QUANTILE, method="lower")) if v else -math.inf
        for k, v in values.items()
    }
    return {"lower_bounds": lower, "present": all(b > 0 for b in lower.values())}


def split_fragments(x: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a track's runs of present samples as fragments."""
    present = np.isfinite(np.asarray(x)) & np.isfinite(np.asarray(y))
    out = []
    t, n = 0, len(present)
    while t < n:
        if not present[t]:
            t += 1
            continue
        u = t
        while u < n and present[u]:
            u += 1
        out.append((np.asarray(x[t:u], dtype=float), np.asarray(y[t:u], dtype=float)))
        t = u
    return out
