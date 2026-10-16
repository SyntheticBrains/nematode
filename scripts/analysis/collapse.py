#!/usr/bin/env python
"""The collapse statistic: one per-seed definition of a learner that reaches a peak and falls off it.

Small PPO learners trained long enough can lose a policy they had. A panel that reads only a
final-quarter plateau scores such a seed as "never learned"; a panel that reads a best block scores it
as if it kept its peak. This module is the one definition every long panel reports beside its
primary, so neither reading hides the loss.

**The definition** (registered, D25). On a run's per-episode success series:

* the **peak** at episode *t* is the highest trailing-200-episode success rate seen by *t*;
* **onset** is the first episode at which the trailing-100-episode rate falls below half the peak
  and stays below it for at least 300 consecutive episodes;
* a run whose fall would be confirmed only after its horizon is **censored**: no collapse observed;
* a collapsed run **recovered** if its trailing-100 rate returns above 0.8 of the peak within
  1,000 episodes of onset.

A run that never succeeds has a peak of zero and cannot fall below half of it, so it does not
collapse; it is a run that never learned, which the panel's own floor gate reports.

Over a panel: incidence with a Wilson interval, and Kaplan-Meier time-to-collapse with every
uncollapsed run censored at its horizon. The sensitivity grid varies the trailing window, the
fraction and the persistence and reports how incidence moves. Logbook 092's own rule (500-episode
blocks; a peak block of at least 30%, and a last block below half of it) is computed beside it.

Usage::

    uv run python scripts/analysis/collapse.py --logs <dir> --stem <config stem> [--stem ...] \
        --out collapse.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PEAK_WINDOW = 200
LEVEL_WINDOW = 100
FRACTION = 0.5
PERSISTENCE = 300
RECOVERY_FRACTION = 0.8
RECOVERY_WINDOW = 1000
# 95% Wilson intervals.
Z = 1.959963984540054
# The sensitivity grid.
PEAK_WINDOWS: tuple[int, ...] = (100, 200, 400)
FRACTIONS: tuple[float, ...] = (0.4, 0.5, 0.6)
PERSISTENCES: tuple[int, ...] = (200, 300, 500)
# Logbook 092's rule.
BLOCK = 500
BLOCK_MIN_PEAK = 0.30

_RUN_LINE = re.compile(r"Run:\s+\d+\s+Status:\s+(\S+)")


@dataclass(frozen=True)
class Collapse:
    """One run's reading under the definition."""

    horizon: int
    peak: float
    final: float | None
    onset: int | None
    peak_at_onset: float | None
    recovered: bool | None

    @property
    def collapsed(self) -> bool:
        """Whether an onset was observed before the horizon."""
        return self.onset is not None


def success_series(log: Path) -> np.ndarray:
    """Every episode's outcome, 1 for success and 0 otherwise, read off a run's ``Run:`` lines."""
    return np.array(
        [
            1.0 if m.group(1) == "SUCCESS" else 0.0
            for line in log.read_text().splitlines()
            if (m := _RUN_LINE.match(line))
        ],
    )


def trailing_mean(series: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over ``window`` episodes, NaN until the first full window."""
    out = np.full(len(series), np.nan)
    if len(series) >= window:
        sums = np.cumsum(np.insert(series.astype(float), 0, 0.0))
        out[window - 1 :] = (sums[window:] - sums[:-window]) / window
    return out


def detect(  # noqa: PLR0913 - the definition's five parameters, each varied by the grid
    series: np.ndarray,
    *,
    peak_window: int = PEAK_WINDOW,
    level_window: int = LEVEL_WINDOW,
    fraction: float = FRACTION,
    persistence: int = PERSISTENCE,
    recovery_window: int = RECOVERY_WINDOW,
) -> Collapse:
    """Read one run's success series under the definition (registered values by default)."""
    horizon = len(series)
    peak_curve = np.fmax.accumulate(np.nan_to_num(trailing_mean(series, peak_window), nan=0.0))
    level = trailing_mean(series, level_window)
    below = np.nan_to_num(level, nan=np.inf) < fraction * peak_curve
    onset: int | None = None
    run = 0
    for t in range(horizon):
        run = run + 1 if below[t] else 0
        if run == persistence:
            onset = t - persistence + 1
            break
    final = float(level[-1]) if horizon and not np.isnan(level[-1]) else None
    if onset is None:
        return Collapse(horizon, float(peak_curve[-1]) if horizon else 0.0, final, None, None, None)
    peak = float(peak_curve[onset])
    after = level[onset : onset + recovery_window]
    recovered = bool(np.any(after > RECOVERY_FRACTION * peak))
    return Collapse(horizon, float(peak_curve[-1]), final, onset + 1, peak, recovered)


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def kaplan_meier(readings: list[Collapse]) -> dict[str, Any]:
    """Time to collapse, each uncollapsed run censored at its horizon.

    Returns the survival curve's steps as ``[episode, surviving fraction]`` and the median time to
    collapse, ``None`` when the curve never falls to one half.
    """
    events = sorted(
        (r.onset if r.onset is not None else r.horizon, r.onset is not None) for r in readings
    )
    at_risk = len(events)
    survival = 1.0
    steps: list[list[float]] = []
    median: int | None = None
    i = 0
    while i < len(events):
        time = events[i][0]
        tied = [e for e in events[i:] if e[0] == time]
        deaths = sum(1 for _, event in tied if event)
        if deaths:
            survival *= 1 - deaths / at_risk
            steps.append([time, survival])
            if median is None and survival <= 0.5:
                median = time
        at_risk -= len(tied)
        i += len(tied)
    return {"steps": steps, "median": median, "n": len(readings)}


def incidence(readings: list[Collapse]) -> dict[str, Any]:
    """Collapsed count over runs, with its Wilson interval."""
    k, n = sum(r.collapsed for r in readings), len(readings)
    lo, hi = wilson(k, n)
    return {"collapsed": k, "n": n, "rate": k / n if n else None, "wilson_95": [lo, hi]}


def rule_092(series: np.ndarray, block: int = BLOCK) -> dict[str, Any]:
    """Logbook 092's rule: the best 500-episode block of at least 30%, then a last block below half.

    A run whose best block is under 30% never learned under this rule and is not counted as a
    collapse.
    """
    blocks = [float(series[i : i + block].mean()) for i in range(0, len(series), block)]
    if not blocks:
        return {"collapsed": False, "never_learned": True, "peak": None, "last": None}
    peak, last = max(blocks), blocks[-1]
    never = peak < BLOCK_MIN_PEAK
    return {
        "collapsed": (not never) and last < FRACTION * peak,
        "never_learned": never,
        "peak": peak,
        "last": last,
    }


def sensitivity(series_by_run: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Incidence over the grid of trailing windows, fractions and persistences."""
    out = []
    for window in PEAK_WINDOWS:
        for fraction in FRACTIONS:
            for persistence in PERSISTENCES:
                readings = [
                    detect(s, peak_window=window, fraction=fraction, persistence=persistence)
                    for s in series_by_run.values()
                ]
                out.append(
                    {
                        "peak_window": window,
                        "fraction": fraction,
                        "persistence": persistence,
                        "registered": (window, fraction, persistence)
                        == (PEAK_WINDOW, FRACTION, PERSISTENCE),
                        **incidence(readings),
                    },
                )
    return out


def auc_to(series: np.ndarray, episodes: int) -> float:
    """Mean success over the first ``episodes`` episodes (the primary to that horizon)."""
    return float(series[:episodes].mean()) if len(series) else 0.0


def panel(series_by_run: dict[str, np.ndarray], early_horizon: int = 3000) -> dict[str, Any]:
    """Every run's reading, the panel's incidence and time to collapse, the grid, and 092's rule."""
    readings = {run: detect(s) for run, s in series_by_run.items()}
    return {
        "definition": {
            "peak_window": PEAK_WINDOW,
            "level_window": LEVEL_WINDOW,
            "fraction": FRACTION,
            "persistence": PERSISTENCE,
            "recovery": [RECOVERY_FRACTION, RECOVERY_WINDOW],
        },
        "runs": {
            run: {
                **asdict(r),
                "collapsed": r.collapsed,
                "rule_092": rule_092(series_by_run[run]),
                f"auc_to_{early_horizon}": auc_to(series_by_run[run], early_horizon),
                "auc_to_horizon": auc_to(series_by_run[run], len(series_by_run[run])),
            }
            for run, r in readings.items()
        },
        "incidence": incidence(list(readings.values())),
        "kaplan_meier": kaplan_meier(list(readings.values())),
        "sensitivity": sensitivity(series_by_run),
        "rule_092_incidence": sum(rule_092(s)["collapsed"] for s in series_by_run.values()),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: read every run of each stem in the log directories, and report each stem's panel."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--logs", type=Path, action="append", required=True)
    ap.add_argument("--stem", action="append", required=True, help="a config stem, one arm")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    result: dict[str, Any] = {}
    for stem in args.stem:
        series = {
            log.stem: success_series(log)
            for log_dir in args.logs
            for log in sorted(log_dir.glob(f"{stem}-seed*.log"))
        }
        result[stem] = panel(series)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
