#!/usr/bin/env python
r"""Measure how a real on-lawn reversal's speed follows the worm's speed just before it.

Reads Scheer & Bargmann 2023's OD 2 wild type from the vendored store: the authors' per-frame
reversal and omega flags at 3 frames/s and the 1-s midbody positions. Bouts follow the kinematics
check's rule (one-frame gaps merged, at least two frames kept).

**The rule, as the primitives apply it.** Each reversal bout that starts on the lawn (the midbody
inside the boundary) is read against the worm's last 3 run seconds before it, wherever they fell. A
run second is one with no reversal or omega frame and a finite 1-s displacement.

- **The speed before:** those seconds' mean 1-s displacement.
- **The reversal's speed:** its path over the bout's whole seconds, as the kinematics check reads it.
- **Its duration:** its frames over 3.

**Reported:**

- for the held-out cohorts (the source of the primitives' values), every on-lawn start and the stratum
  starting more than 0.5 mm inside the lawn;
- for the band cohorts, every on-lawn start, beside as replication;
- per set: the least-squares line of reversal speed on the speed before, with a 95% bootstrap over
  recording dates; and the mean speed before, reversal speed and duration by decile of the speed
  before;
- a paused worm's 1-s displacement in the vendored still-worm library
  (`data/benchmark/scheer_tracker_noise.npz`), its mean and RMS, beside the slowest decile's speed
  before, as a check on how far tracking noise lifts the slowest speeds.

The primitives take the held-out on-lawn line, clipped to its lowest and highest decile's mean
reversal speed. The clip sets the slow end: a still worm reverses at the slowest decile's mean.

Usage::

    uv run python scripts/analysis/reversal_speed_coupling.py \
        --out docs/experiments/logbooks/supporting/132-speed-following-reversals/coupling.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from primitive_kinematics_check import BOUNDARY_BAND_MM, WILD_TYPE, bouts, finite_or_null
from quantumnematode.benchmark import references as refs

DESCRIPTION = "How a real on-lawn reversal's speed follows the worm's speed just before it."

FPS = int(refs.SCHEER_FRAMES_PER_SECOND)
WINDOW_S = 3
DECILES = 10
RESAMPLES = 2000
BOOTSTRAP_SEED = 20261111
NOISE_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "scheer_tracker_noise.npz"


def bout_samples(
    x: np.ndarray,
    y: np.ndarray,
    radii: np.ndarray,
    reversal: np.ndarray,
    omega: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    """Return one animal's on-lawn reversals: speed before, reversal speed, duration, depth.

    Depth is the start's signed distance inside the lawn's boundary, in mm. A bout counts when it
    starts on the lawn and the worm has ``WINDOW_S`` run seconds before it.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    with np.errstate(invalid="ignore"):
        depth = refs.boundary_distance(x, y, radii)
    busy = np.zeros(len(x), dtype=bool)
    for flag in (reversal, omega):
        for start, stop in bouts(flag, filtered=True):
            busy[start // FPS : (stop - 1) // FPS + 1] = True
    step = np.hypot(np.diff(x), np.diff(y))  # step[s] is the motion from second s to s + 1
    runs = np.flatnonzero(~busy[:-1] & np.isfinite(step))
    out = []
    for start, stop in bouts(reversal, filtered=True):
        first_second = start // FPS
        whole_first, whole_last = -(-start // FPS), stop // FPS
        if first_second >= len(depth) or not depth[first_second] > 0:
            continue
        before = runs[runs < first_second][-WINDOW_S:]
        if len(before) < WINDOW_S or whole_last <= whole_first or whole_last > len(step):
            continue
        path = step[whole_first:whole_last]
        if not np.isfinite(path).all():
            continue
        out.append(
            (
                float(step[before].mean()),
                float(path.sum() / (whole_last - whole_first)),
                (stop - start) / FPS,
                float(depth[first_second]),
            ),
        )
    return out


def collect(tracks: refs.ScheerTracks, side: str) -> dict[str, np.ndarray]:
    """Return every counted bout of the wild type's ``side`` cohorts, tagged by recording date."""
    rows = [i for i in tracks.arm(WILD_TYPE) if tracks.side[i] == side]
    names = ("reversal", "omega")
    flags = {
        n: refs.unpack_flag(tracks.frame_flags[rows], tracks.flag_names.index(n)) for n in names
    }
    columns: dict[str, list[float]] = {
        "before": [],
        "reversal": [],
        "duration": [],
        "depth": [],
        "date": [],
    }
    for j, i in enumerate(rows):
        for sample in bout_samples(
            tracks.x[i],
            tracks.y[i],
            tracks.boundary_radius[i],
            flags["reversal"][j],
            flags["omega"][j],
        ):
            for name, value in zip(
                ("before", "reversal", "duration", "depth"),
                sample,
                strict=True,
            ):
                columns[name].append(value)
            columns["date"].append(float(tracks.date[i]))
    return {name: np.asarray(values) for name, values in columns.items()}


def fit(samples: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return the line, its bootstrap over recording dates, and the deciles of the speed before."""
    before, speed = samples["before"], samples["reversal"]
    slope, intercept = np.polyfit(before, speed, 1)
    dates = np.unique(samples["date"])
    by_date = [np.flatnonzero(samples["date"] == d) for d in dates]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lines = []
    for _ in range(RESAMPLES):
        pick = np.concatenate([by_date[k] for k in rng.integers(0, len(dates), len(dates))])
        lines.append(np.polyfit(before[pick], speed[pick], 1))
    lines_array = np.asarray(lines)
    edges = np.quantile(before, np.linspace(0.0, 1.0, DECILES + 1))
    deciles = []
    for low, high in itertools.pairwise(edges):
        keep = (before >= low) & (before <= high)
        deciles.append(
            {
                "before_mm_per_s": float(before[keep].mean()),
                "reversal_mm_per_s": float(speed[keep].mean()),
                "duration_s": float(samples["duration"][keep].mean()),
                "bouts": int(keep.sum()),
            },
        )
    return {
        "bouts": len(speed),
        "dates": len(dates),
        "line": {
            "intercept_mm_per_s": float(intercept),
            "slope": float(slope),
            "intercept_interval": [float(np.percentile(lines_array[:, 1], q)) for q in (2.5, 97.5)],
            "slope_interval": [float(np.percentile(lines_array[:, 0], q)) for q in (2.5, 97.5)],
        },
        "mean_reversal_mm_per_s": float(speed.mean()),
        "mean_duration_s": float(samples["duration"].mean()),
        "deciles": deciles,
    }


def tracker_floor(path: Path = NOISE_PATH) -> dict[str, float]:
    """Return a paused worm's 1-s displacement in the vendored still-worm library, mean and RMS."""
    data = np.load(path)
    x, y, lengths = data["x_um"], data["y_um"], data["lengths"]
    steps, offset = [], 0
    for length in lengths:
        xs, ys = x[offset : offset + length] / 1000.0, y[offset : offset + length] / 1000.0
        offset += int(length)
        steps.append(np.hypot(np.diff(xs), np.diff(ys)))
    joined = np.concatenate(steps)
    joined = joined[np.isfinite(joined)]
    return {
        "mean_mm": float(joined.mean()),
        "rms_mm": float(np.sqrt(np.mean(joined**2))),
        "seconds": len(joined),
    }


def main() -> int:
    """CLI: measure the coupling on the held-out and band cohorts and write its record."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tracks = refs.load_scheer_tracks()
    held_out = collect(tracks, refs.HELD_OUT)
    inside = {k: v[held_out["depth"] > BOUNDARY_BAND_MM] for k, v in held_out.items()}
    band = collect(tracks, refs.BANDS)
    chosen = fit(held_out)
    floor = tracker_floor()
    record = {
        "description": DESCRIPTION,
        "rule": {
            "window_run_seconds": WINDOW_S,
            "run_second": "no reversal or omega frame, and a finite 1-s displacement",
            "start": "the midbody inside the lawn's boundary",
            "bout_rule": "primitive_kinematics_check.bouts, filtered",
        },
        "bootstrap": {"unit": "recording date", "resamples": RESAMPLES, "seed": BOOTSTRAP_SEED},
        "held_out_on_lawn": chosen,
        "held_out_inside_lawn": {"depth_mm": BOUNDARY_BAND_MM, **fit(inside)},
        "band_on_lawn": fit(band),
        "tracker_floor": floor,
        "primitives_entry": {
            "intercept_mm_per_s": chosen["line"]["intercept_mm_per_s"],
            "slope": chosen["line"]["slope"],
            "low_mm_per_s": chosen["deciles"][0]["reversal_mm_per_s"],
            "high_mm_per_s": chosen["deciles"][-1]["reversal_mm_per_s"],
            "window_s": float(WINDOW_S),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(finite_or_null(record), indent=2, allow_nan=False) + "\n")
    print(json.dumps(finite_or_null(record["primitives_entry"]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
