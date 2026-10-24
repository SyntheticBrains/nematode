#!/usr/bin/env python
r"""Check the locomotor primitives' published off-food kinematics against real worms on food.

Reads Scheer & Bargmann 2023's held-out OD 2 wild-type cohorts from the vendored store: the authors'
per-frame reversal, omega and pause flags at 3 frames/s, and the 1-s midbody positions. It never
reads labels, run masks or any roaming statistic.

**Bouts.** A bout is a run of its flag after one-frame gaps are merged, kept if it lasts at least two
frames (0.67 s). The unfiltered runs are reported beside it.

**Quantities**, each in three strata of the distance to the lawn boundary at the bout's start (or
the second's): more than 0.5 mm inside, within 0.5 mm, and more than 0.5 mm outside.

- reversal bout duration;
- reversal speed, path length over the bout's whole seconds;
- omega bout duration;
- forward speed and turning rate, 95th percentiles over seconds outside every bout;
- pause fraction.

Intervals are 95% bootstraps over recording dates, the cohorts.

**The rule**, fixed before the data were read. A quantity is acted on if its source states a mean,
an SD and an n. Its reference span runs from the lowest source mean minus 2 SEM to the highest plus
2 SEM. A disagreement is the inside-lawn stratum's interval lying wholly outside that span. Only
reversal duration qualifies (Zhao et al. 2003: 2.3 ± 1.6 s, n = 249; 3.4 ± 1.8 s, n = 135). Reversal
speed (Zhao, a mean only) and omega duration (Broekmans et al. 2016, escape omegas without an n) are
reported, not acted on. A disagreement is recorded and brought to the maintainer; this script edits
nothing.

Usage::

    uv run python scripts/analysis/primitive_kinematics_check.py \
        --out docs/experiments/logbooks/supporting/105-reference-extraction/kinematics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.benchmark import references as refs

DESCRIPTION = (
    "Check the locomotor primitives' published off-food kinematics against real worms on food."
)

FPS = refs.SCHEER_FRAMES_PER_SECOND
MERGE_GAP_FRAMES = 1
MIN_BOUT_FRAMES = 2
BOUNDARY_BAND_MM = 0.5
STRATA = ("inside", "edge", "outside")
RESAMPLES = 2000
# Percentiles are reported, not acted on, and cost a full sort per resample.
RESAMPLES_PERCENTILE = 200
BOOTSTRAP_SEED = 105
MIN_STEP_MM = 0.01  # a 1-s displacement shorter than this has no heading
WILD_TYPE = "PD1074_od2_Data"

# Sources (mean, SD, n); reported-only quantities have no n.
REVERSAL_DURATION_SOURCES = ((2.3, 1.6, 249), (3.4, 1.8, 135))
REFERENCE = {
    "reversal_duration_s": {
        "acted_on": True,
        "span": (
            min(m - 2 * s / math.sqrt(n) for m, s, n in REVERSAL_DURATION_SOURCES),
            max(m + 2 * s / math.sqrt(n) for m, s, n in REVERSAL_DURATION_SOURCES),
        ),
        "source": "Zhao et al. 2003, Fig 5B: 2.3 +/- 1.6 s (n = 249) and 3.4 +/- 1.8 s (n = 135)",
    },
    "reversal_speed_mm_per_s": {
        "acted_on": False,
        "value": 0.14,
        "source": "Zhao et al. 2003, Fig 5A: mean 140 um/s; no SD or n stated",
    },
    "omega_duration_s": {
        "acted_on": False,
        "value": 7.0,
        "source": "Broekmans et al. 2016: escape omegas 7 +/- 3 s, no n; foraging not reported",
    },
    "forward_speed_p95_mm_per_s": {
        "acted_on": False,
        "value": 0.2,
        "source": "v_max_mm_per_s, a maximum",
    },
    "turning_rate_p95_deg_per_s": {
        "acted_on": False,
        "value": 50.0,
        "source": "omega_max_deg_per_s, a maximum",
    },
    "pause_fraction": {"acted_on": False, "source": "none: a pause is a run at zero speed"},
}


def bouts(flag: np.ndarray, *, filtered: bool) -> list[tuple[int, int]]:
    """Return the ``(start, stop)`` frames of each bout of a per-frame flag."""
    on = np.asarray(flag, dtype=bool).copy()
    if filtered:
        padded = np.concatenate(([True], on, [True]))
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            if start > 0 and stop < len(on) and stop - start <= MERGE_GAP_FRAMES:
                on[start:stop] = True
    edges = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(np.int8), [0]))))
    runs = list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))
    if filtered:
        runs = [(a, b) for a, b in runs if b - a >= MIN_BOUT_FRAMES]
    return runs


def stratum(distance: float) -> str | None:
    """Return the boundary stratum of a signed distance, or None if it is unknown."""
    if not np.isfinite(distance):
        return None
    if distance > BOUNDARY_BAND_MM:
        return "inside"
    if distance < -BOUNDARY_BAND_MM:
        return "outside"
    return "edge"


def _strata_masks(distance: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "inside": distance > BOUNDARY_BAND_MM,
        "edge": np.abs(distance) <= BOUNDARY_BAND_MM,
        "outside": distance < -BOUNDARY_BAND_MM,
    }


def _bout_samples(  # noqa: PLR0913 - one animal's track, its flags and the output
    x: np.ndarray,
    y: np.ndarray,
    distance: np.ndarray,
    flags: dict[str, np.ndarray],
    out: dict[str, dict[str, list[float]]],
    *,
    filtered: bool,
    drop_missing: bool,
) -> np.ndarray:
    """Add one animal's reversal and omega bouts to ``out``; return the frames inside any bout.

    With ``drop_missing``, a bout touching a frame the tracker lost is skipped. Omegas always touch
    one, since the skeleton fails while the worm coils.
    """
    missing = flags["missing"]
    in_bout = np.zeros(len(missing), dtype=bool)
    for name in ("reversal", "omega"):
        for start, stop in bouts(flags[name], filtered=filtered):
            in_bout[start:stop] = True
            where = stratum(distance[min(start // int(FPS), len(x) - 1)])
            if where is None or (drop_missing and missing[start:stop].any()):
                continue
            out[f"{name}_duration_s"][where].append((stop - start) / FPS)
            if name != "reversal":
                continue
            first, last = math.ceil(start / FPS), math.floor((stop - 1) / FPS)
            if last <= first:
                continue
            path = np.hypot(np.diff(x[first : last + 1]), np.diff(y[first : last + 1]))
            if np.all(np.isfinite(path)):
                out["reversal_speed_mm_per_s"][where].append(float(path.sum() / (last - first)))
    return in_bout


def _second_samples(
    x: np.ndarray,
    y: np.ndarray,
    distance: np.ndarray,
    busy_frames: np.ndarray,
    out: dict[str, dict[str, list[float]]],
) -> None:
    """Add one animal's speeds and turns over seconds outside every bout to ``out``."""
    per = int(FPS)
    seconds = len(x) - 1
    busy = busy_frames[: seconds * per].reshape(seconds, per).any(axis=1)
    dx, dy = np.diff(x), np.diff(y)
    step = np.hypot(dx, dy)
    heading = np.degrees(np.arctan2(dy, dx))
    turn = np.abs((np.diff(heading) + 180.0) % 360.0 - 180.0)
    usable = ~busy & np.isfinite(step)
    turnable = usable[1:] & usable[:-1] & (step[1:] > MIN_STEP_MM) & (step[:-1] > MIN_STEP_MM)
    for where, mask in _strata_masks(distance[:seconds]).items():
        out["forward_speed_p95_mm_per_s"][where] += step[usable & mask].tolist()
        out["turning_rate_p95_deg_per_s"][where] += turn[turnable & mask[1:]].tolist()


def animal_samples(  # noqa: PLR0913 - one animal's track, its flags and two switches
    x: np.ndarray,
    y: np.ndarray,
    radii: np.ndarray,
    flags: dict[str, np.ndarray],
    *,
    filtered: bool,
    drop_missing: bool = False,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, tuple[float, int]]]:
    """Return one animal's samples per quantity and stratum, and its pause sums and counts.

    It reads positions, the boundary and the reversal, omega, pause and missing flags only.
    """
    distance = refs.boundary_distance(x, y, radii)
    out: dict[str, dict[str, list[float]]] = {
        q: {w: [] for w in STRATA} for q in REFERENCE if q != "pause_fraction"
    }
    in_bout = _bout_samples(
        x,
        y,
        distance,
        flags,
        out,
        filtered=filtered,
        drop_missing=drop_missing,
    )
    _second_samples(x, y, distance, in_bout | flags["missing"], out)
    second_of_frame = np.minimum(np.arange(len(flags["pause"])) // int(FPS), len(x) - 1)
    pauses = {}
    for where, mask in _strata_masks(distance[second_of_frame]).items():
        kept = mask & ~flags["missing"]
        pauses[where] = (float(flags["pause"][kept].sum()), int(kept.sum()))
    return out, pauses


def _interval(values: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def measure(
    tracks: refs.ScheerTracks,
    *,
    filtered: bool,
    drop_missing: bool = False,
) -> dict[str, Any]:
    """Return every quantity in every stratum, with a 95% bootstrap over recording dates."""
    rows = [i for i in tracks.arm(WILD_TYPE) if tracks.side[i] == refs.HELD_OUT]
    names = ("reversal", "omega", "pause", "missing")
    unpacked = {
        n: refs.unpack_flag(tracks.frame_flags[rows], tracks.flag_names.index(n)) for n in names
    }
    dates = sorted({int(tracks.date[i]) for i in rows})
    index = {d: k for k, d in enumerate(dates)}
    samples = {q: {w: [[] for _ in dates] for w in STRATA} for q in REFERENCE}
    pause_sum = {w: np.zeros(len(dates)) for w in STRATA}
    pause_count = {w: np.zeros(len(dates)) for w in STRATA}
    for j, i in enumerate(rows):
        flags = {n: unpacked[n][j] for n in names}
        out, pauses = animal_samples(
            tracks.x[i],
            tracks.y[i],
            tracks.boundary_radius[i],
            flags,
            filtered=filtered,
            drop_missing=drop_missing,
        )
        k = index[int(tracks.date[i])]
        for quantity, per_stratum in out.items():
            for where, values in per_stratum.items():
                samples[quantity][where][k] += values
        for where, (total, count) in pauses.items():
            pause_sum[where][k] += total
            pause_count[where][k] += count
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(dates), size=(RESAMPLES, len(dates)))
    result: dict[str, Any] = {"animals": len(rows), "cohorts": len(dates)}
    for quantity in REFERENCE:
        result[quantity] = {}
        for where in STRATA:
            if quantity == "pause_fraction":
                total, count = pause_sum[where], pause_count[where]
                value = float(total.sum() / max(count.sum(), 1))
                with np.errstate(invalid="ignore"):
                    boots = total[draws].sum(axis=1) / count[draws].sum(axis=1)
                n = int(count.sum())
            elif quantity.endswith(("_p95_mm_per_s", "_p95_deg_per_s")):
                per_date = [np.asarray(v) for v in samples[quantity][where]]
                pooled = np.concatenate(per_date)
                value = float(np.percentile(pooled, 95)) if len(pooled) else float("nan")
                boots = np.array(
                    [
                        np.percentile(np.concatenate([per_date[d] for d in draw]), 95)
                        if sum(len(per_date[d]) for d in draw)
                        else np.nan
                        for draw in draws[:RESAMPLES_PERCENTILE]
                    ],
                )
                n = len(pooled)
            else:
                sums = np.array([float(np.sum(v)) for v in samples[quantity][where]])
                counts = np.array([len(v) for v in samples[quantity][where]], dtype=float)
                value = float(sums.sum() / counts.sum()) if counts.sum() else float("nan")
                with np.errstate(invalid="ignore"):
                    boots = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
                n = int(counts.sum())
            result[quantity][where] = {"n": n, "value": value, "interval_95": _interval(boots)}
    return result


def verdicts(filtered: dict[str, Any]) -> dict[str, Any]:
    """Apply the fixed rule to the filtered inside-lawn values."""
    out = {}
    for quantity, ref in REFERENCE.items():
        inside = filtered[quantity]["inside"]
        entry: dict[str, Any] = {"acted_on": ref["acted_on"], "source": ref["source"]}
        if ref["acted_on"]:
            low, high = ref["span"]
            lo, hi = inside["interval_95"]
            entry["span"] = [low, high]
            entry["disagreement"] = bool(hi < low or lo > high)
        elif "value" in ref:
            entry["published"] = ref["value"]
        entry["inside"] = inside
        out[quantity] = entry
    return out


def main() -> int:
    """CLI: run the check on the held-out cohorts and write its record."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tracks = refs.load_scheer_tracks()
    filtered = measure(tracks, filtered=True)
    unfiltered = measure(tracks, filtered=False)
    dropped = measure(tracks, filtered=True, drop_missing=True)
    record = {
        "description": DESCRIPTION,
        "cohorts": "held-out OD 2 wild-type dates (data/benchmark/scheer2023_split.json)",
        "bout_rule": {"merge_gap_frames": MERGE_GAP_FRAMES, "min_bout_frames": MIN_BOUT_FRAMES},
        "strata_mm": BOUNDARY_BAND_MM,
        "bootstrap": {"unit": "recording date", "resamples": RESAMPLES, "seed": BOOTSTRAP_SEED},
        "verdicts": verdicts(filtered),
        "filtered": filtered,
        "unfiltered": unfiltered,
        "missing_frames": (
            "Bouts are kept whether or not they touch a frame the tracker lost, as the registered "
            "rule states. A first run of this script skipped such bouts; it is reported beside, as "
            "description, under bouts_touching_missing_frames_dropped. It removed every omega, "
            "since the skeleton fails while the worm coils."
        ),
        "bouts_touching_missing_frames_dropped": {
            "verdicts": verdicts(dropped),
            "filtered": dropped,
        },
    }
    record["disagreements"] = [q for q, v in record["verdicts"].items() if v.get("disagreement")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["verdicts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
