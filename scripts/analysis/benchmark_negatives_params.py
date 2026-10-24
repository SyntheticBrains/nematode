#!/usr/bin/env python
r"""Measure the matched negatives' parameters on the benchmark's band cohorts.

The open-loop HMM worm and the Poisson worm are matched to the real wild type's own marginal
behaviour, read from Scheer & Bargmann 2023's OD 2 wild type on band dates only; no held-out date
is read. Per state (the authors' label on each second's 10-s bin, roaming or dwelling) and pooled:

- **speed**: the mean 1-s displacement over run seconds, on the lawn by the geometric rule and
  outside every reversal and omega bout;
- **curvature noise**: the SD of the 1-s heading change (deg/s), over consecutive run seconds whose
  displacements are both at least the readouts' heading floor (0.01 mm). On a lawn a dwelling worm
  moves a few tens of µm a second against a pixel of about 8.6 µm, so slower seconds' headings are
  tracking noise; how many are left out is recorded;
- **heading persistence**: the lag-1 autocorrelation of those heading changes over consecutive
  seconds;
- **reversal and omega hazards**: bout onsets (one-frame gaps merged, bouts of at least two frames,
  the kinematics check's rule) per second outside every bout, since the primitives draw a start
  only while no bout runs.

The record also carries the authors' HMM transitions per 10 s, which the HMM worm steps by.

Usage::

    uv run python scripts/analysis/benchmark_negatives_params.py \
        --out docs/experiments/logbooks/supporting/110-benchmark-controls/negatives_params.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import references as refs
from quantumnematode.benchmark.readouts import FLOOR_MM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import primitive_kinematics_check as kc

DESCRIPTION = "Measure the matched negatives' parameters on the benchmark's band cohorts."
_REPO = Path(__file__).resolve().parents[2]
REFERENCE_HMM_PATH = _REPO / "data" / "roaming_dwelling" / "reference_hmm.json"
WILD_TYPE = "PD1074_od2_Data"
FPS = round(refs.SCHEER_FRAMES_PER_SECOND)
BIN_S = round(refs.SCHEER_BIN_SECONDS)
STATES = {"dwelling": 0, "roaming": 1}


def band_rows(tracks: refs.ScheerTracks) -> np.ndarray:
    """Return the OD 2 wild type's rows on band dates only: the only rows this script reads."""
    rows = tracks.arm(WILD_TYPE)
    return rows[tracks.side[rows] == refs.BANDS]


def _in_bout_seconds(bouts: list[tuple[int, int]], seconds: int) -> np.ndarray:
    out = np.zeros(seconds, dtype=bool)
    for start, stop in bouts:
        out[start // FPS : min(seconds, -(-stop // FPS))] = True
    return out


def animal_sums(tracks: refs.ScheerTracks, i: int, flags: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return one animal's per-state sums: speeds, curvatures and their pairs, onsets, run seconds."""
    x, y = tracks.x[i], tracks.y[i]
    seconds = len(x)
    lawn = g.real_on_lawn(x, y, tracks.boundary_radius[i])
    labels = np.asarray(tracks.labels[i], dtype=int)
    state = np.full(seconds, -1)
    state[: len(labels) * BIN_S] = np.repeat(labels, BIN_S)[:seconds]
    reversal = kc.bouts(flags["reversal"], filtered=True)
    omega = kc.bouts(flags["omega"], filtered=True)
    in_bout = _in_bout_seconds(reversal + omega, seconds)
    dx, dy = np.diff(x), np.diff(y)
    step = np.hypot(dx, dy)
    heading = np.arctan2(dy, dx)
    usable = lawn[:-1] & lawn[1:] & ~in_bout[:-1] & ~in_bout[1:] & np.isfinite(step)
    out: dict[str, Any] = {}
    for name, code in STATES.items():
        run = usable & (state[:-1] == code)
        speeds = step[run]
        above = run & (step >= FLOOR_MM)
        both = above[1:] & above[:-1]
        turn = np.degrees(np.angle(np.exp(1j * (heading[1:] - heading[:-1]))))
        curv = np.where(both, turn, np.nan)
        pairs = np.isfinite(curv[1:]) & np.isfinite(curv[:-1])
        free = lawn & ~in_bout & (state == code) & np.isfinite(x)
        onsets = {
            kind: sum(
                1
                for start, _ in bouts
                if start // FPS < seconds
                and state[start // FPS] == code
                and lawn[start // FPS]
                and (start // FPS == 0 or not in_bout[start // FPS - 1])
            )
            for kind, bouts in (("reversal", reversal), ("omega", omega))
        }
        out[name] = {
            "speeds": speeds,
            "curvatures": curv[np.isfinite(curv)],
            "pairs": (curv[:-1][pairs], curv[1:][pairs]),
            "run_seconds_below_floor": int(np.sum(run & (step < FLOOR_MM))),
            "free_seconds": int(free.sum()),
            "onsets": onsets,
        }
    return out


def summarise(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool animals' sums into one set of parameters."""
    speeds = np.concatenate([p["speeds"] for p in parts])
    curvatures = np.concatenate([p["curvatures"] for p in parts])
    a = np.concatenate([p["pairs"][0] for p in parts])
    b = np.concatenate([p["pairs"][1] for p in parts])
    free = sum(p["free_seconds"] for p in parts)
    onsets = {k: sum(p["onsets"][k] for p in parts) for k in ("reversal", "omega")}
    return {
        "speed_mm_per_s": float(speeds.mean()),
        "curvature_sd_deg_per_s": float(curvatures.std(ddof=1)),
        "heading_persistence": float(np.corrcoef(a, b)[0, 1]),
        "reversal_hazard_per_s": onsets["reversal"] / free,
        "omega_hazard_per_s": onsets["omega"] / free,
        "run_seconds": len(speeds),
        "curvature_seconds": len(curvatures),
        "run_seconds_below_floor": sum(p["run_seconds_below_floor"] for p in parts),
        "seconds_outside_bouts": free,
        "onsets": onsets,
    }


def measure(tracks: refs.ScheerTracks) -> dict[str, Any]:
    """Return per-state and pooled parameters over the band animals."""
    rows = band_rows(tracks)
    names = ("reversal", "omega")
    unpacked = {n: refs.unpack_flag(tracks.frame_flags[rows], tracks.flag_names.index(n)) for n in names}
    per_state: dict[str, list[dict[str, Any]]] = {name: [] for name in STATES}
    for j, i in enumerate(rows):
        sums = animal_sums(tracks, i, {n: unpacked[n][j] for n in names})
        for name in STATES:
            per_state[name].append(sums[name])
    hmm = json.loads(REFERENCE_HMM_PATH.read_text())
    transitions = np.exp(np.asarray(hmm["log_transitions"], dtype=float))
    return {
        "description": DESCRIPTION,
        "source": "Scheer & Bargmann 2023, OD 2 wild type (PD1074), band dates only",
        "animals": len(rows),
        "dates": len({int(tracks.date[i]) for i in rows}),
        "heading_floor_mm": FLOOR_MM,
        "states": {name: summarise(per_state[name]) for name in STATES},
        "pooled": summarise(per_state["dwelling"] + per_state["roaming"]),
        "hmm": {
            "source": "data/roaming_dwelling/reference_hmm.json (the authors' HMM)",
            "states": hmm["states"],
            "transitions_per_10_s": transitions.tolist(),
            "window_seconds": hmm["window_seconds"],
            "stationary_roaming": float(transitions[0, 1] / (transitions[0, 1] + transitions[1, 0])),
        },
        "delta_hazard": "the primitives' minimum: Scheer has no delta flag",
    }


def _finite(value: Any) -> Any:  # noqa: ANN401 - any JSON-shaped value
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> int:
    """CLI: measure and write the parameters."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = _finite(measure(refs.load_scheer_tracks()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: record[k] for k in ("states", "pooled", "hmm")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
