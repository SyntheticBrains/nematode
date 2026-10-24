#!/usr/bin/env python
r"""López-Cruz 2019's local-search reference: the wild type's decay, fitted on a registered window.

The deposited bins are recording time, which began 4 minutes after removal from food, so a bin
labelled ``[a, a + 2)`` min covers ``[a + 4, a + 6)`` min after removal. The fit starts at the deposited
wild type's peak bin (recording 4-6 min, 8-10 min after removal) and runs to recording 38-40 min
(42-44 min after removal), the last bin a 45-min simulated assay timed from removal also covers:
``alpha * exp(-gamma * (t - t0)) + beta``, t in minutes after removal, t0 the first bin's centre.

Its interval is a parametric bootstrap over the bins, each drawn from a normal at its mean and SEM.
Fits starting one bin earlier and one bin later, and pdfr-1 against its same-dates wild type, are
reported beside it.

Usage::

    uv run python scripts/analysis/local_search_reference.py \
        --out data/benchmark/lopezcruz2019_reference.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.benchmark import readouts as ro
from quantumnematode.benchmark import references as refs

DESCRIPTION = "López-Cruz 2019's local-search reference fit."
OFFSET_MIN = 4.0  # recording began 4 min after removal
BIN_MIN = 2.0
START_RECORDING_MIN = 4.0  # the deposited wild type's peak bin
END_RECORDING_MIN = 40.0  # the last bin ends here (42-44 min after removal)
RESAMPLES = 2000
SEED = 107
WILD_TYPE_FIGURE = "1C"


def _bins(data: dict[str, Any]) -> np.ndarray:
    """Each deposited bin's start in recording minutes."""
    return np.array([float(label.split("-")[0]) for label in data["bins"]])


def fit_row(row: dict[str, Any], starts: np.ndarray, start_min: float) -> dict[str, Any]:
    """Fit one row over [start_min, END_RECORDING_MIN) in recording time, with a bootstrap."""
    keep = (starts >= start_min) & (starts + BIN_MIN <= END_RECORDING_MIN)
    mean = np.array([np.nan if m is None else m for m in row["mean"]], dtype=float)[keep]
    sem = np.array([np.nan if v is None else v for v in row["sem"]], dtype=float)[keep]
    centres = starts[keep] + BIN_MIN / 2 + OFFSET_MIN
    t0 = float(centres[0])
    fit = ro.fit_local_search(centres, mean, t0)
    rng = np.random.default_rng(SEED)
    gammas = [
        ro.fit_local_search(centres, rng.normal(mean, np.nan_to_num(sem)), t0).gamma
        for _ in range(RESAMPLES)
    ]
    return {
        "genotype": row["genotype"],
        "row": row["row"],
        "n": row["n"],
        "window_after_removal_min": [
            float(centres[0] - BIN_MIN / 2),
            float(centres[-1] + BIN_MIN / 2),
        ],
        "bins": int(keep.sum()),
        "t0_min": t0,
        "alpha": fit.alpha,
        "gamma_per_min": fit.gamma,
        "beta": fit.beta,
        "gamma_interval_95": [
            float(np.percentile(gammas, 2.5)),
            float(np.percentile(gammas, 97.5)),
        ],
    }


def build() -> dict[str, Any]:
    """Return the reference record."""
    data = refs.load_lopezcruz()
    starts = _bins(data)
    rows = data["rows"]
    wild = next(r for r in rows if r["genotype"] == "wt" and r["figure"] == WILD_TYPE_FIGURE)
    peak = float(starts[int(np.nanargmax([np.nan if m is None else m for m in wild["mean"]]))])
    if peak != START_RECORDING_MIN:
        msg = f"the wild type's peak bin starts at {peak} min, not the registered {START_RECORDING_MIN}"
        raise ValueError(msg)
    pdfr = next(r for r in rows if r["genotype"].startswith("pdfr-1"))
    control = next(r for r in rows if r["row"] == pdfr["control_row"])
    return {
        "description": (
            "The local-search reference: López-Cruz 2019's Fig. 1C wild type, reorientations per 2-min "
            "bin, fitted as alpha * exp(-gamma * (t - t0)) + beta with t in minutes after removal. "
            "Deposited bins are recording time, which began 4 min after removal."
        ),
        "time_zero": "food removal; recording bin [a, a + 2) min = [a + 4, a + 6) min after removal",
        "fit_window_recording_min": [START_RECORDING_MIN, END_RECORDING_MIN],
        "bootstrap": {
            "kind": "parametric, normal at each bin's mean and SEM",
            "resamples": RESAMPLES,
            "seed": SEED,
        },
        "reference": fit_row(wild, starts, START_RECORDING_MIN),
        "described": {
            "start_one_bin_earlier": fit_row(wild, starts, START_RECORDING_MIN - BIN_MIN),
            "start_one_bin_later": fit_row(wild, starts, START_RECORDING_MIN + BIN_MIN),
            "pdfr-1": fit_row(pdfr, starts, START_RECORDING_MIN),
            "pdfr-1_control": fit_row(control, starts, START_RECORDING_MIN),
        },
    }


def main() -> int:
    """CLI: write the reference record."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = build()
    args.out.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    summary = {"reference": record["reference"]} | {
        k: v["gamma_per_min"] for k, v in record["described"].items()
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
