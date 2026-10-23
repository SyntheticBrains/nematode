#!/usr/bin/env python
"""The real worms' patch-leaving reference, read as simulated worms are read.

From the vendored windows of Scheer & Bargmann 2023's 1,586 wild-type animals
(``data/roaming_dwelling/scheer2023_windows.npz``), each animal's lawn exits are found with the same
rule as simulated worms': the first window of an out-of-lawn run after an in-lawn run.

States are read two ways:

- **The matched reference.** D.1's calibrated classifier on the windows measured at the 5 s step.
  This is the one the registration grades against.
- **Beside it.** The authors' own labels.

For each, the reference reports:

- the roaming share of on-lawn windows 30 to 180 s before each exit (the registered band), and its
  enrichment over the overall on-lawn roaming share;
- the same for the last window before each exit, partly mechanical since leaving means moving;
- the roaming share over the last 10, 30, 60 and 180 s;
- the leaving rate per on-lawn hour.

Usage::

    uv run python scripts/analysis/patch_leaving_reference.py \
        --out data/roaming_dwelling/scheer2023_leaving.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.validation import roaming_dwelling as rd

WINDOWS_PATH = rd.REFERENCE_HMM_PATH.parent / "scheer2023_windows.npz"


def leaving(states: list[np.ndarray], exits: list[list[int]]) -> dict[str, Any]:
    """Pool every animal's exits: band and last-window enrichment, and roaming before exits."""
    on_states = np.concatenate([s[s != rd.OFF_FOOD] for s in states])
    overall = float(np.mean(on_states == rd.ROAMING))
    band = np.concatenate(
        [rd.pre_exit_band(s, e) for s, e in zip(states, exits, strict=True)] or [np.zeros(0)],
    )
    last = np.concatenate(
        [rd.last_window_before_exits(s, e) for s, e in zip(states, exits, strict=True)]
        or [np.zeros(0)],
    )
    before: dict[str, float] = {}
    for seconds in (10, 30, 60, 180):
        span = seconds // int(rd.WINDOW_SECONDS)
        pooled = np.concatenate(
            [rd.pre_exit_band(s, e, near=0, far=span) for s, e in zip(states, exits, strict=True)],
        )
        before[f"{seconds}s"] = (
            float(np.mean(pooled == rd.ROAMING)) if pooled.size else float("nan")
        )
    band_share = float(np.mean(band == rd.ROAMING)) if band.size else float("nan")
    last_share = float(np.mean(last == rd.ROAMING)) if last.size else float("nan")
    return {
        "overall_on_lawn_roaming": overall,
        "band_30_180s_roaming": band_share,
        "band_enrichment": band_share - overall,
        "band_windows": int(band.size),
        "last_window_roaming": last_share,
        "last_window_enrichment": last_share - overall,
        "roaming_before_exit": before,
    }


def reference(windows_path: Path = WINDOWS_PATH) -> dict[str, Any]:
    """Read the real worms' exits and their roaming with the classifier and with the labels."""
    with np.load(windows_path) as f:
        w = {k: f[k] for k in f.files}
    classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm())
    n = w["speed"].shape[0]
    exits = [rd.lawn_exits(w["on_food"][i], w["off_lawn"][i]) for i in range(n)]
    classified = [
        classifier.states(w["speed"][i], w["angular"][i], w["on_food"][i]) for i in range(n)
    ]
    labelled = [np.where(w["on_food"][i], w["labels"][i], rd.OFF_FOOD) for i in range(n)]
    n_exits = sum(len(e) for e in exits)
    on_hours = float(w["on_food"].sum()) * rd.WINDOW_SECONDS / 3600.0
    return {
        "source": "Scheer & Bargmann 2023 wild type (PD1074, OD2 small lawns), 1,586 animals",
        "exit_rule": "the first window of an out-of-lawn run after an in-lawn run",
        "band_seconds": [
            rd.EXIT_NEAR_WINDOWS * rd.WINDOW_SECONDS,
            rd.EXIT_FAR_WINDOWS * rd.WINDOW_SECONDS,
        ],
        "animals": n,
        "animals_leaving": sum(bool(e) for e in exits),
        "exits": n_exits,
        "exits_per_on_lawn_hour": n_exits / on_hours if on_hours else float("nan"),
        "classifier": leaving(classified, exits),
        "authors_labels": leaving(labelled, exits),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: write the reference."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = reference()
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
