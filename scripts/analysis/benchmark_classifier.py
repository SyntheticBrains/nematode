#!/usr/bin/env python
r"""Fit the benchmark's roaming/dwelling classifier on the band cohorts, and gate it.

The model is the two-state Gaussian HMM of ``validation/roaming_dwelling.py``, on each 10-s window's
log speed and angular speed, read from 1-s positions at 5 s. It is fitted against Scheer & Bargmann
2023's own labels, on the OD 2 wild type's recording dates in the benchmark's band cohorts only;
no held-out date is read.

**The split.** The band dates are drawn two thirds to fit and one third to gate, by date
(``default_rng(2029)``).

**Windows.** A window is on the lawn when every 1-s sample in its span lies inside the lawn's
boundary (:func:`quantumnematode.benchmark.grading.window_inputs`). Only windows on the lawn and
labelled roaming or dwelling enter the fit, as runs of consecutive such windows.

**The gate**, fixed before the fit: Cohen's kappa >= 0.6 between the decoded states and the labels
on the gate dates, decoded within the same runs the fit uses: windows on the lawn and labelled. On
failure the script writes nothing
and exits 1; the registered fallback is D.1's model (``data/roaming_dwelling/calibration_retry.json``),
with its overlap with the held-out dates stated.

Usage::

    uv run python scripts/analysis/benchmark_classifier.py --out data/benchmark/classifier.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import references as refs
from quantumnematode.validation import roaming_dwelling as rd

DESCRIPTION = "Fit the benchmark's roaming/dwelling classifier on the band cohorts, and gate it."
WILD_TYPE = "PD1074_od2_Data"
SPLIT_SEED = 2029
FIT_SHARE = 2.0 / 3.0
KAPPA_GATE = 0.6
FALLBACK = "data/roaming_dwelling/calibration_retry.json"


def split_dates(dates: list[int], seed: int = SPLIT_SEED) -> tuple[list[int], list[int]]:
    """Split band dates into fit and gate dates, two thirds and one third, by a seeded draw."""
    order = np.random.default_rng(seed).permutation(sorted(dates))
    k = round(FIT_SHARE * len(order))
    return sorted(int(d) for d in order[:k]), sorted(int(d) for d in order[k:])


def band_rows(tracks: refs.ScheerTracks, arm: str) -> np.ndarray:
    """Return an arm's rows on band dates only."""
    rows = tracks.arm(arm)
    return rows[tracks.side[rows] == refs.BANDS]


def animal_inputs(
    tracks: refs.ScheerTracks,
    i: int,
) -> tuple[g.WindowInputs, np.ndarray]:
    """Return one small-lawn animal's window measures and its labels, aligned to the windows."""
    lawn = g.real_on_lawn(tracks.x[i], tracks.y[i], tracks.boundary_radius[i])
    inputs = g.window_inputs(tracks.x[i], tracks.y[i], lawn)
    return inputs, np.asarray(tracks.labels[i][: len(inputs.speed)], dtype=int)


def labelled_runs(
    inputs: g.WindowInputs,
    labels: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return runs of consecutive windows on the lawn and labelled roaming or dwelling."""
    keep = inputs.on & (labels >= 0)
    features = rd.window_features(inputs.speed, inputs.angular)
    return [(features[a:b], labels[a:b]) for a, b in rd.on_food_runs(keep)]


def calibrate(tracks: refs.ScheerTracks) -> dict[str, Any]:
    """Fit on the fit dates, read the gate on the gate dates; return the record."""
    rows = band_rows(tracks, WILD_TYPE)
    dates = sorted({int(tracks.date[i]) for i in rows})
    fit_dates, gate_dates = split_dates(dates)
    fit_rows = [i for i in rows if int(tracks.date[i]) in set(fit_dates)]
    gate_rows = [i for i in rows if int(tracks.date[i]) in set(gate_dates)]
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in fit_rows:
        runs += labelled_runs(*animal_inputs(tracks, i))
    runs = [r for r in runs if len(r[0])]
    hmm = rd.GaussianHMM.fit_labelled(runs)
    classifier = rd.GaussianClassifier(hmm=hmm)
    predicted, labels = [], []
    for i in gate_rows:
        inputs, lab = animal_inputs(tracks, i)
        # The gate decodes the runs the fit used: windows on the lawn and labelled.
        eligible = inputs.on & (lab >= 0)
        states = classifier.states(inputs.speed, inputs.angular, eligible)
        predicted.append(states)
        labels.append(np.where(eligible, lab, rd.OFF_FOOD))
    gate = rd.agreement(np.concatenate(predicted), np.concatenate(labels))
    return {
        "model": "two-state Gaussian-emission HMM on (log(speed + 0.001), angular speed)",
        "source": "Scheer & Bargmann 2023, OD 2 wild type (PD1074), band cohorts only",
        "window_rule": "on the lawn when every 1-s sample in the window's span lies inside the "
        "lawn's boundary; fit on runs of windows on the lawn and labelled",
        "split": {
            "seed": SPLIT_SEED,
            "unit": "recording date",
            "fit_dates": fit_dates,
            "gate_dates": gate_dates,
            "fit_animals": len(fit_rows),
            "gate_animals": len(gate_rows),
            "fit_windows": int(sum(len(f) for f, _ in runs)),
        },
        "gate": {"kappa_min": KAPPA_GATE, **gate, "passes": bool(gate["kappa"] >= KAPPA_GATE)},
        "fallback": FALLBACK,
        "parameters": {
            "log_pi0": hmm.log_pi0.tolist(),
            "log_transitions": hmm.log_transitions.tolist(),
            "means": hmm.means.tolist(),
            "covariances": hmm.covariances.tolist(),
        },
    }


def main() -> int:
    """CLI: fit, gate, and write the model only if the gate passes."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = calibrate(refs.load_scheer_tracks())
    print(json.dumps({k: record[k] for k in ("split", "gate")}, indent=2))
    if not record["gate"]["passes"]:
        print(
            f"The gate failed (kappa {record['gate']['kappa']:.3f} < {KAPPA_GATE}); nothing was "
            f"written. The registered fallback is {FALLBACK}, with its overlap with the held-out "
            "dates stated.",
            file=sys.stderr,
        )
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
