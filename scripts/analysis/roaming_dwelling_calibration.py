#!/usr/bin/env python
r"""Calibrate the roaming/dwelling line at the point worm's 5-second step against real worms.

Two stages, each a subcommand:

``derive``
    Read Scheer & Bargmann 2023's wild-type pickle (Dryad 10.5061/dryad.47d7wm3jf, CC0) with an
    unpickler that refuses every class outside numpy, pandas, ``slice`` and ``datetime.date``,
    replacing the ``ssm`` model classes by empty stand-ins so none of their code runs. For each
    animal, the midbody position is sampled every 15 frames (5 s at 3 frames/s), converted to mm
    with the animal's own scale, and measured as simulated tracks are
    (:func:`quantumnematode.validation.roaming_dwelling.window_measures`). Each 10-second window
    keeps its speed, angular speed, whether it lies in an in-lawn run, and the authors' label
    (``RD_states_Matrix_exog``: roaming, dwelling, or masked off-lawn). The windows line up with the
    authors' 10-second bins. The result is written as a compressed ``.npz``.

``retry``
    The one registered retry after ``calibrate`` failed its gate: a two-state Gaussian-emission
    model on each window's log speed and angular speed, fitted with the labels on the same
    calibration half, decoded per on-lawn run, read against the same gate on the held-out half.

``calibrate``
    Split the animals in half with a fixed seed. On the first half, choose the slope whose decoded
    states best agree with the authors' labels (Cohen's kappa), with the vendored model. On the
    held-out half, report that agreement, the gate (kappa >= 0.6, fixed before calibration), each
    state's 5-second turning, and the on-lawn roaming fraction beside the authors'.

Usage::

    uv run python scripts/analysis/roaming_dwelling_calibration.py derive \\
        --pickle <PD1074_od2_Fig1_021523.pkl> --out data/roaming_dwelling/scheer2023_windows.npz
    uv run python scripts/analysis/roaming_dwelling_calibration.py calibrate \\
        --windows data/roaming_dwelling/scheer2023_windows.npz \\
        --out data/roaming_dwelling/calibration.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.validation import roaming_dwelling as rd

FRAMES_PER_STEP = 15  # 5 s at 3 frames/s
KAPPA_GATE = 0.6
SPLIT_SEED = 2026
SLOPES = np.round(np.geomspace(5.0, 5000.0, 121), 3)
_ALLOWED_PREFIXES = ("numpy", "pandas")
_ALLOWED = {("builtins", "slice"), ("datetime", "date")}


class _Stub:
    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class _SafeUnpickler(pickle.Unpickler):
    """Allow numpy and pandas, ``slice`` and ``datetime.date``; stand in for ``ssm``; refuse all else."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401 - a class of any type
        if module.startswith("ssm."):
            return type(name, (_Stub,), {})
        if module.split(".", maxsplit=1)[0] in _ALLOWED_PREFIXES or (module, name) in _ALLOWED:
            return super().find_class(module, name)
        msg = f"refused {module}.{name}"
        raise pickle.UnpicklingError(msg)


def derive(pickle_path: Path, out: Path) -> dict[str, Any]:
    """Measure every wild-type animal's windows as simulated tracks are measured; write them."""
    with pickle_path.open("rb") as f:
        _bins, bin_size, data = _SafeUnpickler(f).load()
    frames_per_window = FRAMES_PER_STEP * rd.WINDOW_STEPS
    if bin_size != frames_per_window:
        msg = f"the deposit's bins are {bin_size} frames; the windows need {frames_per_window}"
        raise ValueError(msg)
    x_px, y_px = np.asarray(data["Midbody_cent_x"]), np.asarray(data["Midbody_cent_y"])
    scale = np.asarray(data["pixpermm"], dtype=float).ravel()
    in_run = np.asarray(data["InLawnRunMask"], dtype=bool)
    labels_ma = data["RD_states_Matrix_exog"]
    raw_labels = np.where(np.ma.getdata(labels_ma), rd.ROAMING, rd.DWELLING)
    labels = np.where(np.ma.getmaskarray(labels_ma), rd.OFF_FOOD, raw_labels).astype(np.int8)
    n_animals, n_bins = in_run.shape
    speed = np.full((n_animals, n_bins), np.nan, dtype=np.float32)
    angular = np.full((n_animals, n_bins), np.nan, dtype=np.float32)
    for i in range(n_animals):
        x = x_px[i, ::FRAMES_PER_STEP] / scale[i]
        y = y_px[i, ::FRAMES_PER_STEP] / scale[i]
        s, a = rd.window_measures(x, y)
        speed[i, : len(s)], angular[i, : len(a)] = s, a
    on_food = in_run & np.isfinite(speed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, speed=speed, angular=angular, on_food=on_food, labels=labels)
    return {
        "animals": int(n_animals),
        "windows": int(n_bins),
        "on_food_windows": int(on_food.sum()),
    }


Track = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _tracks(windows: dict[str, np.ndarray], animals: np.ndarray) -> list[Track]:
    return [
        (
            windows["speed"][i],
            windows["angular"][i],
            windows["on_food"][i],
            np.where(windows["on_food"][i], windows["labels"][i], rd.OFF_FOOD),
        )
        for i in animals
    ]


def _turning_by_state(tracks: list[Track]) -> dict[str, dict[str, float]]:
    """Each labelled state's angular speed (degrees/s at the 5 s step), as percentiles."""
    angular = np.concatenate([t[1] for t in tracks])
    labels = np.concatenate([t[3] for t in tracks])
    out: dict[str, dict[str, float]] = {}
    for name, state in (("roaming", rd.ROAMING), ("dwelling", rd.DWELLING)):
        values = angular[(labels == state) & np.isfinite(angular)]
        p50, p90, p99 = np.percentile(values, [50, 90, 99])
        out[name] = {
            "median": float(p50),
            "p90": float(p90),
            "p99": float(p99),
            "n": int(values.size),
        }
    return out


def calibrate(windows_path: Path) -> dict[str, Any]:
    """Fit the slope on half the animals; report agreement and the gate on the other half."""
    with np.load(windows_path) as f:
        windows = {k: f[k] for k in f.files}
    n = windows["speed"].shape[0]
    order = np.random.default_rng(SPLIT_SEED).permutation(n)
    fit_animals, held_animals = np.sort(order[: n // 2]), np.sort(order[n // 2 :])
    hmm = rd.load_reference_hmm()
    fit_tracks, held_tracks = _tracks(windows, fit_animals), _tracks(windows, held_animals)
    slope, fit_kappa = rd.calibrate_slope(fit_tracks, hmm, SLOPES)
    classifier = rd.Classifier(slope=slope, hmm=hmm)
    predicted = np.concatenate([classifier.states(s, a, on) for s, a, on, _ in held_tracks])
    labels = np.concatenate([lab for *_, lab in held_tracks])
    held = rd.agreement(predicted, labels)
    classified = predicted != rd.OFF_FOOD
    authors = labels != rd.OFF_FOOD
    max_turn_deg_per_s = float(np.degrees(np.pi) / rd.STEP_SECONDS)
    turning = _turning_by_state(held_tracks)
    return {
        "source": "Scheer & Bargmann 2023 wild type (PD1074, OD2 small lawns), 1,586 animals",
        "split_seed": SPLIT_SEED,
        "fit_animals": len(fit_animals),
        "held_out_animals": len(held_animals),
        "slope": slope,
        "fit_kappa": fit_kappa,
        "held_out": held,
        "kappa_gate": KAPPA_GATE,
        "gate_passes": bool(held["kappa"] >= KAPPA_GATE),
        "roaming_fraction_on_lawn": {
            "classifier": float(np.mean(predicted[classified] == rd.ROAMING)),
            "authors": float(np.mean(labels[authors] == rd.ROAMING)),
        },
        "turning_deg_per_s": turning,
        "point_worm_max_turn_deg_per_s": max_turn_deg_per_s,
        "dwelling_p99_within_point_worm": bool(turning["dwelling"]["p99"] <= max_turn_deg_per_s),
    }


def _labelled_runs(tracks: list[Track]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split each track's on-food windows into runs whose every window carries a label."""
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for speed, angular, on, labels in tracks:
        usable = on & (labels != rd.OFF_FOOD)
        features = rd.window_features(speed, angular)
        runs.extend((features[a:b], labels[a:b].astype(int)) for a, b in rd.on_food_runs(usable))
    return runs


def retry(windows_path: Path) -> dict[str, Any]:
    """Fit the Gaussian-emission model with the labels on the same calibration half; read the gate."""
    with np.load(windows_path) as f:
        windows = {k: f[k] for k in f.files}
    n = windows["speed"].shape[0]
    order = np.random.default_rng(SPLIT_SEED).permutation(n)
    fit_animals, held_animals = np.sort(order[: n // 2]), np.sort(order[n // 2 :])
    hmm = rd.GaussianHMM.fit_labelled(_labelled_runs(_tracks(windows, fit_animals)))
    classifier = rd.GaussianClassifier(hmm=hmm)
    held_tracks = _tracks(windows, held_animals)
    predicted = np.concatenate([classifier.states(s, a, on) for s, a, on, _ in held_tracks])
    labels = np.concatenate([lab for *_, lab in held_tracks])
    held = rd.agreement(predicted, labels)
    classified, authors = predicted != rd.OFF_FOOD, labels != rd.OFF_FOOD
    return {
        "model": "two-state Gaussian-emission HMM on (log(speed + 0.001), angular speed)",
        "split_seed": SPLIT_SEED,
        "fit_animals": len(fit_animals),
        "held_out_animals": len(held_animals),
        "held_out": held,
        "kappa_gate": KAPPA_GATE,
        "gate_passes": bool(held["kappa"] >= KAPPA_GATE),
        "roaming_fraction_on_lawn": {
            "classifier": float(np.mean(predicted[classified] == rd.ROAMING)),
            "authors": float(np.mean(labels[authors] == rd.ROAMING)),
        },
        "parameters": {
            "log_pi0": hmm.log_pi0.tolist(),
            "log_transitions": hmm.log_transitions.tolist(),
            "means": hmm.means.tolist(),
            "covariances": hmm.covariances.tolist(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: derive the windows, or calibrate on them."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="stage", required=True)
    d = sub.add_parser("derive")
    d.add_argument("--pickle", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    for name in ("calibrate", "retry"):
        c = sub.add_parser(name)
        c.add_argument("--windows", type=Path, required=True)
        c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.stage == "derive":
        result = derive(args.pickle, args.out)
    else:
        result = calibrate(args.windows) if args.stage == "calibrate" else retry(args.windows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
