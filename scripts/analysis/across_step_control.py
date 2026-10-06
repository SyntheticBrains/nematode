#!/usr/bin/env python
"""B.2a's positive controls: whether PPO learns each block-V cell on the leaky substrate.

Two controls, in the roadmap's order. **Control 1**: the cell is solvable by the strongest method.
hard350's is discharged by Logbook 060's committed MLP-PPO runs; thermal at target 35 runs MLP-PPO
on seeds 1201-1208 and passes if every seed's plateau success reaches the 30% competence level.
**Control 2**: on each cell, the dynamical wild type against the settling wild type, paired by seed,
read as non-inferiority on ``auc_success`` against a margin fixed from that cell's committed data.

The time constant is chosen first, on a pilot (seeds 1101-1104), by a rule fixed before the pilot
runs: a tau is eligible where the gate preflight reads both cells ``readable``; tau = 1 step is
chosen unless another eligible tau beats it by more than ``TIE_BAND`` on the worse cell's mean
paired difference. Only the wild type runs, so no wiring gap informs the choice.

Each cell is a level in the gate preflight's ``STEMS`` shape: the dynamical substrate fills the
wild-type slots and the settling substrate the null slots, so the preflight reads both substrates'
floors and the saturation bar. Every statistic is A.2's (``operating_point_surface``).

Usage::

    uv run python scripts/analysis/across_step_control.py --pilot --logs <dir> [--logs <dir>] \
        --out-dir <scratch> --out pilot.json
    uv run python scripts/analysis/across_step_control.py --logs <dir> [--logs <dir>] \
        --out-dir <scratch> --out control.json --csv per-seed.csv
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import boundary_null as bn  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

HALF = "ppo"
PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC

# ── The pilot ────────────────────────────────────────────────────────────────────────────────
TAUS: tuple[float, ...] = (0.2, 1.0, 5.0)
DEFAULT_TAU = 1.0
# About one pilot standard error of the mean paired difference at the proxy spread on four seeds
# (0.062 / 2 on hard350, 0.10 / 2 on thermal): a smaller lead over tau = 1 is not resolved.
TIE_BAND = 0.05
PILOT_SEEDS: tuple[int, ...] = tuple(range(1101, 1105))
# Set at registration from the pilot's selection; ``score`` refuses to run while it is unset.
REGISTERED_TAU: float | None = None

# ── The cells ────────────────────────────────────────────────────────────────────────────────
_HARD = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
_THERMAL = "connectomeppo_small_continuous2d_thermal_klinotaxis"
# Cell -> how the instrument scores it, its band, its margin, and its settling parents. Each margin
# is the minimum the cell's latest panel registered from committed data: 2/3 of A.6's lead over the
# chemical-only null on hard350 (Logbook 079), and Logbook 078's minimum on thermal at target 35.
CELLS: dict[str, dict[str, Any]] = {
    "hard350": {
        "instrument_cell": ops.CELL,
        "seeds": tuple(range(641, 769)),
        "margin": bn.MINIMUM,
        "settling": (_HARD, f"{_HARD}_frozen"),
    },
    "thermal": {
        "instrument_cell": ts.CELL,
        "seeds": tuple(range(513, 641)),
        "margin": ts.MINIMUM,
        "settling": (f"{_THERMAL}_t35", f"{_THERMAL}_frozen_t35"),
    },
}

# ── Control 1 on thermal ─────────────────────────────────────────────────────────────────────
MLP_STEM = f"{_THERMAL}_t35".replace("connectomeppo", "mlpppo")
MLP_SEEDS: tuple[int, ...] = tuple(range(1201, 1209))
# The episode metric's competence level, in percent success.
COMPETENCE = 30.0


class AcrossStepError(ValueError):
    """The runs on disk are not the panel this module scores."""


def tau_tag(tau: float) -> str:
    """Return the stem suffix for a time constant: ``0.2`` -> ``tau0p2``, ``1.0`` -> ``tau1``."""
    return "tau" + f"{tau:g}".replace(".", "p")


def stems(tau: float) -> dict[str, dict[str, str]]:
    """Return ``cell -> arm -> stem`` at one tau, in the gate preflight's shape."""
    out: dict[str, dict[str, str]] = {}
    for cell, spec in CELLS.items():
        learn, frozen = spec["settling"]
        out[cell] = {
            "wt_learn": f"{learn}_leaky_{tau_tag(tau)}",
            "wt_frozen": f"{frozen}_leaky_{tau_tag(tau)}",
            "rn_learn": learn,
            "rn_frozen": frozen,
        }
    return out


STEMS = stems(REGISTERED_TAU if REGISTERED_TAU is not None else DEFAULT_TAU)


# ── Selection: the rule fixed before the pilot ───────────────────────────────────────────────
def select(per_tau: dict[float, dict[str, Any]]) -> dict[str, Any]:
    """Choose tau from each candidate's gate statuses and mean paired differences.

    ``per_tau`` maps each tau to ``{"status": {cell: status}, "difference": {cell: mean}}``.
    """
    eligible = [
        tau
        for tau in TAUS
        if tau in per_tau and all(s == "readable" for s in per_tau[tau]["status"].values())
    ]
    if not eligible:
        return {"tau": None, "eligible": [], "why": "no tau is readable on both cells"}
    score = {tau: min(per_tau[tau]["difference"].values()) for tau in eligible}
    best = max(eligible, key=lambda tau: score[tau])
    if DEFAULT_TAU in eligible and score[best] - score[DEFAULT_TAU] <= TIE_BAND:
        chosen, why = DEFAULT_TAU, "tau = 1 step, no eligible tau beats it by more than the band"
    elif DEFAULT_TAU in eligible:
        chosen, why = best, "beats tau = 1 step by more than the band"
    else:
        chosen, why = best, "tau = 1 step is not eligible; the higher score among the eligible"
    return {"tau": chosen, "eligible": eligible, "score": score, "why": why}


# ── Manifests ────────────────────────────────────────────────────────────────────────────────
def build_manifest(
    table: dict[str, dict[str, str]],
    log_dirs: list[Path],
    seeds_by_cell: dict[str, tuple[int, ...]],
    path: Path,
) -> Path:
    """Write ``<arm> <cell> <seed> <log>`` for every run of the table on each cell's seeds."""
    seen: dict[tuple[str, str, int], Path] = {}
    lines: list[str] = []
    for arm, cell, seed, log in gp.evidence(table, log_dirs):
        if seed not in seeds_by_cell[cell]:
            continue
        prior = seen.setdefault((arm, cell, seed), log)
        if prior != log:
            msg = f"{arm} {cell} seed {seed} has two runs: {prior} and {log}"
            raise AcrossStepError(msg)
        lines.append(f"{arm} {cell} {seed} {log.resolve()}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _difference(manifest: Path, cell: str, tmp: Path, metric: str) -> dict[str, Any]:
    report = ops.score_level(manifest, HALF, cell, tmp, cell=CELLS[cell]["instrument_cell"])
    return ops.wiring_gap(report, metric)


# ── The pilot ────────────────────────────────────────────────────────────────────────────────
def pilot(log_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Every candidate's preflight and mean paired difference, then the selection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = dict.fromkeys(CELLS, PILOT_SEEDS)
    per_tau: dict[float, dict[str, Any]] = {}
    for tau in TAUS:
        table = stems(tau)
        preflight = gp.preflight(table, log_dirs)
        status = {cell: entry["status"] for cell, entry in preflight["levels"].items()}
        manifest = build_manifest(table, log_dirs, seeds, out_dir / f"pilot-{tau_tag(tau)}.txt")
        difference: dict[str, float] = {}
        for cell in CELLS:
            if status[cell] in ("no_evidence", "incomplete_evidence"):
                continue
            gap = _difference(manifest, cell, out_dir / f"tmp-{tau_tag(tau)}", PRIMARY_METRIC)
            difference[cell] = float(gap["gap_mean"])
        per_tau[tau] = {"status": status, "difference": difference, "preflight": preflight}
    return {
        "seeds": list(PILOT_SEEDS),
        "rule": {"default_tau": DEFAULT_TAU, "tie_band": TIE_BAND},
        "candidates": {tau_tag(tau): entry for tau, entry in per_tau.items()},
        "selection": select(per_tau),
    }


# ── Control 1 ────────────────────────────────────────────────────────────────────────────────
def mlp_gate(log_dirs: list[Path]) -> dict[str, Any]:
    """Thermal MLP-PPO: every seed's plateau success at or above the competence level."""
    plateaus: dict[int, float] = {}
    for log_dir in log_dirs:
        for log in sorted(log_dir.glob(f"{MLP_STEM}-seed*.log")):
            seed = int(log.stem.rpartition("-seed")[2])
            tail = wp.plateau_tail(log)
            if seed in MLP_SEEDS and tail is not None:
                plateaus[seed] = float(tail[0])
    complete = set(plateaus) == set(MLP_SEEDS)
    passes = complete and all(p >= COMPETENCE for p in plateaus.values())
    return {
        "seeds": list(MLP_SEEDS),
        "plateau_success": plateaus,
        "complete": complete,
        "verdict": "passes" if passes else ("incomplete" if not complete else "fails"),
    }


# ── Control 2 ────────────────────────────────────────────────────────────────────────────────
def read_cell(gates: dict[str, Any], test: dict[str, Any], margin: float) -> str:
    """Gates first, then non-inferiority of the dynamical substrate against the margin."""
    if not mp.wild_type_learns(gates):
        return "unlearnable"
    if not mp.level_passes(gates):
        return "unreadable"
    if float(test["ci_lo"]) > -margin:
        return "non_inferior"
    if float(test["ci_hi"]) < -margin:
        return "inferior"
    return "unresolved"


def score(log_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Score both cells at the registered tau, and the thermal MLP control."""
    if REGISTERED_TAU is None:
        msg = "REGISTERED_TAU is unset: the pilot chooses it, and the registration records it"
        raise AcrossStepError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = {cell: spec["seeds"] for cell, spec in CELLS.items()}
    manifest = build_manifest(STEMS, log_dirs, seeds, out_dir / "manifest-across-step.txt")
    cells: dict[str, Any] = {}
    for cell, spec in CELLS.items():
        mp.require_complete(manifest, HALF, spec["seeds"], (cell,))
        tmp = out_dir / f"tmp-{cell}"
        report = ops.score_level(manifest, HALF, cell, tmp, cell=spec["instrument_cell"])
        differences = {m: ops.wiring_gap(report, m) for m in (PRIMARY_METRIC, BESIDE_METRIC)}
        gates = ops.learning_gates(manifest, HALF, spec["seeds"], cell, floor_level=cell)
        cells[cell] = {
            "seeds": list(spec["seeds"]),
            "margin": spec["margin"],
            "gates": gates,
            "difference": differences,
            "censoring": ops.censoring_rates(report),
            "verdict": read_cell(gates, differences[PRIMARY_METRIC]["test"], spec["margin"]),
        }
    return {
        "tau": REGISTERED_TAU,
        "primary_metric": PRIMARY_METRIC,
        "beside_metric": BESIDE_METRIC,
        "cells": cells,
        "mlp_thermal": mlp_gate(log_dirs),
        "mlp_hard350": "discharged by Logbook 060's committed MLP-PPO runs",
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per cell and seed: both substrates' plateau and floor, both differences."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    header = ["cell", "seed", "leaky_plateau", "leaky_floor", "settling_plateau", "settling_floor"]
    header += [f"difference_{m}" for m in metrics]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for cell, entry in result["cells"].items():
            for seed in entry["seeds"]:
                row: list[Any] = [cell, seed]
                for wiring in ("wt", "rn"):
                    per_seed = entry["gates"][wiring]["per_seed"].get(seed, {})
                    row += [_fmt(per_seed.get("learn")), _fmt(per_seed.get("floor"))]
                row += [_fmt(entry["difference"][m]["per_seed"].get(seed)) for m in metrics]
                writer.writerow(row)
    return path


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pilot", action="store_true", help="score the tau pilot and select")
    ap.add_argument("--logs", type=Path, action="append", required=True, help="a run-log dir")
    ap.add_argument("--out-dir", type=Path, required=True, help="scratch directory for manifests")
    ap.add_argument("--out", type=Path, help="write the analysis JSON here instead of stdout")
    ap.add_argument("--csv", type=Path, help="write the per-seed CSV here (panel only)")
    args = ap.parse_args(argv)

    result = pilot(args.logs, args.out_dir) if args.pilot else score(args.logs, args.out_dir)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload)
    if args.csv and not args.pilot:
        write_csv(result, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
