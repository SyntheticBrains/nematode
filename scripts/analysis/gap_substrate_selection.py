#!/usr/bin/env python
"""G.0-A: choose the corrected gap model, ohmic coupling or log-compressed counts, by learnability alone.

G.0 found the gap-placement lead on the thermal cell rides on the model's additive, count-scaled gap
coupling (Logbook 095). The maintainer chose ohmic coupling in the settling map, selected against
log-compressed counts in a gated pilot. This module is that pilot's instrument.

**The probe and the calibration** (no run cost, before any run): on untrained brains, with the
calibration probe's 64 fixed inputs and the pilot's seeds, the saturated-unit fraction and the action
mean's spread across inputs, for the gap-off brain (the reference: a known learner), the additive and log
wild types, and the ohmic wild type over a grid of conductance scales. The ohmic scale is the largest on
the grid whose spread is at least half the gap-off brain's on both cells; if none is, the pilot does not
launch. (The additive count model's spread was the first reference; it is inflated by the very gap
amplification G.0 found to be an artefact, and every known learner falls below a quarter of it.)

**The selection** (seeds 9871-9886, both cells, Emmons 2024):

1. The cells must be readable: the MLP control beats its frozen floor on both cells.
2. A model is eligible on a cell when its wild type and its gap-only null each beat their floors and
   the two do not both plateau at or above 90%.
3. Ohmic is chosen if eligible on both cells, unless on either cell its wild type is shown slower than
   log's by more than the cell's margin: the paired 80% interval of ohmic - log ``auc_success`` lies
   entirely below -delta (0.0239 thermal, 0.0143 hard350).
4. Otherwise log, if eligible on both cells; otherwise neither.

The rule never reads the gap between the wild type and the null. The additive-count wild type is read
beside, as description only.

Usage::

    uv run python scripts/analysis/gap_substrate_selection.py plan
    uv run python scripts/analysis/gap_substrate_selection.py probe --out probe.json
    uv run python scripts/analysis/gap_substrate_selection.py select --logs <dir> --out-dir <scratch> \
        --out selection.json
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_gap_substrate_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]

HALF = "ppo"
SEEDS: tuple[int, ...] = tuple(range(9871, 9887))
SMOKE_SEEDS: tuple[int, ...] = (9891, 9892)
CELLS: tuple[str, ...] = ("thermal", "hard350")
# The instrument's cell labels, as the committed harnesses name the two block-V cells.
INSTRUMENT_CELL: dict[str, str] = {"thermal": ts.CELL, "hard350": ops.CELL}
MODELS: tuple[str, ...] = ("ohmic", "log")
# Logbook 080's margins for the same question: a new substrate's wild type against the settling one's.
MARGIN: dict[str, float] = {"thermal": 0.0239, "hard350": 0.0143}
# The ohmic conductance-scale grid, and the share of the gap-off brain's spread a scale must keep.
SCALES: tuple[float, ...] = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
REFERENCE_FRACTION = 0.5
PRIMARY_METRIC = ops.UNCENSORED_METRIC


def levels(cell: str) -> dict[str, dict[str, str]]:
    """Level -> arm -> config stem, in the gate preflight's shape, for one cell.

    ``<model>``: that model's wild type against its gap-only null (its eligibility). ``speed``: the
    ohmic wild type against the log wild type (the selection's comparison). ``mlp``: the MLP
    control's learner and floor on both sides (the cell's readability). ``additive_<model>``: that
    model's wild type against the additive-count wild type (description).
    """
    a = gen.ARMS[cell]
    out: dict[str, dict[str, str]] = {
        model: {
            "wt_learn": a[f"{model}_wt_learn"],
            "wt_frozen": a[f"{model}_wt_frozen"],
            "rn_learn": a[f"{model}_gaponly_learn"],
            "rn_frozen": a[f"{model}_gaponly_frozen"],
        }
        for model in MODELS
    }
    out["speed"] = {
        "wt_learn": a["ohmic_wt_learn"],
        "wt_frozen": a["ohmic_wt_frozen"],
        "rn_learn": a["log_wt_learn"],
        "rn_frozen": a["log_wt_frozen"],
    }
    out["mlp"] = {
        "wt_learn": a["mlp_learn"],
        "wt_frozen": a["mlp_frozen"],
        "rn_learn": a["mlp_learn"],
        "rn_frozen": a["mlp_frozen"],
    }
    for model in MODELS:
        out[f"additive_{model}"] = {
            "wt_learn": a[f"{model}_wt_learn"],
            "wt_frozen": a[f"{model}_wt_frozen"],
            "rn_learn": a["count_wt_learn"],
            "rn_frozen": a[f"{model}_wt_frozen"],
        }
    return out


def runs() -> dict[str, tuple[int, ...]]:
    """Every config stem the pilot runs, on its seeds: 22 configs x 16 seeds."""
    return {stem: SEEDS for cell in CELLS for stem in gen.ARMS[cell].values()}


class SelectionError(ValueError):
    """The runs on disk are not the pilot this module reads."""


# ── The probe (Decision D) ───────────────────────────────────────────────────────────────────
def _connectome_config(stem: str, seed: int) -> Any:  # noqa: ANN401 - a brain config
    from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrainConfig
    from quantumnematode.utils.config_loader import load_simulation_config

    container = load_simulation_config(str(gen.path_of(stem))).brain
    if container is None or not isinstance(container.config, ConnectomePPOBrainConfig):
        msg = f"{stem} does not carry a connectome brain config"
        raise SelectionError(msg)
    return container.config.model_copy(update={"seed": seed})


def probe(seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Measure the untrained operating points, calibrate the ohmic conductance scale, and say go."""
    import sign_scale_calibration as ssc  # pyright: ignore[reportMissingImports]
    from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain
    from quantumnematode.brain.arch.dtypes import DeviceType

    inputs = ssc.inputs()

    def reading(stem: str, **update: object) -> dict[str, float]:
        values = [
            ssc.measure(_connectome_config(stem, s).model_copy(update=update), inputs)
            for s in seeds
        ]
        return {
            "saturated": float(np.mean([v[0] for v in values])),
            "action_spread": float(np.mean([v[1] for v in values])),
        }

    out: dict[str, Any] = {
        "seeds": [seeds[0], seeds[-1]],
        "grid": list(SCALES),
        "reference_fraction": REFERENCE_FRACTION,
        "cells": {},
    }
    passing: list[set[float]] = []
    for cell in CELLS:
        arms = gen.ARMS[cell]
        cell_out: dict[str, Any] = {
            "gap_off": reading(arms["count_wt_learn"], enable_gap_junctions=False),
            "additive": reading(arms["count_wt_learn"]),
            "log": reading(arms["log_wt_learn"]),
            "ohmic": {
                str(scale): reading(arms["ohmic_wt_learn"], gap_weight_scale=scale)
                for scale in SCALES
            },
        }
        floor = REFERENCE_FRACTION * cell_out["gap_off"]["action_spread"]
        cell_out["floor"] = floor
        passing.append(
            {sc for sc in SCALES if cell_out["ohmic"][str(sc)]["action_spread"] >= floor},
        )
        out["cells"][cell] = cell_out
    common = set.intersection(*passing)
    chosen = max(common) if common else None
    out["chosen_scale"] = chosen
    out["launch"] = chosen is not None
    if chosen is not None:
        topology = ConnectomePPOBrain(
            config=_connectome_config(gen.ARMS["thermal"]["ohmic_wt_learn"], seeds[0]).model_copy(
                update={"gap_weight_scale": chosen},
            ),
            device=DeviceType.CPU,
        ).topology
        names = list(topology.neuron_names)
        self_weight = topology.ohmic_inverse.diagonal().double().numpy()
        out["ohmic_self_weight_at_chosen"] = {
            "median": float(np.median(self_weight)),
            "below_half": float((self_weight < 0.5).mean()),
            **{n: float(self_weight[names.index(n)]) for n in ("AVAL", "AVBL", "AFDL", "ALA")},
        }
    return out


# ── The selection (Decision C) ───────────────────────────────────────────────────────────────
def build_manifest(log_dirs: list[Path], path: Path, cell: str) -> Path:
    """Write ``<arm> <level> <seed> <log>`` for one cell's runs that exited 0."""
    seen: dict[tuple[str, str, int], Path] = {}
    lines: list[str] = []
    for arm, level, seed, log in gp.evidence(levels(cell), log_dirs):
        status = log.with_suffix(".exit")
        if seed not in SEEDS or not status.is_file() or status.read_text().strip() != "0":
            continue
        prior = seen.setdefault((arm, level, seed), log)
        if prior != log:
            msg = f"{arm} {level} seed {seed} has two runs: {prior} and {log}"
            raise SelectionError(msg)
        lines.append(f"{arm} {level} {seed} {log.resolve()}")
    path.write_text("\n".join(lines) + "\n")
    return path


def require_complete(manifest: Path, cell: str) -> None:
    """Refuse a partial pilot: every arm of every level on every seed."""
    have: dict[tuple[str, str], set[int]] = {}
    for raw in manifest.read_text().splitlines():
        if raw.strip():
            arm, level, seed, _ = raw.split()
            have.setdefault((arm, level), set()).add(int(seed))
    missing = [
        f"{cell}/{level}/{arm}: {sorted(set(SEEDS) - have.get((arm, level), set()))}"
        for level in levels(cell)
        for arm in gp.ARMS
        if set(SEEDS) - have.get((arm, level), set())
    ]
    if missing:
        msg = "G.0-A pilot is incomplete — " + "; ".join(missing)
        raise SelectionError(msg)


def shown_slower(test: dict[str, Any], margin: float) -> bool:
    """Whether ohmic's wild type is shown slower than log's by more than the margin."""
    return float(test["ci_hi"]) < -margin


def decide(
    readable: dict[str, bool],
    eligible: dict[str, dict[str, bool]],
    slower: dict[str, bool],
) -> str:
    """Apply the registered rule: ``ohmic``, ``log``, ``neither`` or ``unreadable``."""
    if not all(readable.values()):
        return "unreadable"
    if all(eligible["ohmic"].values()) and not any(slower.values()):
        return "ohmic"
    if all(eligible["log"].values()):
        return "log"
    return "neither"


def select(log_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Score both cells' levels and apply the selection rule; the wiring gap is never read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells: dict[str, Any] = {}
    readable: dict[str, bool] = {}
    eligible: dict[str, dict[str, bool]] = {m: {} for m in MODELS}
    slower: dict[str, bool] = {}
    for cell in CELLS:
        manifest = build_manifest(log_dirs, out_dir / f"manifest-{cell}.txt", cell)
        require_complete(manifest, cell)
        gates = {
            level: ops.learning_gates(manifest, HALF, SEEDS, level, floor_level=level)
            for level in ("mlp", *MODELS)
        }
        readable[cell] = bool(gates["mlp"]["gate_passes"])
        for model in MODELS:
            eligible[model][cell] = mp.level_passes(gates[model])
        compared: dict[str, Any] = {}
        for level in ("speed", "additive_ohmic", "additive_log"):
            report = ops.score_level(
                manifest,
                HALF,
                level,
                out_dir / f"tmp-{cell}-{level}",
                cell=INSTRUMENT_CELL[cell],
            )
            compared[level] = ops.wiring_gap(report, PRIMARY_METRIC)
        slower[cell] = shown_slower(compared["speed"]["test"], MARGIN[cell])
        cells[cell] = {
            "mlp_beats_floor": readable[cell],
            "gates": {level: _gate_summary(g) for level, g in gates.items()},
            "eligible": {m: eligible[m][cell] for m in MODELS},
            "ohmic_minus_log": _gap_summary(compared["speed"]),
            "margin": MARGIN[cell],
            "ohmic_shown_slower": slower[cell],
            "described": {
                "ohmic_minus_additive": _gap_summary(compared["additive_ohmic"]),
                "log_minus_additive": _gap_summary(compared["additive_log"]),
            },
        }
    return {
        "seeds": [SEEDS[0], SEEDS[-1]],
        "primary_metric": PRIMARY_METRIC,
        "cells": cells,
        "choice": decide(readable, eligible, slower),
        "never_read": "the gap between the wild type and the gap-only null, on either model",
    }


def _gate_summary(gates: dict[str, Any]) -> dict[str, Any]:
    return {
        "wt_plateau": gates["wt"]["plateau_success"],
        "wt_floor": gates["wt"]["floor_success"],
        "rn_plateau": gates["rn"]["plateau_success"],
        "rn_floor": gates["rn"]["floor_success"],
        "gate_passes": gates["gate_passes"],
        "saturated": gates["saturated"],
    }


def _gap_summary(gap: dict[str, Any]) -> dict[str, Any]:
    test = gap["test"]
    return {
        "mean": gap["gap_mean"],
        "ci": [test["ci_lo"], test["ci_hi"]],
        "n": gap["n_seeds"],
        "per_seed": gap["per_seed"],
    }


def plan() -> dict[str, Any]:
    """Return the pilot's configs: one campaign over the shared seed band."""
    stems = sorted(runs())
    return {
        "n_runs": len(stems) * len(SEEDS),
        "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "configs": [str(gen.path_of(s).relative_to(gen.PROJECT_ROOT)) for s in stems],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    pl = sub.add_parser("plan", help="the pilot's configs and seeds")
    pr = sub.add_parser("probe", help="the untrained operating point and the stop rule")
    se = sub.add_parser("select", help="apply the selection rule to the pilot's runs")
    se.add_argument("--logs", type=Path, action="append", required=True)
    se.add_argument("--out-dir", type=Path, required=True)
    for parser in (pl, pr, se):
        parser.add_argument("--out", type=Path, help="write the JSON here")
    args = ap.parse_args(argv)
    if args.command == "plan":
        result, ok = plan(), True
    elif args.command == "probe":
        result = probe()
        ok = bool(result["launch"])
    else:
        result = select(args.logs, args.out_dir)
        ok = result["choice"] in MODELS
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
