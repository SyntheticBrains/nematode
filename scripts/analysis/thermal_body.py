#!/usr/bin/env python
"""M.9: whether the wild type's gap-junction placement advantage survives embodiment.

The thermal cell (Cook 2019, 500 steps) through the kinematic body under PPO. The wild type is read
against the gap-only null, which differs from it in gap-junction placement alone, and the
chemical-only null is read beside, each wiring with a frozen floor.

**Stage 1, the positive control.** MLP-PPO and its frozen floor at targets 35 and 20, gated on C.1d's
rule (``body_control.control_gate``): the plateau beats the floor with the 80% interval above zero,
and every seed reaches 30%. The target is 35 if it passes, else 20 if it passes, else none and M.9
stops.

**The registered reading.** ``gap = wild type - gap-only null`` on ``auc_success``, paired by seed and
classified at 0.058, 2/3 of the thermal split's committed effect on this cell. One reading, so its q
is the two-sided Wilcoxon's. The chemical-only null's lead is reported beside, with the floors' gap,
``episodes_to_30pct_success``, and C.3's kinematics for the wild type and the gap-only null.

Every statistic is A.2's (``operating_point_surface``); the states are B.1c's
(``measured_prior_contrast``); the completeness check is B.1b's (``measured_prior_pilot``).

Usage::

    uv run python scripts/analysis/thermal_body.py mlp-gate --logs <dir> --out stage1.json
    uv run python scripts/analysis/thermal_body.py score --logs <dir> --seeds 2301 ... \
        --out-dir <scratch> --out panel.json --csv per-seed.csv
    uv run python scripts/analysis/thermal_body.py kinematics --logs <dir> --seeds 2301 ... \
        --out kinematics.json
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

import body_control as bc  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_thermal_body_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]

HALF = "ppo"
CELL = ts.CELL
STAGE1_SEEDS: tuple[int, ...] = tuple(range(9701, 9709))
PILOT_SEEDS: tuple[int, ...] = tuple(range(9711, 9715))
# The positive control's retry at a doubled budget: target 20 only, on seeds disjoint from stage 1's.
RETRY_SEEDS: tuple[int, ...] = tuple(range(9721, 9729))
RETRY_TARGETS: tuple[int, ...] = (20,)
PANEL_FIRST_SEED = 2301
# The target the panel runs at, set by stage 1's rule before stage 2 runs.
TARGET = 35
GAP_ONLY, CHEMICAL_ONLY = "gap_only", "chemical_only"
LEVELS: tuple[str, ...] = (GAP_ONLY, CHEMICAL_ONLY)
_NULL_TAG = {GAP_ONLY: "gaponly", CHEMICAL_ONLY: "chemnull"}

PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC
# The thermal split's committed effect on this cell: how far holding the null's gap junctions moved
# the wild type's lead (on the primary metric). M.8 read against the same minimum.
REFERENCE_EFFECT = 0.08647916666666666
MINIMUM = mc.MINIMUM_FRACTION * abs(REFERENCE_EFFECT)

VERDICTS: dict[str, str] = {
    "move_wt": "gap_advantage_survives",
    "move_null": "null_leads",
    "below": "lead_below_minimum",
    "no_move": "no_lead_at_minimum",
    "unresolved": "unresolved",
}


def stems(target: int = TARGET) -> dict[str, dict[str, str]]:
    """Return ``level -> arm -> stem`` at ``target``: one level per null, the wild type shared."""
    return {
        level: {
            "wt_learn": gen.stem("wt", "ppo", target),
            "wt_frozen": gen.stem("wt", "frozen", target),
            "rn_learn": gen.stem(_NULL_TAG[level], "ppo", target),
            "rn_frozen": gen.stem(_NULL_TAG[level], "frozen", target),
        }
        for level in LEVELS
    }


STEMS = stems()


class ThermalBodyError(ValueError):
    """The runs on disk are not the panel this module scores."""


# ── Stage 1 ──────────────────────────────────────────────────────────────────────────────────
def mlp_gate(
    log_dirs: list[Path],
    seeds: tuple[int, ...] = STAGE1_SEEDS,
    targets: tuple[int, ...] = gen.TARGETS,
) -> dict[str, Any]:
    """Gate the MLP at each of ``targets`` on C.1d's rule, then choose the target by the fixed rule."""
    gates = {
        target: bc.control_gate(
            bc.plateaus(log_dirs, gen.stem("mlp", "ppo", target), seeds),
            bc.plateaus(log_dirs, gen.stem("mlp", "frozen", target), seeds),
            seeds,
        )
        for target in targets
    }
    return {"gates": gates, "target": choose_target({t: g["verdict"] for t, g in gates.items()})}


def choose_target(verdicts: dict[int, str]) -> int | None:
    """Return 35 if its gate passes, else 20 if its gate passes, else None: M.9 stops."""
    for target in gen.TARGETS:
        if verdicts.get(target) == "passes":
            return target
    return None


# ── The panel ────────────────────────────────────────────────────────────────────────────────
def read_panel(gates: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
    """Gates first, then the one registered reading's state and verdict.

    ``primary`` is ``ops.wiring_gap`` on the gap-only level: ``gap_mean`` and a test whose p is
    one-sided and is folded to two-sided. A gap-only level that fails its gates leaves the rung
    unreadable; the chemical-only level only gates its beside reading.
    """
    readable = {level: mp.level_passes(g) for level, g in gates.items()}
    out: dict[str, Any] = {
        "minimum": MINIMUM,
        "reference_effect": REFERENCE_EFFECT,
        "level_readable": readable,
    }
    if not readable[GAP_ONLY]:
        out["verdict"] = "unreadable"
        out["why"] = "the gap-only level fails its floor on a wiring, or both its arms saturate"
        return out
    test = primary["test"]
    q = ops.two_sided(test["wilcoxon_p"])
    state = mc.classify(primary["gap_mean"], test["ci_lo"], test["ci_hi"], q, MINIMUM)
    return out | {
        "mean": primary["gap_mean"],
        "ci": [test["ci_lo"], test["ci_hi"]],
        "q": q,
        "state": state,
        "verdict": VERDICTS[state],
    }


def build_manifest(
    log_dirs: list[Path],
    path: Path,
    seeds: tuple[int, ...],
    target: int,
) -> Path:
    """Write ``<arm> <level> <seed> <log>`` for every panel run on ``seeds`` at ``target``."""
    seen: dict[tuple[str, str, int], Path] = {}
    lines: list[str] = []
    for arm, level, seed, log in gp.evidence(stems(target), log_dirs):
        if seed not in seeds:
            continue
        prior = seen.setdefault((arm, level, seed), log)
        if prior != log:
            msg = f"{arm} {level} seed {seed} has two runs: {prior} and {log}"
            raise ThermalBodyError(msg)
        lines.append(f"{arm} {level} {seed} {log.resolve()}")
    path.write_text("\n".join(lines) + "\n")
    return path


def score(
    log_dirs: list[Path],
    out_dir: Path,
    seeds: tuple[int, ...],
    target: int = TARGET,
) -> dict[str, Any]:
    """Score both levels, read the gates, then the registered reading and the beside ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(log_dirs, out_dir / "manifest-thermal-body.txt", seeds, target)
    mp.require_complete(manifest, HALF, seeds, LEVELS)
    reports = {
        level: ops.score_level(manifest, HALF, level, out_dir / f"tmp-{level}", cell=CELL)
        for level in LEVELS
    }
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    gaps = {m: {lv: ops.wiring_gap(reports[lv], m) for lv in LEVELS} for m in metrics}
    gates = {
        level: ops.learning_gates(manifest, HALF, seeds, level, floor_level=level)
        for level in LEVELS
    }
    return {
        "cell": CELL,
        "target": target,
        "seeds": list(seeds),
        "primary_metric": PRIMARY_METRIC,
        "beside_metric": BESIDE_METRIC,
        "gates": gates,
        "gaps": gaps,
        "reading": read_panel(gates, gaps[PRIMARY_METRIC][GAP_ONLY]),
    }


def kinematics(
    log_dirs: list[Path],
    seeds: tuple[int, ...],
    target: int = TARGET,
) -> dict[str, Any]:
    """Read C.3's instruments for the wild type and the gap-only null, trained and frozen.

    A trained arm reads its run's final weights; a frozen arm reads its seed's untrained policy. If
    the null crawls differently even untrained, a lead is at least partly locomotion.
    """
    out: dict[str, Any] = {}
    for wiring in ("wt", "gaponly"):
        for learner in ("ppo", "frozen"):
            name = gen.stem(wiring, learner, target)
            config = gen.THERMAL / f"{name}.yml"
            readings = []
            for seed in seeds:
                weights = None
                if learner == "ppo":
                    candidates = (d / f"{name}-seed{seed}.log" for d in log_dirs)
                    log = next((c for c in candidates if c.is_file()), None)
                    weights = bc.final_weights(log) if log is not None else None
                    if weights is None:
                        continue
                readings.append(bc._evaluate(config, seed, weights, None))
            out[f"{wiring}_{learner}"] = {"n_seeds": len(readings), **bc.pooled(readings)}
    return out


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per seed: every arm's plateau and floor, and each null's gap on both metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    header = ["seed"]
    for level in LEVELS:
        header += [f"{level}_{c}" for c in ("wt_plateau", "wt_floor", "rn_plateau", "rn_floor")]
        header += [f"{level}_gap_{m}" for m in metrics]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for seed in result["seeds"]:
            row: list[Any] = [seed]
            for level in LEVELS:
                for wiring in ("wt", "rn"):
                    per_seed = result["gates"][level][wiring]["per_seed"].get(seed, {})
                    row += [_fmt(per_seed.get("learn")), _fmt(per_seed.get("floor"))]
                row += [_fmt(result["gaps"][m][level]["per_seed"].get(seed)) for m in metrics]
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
    sub = ap.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("mlp-gate", help="stage 1: the MLP's gate at each target, and the choice")
    sc = sub.add_parser("score", help="score the panel")
    kin = sub.add_parser("kinematics", help="C.3's instruments, wild type and gap-only null")
    for parser in (gate, sc, kin):
        parser.add_argument("--logs", type=Path, action="append", required=True)
        parser.add_argument("--out", type=Path, help="write the JSON here")
    for parser in (sc, kin):
        parser.add_argument("--seeds", type=int, nargs="+", required=True)
        parser.add_argument("--target", type=int, choices=gen.TARGETS, default=TARGET)
    gate.add_argument("--seeds", type=int, nargs="+", default=list(STAGE1_SEEDS))
    gate.add_argument(
        "--targets",
        type=int,
        nargs="+",
        choices=gen.TARGETS,
        default=list(gen.TARGETS),
    )
    sc.add_argument("--out-dir", type=Path, required=True)
    sc.add_argument("--csv", type=Path)
    args = ap.parse_args(argv)
    if args.command == "mlp-gate":
        result = mlp_gate(args.logs, tuple(args.seeds), tuple(args.targets))
    elif args.command == "score":
        result = score(args.logs, args.out_dir, tuple(args.seeds), args.target)
        if args.csv:
            write_csv(result, args.csv)
    else:
        result = kinematics(args.logs, tuple(args.seeds), args.target)
    payload = json.dumps(result, indent=2, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
