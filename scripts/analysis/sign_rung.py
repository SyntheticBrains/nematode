#!/usr/bin/env python
"""The per-connection signs rung: whether the wild type's chemical wiring matters once signs are real.

On the thermal cell at target 35 under PPO, the wild type and the chemical-only null each learn under
three sign models -- random, the per-neuron atlas rule, and the per-connection table -- beside a
frozen floor per wiring and model. Grounded signs are enforced. The wild type's lead over the null,
``gap``, is read per seed on ``auc_success``.

**Three readings**, Holm-adjusted together on two-sided Wilcoxon p values:

* ``lead = gap(per_connection)``: whether the chemical wiring matters under real signs;
* ``grounding = gap(per_connection) - gap(random)``;
* ``receptor = gap(per_connection) - gap(atlas)``: the 342 corrected signs alone.

All three are read against one minimum: 2/3 of 0.0865, the thermal split's committed effect on this
cell, so a pass is a change the size of the cell's largest committed wiring effect. Each sign level
must pass its gates -- both learners beat their floors and do not both saturate -- or the rung is
unreadable.

Every statistic is A.2's (``operating_point_surface``); the states are B.1c's
(``measured_prior_contrast``); the completeness check is B.1b's (``measured_prior_pilot``).

Usage::

    uv run python scripts/analysis/sign_rung.py score --logs <dir> [--logs <dir>] \
        --seeds 2201 ... --out-dir <scratch> --out rung.json --csv per-seed.csv
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

import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_sign_rung_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]

HALF = "ppo"
CELL = ts.CELL
PILOT_SEEDS: tuple[int, ...] = tuple(range(9501, 9505))
PANEL_FIRST_SEED = 2201
RANDOM, ATLAS, PER_CONNECTION = "random", "atlas", "per_connection"
LEVELS: tuple[str, ...] = (RANDOM, ATLAS, PER_CONNECTION)
STEMS: dict[str, dict[str, str]] = {
    level: {
        "wt_learn": gen.stem(level, null=False, frozen=False),
        "wt_frozen": gen.stem(level, null=False, frozen=True),
        "rn_learn": gen.stem(level, null=True, frozen=False),
        "rn_frozen": gen.stem(level, null=True, frozen=True),
    }
    for level in LEVELS
}

PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC
# The thermal split's committed effect on this cell: how far holding the null's gap junctions moved
# the wild type's lead (Logbook 078's split, on the primary metric).
REFERENCE_EFFECT = 0.08647916666666666
MINIMUM = mc.MINIMUM_FRACTION * abs(REFERENCE_EFFECT)
READINGS: tuple[str, ...] = ("lead", "grounding", "receptor")

_LEAD_VERDICTS: dict[str, str] = {
    "move_wt": "wild_type_leads",
    "move_null": "null_leads",
    "below": "lead_below_minimum",
    "no_move": "no_lead_at_minimum",
    "unresolved": "unresolved",
}
_CHANGE_VERDICTS: dict[str, str] = {
    "move_wt": "lead_grows",
    "move_null": "lead_shrinks",
    "below": "change_below_minimum",
    "no_move": "no_change_at_minimum",
    "unresolved": "unresolved",
}
VERDICTS: dict[str, dict[str, str]] = {
    "lead": _LEAD_VERDICTS,
    "grounding": _CHANGE_VERDICTS,
    "receptor": _CHANGE_VERDICTS,
}


class SignRungError(ValueError):
    """The runs on disk are not the panel this module scores."""


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjust a family of p values: step down from the smallest, monotone, capped at 1."""
    ordered = sorted(p_values, key=p_values.__getitem__)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def read_panel(gates: dict[str, Any], readings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Gates first, then the three readings' Holm q values, states and verdicts.

    ``readings`` maps each name in :data:`READINGS` to ``{"mean": float, "test": {...}}``, the test
    being ``wp.paired_seed_wilcoxon_bootstrap``'s, whose p is one-sided and is folded to two-sided.
    """
    readable = {level: mp.level_passes(g) for level, g in gates.items()}
    out: dict[str, Any] = {
        "minimum": MINIMUM,
        "reference_effect": REFERENCE_EFFECT,
        "level_readable": readable,
    }
    if not all(readable.values()):
        out["verdict"] = "unreadable"
        out["why"] = "a sign level fails its floor on a wiring, or both its arms saturate"
        return out
    qs = holm({name: ops.two_sided(readings[name]["test"]["wilcoxon_p"]) for name in READINGS})
    for name in READINGS:
        test = readings[name]["test"]
        state = mc.classify(readings[name]["mean"], test["ci_lo"], test["ci_hi"], qs[name], MINIMUM)
        out[name] = {
            "mean": readings[name]["mean"],
            "ci": [test["ci_lo"], test["ci_hi"]],
            "holm_q": qs[name],
            "state": state,
            "verdict": VERDICTS[name][state],
        }
    return out


def build_manifest(log_dirs: list[Path], path: Path, seeds: tuple[int, ...]) -> Path:
    """Write ``<arm> <level> <seed> <log>`` for every panel run on ``seeds``."""
    seen: dict[tuple[str, str, int], Path] = {}
    lines: list[str] = []
    for arm, level, seed, log in gp.evidence(STEMS, log_dirs):
        if seed not in seeds:
            continue
        prior = seen.setdefault((arm, level, seed), log)
        if prior != log:
            msg = f"{arm} {level} seed {seed} has two runs: {prior} and {log}"
            raise SignRungError(msg)
        lines.append(f"{arm} {level} {seed} {log.resolve()}")
    path.write_text("\n".join(lines) + "\n")
    return path


def score(log_dirs: list[Path], out_dir: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    """Score the three levels, read the gates, then the three Holm-adjusted readings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(log_dirs, out_dir / "manifest-sign-rung.txt", seeds)
    mp.require_complete(manifest, HALF, seeds, LEVELS)
    reports = {
        level: ops.score_level(manifest, HALF, level, out_dir / f"tmp-{level}", cell=CELL)
        for level in LEVELS
    }
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    gaps = {m: {lv: ops.wiring_gap(reports[lv], m) for lv in LEVELS} for m in metrics}
    changes = {
        m: {
            "grounding": ops.interaction(reports[RANDOM], reports[PER_CONNECTION], m),
            "receptor": ops.interaction(reports[ATLAS], reports[PER_CONNECTION], m),
            "atlas_minus_random": ops.interaction(reports[RANDOM], reports[ATLAS], m),
        }
        for m in metrics
    }
    gates = {
        level: ops.learning_gates(manifest, HALF, seeds, level, floor_level=level)
        for level in LEVELS
    }
    primary = gaps[PRIMARY_METRIC][PER_CONNECTION]
    readings = {
        "lead": {"mean": primary["gap_mean"], "test": primary["test"]},
        **{
            name: {
                "mean": changes[PRIMARY_METRIC][name]["interaction_mean"],
                "test": changes[PRIMARY_METRIC][name]["test"],
            }
            for name in ("grounding", "receptor")
        },
    }
    return {
        "cell": CELL,
        "seeds": list(seeds),
        "primary_metric": PRIMARY_METRIC,
        "beside_metric": BESIDE_METRIC,
        "gates": gates,
        "gaps": gaps,
        "changes": changes,
        "reading": read_panel(gates, readings),
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per seed: every arm's plateau and floor, each level's gap, and the two changes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    header = ["seed"]
    for level in LEVELS:
        header += [f"{level}_{c}" for c in ("wt_plateau", "wt_floor", "rn_plateau", "rn_floor")]
        header += [f"{level}_gap_{m}" for m in metrics]
    header += [f"{name}_{m}" for name in ("grounding", "receptor") for m in metrics]
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
            row += [
                _fmt(result["changes"][m][name]["per_seed"].get(seed))
                for name in ("grounding", "receptor")
                for m in metrics
            ]
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
    sc = sub.add_parser("score", help="score the rung")
    sc.add_argument("--logs", type=Path, action="append", required=True)
    sc.add_argument("--seeds", type=int, nargs="+", required=True)
    sc.add_argument("--out-dir", type=Path, required=True)
    sc.add_argument("--csv", type=Path)
    sc.add_argument("--out", type=Path, help="write the JSON here")
    args = ap.parse_args(argv)
    result = score(args.logs, args.out_dir, tuple(args.seeds))
    if args.csv:
        write_csv(result, args.csv)
    payload = json.dumps(result, indent=2, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
