#!/usr/bin/env python
"""M.8: whether the wild type's gap-junction advantage is in placement or in strength.

On the thermal cell at target 35 under PPO at block V's point otherwise, the wild type and the
gap-only null (chemical graph held, gap junctions rewired with their counts) each learn with fixed
gaps and with plastic gaps (a learnable positive multiplier per existing pair), beside one frozen
floor per wiring. Seeds 513-576; the wild type's fixed-gap arms are reused from the thermal split.

**Three readings**, corrected together, each positive when the wild type is ahead:

* ``base = gap(fixed)``: the lead over the gap-only null with fixed strengths;
* ``lead = gap(plastic)``: the lead once each wiring can tune its strengths;
* ``interaction = gap(plastic) - gap(fixed)``.

All three are read against one minimum: 2/3 of the 0.0865 that holding the null's gap junctions moved
the lead on this cell, the thermal split's committed effect. ``base`` must exclude zero above, or the
verdict is ``no_gap_effect``.

Every statistic is A.2's (``operating_point_surface``); the states are B.1c's
(``measured_prior_contrast``); the completeness check is B.1b's (``measured_prior_pilot``).

Usage::

    uv run python scripts/analysis/plastic_gaps.py identity --rerun <campaign> --out identity.json
    uv run python scripts/analysis/plastic_gaps.py plasticity --logs <pilot logs> --out check.json
    uv run python scripts/analysis/plastic_gaps.py score --logs <dir> [--logs <dir>] \
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

import gap_split as gs  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

HALF = "ppo"
CELL = ts.CELL
SEEDS: tuple[int, ...] = tuple(range(513, 577))
PILOT_SEEDS: tuple[int, ...] = tuple(range(1401, 1405))
FIXED, PLASTIC = "fixed", "plastic"
LEVELS: tuple[str, ...] = (FIXED, PLASTIC)

_T = "connectomeppo_small_continuous2d_thermal_klinotaxis"
_WT_FROZEN = f"{_T}_frozen_t35"
_NULL_FROZEN = f"{_T}_rewired_gap_only_null_frozen_t35"
STEMS: dict[str, dict[str, str]] = {
    FIXED: {
        "wt_learn": f"{_T}_t35",
        "wt_frozen": _WT_FROZEN,
        "rn_learn": f"{_T}_rewired_gap_only_null_t35",
        "rn_frozen": _NULL_FROZEN,
    },
    PLASTIC: {
        "wt_learn": f"{_T}_plastic_gaps_t35",
        "wt_frozen": _WT_FROZEN,
        "rn_learn": f"{_T}_rewired_gap_only_null_plastic_gaps_t35",
        "rn_frozen": _NULL_FROZEN,
    },
}

PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC
# The thermal split's committed effect on this cell: how far holding the null's gap junctions moved
# the wild type's lead (Logbook 078's split, on the primary metric).
REFERENCE_EFFECT = 0.08647916666666666
MINIMUM = mc.MINIMUM_FRACTION * abs(REFERENCE_EFFECT)

# The committed thermal split the wild type's fixed-gap arms are reused from, and the identity runs.
COMMITTED_CAMPAIGN = "campaigns/a6t2-thermal-split"
IDENTITY_RUNS: tuple[tuple[str, int], ...] = (
    *((f"{_T}_t35", s) for s in SEEDS[:4]),
    *((_WT_FROZEN, s) for s in SEEDS[:2]),
)


class PlasticGapsError(ValueError):
    """The runs on disk are not the panel this module scores."""


# ── The verdict map ──────────────────────────────────────────────────────────────────────────
# (interaction state, lead state) -> verdict, once the base is present and nothing is unresolved.
_VERDICTS: dict[tuple[str, str], str] = {
    ("move_null", "no_move"): "strength",
    ("move_null", "below"): "strength",
    ("no_move", "move_wt"): "placement",
    ("move_null", "move_wt"): "partly_strength",
}


def verdict(base_test: dict[str, Any], interaction: str, lead: str) -> str:
    """Map the three readings' states to the registered verdict.

    ``interaction`` and ``lead`` are ``mc.classify`` states. ``move_null`` on the interaction means
    the lead shrank once strengths could be tuned; ``move_wt`` on the lead means it remains.
    """
    if float(base_test["ci_lo"]) <= 0.0:
        return "no_gap_effect"
    if "unresolved" in (interaction, lead):
        return "unresolved"
    if interaction == "move_wt":
        return "placement_amplified"
    return _VERDICTS.get((interaction, lead), "mixed")


def read_panel(
    gates: dict[str, Any],
    base: dict[str, Any],
    lead: dict[str, Any],
    interaction: dict[str, Any],
) -> dict[str, Any]:
    """Gates first, then each reading's state, then the verdict."""
    readable = {level: mp.level_passes(g) for level, g in gates.items()}
    out: dict[str, Any] = {
        "minimum": MINIMUM,
        "reference_effect": REFERENCE_EFFECT,
        "level_readable": readable,
    }
    if not all(readable.values()):
        out["verdict"] = "unreadable"
        out["why"] = "a level fails its floor on a wiring, or both its arms saturate"
        return out
    states: dict[str, str] = {}
    for name, mean, test in (
        ("base", base["gap_mean"], base["test"]),
        ("lead", lead["gap_mean"], lead["test"]),
        ("interaction", interaction["interaction_mean"], interaction["test"]),
    ):
        states[name] = mc.classify(mean, test["ci_lo"], test["ci_hi"], test["bh_q"], MINIMUM)
    out["states"] = states
    out["verdict"] = verdict(base["test"], states["interaction"], states["lead"])
    return out


# ── Manifest and scoring ─────────────────────────────────────────────────────────────────────
def build_manifest(log_dirs: list[Path], path: Path, seeds: tuple[int, ...] = SEEDS) -> Path:
    """Write ``<arm> <level> <seed> <log>`` for every panel run on ``seeds``."""
    seen: dict[tuple[str, str, int], Path] = {}
    lines: list[str] = []
    for arm, level, seed, log in gp.evidence(STEMS, log_dirs):
        if seed not in seeds:
            continue
        prior = seen.setdefault((arm, level, seed), log)
        if prior != log:
            msg = f"{arm} {level} seed {seed} has two runs: {prior} and {log}"
            raise PlasticGapsError(msg)
        lines.append(f"{arm} {level} {seed} {log.resolve()}")
    path.write_text("\n".join(lines) + "\n")
    return path


def score(log_dirs: list[Path], out_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Score both levels, correct the three readings together, then read them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(log_dirs, out_dir / "manifest-plastic-gaps.txt", seeds)
    mp.require_complete(manifest, HALF, seeds, LEVELS)
    reports = {
        level: ops.score_level(manifest, HALF, level, out_dir / f"tmp-{level}", cell=CELL)
        for level in LEVELS
    }
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    gaps = {m: {lv: ops.wiring_gap(reports[lv], m) for lv in LEVELS} for m in metrics}
    interactions = {m: ops.interaction(reports[FIXED], reports[PLASTIC], m) for m in metrics}
    for m in metrics:
        ops.apply_family_correction(
            {
                "base": {m: {"interaction": gaps[m][FIXED]}},
                "lead": {m: {"interaction": gaps[m][PLASTIC]}},
                "interaction": {m: {"interaction": interactions[m]}},
            },
        )
    gates = {
        level: ops.learning_gates(manifest, HALF, seeds, level, floor_level=level)
        for level in LEVELS
    }
    return {
        "cell": CELL,
        "seeds": list(seeds),
        "primary_metric": PRIMARY_METRIC,
        "beside_metric": BESIDE_METRIC,
        "gates": gates,
        "gaps": gaps,
        "interaction": interactions,
        "censoring_rule_choice": ops.choose_metric(
            {level: ops.censoring_rates(report) for level, report in reports.items()},
        ),
        "reading": read_panel(
            gates,
            gaps[PRIMARY_METRIC][FIXED],
            gaps[PRIMARY_METRIC][PLASTIC],
            interactions[PRIMARY_METRIC],
        ),
    }


def _final_topology(log: Path) -> dict[str, Any] | None:
    """Return a run's final topology tensors, through its tracked-experiment record."""
    import torch
    from l4_panel import (  # pyright: ignore[reportMissingImports]
        EXPERIMENTS,
        REPO,
        _experiment_json,
    )

    experiment = _experiment_json(log.read_text(), EXPERIMENTS)
    exports = experiment.get("exports_path") if experiment else None
    final = REPO / exports / "weights" / "final.pt" if exports else None
    if final is None or not final.is_file():
        return None
    topology = torch.load(final, weights_only=True).get("topology")
    return topology if isinstance(topology, dict) else None


def multipliers(log_dirs: list[Path], seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Describe how far each plastic run's learned gap multipliers moved from 1.

    Per run, over the existing gap pairs: the median absolute log multiplier, the 5th and 95th
    percentile multipliers, the extremes, and total gap strength relative to the starting strengths.
    Each is summarised over seeds per wiring.
    """
    import numpy as np
    import torch

    out: dict[str, Any] = {}
    for wiring, arm in (("wild_type", "wt_learn"), ("gap_only_null", "rn_learn")):
        rows: list[tuple[float, ...]] = []
        for seed in seeds:
            log = _find(log_dirs, f"{STEMS[PLASTIC][arm]}-seed{seed}.log")
            topology = _final_topology(log) if log is not None else None
            if topology is None:
                continue
            log_multiplier = topology["gap_log_multiplier"]
            strengths = topology["g_gap"]
            pairs = strengths > 0
            multiplier = torch.exp((log_multiplier + log_multiplier.T) / 2)[pairs].double().numpy()
            start = strengths[pairs].double().numpy()
            rows.append(
                (
                    float(np.median(np.abs(np.log(multiplier)))),
                    float(np.percentile(multiplier, 5)),
                    float(np.percentile(multiplier, 95)),
                    float(multiplier.min()),
                    float(multiplier.max()),
                    float((start * multiplier).sum() / start.sum()),
                ),
            )
        table = np.array(rows) if rows else np.zeros((0, 6))
        out[wiring] = {
            "n_seeds": len(rows),
            "median_abs_log_multiplier": float(np.median(table[:, 0])) if rows else None,
            "p5_multiplier": float(np.median(table[:, 1])) if rows else None,
            "p95_multiplier": float(np.median(table[:, 2])) if rows else None,
            "min_multiplier": float(table[:, 3].min()) if rows else None,
            "max_multiplier": float(table[:, 4].max()) if rows else None,
            "total_strength_ratio": float(np.median(table[:, 5])) if rows else None,
            "total_strength_ratio_range": [float(table[:, 5].min()), float(table[:, 5].max())]
            if rows
            else None,
        }
    out["summary"] = "medians over seeds of per-run statistics over existing gap pairs"
    return out


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per seed: every arm's plateau and floor, both gaps, the interaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    header = ["seed"]
    for level in LEVELS:
        header += [f"{level}_{c}" for c in ("wt_plateau", "wt_floor", "rn_plateau", "rn_floor")]
        header += [f"{level}_gap_{m}" for m in metrics]
    header += [f"interaction_{m}" for m in metrics]
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
            row += [_fmt(result["interaction"][m]["per_seed"].get(seed)) for m in metrics]
            writer.writerow(row)
    return path


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


# ── The identity check and the plasticity check ─────────────────────────────────────────────
def identity(rerun_dir: Path, repo: Path = wp.REPO) -> dict[str, Any]:
    """Compare the wild type's fixed-gap re-runs with the committed thermal split, bit for bit."""
    logs = rerun_dir / "logs" if (rerun_dir / "logs").is_dir() else rerun_dir
    committed_logs = repo / COMMITTED_CAMPAIGN / "logs"
    runs: dict[str, Any] = {}
    for stem, seed in IDENTITY_RUNS:
        name = f"{stem}-seed{seed}.log"
        committed, rerun = committed_logs / name, logs / name
        if not committed.is_file() or not rerun.is_file():
            runs[name] = {
                "identical": False,
                "missing": [str(p) for p in (committed, rerun) if not p.is_file()],
            }
            continue
        record = gs.compare_runs(committed, rerun)
        record["committed"] = f"{COMMITTED_CAMPAIGN}/logs/{name}"
        runs[name] = record
    return {
        "runs": runs,
        "all_identical": bool(runs) and all(r["identical"] for r in runs.values()),
    }


def _find(log_dirs: list[Path], name: str) -> Path | None:
    return next((d / name for d in log_dirs if (d / name).is_file()), None)


def plasticity(log_dirs: list[Path], seeds: tuple[int, ...] = PILOT_SEEDS) -> dict[str, Any]:
    """Check each plastic-gap learning run differs from its fixed-gap twin at the same seed.

    If the multipliers never received a gradient the two runs would be bit-identical, so a
    difference shows the plasticity acts.
    """
    pairs = {
        "wild_type": (STEMS[FIXED]["wt_learn"], STEMS[PLASTIC]["wt_learn"]),
        "gap_only_null": (STEMS[FIXED]["rn_learn"], STEMS[PLASTIC]["rn_learn"]),
    }
    out: dict[str, Any] = {}
    for wiring, (fixed, plastic) in pairs.items():
        per_seed: dict[int, bool | None] = {}
        for seed in seeds:
            # The twins may sit in different campaigns (reused fixed-gap runs), so each is found
            # across every directory on its own.
            a = _find(log_dirs, f"{fixed}-seed{seed}.log")
            b = _find(log_dirs, f"{plastic}-seed{seed}.log")
            per_seed[seed] = None if a is None or b is None else gs.run_lines(a) != gs.run_lines(b)
        out[wiring] = per_seed
    complete = all(v is not None for seeds_ in out.values() for v in seeds_.values())
    acts = complete and all(all(seeds_.values()) for seeds_ in out.values())
    return {"per_seed": out, "complete": complete, "plasticity_acts": acts}


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    ident = sub.add_parser("identity", help="compare the wild type's re-runs with the split")
    ident.add_argument("--rerun", type=Path, required=True)
    plast = sub.add_parser("plasticity", help="check plastic runs differ from their fixed twins")
    plast.add_argument("--logs", type=Path, action="append", required=True)
    plast.add_argument(
        "--panel",
        action="store_true",
        help="check the panel's seeds, not the pilot's",
    )
    mult = sub.add_parser("multipliers", help="how far the learned gap multipliers moved")
    mult.add_argument("--logs", type=Path, action="append", required=True)
    sc = sub.add_parser("score", help="score the panel")
    sc.add_argument("--logs", type=Path, action="append", required=True)
    sc.add_argument("--out-dir", type=Path, required=True)
    sc.add_argument("--csv", type=Path)
    for parser in (ident, plast, mult, sc):
        parser.add_argument("--out", type=Path, help="write the JSON here")
    args = ap.parse_args(argv)

    if args.command == "identity":
        result = identity(args.rerun)
        ok = result["all_identical"]
    elif args.command == "plasticity":
        result = plasticity(args.logs, SEEDS if args.panel else PILOT_SEEDS)
        ok = result["plasticity_acts"]
    elif args.command == "multipliers":
        result = multipliers(args.logs)
        ok = all(result[w]["n_seeds"] == len(SEEDS) for w in ("wild_type", "gap_only_null"))
    else:
        result = score(args.logs, args.out_dir)
        ok = True
        if args.csv:
            write_csv(result, args.csv)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
