#!/usr/bin/env python
"""A.6: block V's wiring gap, read against a null that differs from the wild type in chemical placement.

The degree-preserving null behind every wiring result here preserves degree and nothing else: gap
junction counts are coupling weights and travel with their edges, so each neuron's total gap strength
moves, and the swap can remove autapses. The chemical-only null swaps the chemical graph alone and
holds the gap junctions (pairs and counts) and the autapses at the wild type's.

Three wirings, each learning and frozen, on hard350, under two learners:

* **PPO** at block V's committed point (edge-order draw, pooled readout, depth 4), seeds 305-336.
* **The reading learner** (``readout_only``) at A.2's centre, seeds 337-384.

Two levels -- ``full`` (wild type against the current null) and ``chemical`` (wild type against the
chemical-only null) -- share the wild-type runs. **One interaction per learner**,
``gap(chemical) - gap(full)``, paired by seed; the wild type cancels, so it is the full null minus the
chemical null, seed by seed. It is positive when the wild type stands better against the chemical
null.

**It is a combined control.** Gap placement, gap strength and autapses are held together, so a move
is attributed to them jointly and to none of them alone; the gap-only null, which reproduces the
current null's chemical graph exactly, is the paired follow-up. And the two nulls are different
random chemical graphs at each seed, so the interaction carries graph-sampling variance.

**A verdict attributes a gap only where one exists.** ``chemical`` needs the gap against the chemical
null to exclude zero on the reference effect's side; otherwise the verdict is ``no_gap_to_attribute``.

Every statistic is A.2's (``operating_point_surface``); the states and the drift rule are B.1c's
(``measured_prior_contrast``); the completeness check is B.1b's (``measured_prior_pilot``).
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
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

# ── Seeds ────────────────────────────────────────────────────────────────────────────────────
# Fresh: B.1c ended at 304. The learners take disjoint bands, the reading learner's larger because
# its per-seed spread is larger.
SEEDS_BY_HALF: dict[str, tuple[int, ...]] = {
    "ppo": tuple(range(305, 337)),
    "reading": tuple(range(337, 385)),
}
HALVES = tuple(SEEDS_BY_HALF)

# ── Levels and stems ─────────────────────────────────────────────────────────────────────────
FULL = "full"
CHEMICAL = "chemical"
LEVELS: tuple[str, ...] = (FULL, CHEMICAL)
ARMS = mp.ARMS

_PPO = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
_READ = f"{_PPO}_eprop_readout_only"
_READ_FROZEN = f"{_PPO}_eprop_frozen"

# An explicit table, not a rule: the committed spellings are not uniform -- PPO puts `_frozen` after
# the wiring tag, the reading family puts it before -- and the new stems follow each family's order.
STEMS: dict[str, dict[str, dict[str, str]]] = {
    "ppo": {
        FULL: {
            "wt_learn": _PPO,
            "wt_frozen": f"{_PPO}_frozen",
            "rn_learn": f"{_PPO}_rewired_null",
            "rn_frozen": f"{_PPO}_rewired_null_frozen",
        },
        CHEMICAL: {
            "wt_learn": _PPO,
            "wt_frozen": f"{_PPO}_frozen",
            "rn_learn": f"{_PPO}_rewired_chemical_null",
            "rn_frozen": f"{_PPO}_rewired_chemical_null_frozen",
        },
    },
    "reading": {
        FULL: {
            "wt_learn": _READ,
            "wt_frozen": _READ_FROZEN,
            "rn_learn": f"{_READ}_rewired_null",
            "rn_frozen": f"{_READ_FROZEN}_rewired_null",
        },
        CHEMICAL: {
            "wt_learn": _READ,
            "wt_frozen": _READ_FROZEN,
            "rn_learn": f"{_READ}_rewired_chemical_null",
            "rn_frozen": f"{_READ_FROZEN}_rewired_chemical_null",
        },
    },
}
# The new configs: the chemical-null arms, each derived from the full-null arm of the same half.
NEW_ARMS: dict[str, tuple[str, str]] = {
    STEMS[half][CHEMICAL][arm]: (half, STEMS[half][FULL][arm])
    for half in HALVES
    for arm in ("rn_learn", "rn_frozen")
}


def _levels_by_stem() -> dict[str, tuple[str, str, tuple[str, ...]]]:
    """Config stem -> ``(half, arm, levels it serves)``: a wild-type run serves both levels."""
    out: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    for half in HALVES:
        for level in LEVELS:
            for arm, stem in STEMS[half][level].items():
                prior = out.get(stem)
                levels = (*prior[2], level) if prior else (level,)
                if prior and (prior[0], prior[1]) != (half, arm):
                    msg = f"stem {stem!r} is claimed by {prior[:2]} and by {(half, arm)}"
                    raise ValueError(msg)
                out[stem] = (half, arm, levels)
    return out


LEVELS_BY_STEM = _levels_by_stem()

# ── Metric, reference, minimum ───────────────────────────────────────────────────────────────
PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC
# Each learner's committed wiring effect on hard350 at this point: A.1's edge-order baseline
# (Logbook 070's JSON, `baseline_gap_mean`), and A.2's reading centre (Logbook 071's reading JSON).
REFERENCE_EFFECT: dict[str, float] = {"ppo": 0.06098958333333333, "reading": -0.2105}


def minimum(half: str) -> float:
    """Return the registered minimum on the primary: 2/3 of the reference effect's size."""
    return mc.MINIMUM_FRACTION * abs(REFERENCE_EFFECT[half])


class ControlError(ValueError):
    """The panel on disk is not the panel this module scores."""


# ── Verdict ──────────────────────────────────────────────────────────────────────────────────
_VERDICT_BY_STATE = {
    "move_null": "gap_or_autapse",
    "move_wt": "amplified",
    "below": "below_minimum",
    "unresolved": "unresolved",
}


def gap_to_attribute(half: str, chemical_gap_test: dict[str, Any]) -> bool:
    """Say whether the gap against the chemical null excludes zero on the reference's side."""
    if REFERENCE_EFFECT[half] > 0.0:
        return float(chemical_gap_test["ci_lo"]) > 0.0
    return float(chemical_gap_test["ci_hi"]) < 0.0


def verdict(half: str, state: str, chemical_gap_test: dict[str, Any]) -> str:
    """Map a state to the registered verdict, with the attribution gate on ``chemical``."""
    if state == "no_move":
        return "chemical" if gap_to_attribute(half, chemical_gap_test) else "no_gap_to_attribute"
    return _VERDICT_BY_STATE[state]


def read_learner(
    half: str,
    gates: dict[str, Any],
    interaction: dict[str, Any],
    chemical_gap_test: dict[str, Any],
) -> dict[str, Any]:
    """Gates first, then the state and the verdict, for one learner."""
    readable = {level: mp.level_passes(g) for level, g in gates.items()}
    out: dict[str, Any] = {
        "minimum": minimum(half),
        "reference_effect": REFERENCE_EFFECT[half],
        "level_readable": readable,
    }
    if not all(readable.values()):
        out["verdict"] = "unreadable"
        out["why"] = "a level fails its floor on a wiring, or both its arms saturate"
        return out
    test = interaction["test"]
    out["state"] = mc.classify(
        interaction["interaction_mean"],
        test["ci_lo"],
        test["ci_hi"],
        test["bh_q"],
        minimum(half),
    )
    out["gap_to_attribute"] = gap_to_attribute(half, chemical_gap_test)
    out["verdict"] = verdict(half, out["state"], chemical_gap_test)
    return out


def correct_family(results: dict[str, dict[str, Any]], metric: str) -> None:
    """BH-FDR one metric's interactions across both learners, in place."""
    ops.apply_family_correction(
        {
            half: {metric: {"interaction": results[half]["interactions"][metric]}}
            for half in results
        },
    )


# ── Manifest ─────────────────────────────────────────────────────────────────────────────────
def build_manifest(campaign_dir: Path, path: Path, half: str, seeds: tuple[int, ...]) -> Path:
    """Write ``<arm> <level> <seed> <log>`` lines, a wild-type run once under each level."""
    logs = campaign_dir / "logs"
    if not logs.is_dir():
        logs = campaign_dir
    seen: set[tuple[str, str, int]] = set()
    lines: list[str] = []
    for log in sorted(logs.glob("*.log")):
        stem, sep, seed_part = log.stem.rpartition("-seed")
        if not sep:
            msg = f"{log.name} has no -seedN suffix, so it cannot be placed in the panel"
            raise ControlError(msg)
        entry = LEVELS_BY_STEM.get(stem)
        if entry is None:
            msg = f"{log.name} names config {stem!r}, which this panel does not have"
            raise ControlError(msg)
        log_half, arm, levels = entry
        seed = int(seed_part)
        if log_half != half or seed not in seeds:
            continue
        resolved = log.resolve()
        try:
            out = str(resolved.relative_to(wp.REPO))
        except ValueError:
            out = str(resolved)
        for level in levels:
            key = (arm, level, seed)
            if key in seen:
                msg = f"{key} appears twice in {campaign_dir}"
                raise ControlError(msg)
            seen.add(key)
            lines.append(f"{arm} {level} {seed} {out}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Scoring ──────────────────────────────────────────────────────────────────────────────────
def score_half(
    campaign_dir: Path,
    half: str,
    out_dir: Path,
    seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Score one learner's gates, gaps and interaction. The family correction comes after."""
    seeds = seeds if seeds is not None else SEEDS_BY_HALF[half]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(campaign_dir, out_dir / f"manifest-{half}.txt", half, seeds)
    mp.require_complete(manifest, half, seeds, LEVELS)

    reports = {
        level: ops.score_level(manifest, half, level, out_dir / f"tmp-{half}-{level}")
        for level in LEVELS
    }
    rates = {level: ops.censoring_rates(report) for level, report in reports.items()}
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    drift = ops.substrate_drift(manifest, half, seeds, LEVELS, floor_levels={s: s for s in LEVELS})
    drift["obligation_applies"] = half == "reading"
    return {
        "half": half,
        "cell": ops.CELL,
        "seeds": list(seeds),
        "gates": {
            level: ops.learning_gates(manifest, half, seeds, level, floor_level=level)
            for level in LEVELS
        },
        "gaps": {m: {lv: ops.wiring_gap(reports[lv], m) for lv in LEVELS} for m in metrics},
        "interactions": {m: ops.interaction(reports[FULL], reports[CHEMICAL], m) for m in metrics},
        # The censoring rule's own choice, recorded beside the registered departure from it.
        "censoring_rule_choice": ops.choose_metric(rates),
        "substrate_drift": drift,
        "verdicts": {level: report.get("verdict") for level, report in reports.items()},
    }


def score(campaigns: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    """Score both learners, correct each metric's family across them, then read the verdicts."""
    missing = set(HALVES) - set(campaigns)
    if missing:
        msg = f"the family spans both learners; missing {sorted(missing)}"
        raise ControlError(msg)
    halves = {half: score_half(path, half, out_dir) for half, path in campaigns.items()}
    for metric in (PRIMARY_METRIC, BESIDE_METRIC):
        correct_family(halves, metric)
    for half, result in halves.items():
        result["reading"] = mc.honour_drift(
            half,
            read_learner(
                half,
                result["gates"],
                result["interactions"][PRIMARY_METRIC],
                result["gaps"][PRIMARY_METRIC][CHEMICAL]["test"],
            ),
            result["substrate_drift"],
        )
    return {"primary_metric": PRIMARY_METRIC, "beside_metric": BESIDE_METRIC, "halves": halves}


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per (learner, seed): every arm's plateau and floor, both gaps, the interaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    header = ["half", "seed"]
    for level in LEVELS:
        header += [f"{level}_{c}" for c in ("wt_plateau", "wt_floor", "rn_plateau", "rn_floor")]
        header += [f"{level}_gap_{m}" for m in metrics]
    header += [f"interaction_{m}" for m in metrics]
    with path.open("w", newline="") as handle:
        # csv defaults to CRLF, which would make every regeneration read as a whole-file diff.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for half, res in result["halves"].items():
            for seed in res["seeds"]:
                row: list[Any] = [half, seed]
                for level in LEVELS:
                    gates = res["gates"][level]
                    for wiring in ("wt", "rn"):
                        per_seed = gates[wiring]["per_seed"].get(seed, {})
                        row += [_fmt(per_seed.get("learn")), _fmt(per_seed.get("floor"))]
                    row += [_fmt(res["gaps"][m][level]["per_seed"].get(seed)) for m in metrics]
                row += [_fmt(res["interactions"][m]["per_seed"].get(seed)) for m in metrics]
                writer.writerow(row)
    return path


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI: score both learners' campaigns together, since the family spans them."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ppo-campaign", type=Path, required=True)
    ap.add_argument("--reading-campaign", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="scratch directory for manifests")
    ap.add_argument("--out", type=Path, help="write the analysis JSON here instead of stdout")
    ap.add_argument("--csv", type=Path, help="write the per-seed CSV here")
    args = ap.parse_args(argv)

    result = score({"ppo": args.ppo_campaign, "reading": args.reading_campaign}, args.out_dir)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload)
    if args.csv:
        write_csv(result, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
