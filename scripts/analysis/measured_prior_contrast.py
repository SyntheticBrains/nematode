#!/usr/bin/env python
"""The wiring x weight-prior 2x3: whether the animal's own weights make its wiring legible.

Wiring {wild type, rewired null} x prior {random, measured, measured-shuffled}, each arm learning and
frozen, on hard350, under two learners:

* **PPO** under the per-neuron fan-in draw, with the pooled readout, seeds 225-256.
* **The reading learner** (``readout_only``) at A.2's centre under the edge-order draw, seeds 257-304.

Two interactions per learner, both paired by seed on the wiring gap (wild type minus null, positive
when the wild type is better):

1. **measured x wiring** -- ``gap(measured) - gap(random)``: does the measured prior move the gap?
2. **placement x wiring** -- ``gap(measured) - gap(shuffled)``: does it matter which synapse carries
   which fitted value? The shuffled arm keeps the values and destroys their assignment, so only this
   contrast can say "the animal's weights" rather than "a distribution of values".

**The primary is ``auc_success`` on both learners**, a departure from A.2's censoring rule registered
before launch: on episodes, PPO's detectable effect is 1.14 times its committed wiring effect under
this draw, so the panel could not see even a sign move there. Episodes are reported beside, with the
censoring rule's own choice recorded next to them.

The minimum is 2/3 of each learner's **committed** wiring effect on the draw it runs -- A.1's fan-in
figure for PPO, A.2's reading centre for the reading learner -- never an in-campaign gap, which would
let the result size its own bar. Each interaction is classified into a state, and the two states map to
a verdict through a table fixed before launch.

Every statistic is A.2's (``operating_point_surface``) and every stem is B.1b's
(``measured_prior_pilot``), imported rather than copied.
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

import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]

# ── Seeds ────────────────────────────────────────────────────────────────────────────────────
# B.1b fixed 225 as this panel's first seed. The learners take disjoint bands, as every Phase 8
# campaign has; the reading learner's is larger because its per-seed spread is larger.
SEEDS_BY_HALF: dict[str, tuple[int, ...]] = {
    "ppo": tuple(range(225, 257)),
    "reading": tuple(range(257, 305)),
}
HALVES = tuple(SEEDS_BY_HALF)

# ── Levels ───────────────────────────────────────────────────────────────────────────────────
RANDOM = mp.RANDOM
MEASURED = mp.DEFAULT_LEVEL
SHUFFLED = mp.SHUFFLED_LEVEL
LEVELS: tuple[str, ...] = (RANDOM, MEASURED, SHUFFLED)
ARMS = mp.ARMS


def _arms_by_stem() -> dict[str, tuple[str, str, str]]:
    """Config stem -> ``(half, arm, level)``, from B.1b's shared stem rule."""
    out: dict[str, tuple[str, str, str]] = {}
    for half in HALVES:
        for level in LEVELS:
            for arm in ARMS:
                stem = mp.stem_for(half, arm, level)
                if stem in out:
                    msg = f"stem {stem!r} is claimed by {out[stem]} and by {(half, arm, level)}"
                    raise ValueError(msg)
                out[stem] = (half, arm, level)
    return out


ARM_BY_STEM: dict[str, tuple[str, str, str]] = _arms_by_stem()

# ── The two interactions: (name, level, reference level) ────────────────────────────────────
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("measured", MEASURED, RANDOM),
    ("placement", MEASURED, SHUFFLED),
)

# ── Metric, reference and minimum ────────────────────────────────────────────────────────────
PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC
# Each learner's committed wiring effect on hard350, on the draw this panel runs it under:
# A.1's shared-initialisation gap under per_neuron_fanin (Logbook 070's JSON, `shared_gap_mean`),
# and A.2's reading centre (Logbook 071's reading JSON, the centre's `gap_mean`).
REFERENCE_EFFECT: dict[str, float] = {"ppo": 0.055020833, "reading": -0.2105}
MINIMUM_FRACTION = 2.0 / 3.0
SIGNIFICANCE = 0.05


def minimum(half: str) -> float:
    """Return the registered minimum on the primary: 2/3 of the reference effect's size."""
    return MINIMUM_FRACTION * abs(REFERENCE_EFFECT[half])


class ContrastError(ValueError):
    """The panel on disk is not the panel this module scores."""


# ── Classification and verdict: pure functions of the statistics ─────────────────────────────
def classify(mean: float, ci_lo: float, ci_hi: float, q: float, floor: float) -> str:
    """Place one interaction in its registered state.

    ``unresolved`` covers three cases: the interval spans the minimum; q is not significant while
    the interval excludes zero; q is significant while the interval spans zero, where the Wilcoxon
    and the bootstrap disagree.
    """
    significant = q < SIGNIFICANCE
    if significant and ci_lo > 0.0 and mean >= floor:
        return "move_wt"
    if significant and ci_hi < 0.0 and mean <= -floor:
        return "move_null"
    if significant and (ci_lo > 0.0 or ci_hi < 0.0):
        return "below"
    if not significant and -floor < ci_lo <= 0.0 <= ci_hi < floor:
        return "no_move"
    return "unresolved"


# The registered verdict map. A move needs the placement control to move the same way before it is
# legibility or hiding; otherwise it is a distribution effect in the direction it moved.
_MOVE_VERDICT: dict[str, tuple[str, str]] = {
    "move_wt": ("legible", "value_distribution_wt"),
    "move_null": ("hides", "value_distribution_null"),
}
_OTHER_VERDICT: dict[str, str] = {"below": "below_minimum", "unresolved": "unresolved"}


def verdict(measured: str, placement: str) -> str:
    """Map the two states to the registered verdict."""
    if measured in _MOVE_VERDICT:
        both, distribution = _MOVE_VERDICT[measured]
        return both if placement == measured else distribution
    if measured == "no_move":
        if placement == "no_move":
            return "null"
        return "placement_only" if placement in _MOVE_VERDICT else "null_placement_unresolved"
    return _OTHER_VERDICT[measured]


def read_learner(half: str, gates: dict[str, Any], interactions: dict[str, Any]) -> dict[str, Any]:
    """Apply the gates, then the states and the verdict, for one learner.

    ``gates`` maps each level to ``learning_gates`` output; ``interactions`` maps each contrast name
    to its primary-metric interaction with ``bh_q`` set.
    """
    wt_at_measured = mp.wild_type_learns(gates[MEASURED])
    readable = {lvl: mp.level_passes(g) for lvl, g in gates.items()}
    out: dict[str, Any] = {
        "minimum": minimum(half),
        "reference_effect": REFERENCE_EFFECT[half],
        "level_readable": readable,
        "states": {},
    }
    if not wt_at_measured:
        out["verdict"] = "lee_unlearnable"
        out["why"] = "the wild type does not beat its floor under the measured prior"
        return out
    unreadable = [name for name, level, ref in CONTRASTS if not (readable[level] and readable[ref])]
    if unreadable:
        out["verdict"] = "unreadable"
        out["why"] = f"a level in {unreadable} fails its floor on a wiring or saturates"
        return out
    for name, _, _ in CONTRASTS:
        entry = interactions[name]
        test = entry["test"]
        out["states"][name] = classify(
            entry["interaction_mean"],
            test["ci_lo"],
            test["ci_hi"],
            test["bh_q"],
            minimum(half),
        )
    out["verdict"] = verdict(out["states"]["measured"], out["states"]["placement"])
    return out


def honour_drift(half: str, reading: dict[str, Any], drift: dict[str, Any]) -> dict[str, Any]:
    """Void a learner whose fixed substrate moved, whatever its verdict would have been.

    The reading learner reads a chemical matrix it must never write; drift on any scored seed, or
    missing evidence, voids that half. PPO writes the matrix by design, so its drift voids nothing.
    """
    if half == "reading" and drift.get("void"):
        return {
            **reading,
            "verdict": "void",
            "why": "w_chem drifted from its frozen floor, or its evidence is missing, on at least "
            "one scored seed",
        }
    return reading


def correct_family(results: dict[str, dict[str, Any]], metric: str) -> None:
    """BH-FDR one metric's interactions across both learners and both contrasts, in place."""
    family = {
        f"{half}:{name}": {metric: {"interaction": results[half]["interactions"][metric][name]}}
        for half in results
        for name, _, _ in CONTRASTS
    }
    ops.apply_family_correction(family)


# ── Scoring ──────────────────────────────────────────────────────────────────────────────────
def score_half(
    campaign_dir: Path,
    half: str,
    out_dir: Path,
    seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Score one learner's gates, gaps and interactions. The family correction comes after."""
    seeds = seeds if seeds is not None else SEEDS_BY_HALF[half]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = mp.build_manifest(
        campaign_dir,
        out_dir / f"manifest-{half}.txt",
        half,
        seeds,
        arm_by_stem=ARM_BY_STEM,
    )
    mp.require_complete(manifest, half, seeds, LEVELS)

    reports = {
        level: ops.score_level(manifest, half, level, out_dir / f"tmp-{half}-{level}")
        for level in LEVELS
    }
    rates = {level: ops.censoring_rates(report) for level, report in reports.items()}
    gates = {
        level: ops.learning_gates(manifest, half, seeds, level, floor_level=level)
        for level in LEVELS
    }
    gaps = {
        metric: {level: ops.wiring_gap(reports[level], metric) for level in LEVELS}
        for metric in (PRIMARY_METRIC, BESIDE_METRIC)
    }
    interactions = {
        metric: {
            name: ops.interaction(reports[ref], reports[level], metric)
            for name, level, ref in CONTRASTS
        }
        for metric in (PRIMARY_METRIC, BESIDE_METRIC)
    }
    drift = ops.substrate_drift(manifest, half, seeds, LEVELS, floor_levels={s: s for s in LEVELS})
    drift["obligation_applies"] = half == "reading"
    return {
        "half": half,
        "cell": ops.CELL,
        "seeds": list(seeds),
        "gates": gates,
        "gaps": gaps,
        "interactions": interactions,
        # The censoring rule's own choice, recorded beside the registered departure from it.
        "censoring_rule_choice": ops.choose_metric(rates),
        "substrate_drift": drift,
        "verdicts": {level: report.get("verdict") for level, report in reports.items()},
    }


def coverage_report() -> dict[str, int]:
    """Where the fitted table lands on Cook 2019, at head scope and at full scope."""
    from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
    from quantumnematode.connectome.measured_weights import coverage, measured_weights

    c = coverage(measured_weights(), load_cook_2019_hermaphrodite())
    return {
        "covered": len(c.covered),
        "full_scope": len(c.full_scope),
        "head_scope": len(c.head_scope),
        "coverable_head_scope": len(c.coverable_head_scope),
        "gap_junction_only": len(c.gap_junction_only),
        "no_connection": len(c.no_connection),
    }


def score(campaigns: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    """Score both learners, correct each metric's family across them, then read the verdicts."""
    halves = {half: score_half(path, half, out_dir) for half, path in campaigns.items()}
    missing = set(HALVES) - set(halves)
    if missing:
        msg = f"the family spans both learners; missing {sorted(missing)}"
        raise ContrastError(msg)
    for metric in (PRIMARY_METRIC, BESIDE_METRIC):
        correct_family(halves, metric)
    for half, result in halves.items():
        result["reading"] = honour_drift(
            half,
            read_learner(half, result["gates"], result["interactions"][PRIMARY_METRIC]),
            result["substrate_drift"],
        )
    return {
        "primary_metric": PRIMARY_METRIC,
        "beside_metric": BESIDE_METRIC,
        "coverage": coverage_report(),
        "halves": halves,
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per (learner, seed): every arm's plateau and floor, every gap, both interactions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    header = ["half", "seed"]
    for level in LEVELS:
        header += [f"{level}_{c}" for c in ("wt_plateau", "wt_floor", "rn_plateau", "rn_floor")]
        header += [f"{level}_gap_{m}" for m in metrics]
    header += [f"{name}_interaction_{m}" for name, _, _ in CONTRASTS for m in metrics]
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
                row += [
                    _fmt(res["interactions"][m][name]["per_seed"].get(seed))
                    for name, _, _ in CONTRASTS
                    for m in metrics
                ]
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
