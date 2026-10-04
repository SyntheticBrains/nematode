#!/usr/bin/env python
"""A.6t: the null-strength control and its gap-only split, on block V's thermal cell.

A.6 and its split ran on hard350 alone. There, about half of block V's ``auc_success`` lead over the
degree-preserving null came from that null's rewired gap junctions. On block V's second cell, the
thermal-plus-foraging cell at target 20, only A.1's initialisation control had run.

Four wirings, each learning and frozen, under PPO at block V's committed point (edge-order draw,
pooled readout, depth 4), seeds 385-512:

* the wild type and the **current** degree-preserving null, block V's own thermal configs;
* the **chemical-only** null: the chemical graph rewired, gap junctions (pairs and counts) and the
  38 autapses held at the wild type's;
* the **gap-held** null: the current null's chemical graph exactly, autapses lost as there, gap
  junctions held at the wild type's.

Three levels share the wild-type runs. **Two primary interactions**, each paired by seed:

* ``combined = gap(chemical) - gap(full)``: the gap junctions and autapses held together, as A.6;
* ``split = gap(gap_held) - gap(full)``: the gap junctions alone, on the current null's exact
  chemical graph, as A.6's split.

Each is positive when the wild type stands better against the narrower null. Both are read against
one registered minimum, 2/3 of A.1's thermal effect, and corrected together as one family. Neither
minimum is taken from this campaign's own data or from the hard350 cell.

Every statistic is A.2's (``operating_point_surface``); the states and the drift rule are B.1c's
(``measured_prior_contrast``); the completeness check is B.1b's (``measured_prior_pilot``); the two
verdict maps are A.6's and its split's.
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

# ── Panel ────────────────────────────────────────────────────────────────────────────────────
HALF = "ppo"
CELL = "thermal"
# Fresh: A.6 and its split ended at 384.
SEEDS: tuple[int, ...] = tuple(range(385, 513))
FULL, CHEMICAL, GAP_HELD = "full", "chemical", "gap_held"
LEVELS: tuple[str, ...] = (FULL, CHEMICAL, GAP_HELD)
ARMS = mp.ARMS

_T = "connectomeppo_small_continuous2d_thermal_klinotaxis"
_WILD = {"wt_learn": f"{_T}_t20", "wt_frozen": f"{_T}_frozen_t20"}
# The thermal family writes the wiring tag, then `_frozen`, then the target suffix.
STEMS: dict[str, dict[str, str]] = {
    FULL: {
        **_WILD,
        "rn_learn": f"{_T}_rewired_null_t20",
        "rn_frozen": f"{_T}_rewired_null_frozen_t20",
    },
    CHEMICAL: {
        **_WILD,
        "rn_learn": f"{_T}_rewired_chemical_null_t20",
        "rn_frozen": f"{_T}_rewired_chemical_null_frozen_t20",
    },
    GAP_HELD: {
        **_WILD,
        "rn_learn": f"{_T}_rewired_gap_held_null_t20",
        "rn_frozen": f"{_T}_rewired_gap_held_null_frozen_t20",
    },
}
# The new configs: each narrower-null arm from the current-null arm it narrows.
NEW_ARMS: dict[str, tuple[str, str]] = {
    STEMS[level][arm]: (HALF, STEMS[FULL][arm])
    for level in (CHEMICAL, GAP_HELD)
    for arm in ("rn_learn", "rn_frozen")
}
NEW_WIRING: dict[str, str] = {
    **{STEMS[CHEMICAL][a]: "rewired_chemical_only" for a in ("rn_learn", "rn_frozen")},
    **{STEMS[GAP_HELD][a]: "rewired_gap_junctions_held" for a in ("rn_learn", "rn_frozen")},
}


def _levels_by_stem() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Config stem -> ``(arm, levels it serves)``: a wild-type run serves every level."""
    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for level in LEVELS:
        for arm, stem in STEMS[level].items():
            prior = out.get(stem)
            if prior and prior[0] != arm:
                msg = f"stem {stem!r} is claimed by {prior[0]} and by {arm}"
                raise ValueError(msg)
            out[stem] = (arm, (*prior[1], level) if prior else (level,))
    return out


LEVELS_BY_STEM = _levels_by_stem()

# ── Metric, reference, minimum ───────────────────────────────────────────────────────────────
PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC
# A.1's thermal effect at this point (Logbook 070's JSON, the thermal `baseline_gap_mean` on the
# primary): the thermal cell's own, since its magnitude has not matched hard350's on any seed set.
REFERENCE_EFFECT = 0.08289583333333334
MINIMUM = mc.MINIMUM_FRACTION * abs(REFERENCE_EFFECT)
INTERACTIONS: dict[str, str] = {"combined": CHEMICAL, "split": GAP_HELD}

_COMBINED_VERDICT = {
    "move_null": "gap_or_autapse",
    "move_wt": "amplified",
    "below": "below_minimum",
    "unresolved": "unresolved",
}
_SPLIT_VERDICT = {
    "move_null": "gap_junctions",
    "below": "partial",
    "no_move": "not_gap_junctions",
    "move_wt": "opposite",
    "unresolved": "unresolved",
}


class ThermalControlError(ValueError):
    """The panel on disk is not the panel this module scores."""


# ── Verdicts ─────────────────────────────────────────────────────────────────────────────────
def gap_to_attribute(chemical_gap_test: dict[str, Any]) -> bool:
    """Say whether the gap against the chemical-only null excludes zero on the wild type's side."""
    return float(chemical_gap_test["ci_lo"]) > 0.0


def verdict(name: str, state: str, chemical_gap_test: dict[str, Any]) -> str:
    """Map one interaction's state to its registered verdict."""
    if name == "combined":
        if state == "no_move":
            return "chemical" if gap_to_attribute(chemical_gap_test) else "no_gap_to_attribute"
        return _COMBINED_VERDICT[state]
    return _SPLIT_VERDICT[state]


def read_panel(
    gates: dict[str, Any],
    interactions: dict[str, dict[str, Any]],
    chemical_gap_test: dict[str, Any],
) -> dict[str, Any]:
    """Gates first, then each interaction's state and verdict."""
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
    out["gap_to_attribute"] = gap_to_attribute(chemical_gap_test)
    for name, inter in interactions.items():
        test = inter["test"]
        state = mc.classify(
            inter["interaction_mean"],
            test["ci_lo"],
            test["ci_hi"],
            test["bh_q"],
            MINIMUM,
        )
        out[name] = {"state": state, "verdict": verdict(name, state, chemical_gap_test)}
    return out


def split_share(interactions: dict[str, dict[str, Any]]) -> float | None:
    """Describe the split's mean as a share of the combined mean; never read as a verdict."""
    combined = interactions["combined"]["interaction_mean"]
    split = interactions["split"]["interaction_mean"]
    if combined in (None, 0.0) or split is None:
        return None
    return split / combined


# ── Manifest ─────────────────────────────────────────────────────────────────────────────────
def build_manifest(campaign_dir: Path, path: Path, seeds: tuple[int, ...] = SEEDS) -> Path:
    """Write ``<arm> <level> <seed> <log>`` lines, a wild-type run once under each level."""
    logs = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    seen: set[tuple[str, str, int]] = set()
    lines: list[str] = []
    for log in sorted(logs.glob("*.log")):
        stem, sep, seed_part = log.stem.rpartition("-seed")
        if not sep:
            msg = f"{log.name} has no -seedN suffix, so it cannot be placed in the panel"
            raise ThermalControlError(msg)
        entry = LEVELS_BY_STEM.get(stem)
        if entry is None:
            msg = f"{log.name} names config {stem!r}, which this panel does not have"
            raise ThermalControlError(msg)
        arm, levels = entry
        seed = int(seed_part)
        if seed not in seeds:
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
                raise ThermalControlError(msg)
            seen.add(key)
            lines.append(f"{arm} {level} {seed} {out}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Scoring ──────────────────────────────────────────────────────────────────────────────────
def score(campaign_dir: Path, out_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Score the gates, the three gaps and the two interactions, then read the verdicts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(campaign_dir, out_dir / "manifest-thermal.txt", seeds)
    mp.require_complete(manifest, HALF, seeds, LEVELS)

    reports = {
        level: ops.score_level(manifest, HALF, level, out_dir / f"tmp-{level}", cell=CELL)
        for level in LEVELS
    }
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    interactions = {
        m: {
            name: ops.interaction(reports[FULL], reports[level], m)
            for name, level in INTERACTIONS.items()
        }
        for m in metrics
    }
    # One family per metric: the two interactions, every test computed before any is read.
    for m in metrics:
        ops.apply_family_correction(
            {name: {m: {"interaction": interactions[m][name]}} for name in INTERACTIONS},
        )
    gates = {
        level: ops.learning_gates(manifest, HALF, seeds, level, floor_level=level)
        for level in LEVELS
    }
    gaps = {m: {lv: ops.wiring_gap(reports[lv], m) for lv in LEVELS} for m in metrics}
    drift = ops.substrate_drift(manifest, HALF, seeds, LEVELS, floor_levels={s: s for s in LEVELS})
    drift["obligation_applies"] = False
    reading = mc.honour_drift(
        HALF,
        read_panel(gates, interactions[PRIMARY_METRIC], gaps[PRIMARY_METRIC][CHEMICAL]["test"]),
        drift,
    )
    return {
        "cell": CELL,
        "half": HALF,
        "seeds": list(seeds),
        "primary_metric": PRIMARY_METRIC,
        "beside_metric": BESIDE_METRIC,
        "gates": gates,
        "gaps": gaps,
        "interactions": interactions,
        "split_share_of_combined": split_share(interactions[PRIMARY_METRIC]),
        "censoring_rule_choice": ops.choose_metric(
            {level: ops.censoring_rates(report) for level, report in reports.items()},
        ),
        "substrate_drift": drift,
        "reading": reading,
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per seed: every arm's plateau and floor, the three gaps, the two interactions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    header = ["seed"]
    for level in LEVELS:
        header += [f"{level}_{c}" for c in ("wt_plateau", "wt_floor", "rn_plateau", "rn_floor")]
        header += [f"{level}_gap_{m}" for m in metrics]
    header += [f"{name}_{m}" for name in INTERACTIONS for m in metrics]
    with path.open("w", newline="") as handle:
        # csv defaults to CRLF, which would make every regeneration read as a whole-file diff.
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
                _fmt(result["interactions"][m][name]["per_seed"].get(seed))
                for name in INTERACTIONS
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
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="scratch directory for manifests")
    ap.add_argument("--out", type=Path, help="write the analysis JSON here instead of stdout")
    ap.add_argument("--csv", type=Path, help="write the per-seed CSV here")
    args = ap.parse_args(argv)

    result = score(args.campaign, args.out_dir)
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
