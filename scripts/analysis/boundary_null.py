#!/usr/bin/env python
"""A.3's boundary-preserving null: block V's lead once a null keeps the sensory-motor boundary.

A degree-preserving rewiring can route an injected sensor straight onto a readout motor neuron. The
wild type has no motor neuron one hop from a food sensor; the current null has about nine and the
chemical-only null about eight, which is A.2's depth mechanism: at a settling depth of two only those
shortcuts reach the motor layer, and the null wins. The boundary-preserving null is the chemical-only
null with every chemical edge out of an injected sensor or into a readout motor neuron held at the
wild type's, so it has the wild type's one- and two-hop routes and differs only in its interior.

Three wirings, each learning and frozen, on hard350 under PPO at block V's committed point
(edge-order draw, pooled readout, depth 4), seeds 641-768: the wild type, the chemical-only null and
the boundary-preserving null. Two levels share the wild-type runs. **Two registered readings**,
corrected together:

* ``interaction = gap(boundary) - gap(chemical)``, paired by seed: how much holding the boundary moves
  the wild type's lead. Positive when the wild type stands better against the boundary null.
* ``lead = gap(boundary)``: whether the wild type leads a null that differs from it in its interior
  chemical wiring alone.

Both are read against one registered minimum: 2/3 of A.6's committed lead over the chemical-only null
at this point.

Every statistic is A.2's (``operating_point_surface``); the states are B.1c's
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

# ── Panel ────────────────────────────────────────────────────────────────────────────────────
HALF = "ppo"
CELL = ops.CELL
# Fresh: A.6t's follow-up ended at 640.
SEEDS: tuple[int, ...] = tuple(range(641, 769))
CHEMICAL, BOUNDARY = "chemical", "boundary"
LEVELS: tuple[str, ...] = (CHEMICAL, BOUNDARY)
ARMS = mp.ARMS

_PPO = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
_WILD = {"wt_learn": _PPO, "wt_frozen": f"{_PPO}_frozen"}
STEMS: dict[str, dict[str, str]] = {
    CHEMICAL: {
        **_WILD,
        "rn_learn": f"{_PPO}_rewired_chemical_null",
        "rn_frozen": f"{_PPO}_rewired_chemical_null_frozen",
    },
    BOUNDARY: {
        **_WILD,
        "rn_learn": f"{_PPO}_rewired_boundary_held_null",
        "rn_frozen": f"{_PPO}_rewired_boundary_held_null_frozen",
    },
}
# The new configs: each boundary-null arm from the current-null arm, as the other narrower nulls.
NEW_ARMS: dict[str, tuple[str, str]] = {
    STEMS[BOUNDARY]["rn_learn"]: (HALF, f"{_PPO}_rewired_null"),
    STEMS[BOUNDARY]["rn_frozen"]: (HALF, f"{_PPO}_rewired_null_frozen"),
}


def _levels_by_stem() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Config stem -> ``(arm, levels it serves)``: a wild-type run serves both levels."""
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
# A.6's committed lead of the wild type over the chemical-only null at this point under PPO
# (Logbook 074's control.json, the PPO chemical gap on the primary): the base this contrast moves.
REFERENCE_EFFECT = 0.02146875
MINIMUM = mc.MINIMUM_FRACTION * abs(REFERENCE_EFFECT)

_INTERACTION_VERDICT = {
    "move_null": "boundary",
    "move_wt": "shortcuts_helped_null",
    "below": "below_minimum",
    "unresolved": "unresolved",
}
_LEAD_VERDICT = {
    "move_wt": "lead_remains",
    "below": "lead_below_minimum",
    "no_move": "no_lead",
    "move_null": "null_leads",
    "unresolved": "unresolved",
}


class BoundaryNullError(ValueError):
    """The panel on disk is not the panel this module scores."""


def gap_to_attribute(lead_test: dict[str, Any]) -> bool:
    """Say whether the wild type's lead over the boundary null excludes zero above."""
    return float(lead_test["ci_lo"]) > 0.0


def read_panel(
    gates: dict[str, Any],
    interaction: dict[str, Any],
    lead: dict[str, Any],
    base_test: dict[str, Any],
) -> dict[str, Any]:
    """Gates first, then each registered reading's state and verdict.

    The interaction moves the wild type's lead over the chemical-only null, so it reads only where
    that lead exists on this panel's own seeds: ``base_test`` (the chemical-only gap) must exclude
    zero above, or the interaction's verdict is ``no_base_effect``. The lead is read either way.
    """
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
    test = interaction["test"]
    state = mc.classify(
        interaction["interaction_mean"],
        test["ci_lo"],
        test["ci_hi"],
        test["bh_q"],
        MINIMUM,
    )
    if float(base_test["ci_lo"]) <= 0.0:
        verdict = "no_base_effect"
    elif state == "no_move":
        verdict = "interior" if gap_to_attribute(lead["test"]) else "no_gap_to_attribute"
    else:
        verdict = _INTERACTION_VERDICT[state]
    out["interaction"] = {"state": state, "verdict": verdict}
    lt = lead["test"]
    lead_state = mc.classify(lead["gap_mean"], lt["ci_lo"], lt["ci_hi"], lt["bh_q"], MINIMUM)
    out["lead"] = {"state": lead_state, "verdict": _LEAD_VERDICT[lead_state]}
    return out


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
            raise BoundaryNullError(msg)
        entry = LEVELS_BY_STEM.get(stem)
        if entry is None:
            msg = f"{log.name} names config {stem!r}, which this panel does not have"
            raise BoundaryNullError(msg)
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
                raise BoundaryNullError(msg)
            seen.add(key)
            lines.append(f"{arm} {level} {seed} {out}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Scoring ──────────────────────────────────────────────────────────────────────────────────
def score(campaign_dir: Path, out_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Score the gates, both gaps and the interaction, correct the two readings, then read them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(campaign_dir, out_dir / "manifest-boundary.txt", seeds)
    mp.require_complete(manifest, HALF, seeds, LEVELS)

    reports = {
        level: ops.score_level(manifest, HALF, level, out_dir / f"tmp-{level}", cell=CELL)
        for level in LEVELS
    }
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    gaps = {m: {lv: ops.wiring_gap(reports[lv], m) for lv in LEVELS} for m in metrics}
    splits = {m: ops.interaction(reports[CHEMICAL], reports[BOUNDARY], m) for m in metrics}
    # One family per metric: the interaction and the lead, every test computed before any is read.
    for m in metrics:
        ops.apply_family_correction(
            {
                "interaction": {m: {"interaction": splits[m]}},
                "lead": {m: {"interaction": gaps[m][BOUNDARY]}},
            },
        )
    gates = {
        level: ops.learning_gates(manifest, HALF, seeds, level, floor_level=level)
        for level in LEVELS
    }
    drift = ops.substrate_drift(manifest, HALF, seeds, LEVELS, floor_levels={s: s for s in LEVELS})
    drift["obligation_applies"] = False
    reading = mc.honour_drift(
        HALF,
        read_panel(
            gates,
            splits[PRIMARY_METRIC],
            gaps[PRIMARY_METRIC][BOUNDARY],
            gaps[PRIMARY_METRIC][CHEMICAL]["test"],
        ),
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
        "interaction": splits,
        "censoring_rule_choice": ops.choose_metric(
            {level: ops.censoring_rates(report) for level, report in reports.items()},
        ),
        "substrate_drift": drift,
        "reading": reading,
    }


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
            row += [_fmt(result["interaction"][m]["per_seed"].get(seed)) for m in metrics]
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
