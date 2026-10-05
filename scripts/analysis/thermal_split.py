#!/usr/bin/env python
"""A.6t's follow-up: block V's thermal gap against a null with the wild type's gap junctions.

At block V's thermal point (target 20) every learning arm saturates above the 90% bar, so the
null-strength panel there was unreadable. This panel runs the gap-only split at target 35, the
lowest target a gate-only pilot found readable, under PPO at block V's committed point otherwise
(edge-order draw, pooled readout, depth 4), seeds 513-640.

Three wirings, each learning and frozen: the wild type, the current degree-preserving null, and the
gap-held null, which has the current null's chemical graph exactly and the wild type's gap junctions.
Two levels share the wild-type runs. **Two registered readings**, corrected together:

* ``split = gap(gap_held) - gap(full)``, paired by seed: how much holding the gap junctions moves the
  wiring gap. Positive when the wild type stands better against the gap-held null.
* ``gap(gap_held)`` itself: whether the wild type leads a null with its gap junctions at all.

Both are read against one registered minimum: 2/3 of A.1's thermal effect, scaled to this target by
the wild type's own ``auc_success`` ratio between the targets.

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
CELL = "thermal"
TARGET = 35
# Fresh: A.6t ended at 512. The target pilot ran on 1001-1004.
SEEDS: tuple[int, ...] = tuple(range(513, 641))
FULL, GAP_HELD = "full", "gap_held"
LEVELS: tuple[str, ...] = (FULL, GAP_HELD)
ARMS = mp.ARMS

_T = "connectomeppo_small_continuous2d_thermal_klinotaxis"
_WILD = {"wt_learn": f"{_T}_t{TARGET}", "wt_frozen": f"{_T}_frozen_t{TARGET}"}
STEMS: dict[str, dict[str, str]] = {
    FULL: {
        **_WILD,
        "rn_learn": f"{_T}_rewired_null_t{TARGET}",
        "rn_frozen": f"{_T}_rewired_null_frozen_t{TARGET}",
    },
    GAP_HELD: {
        **_WILD,
        "rn_learn": f"{_T}_rewired_gap_held_null_t{TARGET}",
        "rn_frozen": f"{_T}_rewired_gap_held_null_frozen_t{TARGET}",
    },
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
# A.1's thermal effect at target 20 on the primary (Logbook 070's thermal `baseline_gap_mean`), and
# the wild type's mean `auc_success` on A.1's 32 seeds at target 20 and on the pilot's 4 at target
# 35. The ratio uses the wild type alone, so no wiring gap informs the minimum.
A1_EFFECT_T20 = 0.08289583333333334
WILD_AUC_T20 = 0.720760
WILD_AUC_T35 = 0.312333
REFERENCE_EFFECT = A1_EFFECT_T20 * WILD_AUC_T35 / WILD_AUC_T20
MINIMUM = mc.MINIMUM_FRACTION * abs(REFERENCE_EFFECT)

_SPLIT_VERDICT = {
    "move_null": "gap_junctions",
    "below": "partial",
    "no_move": "not_gap_junctions",
    "move_wt": "opposite",
    "unresolved": "unresolved",
}
_LEAD_VERDICT = {
    "move_wt": "lead_remains",
    "below": "lead_below_minimum",
    "no_move": "no_lead",
    "move_null": "null_leads",
    "unresolved": "unresolved",
}


class ThermalSplitError(ValueError):
    """The panel on disk is not the panel this module scores."""


def read_panel(
    gates: dict[str, Any],
    split: dict[str, Any],
    held_gap: dict[str, Any],
) -> dict[str, Any]:
    """Gates first, then each registered reading's state and verdict."""
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
    for name, mean, test, table in (
        ("split", split["interaction_mean"], split["test"], _SPLIT_VERDICT),
        ("lead", held_gap["gap_mean"], held_gap["test"], _LEAD_VERDICT),
    ):
        state = mc.classify(mean, test["ci_lo"], test["ci_hi"], test["bh_q"], MINIMUM)
        out[name] = {"state": state, "verdict": table[state]}
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
            raise ThermalSplitError(msg)
        entry = LEVELS_BY_STEM.get(stem)
        if entry is None:
            msg = f"{log.name} names config {stem!r}, which this panel does not have"
            raise ThermalSplitError(msg)
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
                raise ThermalSplitError(msg)
            seen.add(key)
            lines.append(f"{arm} {level} {seed} {out}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Scoring ──────────────────────────────────────────────────────────────────────────────────
def score(campaign_dir: Path, out_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Score the gates, both gaps and the split, correct the two readings, then read them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(campaign_dir, out_dir / "manifest-thermal-split.txt", seeds)
    mp.require_complete(manifest, HALF, seeds, LEVELS)

    reports = {
        level: ops.score_level(manifest, HALF, level, out_dir / f"tmp-{level}", cell=CELL)
        for level in LEVELS
    }
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    gaps = {m: {lv: ops.wiring_gap(reports[lv], m) for lv in LEVELS} for m in metrics}
    splits = {m: ops.interaction(reports[FULL], reports[GAP_HELD], m) for m in metrics}
    # One family per metric: the split and the lead, every test computed before any is read.
    for m in metrics:
        ops.apply_family_correction(
            {
                "split": {m: {"interaction": splits[m]}},
                "lead": {m: {"interaction": gaps[m][GAP_HELD]}},
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
        read_panel(gates, splits[PRIMARY_METRIC], gaps[PRIMARY_METRIC][GAP_HELD]),
        drift,
    )
    return {
        "cell": CELL,
        "target": TARGET,
        "half": HALF,
        "seeds": list(seeds),
        "primary_metric": PRIMARY_METRIC,
        "beside_metric": BESIDE_METRIC,
        "gates": gates,
        "gaps": gaps,
        "split": splits,
        "censoring_rule_choice": ops.choose_metric(
            {level: ops.censoring_rates(report) for level, report in reports.items()},
        ),
        "substrate_drift": drift,
        "reading": reading,
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per seed: every arm's plateau and floor, both gaps, the split."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    header = ["seed"]
    for level in LEVELS:
        header += [f"{level}_{c}" for c in ("wt_plateau", "wt_floor", "rn_plateau", "rn_floor")]
        header += [f"{level}_gap_{m}" for m in metrics]
    header += [f"split_{m}" for m in metrics]
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
            row += [_fmt(result["split"][m]["per_seed"].get(seed)) for m in metrics]
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
