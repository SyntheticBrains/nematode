"""Wiring premise - the wild-type wiring against its rewired null, on behaviours the animal performs.

Every wiring contrast in this project has run on the integrated C3 cell (food + predator +
thermotaxis, 2400 steps), and the connectome's measured deficit sits in the predator component.
This scores the same contrast on cells matched to behaviours the animal performs: the C1 klinotaxis
foraging cell, and the thermal-plus-foraging survival cell.

**Measured identically to the 029 ranking and the 034 control**: the metric is
``t7_continuous_ranking.plateau_tail`` (final-quarter plateau) and the statistics layer is
``weight_search_architecture_ranking.paired_seed_wilcoxon_bootstrap`` + ``bh_fdr`` - the same
functions ``connectome_structure_controls`` calls, with that control's verdict vocabulary and its
minimum-seeds threshold, so a result here is commensurable with the committed record by
construction rather than by care.

Registered structure (fixed before the campaign ran):

* Four arms per cell - ``wt_ppo``, ``rn_ppo``, ``wt_frozen``, ``rn_frozen`` - plus a descriptive
  ``mlp_ppo`` reference that no test may read.
* Eight tests, both cells corrected together under BH-FDR: the contrast, two learning gates and the
  untrained prior per cell.
* **The gates are read before the contrast.** A contrast against a null presupposes that the arm
  carrying the claim learned; where it did not, the cell yields ``no_learning`` and no wiring
  verdict is assigned.
* A ceiling clause and a registered minimum effect, both fixed in advance: a significant contrast
  below the minimum is named as such and licenses nothing on its own.

The klinotaxis cell is scored on plateau-tail full-clear success (%); the thermal cell is scored on
plateau-tail mean foods, because its satiety recipe (``satiety_gain_per_food: 0.2``) makes it a
survival cell rather than a collect-10 budget. Both metrics come from the same tail and both are
reported for every arm.

Usage::

    uv run python scripts/analysis/wiring_premise.py \
        --manifest <run-dir>/_manifest.txt --out <run-dir>/wiring_premise.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the committed metric + statistics layers verbatim - the same ones the 034 control calls.
from t7_continuous_ranking import plateau_tail
from weight_search_architecture_ranking import bh_fdr, paired_seed_wilcoxon_bootstrap

REPO = Path(__file__).resolve().parents[2]

CELLS = ("klinotaxis", "thermal")
TESTED_ARMS = ("wt_ppo", "rn_ppo", "wt_frozen", "rn_frozen")
REFERENCE_ARMS = ("mlp_ppo",)  # descriptive only - no test may read these
ARMS = (*TESTED_ARMS, *REFERENCE_ARMS)

# Scored metric per cell, fixed from the configs before any data existed. The thermal cell's
# `satiety_gain_per_food: 0.2` makes collect-10 not its task, so it is scored on the graded metric.
SCORED: dict[str, str] = {"klinotaxis": "success", "thermal": "foods"}

# Registered minimum effect on each cell's scored metric, beside significance: at n = 16 a paired
# rank test fires on the consistency of the sign rather than the size of the shift. The foods
# figure is the one I.3b registered for a foods-scored contrast.
MIN_EFFECT: dict[str, float] = {"klinotaxis": 5.0, "thermal": 0.5}

# Ceiling: both PPO arms at or above this full-clear mean means the cell cannot discriminate.
# Measured on full clears for both cells - it is the cell's exhaustion, not the scored metric's.
SATURATION_SUCCESS = 90.0

# 034's thresholds, carried so the two harnesses agree.
MIN_PAIRED_SEEDS = 2
MIN_SEEDS_FOR_SIGNIFICANCE = 5
SIG_Q = 0.05

# (test id, cell, arm a, arm b, role) - a one-sided paired test of `a - b` in each case.
FAMILY: tuple[tuple[str, str, str, str, str], ...] = (
    ("V1", "klinotaxis", "wt_ppo", "rn_ppo", "primary"),
    ("V2", "klinotaxis", "wt_ppo", "wt_frozen", "gate"),
    ("V3", "klinotaxis", "rn_ppo", "rn_frozen", "gate_null"),
    ("V4", "klinotaxis", "wt_frozen", "rn_frozen", "prior"),
    ("V5", "thermal", "wt_ppo", "rn_ppo", "secondary"),
    ("V6", "thermal", "wt_ppo", "wt_frozen", "gate"),
    ("V7", "thermal", "rn_ppo", "rn_frozen", "gate_null"),
    ("V8", "thermal", "wt_frozen", "rn_frozen", "prior"),
)

_CONTRAST_ROLES = ("primary", "secondary")


class ManifestError(ValueError):
    """A manifest line names a cell or arm the registered structure does not have."""


def load(manifest: Path) -> dict[str, dict[str, dict[int, tuple[float, float]]]]:
    """Return ``{cell: {arm: {seed: (success, foods)}}}`` from the campaign manifest.

    Each line is ``<cell> <arm> <seed> <out_path>``. Blank and ``#``-comment lines are skipped;
    a line whose cell or arm is not in the registered structure raises rather than being dropped,
    because a typo there would silently remove an arm from a paired test.
    """
    cells: dict[str, dict[str, dict[int, tuple[float, float]]]] = {}
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4 or not parts[2].isdigit():
            msg = f"malformed manifest line (expected `<cell> <arm> <seed> <out>`): {raw!r}"
            raise ManifestError(msg)
        cell, arm, seed, out_path = parts[0], parts[1], int(parts[2]), Path(parts[3])
        if cell not in CELLS:
            msg = f"unknown cell {cell!r} (expected one of {CELLS}): {raw!r}"
            raise ManifestError(msg)
        if arm not in ARMS:
            msg = f"unknown arm {arm!r} (expected one of {ARMS}): {raw!r}"
            raise ManifestError(msg)
        tail = plateau_tail(REPO / out_path)
        if tail is None:
            print(f"  WARN {cell}/{arm} seed {seed}: no parseable plateau in {out_path} - dropped")
            continue
        seeds = cells.setdefault(cell, {}).setdefault(arm, {})
        if seed in seeds:
            msg = f"duplicate entry for {cell}/{arm} seed {seed}: {raw!r}"
            raise ManifestError(msg)
        seeds[seed] = (float(tail[0]), float(tail[1]))
    return cells


def _scored(
    cells: dict[str, dict[str, dict[int, tuple[float, float]]]],
    cell: str,
    arm: str,
) -> dict[int, float]:
    """Per-seed values of the cell's scored metric for one arm."""
    index = 0 if SCORED[cell] == "success" else 1
    return {seed: vals[index] for seed, vals in cells.get(cell, {}).get(arm, {}).items()}


def _mean(values: dict[int, float]) -> float:
    return float(np.mean(list(values.values()))) if values else float("nan")


def contrasts(cells: dict[str, dict[str, dict[int, tuple[float, float]]]]) -> list[dict[str, Any]]:
    """Compute the eight registered tests, corrected together under BH-FDR.

    Every test is computed before any is read, so the correction is over the whole family and the
    gates cannot be chosen after seeing the contrast.
    """
    rows: list[dict[str, Any]] = []
    for test, cell, arm_a, arm_b, role in FAMILY:
        a, b = _scored(cells, cell, arm_a), _scored(cells, cell, arm_b)
        common = sorted(set(a) & set(b))
        row: dict[str, Any] = {
            "test": test,
            "cell": cell,
            "contrast": f"{arm_a} - {arm_b}",
            "role": role,
            "metric": SCORED[cell],
            "n_common": len(common),
            "complete": len(common) >= MIN_PAIRED_SEEDS,
        }
        if common:
            deltas = [a[s] - b[s] for s in common]
            stats = paired_seed_wilcoxon_bootstrap(deltas)
            row.update(
                {
                    "mean_delta": stats["mean_delta"],
                    "ci_lo": stats["ci_lo"],
                    "ci_hi": stats["ci_hi"],
                    "wilcoxon_p": stats["wilcoxon_p"],
                    "positive_seeds": sum(1 for d in deltas if d > 0),
                },
            )
        rows.append(row)

    testable = [r for r in rows if "wilcoxon_p" in r]
    for row, q in zip(testable, bh_fdr([r["wilcoxon_p"] for r in testable]), strict=True):
        row["bh_q"] = q
    return rows


def _significant(row: dict[str, Any] | None) -> bool:
    """One-sided significance in the tested direction, after correction."""
    return bool(row and row.get("bh_q", 1.0) < SIG_Q and row.get("mean_delta", 0.0) > 0)


def verdict(
    cells: dict[str, dict[str, dict[int, tuple[float, float]]]],
    rows: list[dict[str, Any]],
    cell: str,
) -> dict[str, Any]:
    """Assign one cell's registered verdict, in order, with the gates read before the contrast."""
    by_role = {r["role"]: r for r in rows if r["cell"] == cell}
    contrast = by_role.get("primary") or by_role.get("secondary")
    gate = by_role.get("gate")

    missing = [arm for arm in TESTED_ARMS if not cells.get(cell, {}).get(arm)]
    if missing or contrast is None or gate is None or not contrast["complete"]:
        return {
            "verdict": "insufficient_seeds",
            "missing_arms": missing,
            "n_common": contrast["n_common"] if contrast else 0,
        }

    underpowered = contrast["n_common"] < MIN_SEEDS_FOR_SIGNIFICANCE

    # 1. The gate. A contrast against a null presupposes the claim-carrying arm learned.
    if not _significant(gate):
        return {
            "verdict": "no_learning",
            "gate_test": gate["test"],
            "gate_delta": gate.get("mean_delta"),
            "gate_q": gate.get("bh_q"),
            "note": "the wild type does not beat its own frozen floor; this is a finding about the "
            "platform and licenses nothing about the wiring",
        }

    # 2. The ceiling, on full clears, whatever the cell's scored metric is.
    wt_clear = _mean({s: v[0] for s, v in cells[cell]["wt_ppo"].items()})
    rn_clear = _mean({s: v[0] for s, v in cells[cell]["rn_ppo"].items()})
    if wt_clear >= SATURATION_SUCCESS and rn_clear >= SATURATION_SUCCESS:
        return {
            "verdict": "saturated",
            "wt_full_clear": wt_clear,
            "rn_full_clear": rn_clear,
            "remedy": "re-run this cell once at target_foods_to_collect 20; do not adjust the recipe",
        }

    # 3. The contrast, in 034's vocabulary, with the registered minimum effect beside significance.
    delta, ci_lo, ci_hi = contrast["mean_delta"], contrast["ci_lo"], contrast["ci_hi"]
    minimum = MIN_EFFECT[cell]
    if _significant(contrast):
        name = "specific_wiring" if delta >= minimum else "below_min_effect"
    elif ci_hi < 0.0:
        name = "rewired_beats_wildtype"
    elif ci_lo <= 0.0 <= ci_hi:
        name = "degree_statistics"
    else:
        name = "inconclusive"

    return {
        "verdict": name,
        "n_common": contrast["n_common"],
        "underpowered": underpowered,
        "metric": SCORED[cell],
        "mean_delta": delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "bh_q": contrast["bh_q"],
        "min_effect": minimum,
        "meets_min_effect": delta >= minimum,
    }


def analyse(cells: dict[str, dict[str, dict[int, tuple[float, float]]]], out: dict) -> None:
    """Print and record the per-arm table, the registered family and the verdict per cell."""
    rows = contrasts(cells)
    out["family"] = rows
    out["per_arm"] = {}
    out["verdicts"] = {}

    for cell in CELLS:
        print("\n" + "=" * 78)
        print(f"WIRING PREMISE - {cell} cell, scored on plateau-tail {SCORED[cell]}")
        print("=" * 78)
        out["per_arm"][cell] = {}
        for arm in ARMS:
            per_seed = cells.get(cell, {}).get(arm, {})
            if not per_seed:
                continue
            clears = {s: v[0] for s, v in per_seed.items()}
            foods = {s: v[1] for s, v in per_seed.items()}
            out["per_arm"][cell][arm] = {
                "n": len(per_seed),
                "full_clear_mean": _mean(clears),
                "foods_mean": _mean(foods),
                "per_seed": {s: list(v) for s, v in sorted(per_seed.items())},
                "reference_only": arm in REFERENCE_ARMS,
            }
            tag = "  (reference, no test reads this)" if arm in REFERENCE_ARMS else ""
            print(
                f"  {arm:10} n={len(per_seed):2}  full-clear {_mean(clears):6.2f}%  "
                f"foods {_mean(foods):5.2f}{tag}",
            )

        print()
        for row in [r for r in rows if r["cell"] == cell]:
            if "mean_delta" not in row:
                print(f"  {row['test']} {row['contrast']:24} INCOMPLETE (n={row['n_common']})")
                continue
            print(
                f"  {row['test']} {row['contrast']:24} d={row['mean_delta']:+7.2f}  "
                f"CI[{row['ci_lo']:+7.2f},{row['ci_hi']:+7.2f}]  q={row['bh_q']:.3f}  "
                f"+{row['positive_seeds']}/{row['n_common']}  [{row['role']}]",
            )

        result = verdict(cells, rows, cell)
        out["verdicts"][cell] = result
        print("-" * 78)
        print(f"  VERDICT ({cell}): {result['verdict'].upper().replace('_', '-')}")
        if result["verdict"] == "below_min_effect":
            print(
                f"    significant at q={result['bh_q']:.3f} but d={result['mean_delta']:+.2f} is "
                f"below the registered minimum of {result['min_effect']:+.2f} - licenses nothing.",
            )
        if result.get("underpowered"):
            print(
                f"    NOTE: n={result['n_common']} paired seeds - below {MIN_SEEDS_FOR_SIGNIFICANCE}"
                " the one-sided signed-rank floor exceeds 0.05 and significance is unreachable.",
            )


def main() -> None:
    """Load the manifest, compute, print and write the wiring-premise analysis."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True, help="<cell> <arm> <seed> <out> a line")
    ap.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    args = ap.parse_args()

    cells = load(args.manifest)
    out: dict = {}
    analyse(cells, out)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
