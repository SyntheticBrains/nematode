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

# Reuse the committed metric + statistics layers verbatim - the same ones the 034 control calls.
import connectome_structure_efficiency as efficiency
import numpy as np
from t7_continuous_ranking import plateau_tail
from weight_search_architecture_ranking import bh_fdr, paired_seed_wilcoxon_bootstrap

REPO = Path(__file__).resolve().parents[2]

CELLS = ("klinotaxis", "thermal", "hard_food")
TESTED_ARMS = ("wt_ppo", "rn_ppo", "wt_frozen", "rn_frozen")
REFERENCE_ARMS = ("mlp_ppo",)  # descriptive only - no test may read these
ARMS = (*TESTED_ARMS, *REFERENCE_ARMS)

# Scored metric per cell, fixed from the configs before any data existed. The thermal cell's
# `satiety_gain_per_food: 0.2` makes collect-10 not its task, so it is scored on the graded metric.
SCORED: dict[str, str] = {"klinotaxis": "success", "thermal": "foods", "hard_food": "success"}

# Registered minimum effect on each cell's scored metric, beside significance: at n = 16 a paired
# rank test fires on the consistency of the sign rather than the size of the shift. The foods
# figure is the one I.3b registered for a foods-scored contrast.
MIN_EFFECT: dict[str, float] = {"klinotaxis": 5.0, "thermal": 0.5, "hard_food": 5.0}

# Ceiling: both PPO arms at or above this full-clear mean means the cell cannot discriminate on the
# peak axis. Measured on full clears for both cells - it is the cell's exhaustion, not the scored
# metric's.
SATURATION_SUCCESS = 90.0

# Amendment 2026-09-12 (recorded in the change's design and in the launch record). The registered
# saturation remedy was applied once, on disjoint pilot seeds, and did not unsaturate either cell:
# both wirings reach 100.00% full clear on the klinotaxis cell at `target_foods_to_collect: 20`,
# a contrast of exactly zero, and 19.80 against 19.69 foods of 20 on the thermal cell. The peak axis
# therefore cannot answer the question on these cells, and the primary moves to the **efficiency**
# axis, read through the committed `connectome_structure_efficiency` harness - 034's own follow-up,
# its four-metric BH-FDR family and its verdict rule unchanged. The learning gates stay on the peak
# axis, where "did this arm learn at all" is what they ask.
# Cells whose *efficiency* contrast decides their own campaign's verdict. The klinotaxis cell is
# never one: it saturates, and 057's pilot showed both wirings clearing 100%. This set feeds a
# printed label only - no committed number depends on it.
PRIMARY_CELLS = frozenset({"thermal", "hard_food"})
EFFICIENCY_ARMS = {"wt_ppo": efficiency._WILD, "rn_ppo": efficiency._REWIRED}

# Minimum effect on the efficiency primary, registered with the amendment and before the registered
# seeds ran: a significant contrast that shortens time-to-competence by less than this is named and
# licenses nothing. The pilot's direction, at an unresolvable n = 4, was about 48%.
MIN_EFFICIENCY_GAIN = 0.20

# The lower edge of the band the primary metric needs. `episodes_to_30pct_success` returns the horizon
# for a seed that never crosses the threshold, so a cell too hard to reach it censors the metric and
# reads exactly like a null. Below this fraction of seeds crossing, in either arm, the contrast is
# recorded as materially censored. Set at 0.8 rather than 1.0 because a single non-crossing seed is
# not censoring: 057's committed thermal panel sits at 98% and 100%.
CROSSING_FLOOR = 0.8

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
    # V.3: the hard food-only cell - difficulty raised by the episode budget, no temperature.
    ("V9", "hard_food", "wt_ppo", "rn_ppo", "primary"),
    ("V10", "hard_food", "wt_ppo", "wt_frozen", "gate"),
    ("V11", "hard_food", "rn_ppo", "rn_frozen", "gate_null"),
    ("V12", "hard_food", "wt_frozen", "rn_frozen", "prior"),
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


def efficiency_contrast(manifest: Path, cell: str, tmp_dir: Path) -> dict[str, Any] | None:
    """Score one cell's wild-vs-rewired contrast on the committed efficiency harness.

    Writes the two PPO arms of ``cell`` out as that harness's own ``<arm> <seed> <out>`` manifest
    and calls it unchanged, so the four metrics, the BH-FDR family and the verdict rule are 034's.
    Returns ``None`` when the cell has no paired PPO arms in the manifest.
    """
    lines = []
    for raw in manifest.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) == 4 and parts[0] == cell and parts[1] in EFFICIENCY_ARMS:
            lines.append(f"{EFFICIENCY_ARMS[parts[1]]} {parts[2]} {parts[3]}")
    if not lines:
        return None

    cell_manifest = tmp_dir / f"_efficiency_{cell}.txt"
    cell_manifest.write_text("\n".join(lines) + "\n")
    report = efficiency.analyse(cell_manifest)

    return apply_min_effect(report)


def apply_min_effect(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the registered minimum effect to an efficiency report, in place.

    The gain is the shortening of time-to-competence as a fraction of the null's own time - the most
    direct reading of "learns faster". A verdict that is significant but under the registered
    minimum becomes ``below_min_effect``, which names the result and licenses nothing on its own.
    """
    speed = report["metrics"]["episodes_to_30pct_success"]
    rewired_mean = speed["rewired_mean"]
    gain = speed["wild_minus_rewired_oriented"] / rewired_mean if rewired_mean > 0 else 0.0
    report["episodes_to_competence_gain"] = gain
    report["min_gain"] = MIN_EFFICIENCY_GAIN
    report["meets_min_effect"] = gain >= MIN_EFFICIENCY_GAIN
    if report["verdict"] == "specific_wiring_efficiency" and not report["meets_min_effect"]:
        report["verdict"] = "below_min_effect"
    return report


def crossing_rate(report: dict[str, Any], arm: str) -> float:
    """Fraction of seeds whose arm actually crossed the 30% threshold the primary metric needs.

    `episodes_to_30pct_success` returns the horizon for a seed that never crosses, so a cell too hard
    to reach the threshold censors the metric at the horizon for every seed and reads exactly like a
    null. The rate is reported so that case is visible rather than inferred.
    """
    horizon = report["horizon_episodes"]
    per_seed = report["per_seed"][arm]
    crossed = sum(1 for m in per_seed.values() if m["episodes_to_30pct_success"] < horizon)
    return crossed / len(per_seed) if per_seed else float("nan")


def _print_efficiency(cell: str, report: dict[str, Any]) -> None:
    """Print one cell's efficiency table, its crossing rates and its verdict."""
    print(f"\n  EFFICIENCY axis (034's harness, n={report['n_paired_seeds']} paired):")
    for name, entry in report["metrics"].items():
        print(
            f"    {name:34} wild {entry['wild_mean']:8.2f}  rewired {entry['rewired_mean']:8.2f}  "
            f"d={entry['wild_minus_rewired_oriented']:+8.2f}  q={entry['bh_fdr_q']:.3f}  "
            f"wild-better {entry['wild_better_seeds']}/{report['n_paired_seeds']}",
        )
    tag = " (PRIMARY)" if cell in PRIMARY_CELLS else ""
    print(
        f"    time-to-competence gain {report['episodes_to_competence_gain']:+.1%} against a "
        f"registered minimum of {report['min_gain']:+.0%}",
    )
    wild_rate = crossing_rate(report, efficiency._WILD)
    rewired_rate = crossing_rate(report, efficiency._REWIRED)
    censored = min(wild_rate, rewired_rate) < CROSSING_FLOOR
    flag = f"  <-- below the {CROSSING_FLOOR:.0%} floor: materially censored" if censored else ""
    print(f"    crossed the 30% threshold: wild {wild_rate:.0%}, rewired {rewired_rate:.0%}{flag}")
    print(f"  VERDICT ({cell}, efficiency{tag}): {report['verdict'].upper().replace('_', '-')}")


def analyse(
    cells: dict[str, dict[str, dict[int, tuple[float, float]]]],
    out: dict,
    manifest: Path | None = None,
) -> None:
    """Print and record the per-arm table, the registered family and the verdict per cell.

    With ``manifest``, the efficiency axis is scored beside the peak axis through the committed
    034 harness; the primary verdict is that axis on each cell in :data:`PRIMARY_CELLS`.
    """
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

        if manifest is not None:
            report = efficiency_contrast(manifest, cell, manifest.parent)
            if report is not None:
                out.setdefault("efficiency", {})[cell] = report
                _print_efficiency(cell, report)


def main() -> None:
    """Load the manifest, compute, print and write the wiring-premise analysis."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True, help="<cell> <arm> <seed> <out> a line")
    ap.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    args = ap.parse_args()

    cells = load(args.manifest)
    out: dict = {}
    analyse(cells, out, args.manifest)
    if args.out:
        # Trailing newline so a regenerated record does not fail the end-of-file hook,
        # matching `connectome_structure_efficiency`.
        args.out.write_text(json.dumps(out, indent=2, default=str) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
