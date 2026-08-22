"""Aggregator for the transgenerational pilot/full campaign.

Reads ``per_gen_choice_index.csv`` (produced by
``transgenerational_per_gen_eval.py``) and produces:

1. **Per-generation retention table** per arm x seed (mean choice index
   across episodes for each generation).
2. **Decision-gate evaluation** per seed:
   - F1 ≥ 0.40 x F0
   - F2 ≥ 0.25 x F0
   - F3 ≥ 0.15 x F0
   - Monotone non-increasing: F0 ≥ F1 ≥ F2 ≥ F3
3. **Cross-seed verdict**:
   - **GO** iff ≥2 seeds pass all four checks
   - **PIVOT** iff exactly 1 seed passes
   - **STOP** iff 0 seeds pass
4. **TEI-on vs TEI-off retention comparison** (paired-arm signal).
5. **Markdown summary** suitable for inclusion in the logbook.

Outputs (under ``--output-dir``):
  - ``retention_table.csv`` (per arm x seed x generation: mean choice index)
  - ``decision_gate.csv`` (per seed: gate pass/fail per check, overall verdict)
  - ``summary.md`` (human-readable markdown summary)

Usage:
  scripts/campaigns/aggregate_m6_pilot.py \
      --per-gen-csv evaluations/m6_transgenerational/per_gen_choice_index.csv \
      --output-dir evaluations/m6_transgenerational
"""
# pragma: no cover

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

# These aggregators are executed directly (``uv run python scripts/campaigns/...``),
# so the repo root is not on ``sys.path`` and ``scripts.campaigns`` is not
# importable; the tests load them by file path for the same reason. Put the repo
# root on the path before importing the shared helpers.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.campaigns._common import (  # noqa: E402
    GATE_F1_RATIO,
    GATE_F2_RATIO,
    GATE_F3_RATIO,
    VERDICT_GO_MIN_SEEDS,
    VERDICT_PIVOT_MIN_SEEDS,
    build_survival_table,
    evaluate_decision_gate_one_seed,
    load_f0_training_fitness_per_seed,
    mean,
    read_per_gen_csv,
)

logger = logging.getLogger(__name__)


def build_retention_table(rows: list[dict]) -> dict[tuple[str, int, int], float]:
    """Aggregate per-episode rows into ``(arm, seed, generation) -> mean choice_index``.

    Averages across episodes for each (arm, seed, generation) triple.
    """
    bucket: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["arm"]), int(row["seed"]), int(row["generation"]))
        bucket[key].append(float(row["choice_index"]))
    return {k: mean(v) for k, v in bucket.items()}


def aggregate_verdict(seed_evaluations: list[dict]) -> str:
    """Aggregate per-seed evaluations into a cross-seed verdict.

    ``GO`` iff ≥``VERDICT_GO_MIN_SEEDS`` seeds pass; ``PIVOT`` iff
    exactly ``VERDICT_PIVOT_MIN_SEEDS`` seed passes; ``STOP`` otherwise.
    Skipped seeds (incomplete generations) count as failures.
    """
    pass_count = sum(1 for s in seed_evaluations if s["overall_pass"])
    if pass_count >= VERDICT_GO_MIN_SEEDS:
        return "GO"
    if pass_count >= VERDICT_PIVOT_MIN_SEEDS:
        return "PIVOT"
    return "STOP"


def _write_retention_csv(
    retention: dict[tuple[str, int, int], float],
    path: Path,
) -> None:
    """Write the retention table to CSV: ``arm, seed, generation, mean_choice_index``."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("arm", "seed", "generation", "mean_choice_index"))
        for (arm, seed, gen), value in sorted(retention.items()):
            writer.writerow((arm, seed, gen, f"{value:.6f}"))


def _write_decision_gate_csv(seed_evaluations: list[dict], path: Path) -> None:
    """Write the decision-gate per-seed evaluation to CSV."""
    fieldnames = (
        "arm",
        "seed",
        "f0",
        "f1",
        "f2",
        "f3",
        "f1_ratio_pass",
        "f2_ratio_pass",
        "f3_ratio_pass",
        "monotone_pass",
        "overall_pass",
        "skipped",
        "skip_reason",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in seed_evaluations:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_summary_md(
    *,
    seed_evaluations_per_arm: dict[str, list[dict]],
    verdict_per_arm: dict[str, str],
    retention: dict[tuple[str, int, int], float],
    path: Path,
) -> None:
    """Write a human-readable markdown summary."""
    lines: list[str] = []
    lines.append("# Transgenerational pilot aggregator — decision-gate summary\n")

    for arm in sorted(verdict_per_arm):
        verdict = verdict_per_arm[arm]
        lines.append(f"## Arm: `{arm}` — verdict: **{verdict}**\n")
        lines.append(
            "| seed | F0 | F1 | F2 | F3 | F1≥40%xF0 | F2≥25%xF0 | F3≥15%xF0 | monotone | overall |",
        )
        lines.append(
            "|------|----|----|----|----|-----------|-----------|-----------|----------|---------|",
        )
        for s in seed_evaluations_per_arm.get(arm, []):
            if s["skipped"]:
                lines.append(
                    f"| {s['seed']} | (incomplete) |  |  |  |  |  |  |  | **skipped** |",
                )
                continue
            lines.append(
                f"| {s['seed']} | {s['f0']:.3f} | {s['f1']:.3f} | "
                f"{s['f2']:.3f} | {s['f3']:.3f} | "
                f"{'✓' if s['f1_ratio_pass'] else '✗'} | "
                f"{'✓' if s['f2_ratio_pass'] else '✗'} | "
                f"{'✓' if s['f3_ratio_pass'] else '✗'} | "
                f"{'✓' if s['monotone_pass'] else '✗'} | "
                f"{'**PASS**' if s['overall_pass'] else 'FAIL'} |",
            )
        lines.append("")

    # Paired-arm retention comparison (only meaningful if BOTH arms present).
    arms = sorted({a for (a, _, _) in retention})
    if "tei_on" in arms and "tei_off" in arms:
        lines.append("## TEI-on vs TEI-off paired-arm retention\n")
        lines.append("Mean choice index per generation (averaged across seeds):\n")
        lines.append("| arm | F0 | F1 | F2 | F3 |")
        lines.append("|-----|----|----|----|----|")
        for arm in ("tei_on", "tei_off"):
            per_gen_means: dict[int, list[float]] = defaultdict(list)
            for (a, _seed, gen), v in retention.items():
                if a == arm:
                    per_gen_means[gen].append(v)
            gen_strs = [
                f"{mean(per_gen_means[g]):.3f}" if per_gen_means.get(g) else "—"
                for g in (0, 1, 2, 3)
            ]
            lines.append(
                f"| {arm} | {gen_strs[0]} | {gen_strs[1]} | {gen_strs[2]} | {gen_strs[3]} |",
            )
        lines.append("")
        lines.append(
            "Substrate is the only cross-arm difference (pairing validator enforces "
            "`enabled=true ⇔ inheritance=transgenerational`, `enabled=false ⇔ "
            "inheritance=none`). Any F1+ retention in `tei_on` but absent in `tei_off` "
            "is attributable to the substrate.",
        )
        lines.append("")

    lines.append("## Gate thresholds\n")
    lines.append(f"- F1 ≥ {GATE_F1_RATIO:.0%} x F0")
    lines.append(f"- F2 ≥ {GATE_F2_RATIO:.0%} x F0")
    lines.append(f"- F3 ≥ {GATE_F3_RATIO:.0%} x F0")
    lines.append("- Monotone non-increasing: F0 ≥ F1 ≥ F2 ≥ F3\n")
    lines.append(
        f"- **GO** iff ≥{VERDICT_GO_MIN_SEEDS} seeds pass; "
        f"**PIVOT** iff exactly {VERDICT_PIVOT_MIN_SEEDS}; **STOP** otherwise.",
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:  # noqa: C901 - linear orchestration; nested loops are clearer than helpers
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Transgenerational pilot aggregator. Reads per_gen_choice_index.csv, "
            "produces a per-seed retention table, decision-gate evaluation, paired-arm "
            "retention comparison, and a markdown summary."
        ),
    )
    parser.add_argument(
        "--per-gen-csv",
        type=Path,
        required=True,
        help="Path to per_gen_choice_index.csv (output of transgenerational_per_gen_eval.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write retention_table.csv, decision_gate.csv, summary.md into.",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=None,
        help=(
            "Campaign output root (e.g. ``evolution_results/m6_transgenerational``). "
            "When provided, the aggregator loads each (arm, seed)'s F0 training-time "
            "fitness from ``per_gen_elites.jsonl`` and uses it as the F0 baseline for "
            "the survival_rate decision gate, replacing the post-hoc evaluator's "
            "untrained-brain F0 measurement. This is the biologically-correct gate "
            "baseline: F0 retention should be measured against the substrate's source "
            "(the trained F0 elite), not against an untrained brain."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_per_gen_csv(args.per_gen_csv)
    if not rows:
        print(f"No rows in {args.per_gen_csv}. Nothing to aggregate.")
        return 1

    retention = build_retention_table(rows)
    survival = build_survival_table(rows)  # higher dynamic range than choice_index

    # Per-arm seed-by-seed gate evaluation on BOTH metrics.
    arms = sorted({k[0] for k in retention})
    seeds = sorted({k[1] for k in retention})

    # F0 baseline override: when --campaign-root is provided, load each
    # (arm, seed)'s F0 training-time fitness from per_gen_elites.jsonl
    # and use it as the F0 baseline for the survival_rate gate.
    f0_override: dict[tuple[str, int], float] | None = None
    if args.campaign_root is not None and args.campaign_root.is_dir():
        f0_override = load_f0_training_fitness_per_seed(
            args.campaign_root,
            arms=arms,
        )
        print(f"\nF0 baseline override loaded for {len(f0_override)} (arm, seed) pairs.")

    seed_evaluations_all: list[dict] = []
    seed_evaluations_per_arm: dict[str, list[dict]] = defaultdict(list)
    verdict_per_arm: dict[str, str] = {}
    for arm in arms:
        per_arm = [
            evaluate_decision_gate_one_seed(retention=retention, arm=arm, seed=seed)
            for seed in seeds
            if (arm, seed, 0) in retention
        ]
        seed_evaluations_per_arm[arm] = per_arm
        seed_evaluations_all.extend(per_arm)
        # Verdict only meaningful for TEI-on (TEI-off arm shouldn't satisfy
        # the gates by construction). Compute for all arms for symmetry,
        # but the logbook reads the tei_on row.
        verdict_per_arm[arm] = aggregate_verdict(per_arm)

    # Same gate logic, applied to the survival_rate metric instead of
    # choice_index. Only emitted if the CSV actually had a
    # termination_reason column (older CSVs predate it). The
    # ``f0_override`` (if provided) replaces the post-hoc F0 row with
    # the training-time F0 fitness — see ``load_f0_training_fitness_per_seed``
    # docstring for why this is the biologically-correct baseline.
    survival_evaluations_per_arm: dict[str, list[dict]] = defaultdict(list)
    survival_verdict_per_arm: dict[str, str] = {}
    if survival:
        for arm in arms:
            per_arm_surv = [
                evaluate_decision_gate_one_seed(
                    retention=survival,
                    arm=arm,
                    seed=seed,
                    f0_baseline_override=f0_override,
                )
                for seed in seeds
                if (arm, seed, 0) in survival
            ]
            survival_evaluations_per_arm[arm] = per_arm_surv
            survival_verdict_per_arm[arm] = aggregate_verdict(per_arm_surv)

    _write_retention_csv(retention, args.output_dir / "retention_table.csv")
    _write_decision_gate_csv(seed_evaluations_all, args.output_dir / "decision_gate.csv")
    if survival:
        _write_retention_csv(survival, args.output_dir / "survival_retention_table.csv")
        all_surv_evals = [
            s for arm_evals in survival_evaluations_per_arm.values() for s in arm_evals
        ]
        _write_decision_gate_csv(all_surv_evals, args.output_dir / "survival_decision_gate.csv")
    _write_summary_md(
        seed_evaluations_per_arm=seed_evaluations_per_arm,
        verdict_per_arm=verdict_per_arm,
        retention=retention,
        path=args.output_dir / "summary.md",
    )

    print("\nchoice_index verdicts per arm:")
    for arm in arms:
        print(f"  {arm}: {verdict_per_arm[arm]}")
    if survival:
        print("\nsurvival_rate verdicts per arm:")
        for arm in arms:
            print(f"  {arm}: {survival_verdict_per_arm.get(arm, 'n/a')}")
    print(f"\nArtefacts written to {args.output_dir}/")
    print("  - retention_table.csv (choice_index)")
    print("  - decision_gate.csv (choice_index)")
    if survival:
        print("  - survival_retention_table.csv (survival_rate)")
        print("  - survival_decision_gate.csv (survival_rate)")
    print("  - summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
