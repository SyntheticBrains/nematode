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
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# Decision-gate thresholds — locked by the transgenerational-memory OpenSpec change.
GATE_F1_RATIO = 0.40
GATE_F2_RATIO = 0.25
GATE_F3_RATIO = 0.15

# Cross-seed verdict thresholds.
VERDICT_GO_MIN_SEEDS = 2  # ≥2 of N → GO
VERDICT_PIVOT_MIN_SEEDS = 1  # exactly 1 → PIVOT (anything below → STOP)


def _read_per_gen_csv(path: Path) -> list[dict]:
    """Read the per-gen choice-index CSV into a list of dict rows."""
    if not path.exists():
        msg = f"per-gen CSV not found: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_retention_table(rows: list[dict]) -> dict[tuple[str, int, int], float]:
    """Aggregate per-episode rows into ``(arm, seed, generation) -> mean choice_index``.

    Averages across episodes for each (arm, seed, generation) triple.
    """
    bucket: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["arm"]), int(row["seed"]), int(row["generation"]))
        bucket[key].append(float(row["choice_index"]))
    return {k: _mean(v) for k, v in bucket.items()}


def evaluate_decision_gate_one_seed(
    *,
    retention: dict[tuple[str, int, int], float],
    arm: str,
    seed: int,
) -> dict:
    """Evaluate the decision gate for one (arm, seed).

    Returns a dict with:
      - ``f0``..``f3``: per-gen mean choice indices (or None if missing)
      - ``f1_ratio_pass``: bool — F1 ≥ ``GATE_F1_RATIO`` x F0
      - ``f2_ratio_pass``: bool
      - ``f3_ratio_pass``: bool
      - ``monotone_pass``: bool — F0 ≥ F1 ≥ F2 ≥ F3
      - ``overall_pass``: bool — AND of all four
    """
    f0 = retention.get((arm, seed, 0))
    f1 = retention.get((arm, seed, 1))
    f2 = retention.get((arm, seed, 2))
    f3 = retention.get((arm, seed, 3))
    if any(v is None for v in (f0, f1, f2, f3)):
        return {
            "arm": arm,
            "seed": seed,
            "f0": f0,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f1_ratio_pass": False,
            "f2_ratio_pass": False,
            "f3_ratio_pass": False,
            "monotone_pass": False,
            "overall_pass": False,
            "skipped": True,
            "skip_reason": "incomplete-generations",
        }
    f0_v = float(f0)  # type: ignore[arg-type]
    f1_v = float(f1)  # type: ignore[arg-type]
    f2_v = float(f2)  # type: ignore[arg-type]
    f3_v = float(f3)  # type: ignore[arg-type]
    f1_pass = f1_v >= GATE_F1_RATIO * f0_v
    f2_pass = f2_v >= GATE_F2_RATIO * f0_v
    f3_pass = f3_v >= GATE_F3_RATIO * f0_v
    monotone_pass = f0_v >= f1_v >= f2_v >= f3_v
    overall = f1_pass and f2_pass and f3_pass and monotone_pass
    return {
        "arm": arm,
        "seed": seed,
        "f0": f0_v,
        "f1": f1_v,
        "f2": f2_v,
        "f3": f3_v,
        "f1_ratio_pass": f1_pass,
        "f2_ratio_pass": f2_pass,
        "f3_ratio_pass": f3_pass,
        "monotone_pass": monotone_pass,
        "overall_pass": overall,
        "skipped": False,
        "skip_reason": "",
    }


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
                f"{_mean(per_gen_means[g]):.3f}" if per_gen_means.get(g) else "—"
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


def main() -> int:
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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_per_gen_csv(args.per_gen_csv)
    if not rows:
        print(f"No rows in {args.per_gen_csv}. Nothing to aggregate.")
        return 1

    retention = build_retention_table(rows)

    # Per-arm seed-by-seed gate evaluation.
    arms = sorted({k[0] for k in retention})
    seeds = sorted({k[1] for k in retention})
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

    _write_retention_csv(retention, args.output_dir / "retention_table.csv")
    _write_decision_gate_csv(seed_evaluations_all, args.output_dir / "decision_gate.csv")
    _write_summary_md(
        seed_evaluations_per_arm=seed_evaluations_per_arm,
        verdict_per_arm=verdict_per_arm,
        retention=retention,
        path=args.output_dir / "summary.md",
    )

    print("\nVerdicts per arm:")
    for arm in arms:
        print(f"  {arm}: {verdict_per_arm[arm]}")
    print(f"\nArtefacts written to {args.output_dir}/")
    print("  - retention_table.csv")
    print("  - decision_gate.csv")
    print("  - summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
