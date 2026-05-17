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
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def _mean(values: Sequence[float]) -> float:
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


def build_survival_table(rows: list[dict]) -> dict[tuple[str, int, int], float]:
    """Aggregate per-episode rows into ``(arm, seed, generation) -> survival_rate``.

    ``survival_rate = 1 - (n_episodes_ending_in_HEALTH_DEPLETED / n_episodes)``.
    Directionally aligned with ``choice_index`` (higher = better avoidance),
    so the same decision-gate logic applies. Has higher dynamic range than
    ``choice_index`` on envs where geometry alone keeps a wandering agent
    mostly outside damage radius — death from accumulated damage is a much
    sharper signal of "the agent walks into pathogens" than fractional
    step-time inside damage radius.

    Skips rows that lack a ``termination_reason`` column (backwards-compat
    with older per_gen_choice_index.csv files that predate the column).
    """
    bucket: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row in rows:
        if "termination_reason" not in row or row["termination_reason"] == "":
            continue
        key = (str(row["arm"]), int(row["seed"]), int(row["generation"]))
        # 1 = died from health depletion (bad); 0 = survived (good)
        died = 1 if str(row["termination_reason"]).lower() == "health_depleted" else 0
        bucket[key].append(died)
    return {k: 1.0 - _mean(v) for k, v in bucket.items()}


def load_f0_training_fitness_per_seed(
    campaign_root: Path,
    *,
    arms: list[str] | None = None,
) -> dict[tuple[str, int], float]:
    """Locate each (arm, seed)'s ``per_gen_elites.jsonl`` and extract F0 training fitness.

    The training-time composite fitness recorded in
    ``per_gen_elites.jsonl`` is the canonical F0 retention baseline for
    transgenerational decision-gate evaluation:

    - F0's trained brain achieved this fitness on the training-eval set
      that TPE used to select it.
    - The post-hoc per-gen evaluator can NOT reproduce that measurement
      because the F0 weight ``.pt`` is GC'd by the substrate-extraction
      pipeline (only the ``.tei.pt`` is retained); the evaluator's F0
      row is an untrained-brain baseline, not a measurement of the
      substrate's source policy.

    Using training-time F0 fitness as the gate baseline gives a
    biologically-correct retention ratio (F1+ post-hoc survival vs F0
    trained survival), matching the wet-lab Kaletsky/Vidal-Gadea
    convention.

    Expected directory layout (mirrors campaign shell output):
    ``<campaign_root>/{arm}/seed-{N}/[<session_id>/]per_gen_elites.jsonl``.

    Returns a dict ``{(arm, seed): f0_fitness}``. Missing files / parse
    errors are skipped with a warning so the rest of the campaign's
    seeds are still evaluable.
    """
    arms_to_scan = (
        arms if arms is not None else [d.name for d in campaign_root.iterdir() if d.is_dir()]
    )
    out: dict[tuple[str, int], float] = {}
    for arm in arms_to_scan:
        arm_dir = campaign_root / arm
        if not arm_dir.is_dir():
            continue
        for seed_dir in sorted(arm_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed-"):
                continue
            try:
                seed = int(seed_dir.name.split("-", 1)[1])
            except (IndexError, ValueError):
                logger.warning("Skipping non-seed directory: %s", seed_dir)
                continue
            # Direct layout: <seed_dir>/per_gen_elites.jsonl
            direct = seed_dir / "per_gen_elites.jsonl"
            jsonl_path: Path | None = None
            if direct.exists():
                jsonl_path = direct
            else:
                # Nested layout: <seed_dir>/<session_id>/per_gen_elites.jsonl
                candidates = [
                    p
                    for p in seed_dir.iterdir()
                    if p.is_dir() and (p / "per_gen_elites.jsonl").is_file()
                ]
                if candidates:
                    jsonl_path = (
                        max(candidates, key=lambda p: p.stat().st_mtime) / "per_gen_elites.jsonl"
                    )
            if jsonl_path is None:
                logger.warning(
                    "No per_gen_elites.jsonl for arm=%s seed=%d under %s; skipping.",
                    arm,
                    seed,
                    seed_dir,
                )
                continue
            f0_fitness = _read_f0_training_fitness(jsonl_path)
            if f0_fitness is not None:
                out[(arm, seed)] = f0_fitness
    return out


def _read_f0_training_fitness(jsonl_path: Path) -> float | None:
    """Return the F0 (``generation == 0``) elite's training-time ``fitness`` field, or None."""
    try:
        with jsonl_path.open(encoding="utf-8") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if int(row.get("generation", -1)) == 0:
                    return float(row.get("fitness", 0.0))
    except OSError as exc:
        logger.warning("Failed to read %s: %s", jsonl_path, exc)
    return None


def evaluate_decision_gate_one_seed(
    *,
    retention: dict[tuple[str, int, int], float],
    arm: str,
    seed: int,
    f0_baseline_override: dict[tuple[str, int], float] | None = None,
) -> dict:
    """Evaluate the decision gate for one (arm, seed).

    Returns a dict with:
      - ``f0``..``f3``: per-gen mean choice indices (or None if missing)
      - ``f1_ratio_pass``: bool — F1 ≥ ``GATE_F1_RATIO`` x F0
      - ``f2_ratio_pass``: bool
      - ``f3_ratio_pass``: bool
      - ``monotone_pass``: bool — F0 ≥ F1 ≥ F2 ≥ F3
      - ``overall_pass``: bool — AND of all four

    When ``f0_baseline_override`` is provided AND contains the
    ``(arm, seed)`` key, its value replaces the post-hoc retention
    table's F0 entry. This is the biologically-correct path for
    transgenerational decision-gate evaluation: F0 retention is the
    training-time avoidance behaviour the substrate was extracted
    from, not the untrained-brain baseline that the post-hoc
    evaluator measures (the F0 ``.pt`` weights are GC'd by the
    substrate-extraction pipeline).
    """
    if f0_baseline_override is not None and (arm, seed) in f0_baseline_override:
        f0: float | None = f0_baseline_override[(arm, seed)]
    else:
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

    rows = _read_per_gen_csv(args.per_gen_csv)
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
