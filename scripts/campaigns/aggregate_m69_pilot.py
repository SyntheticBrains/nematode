r"""Aggregator for the M6.9+ TEI re-evaluation three-arm campaign.

Forks ``aggregate_m6_pilot.py`` patterns + adds:

- **Three-arm awareness**. Reads ``per_gen_choice_index.csv`` rows
  with ``arm`` ∈ ``{tei_on, weights_only, control}`` (vs M6's two-arm
  ``{tei_on, tei_off}``). The per-arm gate logic is unchanged from M6
  (F1 >= 40% x F0, F2 >= 25%, F3 >= 15%, monotone non-increasing).

- **Cross-arm primary verdict** (spec § "Cross-Arm Statistical Verdict
  (n=4 Noise-Aware)"): GO iff ``tei_on`` passes its per-arm gate AND
  the paired-seed delta ``tei_on - control`` on F1+ retention is
  statistically distinguishable from zero via BOTH (one-sided Wilcoxon
  signed-rank with p < 0.10) AND (≥ 5pp delta with non-overlapping 80%
  bootstrap CIs over 1000 resamples). Both checks MUST agree on
  direction. A bare 5pp threshold at n=4 is noise-bounded.

- **PR-B trigger decision**: if primary verdict is GO, emit
  ``pr_b_trigger.md`` recommending the PR-B (transgenerational+weights)
  scaffold. Otherwise emit ``m6_13_punt_note.md`` documenting the null
  finding and noting PR-B is deferred to M6.13+.

- **Pilot pivot decision** (``--mode pilot``): emits
  ``pilot_pivot_decision.md`` populated from the design.md § D6
  pivot table based on the observed pilot outcomes.

Outputs (under ``--output-dir``):
  - ``retention_table.csv`` (per arm x seed x gen: mean survival_rate)
  - ``decision_gate.csv`` (per arm x seed: gate pass/fail + overall)
  - ``cross_arm_verdict.csv`` (Wilcoxon p, bootstrap CI bounds per arm-pair)
  - ``summary.md`` (markdown summary; per-arm + cross-arm tables)
  - ``pilot_pivot_decision.md`` (pilot mode only)
  - ``pr_b_trigger.md`` OR ``m6_13_punt_note.md`` (based on primary verdict)

scipy is required (Wilcoxon + bootstrap CI). The campaign shell at
``phase5_m69_transgenerational_lstmppo_klinotaxis.sh`` checks scipy
availability at launch; this script's top-level import will raise
``ImportError`` with the ``uv sync --extra analysis`` pointer if
scipy is missing at execution time.

Usage:
  scripts/campaigns/aggregate_m69_pilot.py \\
      --per-gen-csv evaluations/m69_transgenerational/per_gen_choice_index.csv \\
      --output-dir evaluations/m69_transgenerational \\
      --campaign-root evolution_results/m69_transgenerational \\
      --mode full
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import numpy as np
    from scipy.stats import wilcoxon  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    msg = (
        "aggregate_m69_pilot requires scipy + numpy. Install via:\n"
        "  uv sync --extra analysis\n"
        f"(import error: {exc})"
    )
    raise ImportError(msg) from exc

logger = logging.getLogger(__name__)


# Decision-gate thresholds — same as M6 (per-arm gate is unchanged).
GATE_F1_RATIO = 0.40
GATE_F2_RATIO = 0.25
GATE_F3_RATIO = 0.15

# Per-arm cross-seed verdict thresholds (mirrors M6).
VERDICT_GO_MIN_SEEDS = 2
VERDICT_PIVOT_MIN_SEEDS = 1

# Cross-arm primary-verdict thresholds (M6.9+ noise-aware verdict).
CROSS_ARM_WILCOXON_P_THRESHOLD = 0.10
CROSS_ARM_MIN_DELTA_PP = 0.05  # 5 percentage points
CROSS_ARM_BOOTSTRAP_RESAMPLES = 1000
CROSS_ARM_BOOTSTRAP_CI_LEVEL = 0.80  # 80% CI ⇒ alpha=0.20

# Three-arm campaign arms.
ARM_TEI_ON = "tei_on"
ARM_WEIGHTS_ONLY = "weights_only"
ARM_CONTROL = "control"
EXPECTED_ARMS = (ARM_TEI_ON, ARM_WEIGHTS_ONLY, ARM_CONTROL)


def _read_per_gen_csv(path: Path) -> list[dict]:
    """Read the per-gen choice-index CSV into a list of dict rows."""
    if not path.exists():
        msg = f"per-gen CSV not found: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_survival_table(rows: list[dict]) -> dict[tuple[str, int, int], float]:
    """Aggregate per-episode rows into ``(arm, seed, generation) -> survival_rate``.

    ``survival_rate = 1 - (n_episodes_ending_in_HEALTH_DEPLETED / n_episodes)``.
    Mirrors ``aggregate_m6_pilot.build_survival_table``. Skips rows
    without ``termination_reason`` (backwards-compat with older CSVs).
    """
    bucket: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row in rows:
        if "termination_reason" not in row or row["termination_reason"] == "":
            continue
        key = (str(row["arm"]), int(row["seed"]), int(row["generation"]))
        died = 1 if str(row["termination_reason"]).lower() == "health_depleted" else 0
        bucket[key].append(died)
    return {k: 1.0 - _mean(v) for k, v in bucket.items()}


def load_f0_training_fitness_per_seed(
    campaign_root: Path,
    *,
    arms: list[str] | None = None,
) -> dict[tuple[str, int], float]:
    """Locate each (arm, seed)'s ``per_gen_elites.jsonl`` and extract F0 training fitness.

    Mirrors M6's loader. Returns ``{(arm, seed): f0_fitness}``. Missing
    files / parse errors are skipped with a warning.
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
            direct = seed_dir / "per_gen_elites.jsonl"
            jsonl_path: Path | None = None
            if direct.exists():
                jsonl_path = direct
            else:
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
    """Return the F0 (``generation == 0``) elite's training-time ``fitness`` field, or None.

    Skips rows missing the ``fitness`` key OR with non-finite values
    (NaN, inf, non-numeric string). Mirrors the M6 hardened loader.
    """
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
                if int(row.get("generation", -1)) != 0:
                    continue
                if "fitness" not in row:
                    return None
                try:
                    value = float(row["fitness"])
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(value):
                    return None
                return value
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
    """Evaluate the per-arm decision gate for one (arm, seed).

    Mirrors M6's `evaluate_decision_gate_one_seed` exactly. The gate
    is on survival_rate (the M6.9+ primary metric); choice_index is
    not used at the per-arm gate.
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


def aggregate_per_arm_verdict(seed_evaluations: list[dict]) -> str:
    """Aggregate per-seed evaluations into a per-arm cross-seed verdict.

    GO iff ≥2 seeds pass; PIVOT iff 1; STOP otherwise. Mirrors M6.
    """
    pass_count = sum(1 for s in seed_evaluations if s["overall_pass"])
    if pass_count >= VERDICT_GO_MIN_SEEDS:
        return "GO"
    if pass_count >= VERDICT_PIVOT_MIN_SEEDS:
        return "PIVOT"
    return "STOP"


def compute_cross_arm_delta_stats(
    survival_table: dict[tuple[str, int, int], float],
    arm_a: str,
    arm_b: str,
    seeds: list[int],
    *,
    f0_baseline_override: dict[tuple[str, int], float] | None = None,
) -> dict:
    """Compute paired-seed F1+ retention deltas + Wilcoxon + bootstrap CI between two arms.

    For each seed: compute F1+ mean survival_rate per arm (averaged
    across F1, F2, F3 — the post-F0 retention window). Delta is
    ``arm_a - arm_b`` per seed. Reports:

    - Mean delta across seeds.
    - One-sided Wilcoxon signed-rank p (alternative: arm_a > arm_b).
    - 80% bootstrap CI of the mean delta (1000 resamples).

    Returns a dict with ``mean_delta``, ``wilcoxon_p``,
    ``bootstrap_ci_lo``, ``bootstrap_ci_hi``, plus the raw
    ``per_seed_deltas`` list for downstream diagnostics.
    """

    def _f1plus_mean(arm: str, seed: int) -> float | None:
        f1 = survival_table.get((arm, seed, 1))
        f2 = survival_table.get((arm, seed, 2))
        f3 = survival_table.get((arm, seed, 3))
        if any(v is None for v in (f1, f2, f3)):
            return None
        return _mean([float(f1), float(f2), float(f3)])  # type: ignore[arg-type]

    per_seed_deltas: list[float] = []
    skipped_seeds: list[int] = []
    for seed in seeds:
        a_mean = _f1plus_mean(arm_a, seed)
        b_mean = _f1plus_mean(arm_b, seed)
        if a_mean is None or b_mean is None:
            skipped_seeds.append(seed)
            continue
        per_seed_deltas.append(a_mean - b_mean)
    if not per_seed_deltas:
        return {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "per_seed_deltas": [],
            "mean_delta": 0.0,
            "wilcoxon_p": 1.0,
            "bootstrap_ci_lo": 0.0,
            "bootstrap_ci_hi": 0.0,
            "skipped_seeds": skipped_seeds,
            "_override_used": f0_baseline_override is not None,
        }
    mean_delta = _mean(per_seed_deltas)
    # Wilcoxon signed-rank: one-sided alternative arm_a > arm_b
    # requires at least one non-zero delta. The all-zero short-circuit
    # both guards against scipy's all-zero RuntimeWarning path (which
    # would return p=1.0 anyway) and skips the function-call overhead.
    if all(abs(d) < 1e-12 for d in per_seed_deltas):
        wilcoxon_p = 1.0
    else:
        result = wilcoxon(per_seed_deltas, alternative="greater")
        # scipy returns a NamedTuple-ish ``WilcoxonResult`` with
        # ``.pvalue``; pyright can't statically introspect it, so
        # getattr keeps the runtime path intact + the type-check quiet.
        wilcoxon_p = float(getattr(result, "pvalue", 1.0))
    # Bootstrap CI: resample with replacement N times; compute mean
    # per resample; take alpha/2 and 1-alpha/2 percentiles. Seeded
    # numpy generator for reproducibility.
    rng = np.random.default_rng(42)
    arr = np.asarray(per_seed_deltas, dtype=float)
    boots = np.array(
        [
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(CROSS_ARM_BOOTSTRAP_RESAMPLES)
        ],
    )
    alpha = 1.0 - CROSS_ARM_BOOTSTRAP_CI_LEVEL
    ci_lo = float(np.quantile(boots, alpha / 2))
    ci_hi = float(np.quantile(boots, 1.0 - alpha / 2))
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "per_seed_deltas": per_seed_deltas,
        "mean_delta": mean_delta,
        "wilcoxon_p": wilcoxon_p,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "skipped_seeds": skipped_seeds,
        "_override_used": f0_baseline_override is not None,
    }


def compute_cross_arm_primary_verdict(
    tei_on_seed_evaluations: list[dict],
    cross_arm_stats: dict,
) -> dict:
    """Cross-arm primary verdict: GO iff per-arm gate passes AND noise-aware delta is positive.

    Per the evolution-framework spec § "Cross-Arm Statistical Verdict":
    GO requires ALL of:
      1. tei_on per-arm gate passes (≥ 2/4 seeds pass).
      2. Wilcoxon p < 0.10 (one-sided, tei_on > control).
      3. Mean delta ≥ 5pp absolute.
      4. 80% bootstrap CI does NOT include zero (lo > 0).

    Returns a verdict dict with ``verdict`` ∈ {"GO", "STOP"} +
    per-check pass/fail flags + rationale.
    """
    tei_on_arm_verdict = aggregate_per_arm_verdict(tei_on_seed_evaluations)
    per_arm_gate_pass = tei_on_arm_verdict == "GO"
    wilcoxon_pass = cross_arm_stats["wilcoxon_p"] < CROSS_ARM_WILCOXON_P_THRESHOLD
    delta_pass = cross_arm_stats["mean_delta"] >= CROSS_ARM_MIN_DELTA_PP
    ci_pass = cross_arm_stats["bootstrap_ci_lo"] > 0.0
    overall_pass = per_arm_gate_pass and wilcoxon_pass and delta_pass and ci_pass
    return {
        "verdict": "GO" if overall_pass else "STOP",
        "per_arm_gate_pass": per_arm_gate_pass,
        "tei_on_arm_verdict": tei_on_arm_verdict,
        "wilcoxon_pass": wilcoxon_pass,
        "wilcoxon_p": cross_arm_stats["wilcoxon_p"],
        "delta_pass": delta_pass,
        "mean_delta": cross_arm_stats["mean_delta"],
        "ci_pass": ci_pass,
        "bootstrap_ci_lo": cross_arm_stats["bootstrap_ci_lo"],
        "bootstrap_ci_hi": cross_arm_stats["bootstrap_ci_hi"],
    }


def _write_retention_csv(
    survival_table: dict[tuple[str, int, int], float],
    path: Path,
) -> None:
    """Write per-(arm, seed, gen) retention table to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "generation", "survival_rate"])
        for (arm, seed, gen), value in sorted(survival_table.items()):
            writer.writerow([arm, seed, gen, f"{value:.6f}"])


def _write_decision_gate_csv(seed_evaluations: list[dict], path: Path) -> None:
    """Write per-(arm, seed) gate evaluation to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
            ],
        )
        for s in seed_evaluations:
            writer.writerow(
                [
                    s["arm"],
                    s["seed"],
                    s["f0"],
                    s["f1"],
                    s["f2"],
                    s["f3"],
                    s["f1_ratio_pass"],
                    s["f2_ratio_pass"],
                    s["f3_ratio_pass"],
                    s["monotone_pass"],
                    s["overall_pass"],
                    s["skipped"],
                    s["skip_reason"],
                ],
            )


def _write_cross_arm_verdict_csv(
    cross_arm_results: list[dict],
    path: Path,
) -> None:
    """Write per-arm-pair Wilcoxon + bootstrap CI stats to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arm_a",
                "arm_b",
                "n_seeds",
                "mean_delta",
                "wilcoxon_p",
                "bootstrap_ci_lo",
                "bootstrap_ci_hi",
                "skipped_seeds",
            ],
        )
        for r in cross_arm_results:
            writer.writerow(
                [
                    r["arm_a"],
                    r["arm_b"],
                    len(r["per_seed_deltas"]),
                    f"{r['mean_delta']:.6f}",
                    f"{r['wilcoxon_p']:.6f}",
                    f"{r['bootstrap_ci_lo']:.6f}",
                    f"{r['bootstrap_ci_hi']:.6f}",
                    ";".join(str(s) for s in r["skipped_seeds"]),
                ],
            )


def _write_summary_md(
    per_arm_verdicts: dict[str, str],
    primary_verdict: dict,
    cross_arm_results: list[dict],
    path: Path,
) -> None:
    """Write a human-readable markdown summary of the campaign verdict."""
    lines: list[str] = []
    lines.append("# M6.9+ TEI re-evaluation — aggregator summary\n")
    lines.append("## Per-arm cross-seed verdicts\n")
    lines.append("| arm | verdict |")
    lines.append("|---|---|")
    for arm in EXPECTED_ARMS:
        verdict = per_arm_verdicts.get(arm, "NO_DATA")
        lines.append(f"| `{arm}` | **{verdict}** |")
    lines.append("\n## Cross-arm primary verdict\n")
    lines.append(f"**Verdict: {primary_verdict['verdict']}**\n")
    lines.append("Per-check breakdown (GO requires ALL four):\n")
    lines.append(
        f"- Per-arm gate (tei_on ≥ 2/4 seeds pass): **{primary_verdict['per_arm_gate_pass']}** (tei_on arm verdict: {primary_verdict['tei_on_arm_verdict']})",
    )
    lines.append(
        f"- Wilcoxon p < {CROSS_ARM_WILCOXON_P_THRESHOLD}: **{primary_verdict['wilcoxon_pass']}** (p = {primary_verdict['wilcoxon_p']:.4f})",
    )
    lines.append(
        f"- Mean delta ≥ {CROSS_ARM_MIN_DELTA_PP * 100:.0f}pp: **{primary_verdict['delta_pass']}** (mean = {primary_verdict['mean_delta'] * 100:.2f}pp)",
    )
    lines.append(
        f"- 80% bootstrap CI excludes zero: **{primary_verdict['ci_pass']}** (CI = [{primary_verdict['bootstrap_ci_lo'] * 100:.2f}pp, {primary_verdict['bootstrap_ci_hi'] * 100:.2f}pp])",
    )
    lines.append("\n## Cross-arm pairwise statistics\n")
    lines.append("| arm_a | arm_b | n | mean Δ | Wilcoxon p | 80% CI lo | 80% CI hi |")
    lines.append("|---|---|--:|--:|--:|--:|--:|")
    lines.extend(
        f"| `{r['arm_a']}` | `{r['arm_b']}` | {len(r['per_seed_deltas'])} | "
        f"{r['mean_delta'] * 100:+.2f}pp | {r['wilcoxon_p']:.4f} | "
        f"{r['bootstrap_ci_lo'] * 100:+.2f}pp | {r['bootstrap_ci_hi'] * 100:+.2f}pp |"
        for r in cross_arm_results
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pilot_pivot_decision(
    pilot_observations: dict,
    path: Path,
) -> None:
    """Emit pilot_pivot_decision.md populated from design.md § D6 pivot table.

    Classifies the pilot observation against the six pre-declared
    pivots and writes a markdown summary the user reviews BEFORE
    unblocking the full campaign.
    """
    tei_on_v = pilot_observations.get("tei_on_per_arm_verdict", "UNKNOWN")
    primary = pilot_observations.get("primary_verdict_dict", {})
    mean_delta = float(primary.get("mean_delta", 0.0))
    per_arm_gate_pass = bool(primary.get("per_arm_gate_pass", False))
    wilcoxon_pass = bool(primary.get("wilcoxon_pass", False))

    # Match against the six pre-declared pilot observations (design.md § D6).
    pivot_lines = ["# Pilot pivot decision\n"]
    pivot_lines.append(
        "Per design.md § D6, the pilot's outcome is classified against six pre-declared pivots:\n",
    )
    if mean_delta > 0.05 and per_arm_gate_pass and wilcoxon_pass:
        pivot_lines.append(
            "**Pilot signal: STRONG.** `tei_on > control` ≥ 5pp AND Wilcoxon significant.",
        )
        pivot_lines.append("Pivot: NONE. Proceed to full campaign with no config changes.\n")
    elif mean_delta < 0.01 and tei_on_v == "STOP":
        pivot_lines.append("**Pilot signal: substrate likely inert.** `tei_on ≈ control` at F1+.")
        pivot_lines.append(
            "Pivot: widen `bias_network.hidden_dim` 8→16 OR add features to `input_features` (e.g. `stam_state_mean`). Re-run pilot.\n",
        )
    elif tei_on_v == "PIVOT":
        pivot_lines.append("**Pilot signal: marginal — 1/4 seed passes tei_on per-arm gate.**")
        pivot_lines.append(
            "Pivot: review per-seed retention curves. If F1 retains but F2/F3 collapse, try `decay_shape: linear` or `decay_factor: 0.8`. Re-run pilot if config changes.\n",
        )
    else:
        pivot_lines.append(
            "**Pilot signal: ambiguous.** Inspect retention curves + cross-arm deltas manually.",
        )
        pivot_lines.append("Pivot decision: USER REVIEW REQUIRED before proceeding.\n")
    pivot_lines.append("\nObserved metrics:\n")
    pivot_lines.append(f"- `tei_on` per-arm verdict: **{tei_on_v}**")
    pivot_lines.append(f"- Cross-arm mean delta: {mean_delta * 100:+.2f}pp")
    pivot_lines.append(f"- Per-arm gate pass (tei_on): {per_arm_gate_pass}")
    pivot_lines.append(f"- Wilcoxon p-threshold met: {wilcoxon_pass}")
    pivot_lines.append("\nThe full pivot table lives in design.md § D6.")
    path.write_text("\n".join(pivot_lines) + "\n", encoding="utf-8")


def _write_pr_b_trigger(path: Path) -> None:
    """Emit pr_b_trigger.md when the primary verdict is GO."""
    body = (
        "# PR-B trigger\n\n"
        "The M6.9+ PR-A primary verdict was **GO**: the pure-TEI floor\n"
        "(tei_on) outperforms the no-inheritance control by a\n"
        "statistically distinguishable margin (Wilcoxon p < 0.10 AND\n"
        "≥ 5pp mean delta AND non-overlapping 80% bootstrap CIs).\n\n"
        "**Recommended next step**: scaffold the PR-B OpenSpec change\n"
        "``add-transgenerational-memory-weights`` per design.md § D4.\n"
        "PR-B adds the `transgenerational+weights` symmetric-compute\n"
        "control and confirms whether the substrate adds value on top\n"
        "of trained weights — the strongest scientific claim M6.9+\n"
        "supports.\n\n"
        "Use `/openspec:new-change add-transgenerational-memory-weights`.\n"
    )
    path.write_text(body, encoding="utf-8")


def _write_m6_13_punt_note(path: Path) -> None:
    """Emit m6_13_punt_note.md when the primary verdict is STOP."""
    body = (
        "# M6.13 punt note\n\n"
        "The M6.9+ PR-A primary verdict was **STOP**: the pure-TEI floor\n"
        "(tei_on) is not statistically distinguishable from the\n"
        "no-inheritance control on F1+ retention.\n\n"
        "**Recommended next step**: PR-B is **NOT** scaffolded — running\n"
        "the `transgenerational+weights` symmetric-compute control\n"
        "without a pure-TEI floor signal would be uninterpretable\n"
        "(the same structural issue M6's audit D flagged). The TEI\n"
        "hypothesis is deferred to M6.13+ unless follow-up evidence\n"
        "(e.g. pilot pivots on substrate architecture or decay shape)\n"
        "revives the signal.\n\n"
        "See design.md § D4 for the PR-B trigger criterion.\n"
    )
    path.write_text(body, encoding="utf-8")


def main() -> int:  # noqa: C901 - linear orchestration; nested helpers would obscure flow
    """Entry point for the M6.9+ three-arm aggregator."""
    parser = argparse.ArgumentParser(description="M6.9+ TEI three-arm pilot/full aggregator.")
    parser.add_argument(
        "--per-gen-csv",
        type=Path,
        required=True,
        help="Path to per_gen_choice_index.csv produced by transgenerational_per_gen_eval.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write retention_table.csv / decision_gate.csv / etc.",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=None,
        help=(
            "Optional. When provided, the F0 training-time fitness is read from "
            "<campaign-root>/{arm}/seed-{N}/per_gen_elites.jsonl and used as the F0 "
            "gate baseline (the biologically-correct retention reference; post-hoc "
            "F0 measures an untrained brain since F0 weights are GC'd)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="full",
        help="Pilot mode emits pilot_pivot_decision.md; full skips it.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_per_gen_csv(args.per_gen_csv)
    survival_table = build_survival_table(rows)
    if not survival_table:
        logger.error(
            "Per-gen CSV %s has no rows with termination_reason; cannot compute "
            "survival_rate. Did transgenerational_per_gen_eval.py run with the "
            "M6.9+ termination tracker enabled?",
            args.per_gen_csv,
        )
        return 1

    # F0 training-time fitness override (biologically-correct retention baseline).
    f0_override: dict[tuple[str, int], float] | None = None
    if args.campaign_root is not None:
        f0_override = load_f0_training_fitness_per_seed(
            args.campaign_root,
            arms=list(EXPECTED_ARMS),
        )
        logger.info(
            "Loaded F0 training-time fitness override for %d (arm, seed) pairs from %s.",
            len(f0_override),
            args.campaign_root,
        )

    # Per-arm gates.
    seeds_per_arm: dict[str, list[int]] = defaultdict(list)
    for arm, seed, _gen in survival_table:
        if seed not in seeds_per_arm[arm]:
            seeds_per_arm[arm].append(seed)
    for arm_seeds in seeds_per_arm.values():
        arm_seeds.sort()

    all_evals: list[dict] = []
    per_arm_evals: dict[str, list[dict]] = defaultdict(list)
    for arm in EXPECTED_ARMS:
        for seed in seeds_per_arm.get(arm, []):
            ev = evaluate_decision_gate_one_seed(
                retention=survival_table,
                arm=arm,
                seed=seed,
                f0_baseline_override=f0_override,
            )
            all_evals.append(ev)
            per_arm_evals[arm].append(ev)

    per_arm_verdicts = {arm: aggregate_per_arm_verdict(evs) for arm, evs in per_arm_evals.items()}

    # Cross-arm pairwise stats. The primary verdict uses (tei_on, control);
    # secondary stats are (weights_only, control) and (tei_on, weights_only).
    all_seeds = sorted({s for seeds in seeds_per_arm.values() for s in seeds})
    cross_arm_pairs: list[tuple[str, str]] = [
        (ARM_TEI_ON, ARM_CONTROL),
        (ARM_WEIGHTS_ONLY, ARM_CONTROL),
        (ARM_TEI_ON, ARM_WEIGHTS_ONLY),
    ]
    cross_arm_results: list[dict] = []
    for arm_a, arm_b in cross_arm_pairs:
        stats = compute_cross_arm_delta_stats(
            survival_table,
            arm_a=arm_a,
            arm_b=arm_b,
            seeds=all_seeds,
            f0_baseline_override=f0_override,
        )
        cross_arm_results.append(stats)

    # Primary verdict: tei_on vs control (the first pair).
    primary_stats = cross_arm_results[0]
    primary_verdict = compute_cross_arm_primary_verdict(
        per_arm_evals[ARM_TEI_ON],
        primary_stats,
    )

    # Emit outputs.
    _write_retention_csv(survival_table, args.output_dir / "retention_table.csv")
    _write_decision_gate_csv(all_evals, args.output_dir / "decision_gate.csv")
    _write_cross_arm_verdict_csv(cross_arm_results, args.output_dir / "cross_arm_verdict.csv")
    _write_summary_md(
        per_arm_verdicts=per_arm_verdicts,
        primary_verdict=primary_verdict,
        cross_arm_results=cross_arm_results,
        path=args.output_dir / "summary.md",
    )
    if args.mode == "pilot":
        _write_pilot_pivot_decision(
            pilot_observations={
                "tei_on_per_arm_verdict": per_arm_verdicts.get(ARM_TEI_ON, "UNKNOWN"),
                "primary_verdict_dict": primary_verdict,
            },
            path=args.output_dir / "pilot_pivot_decision.md",
        )
    if primary_verdict["verdict"] == "GO":
        _write_pr_b_trigger(args.output_dir / "pr_b_trigger.md")
    else:
        _write_m6_13_punt_note(args.output_dir / "m6_13_punt_note.md")

    logger.info("M6.9+ aggregator output written to %s", args.output_dir)
    logger.info("Per-arm verdicts: %s", per_arm_verdicts)
    logger.info("Cross-arm primary verdict: %s", primary_verdict["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
