r"""Aggregator for the M6.13 TEI-as-prior-on-M3 three-arm campaign.

Forks ``aggregate_m69_pilot.py`` patterns. The four-method skeleton
(per-gen CSV read → survival table → per-arm gate → cross-arm verdict)
is byte-equivalent to M6.9+; the M6.13 deltas are:

- **Reframed arm naming**. ``tei_weights`` (composed mode:
  weights+transgenerational) replaces M6.9+'s ``tei_on`` (pure-TEI).
  ``weights_only`` (M3 baseline at K_test) and ``control`` (TPE-fresh
  at K_test) keep their names but the role of ``weights_only``
  flips: in M6.9+ it was a secondary M3-reproduction control, in
  M6.13 it is the PRIMARY comparator (does substrate prior accelerate
  M3 retraining?).

- **Reframed primary verdict pair**. M6.9+ tested
  ``tei_on - control`` (pure-TEI vs from-scratch floor at K=0). M6.13
  tests ``tei_weights - weights_only`` (composed vs M3 alone at K_test).
  Per design.md § D4 the cross-arm verdict requires Wilcoxon p < 0.10
  AND ≥ 5pp delta with non-overlapping 80% bootstrap CIs. Both checks
  MUST agree on direction.

- **Reframed pivot table** (design.md § D6). Six rows:
    1. tei_weights ≈ weights_only (|Δ| < 2pp) → STOP, substrate inert
    2. tei_weights > weights_only by ≥5pp at K_test → GO
    3. tei_weights > weights_only by 2-5pp at K_test → K-sensitivity
    4. tei_weights < weights_only (Δ < -2pp) → STOP, substrate interferes
    5. weights_only F0 ≈ F1 → PIVOT, K_test too large (M3 saturation)
    6. tei_weights F1 collapses to ~0 → STOP, PPO destabilised

- **M6.14 trigger decision** replaces PR-B trigger. If primary verdict
  is GO, emit ``m614_frequency_prior_trigger.md`` recommending the
  M6.14 ablation (frequency-prior substrate variant). Otherwise emit
  ``m613_null_finding_note.md`` documenting the null finding and
  closing the M6 thread (M6.14 NOT triggered when M6.13 is null).

Outputs (under ``--output-dir``):
  - ``retention_table.csv`` (per arm x seed x gen: mean survival_rate)
  - ``decision_gate.csv`` (per arm x seed: gate pass/fail + overall)
  - ``cross_arm_verdict.csv`` (Wilcoxon p, bootstrap CI bounds per arm-pair)
  - ``summary.md`` (markdown summary; per-arm + cross-arm tables)
  - ``pilot_pivot_decision.md`` (pilot mode only)
  - ``m614_frequency_prior_trigger.md`` OR ``m613_null_finding_note.md``
    (based on primary verdict; full mode only)

scipy is required (Wilcoxon + bootstrap CI). The campaign shell at
``phase5_m613_tei_prior_lstmppo_klinotaxis.sh`` checks scipy
availability at launch; this script's top-level import will raise
``ImportError`` with the ``uv sync --extra analysis`` pointer if
scipy is missing at execution time.

Usage:
  scripts/campaigns/aggregate_m613_pilot.py \\
      --per-gen-csv evaluations/m613_tei_prior/per_gen_choice_index.csv \\
      --output-dir evaluations/m613_tei_prior \\
      --campaign-root evolution_results/m613_tei_prior \\
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
        "aggregate_m613_pilot requires scipy + numpy. Install via:\n"
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

# Cross-arm primary-verdict thresholds (n=4 noise-aware verdict —
# inherited unchanged from M6.9+; reframed pair under M6.13).
CROSS_ARM_WILCOXON_P_THRESHOLD = 0.10
CROSS_ARM_MIN_DELTA_PP = 0.05  # 5 percentage points
CROSS_ARM_BOOTSTRAP_RESAMPLES = 1000
CROSS_ARM_BOOTSTRAP_CI_LEVEL = 0.80  # 80% CI ⇒ alpha=0.20

# Full-campaign target n. When the primary pair (tei_weights - weights_only)
# has fewer than this many paired-seed deltas the cross-arm verdict
# is INDETERMINATE — the one-sided Wilcoxon at n=3 has minimum
# achievable p = 0.125, which can never satisfy
# CROSS_ARM_WILCOXON_P_THRESHOLD = 0.10. A null verdict in that
# regime cannot be distinguished from a real null, so we surface
# the under-powered state explicitly rather than silently labelling
# it STOP. M6 closed INCONCLUSIVE precisely because this distinction
# was not made.
CROSS_ARM_FULL_N_SEEDS = 4

# Three-arm campaign arms (M6.13 reframe).
# - ARM_TEI_WEIGHTS: composed-mode (weights+transgenerational) arm.
#   M3 weight inheritance + F0-extracted substrate prior; F1+ retrains
#   K_test episodes with both signals active.
# - ARM_WEIGHTS_ONLY: M3 baseline (lamarckian) at K_test. PRIMARY
#   COMPARATOR for the M6.13 cross-arm verdict (vs PR-A's secondary
#   role as M3-reproduction control).
# - ARM_CONTROL: TPE-fresh (no inheritance) at K_test. Floor reference
#   for secondary verdicts (weights_only - control = M3 re-reproduction
#   at K_test; tei_weights - control = composed vs floor sanity check).
ARM_TEI_WEIGHTS = "tei_weights"
ARM_WEIGHTS_ONLY = "weights_only"
ARM_CONTROL = "control"
EXPECTED_ARMS = (ARM_TEI_WEIGHTS, ARM_WEIGHTS_ONLY, ARM_CONTROL)


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
    tei_weights_seed_evaluations: list[dict],
    cross_arm_stats: dict,
    *,
    mode: str = "full",
) -> dict:
    """Cross-arm primary verdict: GO iff per-arm gate passes AND noise-aware delta is positive.

    Per the M6.13 evolution-framework spec § "M6.13 Cross-Arm Primary
    Verdict (Reframed)": GO requires ALL of:
      1. tei_weights per-arm gate passes (≥ 2/4 seeds pass).
      2. Wilcoxon p < 0.10 (one-sided, tei_weights > weights_only).
      3. Mean delta ≥ 5pp absolute.
      4. 80% bootstrap CI does NOT include zero (lo > 0).

    Under ``mode == "full"``, when fewer than ``CROSS_ARM_FULL_N_SEEDS``
    paired-seed deltas are available on the primary pair the verdict
    is "INDETERMINATE" rather than "STOP" — a one-sided Wilcoxon at
    n=3 has minimum p=0.125, making the GO threshold structurally
    unreachable. The operator MUST distinguish this from a real null.
    Under ``mode == "pilot"`` the n=1 single-seed run is expected to
    fail Wilcoxon; the pilot's primary artefact is
    ``pilot_pivot_decision.md`` rather than this verdict.

    Returns a verdict dict with ``verdict`` ∈ {"GO", "STOP", "INDETERMINATE"}
    + per-check pass/fail flags + rationale.
    """
    tei_weights_arm_verdict = aggregate_per_arm_verdict(tei_weights_seed_evaluations)
    per_arm_gate_pass = tei_weights_arm_verdict == "GO"
    wilcoxon_pass = cross_arm_stats["wilcoxon_p"] < CROSS_ARM_WILCOXON_P_THRESHOLD
    delta_pass = cross_arm_stats["mean_delta"] >= CROSS_ARM_MIN_DELTA_PP
    ci_pass = cross_arm_stats["bootstrap_ci_lo"] > 0.0
    n_seeds = len(cross_arm_stats.get("per_seed_deltas", []))
    overall_pass = per_arm_gate_pass and wilcoxon_pass and delta_pass and ci_pass
    if overall_pass:
        verdict = "GO"
    elif mode == "full" and n_seeds < CROSS_ARM_FULL_N_SEEDS:
        verdict = "INDETERMINATE"
    else:
        verdict = "STOP"
    return {
        "verdict": verdict,
        "per_arm_gate_pass": per_arm_gate_pass,
        "tei_weights_arm_verdict": tei_weights_arm_verdict,
        "wilcoxon_pass": wilcoxon_pass,
        "wilcoxon_p": cross_arm_stats["wilcoxon_p"],
        "delta_pass": delta_pass,
        "mean_delta": cross_arm_stats["mean_delta"],
        "ci_pass": ci_pass,
        "bootstrap_ci_lo": cross_arm_stats["bootstrap_ci_lo"],
        "bootstrap_ci_hi": cross_arm_stats["bootstrap_ci_hi"],
        "n_seeds": n_seeds,
        "indeterminate_under_powered": (mode == "full" and n_seeds < CROSS_ARM_FULL_N_SEEDS),
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
    lines.append("# M6.13 TEI-as-prior-on-M3 — aggregator summary\n")
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
        f"- Per-arm gate (tei_weights ≥ 2/4 seeds pass): "
        f"**{primary_verdict['per_arm_gate_pass']}** "
        f"(tei_weights arm verdict: {primary_verdict['tei_weights_arm_verdict']})",
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


# ============================================================================
# M6.13 D6 pivot-table detectors. Six rows per design.md § D6:
#
#   Row 1: tei_weights ≈ weights_only (|Δ| < 2pp) → STOP, substrate inert
#   Row 2: tei_weights > weights_only by ≥5pp at K_test → GO
#   Row 3: tei_weights > weights_only by 2-5pp at K_test → K-sensitivity
#   Row 4: tei_weights < weights_only (Δ < -2pp) → STOP, substrate interferes
#   Row 5: weights_only F0 ≈ F1 → PIVOT, K_test too large (M3 saturation)
#   Row 6: tei_weights F1 collapses to ~0 → STOP, PPO destabilised
# ============================================================================


# Primary-pair delta thresholds (per design.md § D6).
PIVOT_DELTA_INERT = 0.02  # |Δ| < 2pp ⇒ substrate inert (row 1)
PIVOT_DELTA_K_SENSITIVITY = 0.05  # 2-5pp ⇒ K-sensitivity pivot (row 3)
PIVOT_DELTA_INTERFERES = -0.02  # Δ < -2pp ⇒ substrate interferes (row 4)
# M3-saturation: weights_only F1 within ε of F0.
PIVOT_M3_SATURATION_EPSILON = 0.02
# PPO-destabilised: tei_weights F1 below this absolute threshold.
PIVOT_PPO_COLLAPSE_THRESHOLD = 0.05


def _detect_substrate_inert(cross_arm_results: list[dict]) -> bool:
    """D6 row 1: ``tei_weights ≈ weights_only`` (|Δ| < 2pp).

    Substrate prior carries no measurable signal under retraining.
    The composed mode collapses to pure M3; the M6.13 hypothesis is
    falsified. STOP.
    """
    primary = next(
        (
            r
            for r in cross_arm_results
            if r["arm_a"] == ARM_TEI_WEIGHTS and r["arm_b"] == ARM_WEIGHTS_ONLY
        ),
        None,
    )
    if primary is None:
        return False
    return abs(primary["mean_delta"]) < PIVOT_DELTA_INERT


def _detect_substrate_interferes(cross_arm_results: list[dict]) -> bool:
    """D6 row 4: ``tei_weights < weights_only`` (Δ < -2pp).

    Substrate prior actively interferes with M3 retraining — the
    prior misleads early F1+ exploration before inherited weights
    take over. STOP with a useful negative result: future work
    might investigate substrate-policy alignment.
    """
    primary = next(
        (
            r
            for r in cross_arm_results
            if r["arm_a"] == ARM_TEI_WEIGHTS and r["arm_b"] == ARM_WEIGHTS_ONLY
        ),
        None,
    )
    if primary is None:
        return False
    return primary["mean_delta"] < PIVOT_DELTA_INTERFERES


def _detect_m3_saturation(survival_table: dict, seeds: list[int]) -> bool:
    """D6 row 5: ``weights_only F0 ≈ F1`` (K_test too large).

    M3 has saturated at the chosen K_test — no headroom for the
    substrate prior to help (or hurt). PIVOT: drop K_test to 500
    and rerun the pilot. The T3' smoke tripwire was supposed to
    catch this BEFORE the pilot; firing here means T3' was bypassed
    or the env shifted between smoke and pilot.
    """
    for seed in seeds:
        f0 = survival_table.get((ARM_WEIGHTS_ONLY, seed, 0))
        f1 = survival_table.get((ARM_WEIGHTS_ONLY, seed, 1))
        if f0 is None or f1 is None:
            continue
        # Saturated when F1 is within ε of F0 (M3 hasn't improved meaningfully).
        if f0 > 0 and abs(f1 - f0) < PIVOT_M3_SATURATION_EPSILON:
            return True
    return False


def _detect_ppo_destabilised(survival_table: dict, seeds: list[int]) -> bool:
    """D6 row 6: tei_weights F1 collapses to ~0 (substrate breaks PPO update).

    The substrate prior destabilises PPO retraining — either the
    clamp is too aggressive or the prior misdirects gradient updates.
    STOP; investigate clamp / freeze-during-update as future work.
    """
    for seed in seeds:
        f1 = survival_table.get((ARM_TEI_WEIGHTS, seed, 1))
        if f1 is not None and f1 < PIVOT_PPO_COLLAPSE_THRESHOLD:
            return True
    return False


def _write_pilot_pivot_decision(
    pilot_observations: dict,
    path: Path,
) -> None:
    """Emit pilot_pivot_decision.md populated from design.md § D6 pivot table.

    Classifies the pilot observation against the six M6.13-specific
    pivots and writes a markdown summary the user reviews BEFORE
    unblocking the full campaign.

    Branch order is by SPECIFICITY: rows 5 + 6 (catastrophic) → row 4
    (interferes) → row 2/3 (positive delta variants) → row 1 (inert)
    → ambiguous fallback. Earlier branches absorb cases that later
    rows would also match if checked alone (e.g. PPO collapse → row 6,
    not row 1 even though Δ may be ≈0).

    Required keys in ``pilot_observations``:
        - ``tei_weights_per_arm_verdict``: per-arm GO/PIVOT/STOP
        - ``primary_verdict_dict``: cross-arm primary-verdict dict
        - ``survival_table``: ``{(arm, seed, gen): mean_survival}``
        - ``seeds``: observed seeds
        - ``cross_arm_results``: list of per-pair cross-arm stats
    """
    tei_weights_v = pilot_observations.get("tei_weights_per_arm_verdict", "UNKNOWN")
    primary = pilot_observations.get("primary_verdict_dict", {})
    mean_delta = float(primary.get("mean_delta", 0.0))
    survival_table = pilot_observations.get("survival_table", {})
    seeds = pilot_observations.get("seeds", [])
    cross_arm_results = pilot_observations.get("cross_arm_results", [])

    pivot_lines = ["# Pilot pivot decision (M6.13)\n"]
    pivot_lines.append(
        "Per design.md § D6, the pilot's outcome is classified against "
        "six pre-declared pivots. Branch order is by specificity — "
        "catastrophic failure modes (rows 5, 6) match before positive- "
        "or null-delta variants.\n",
    )
    # Row 5: M3-saturation (K_test too large) — check first because
    # it disqualifies the entire comparison frame (no headroom for
    # substrate to demonstrate anything).
    if _detect_m3_saturation(survival_table, seeds):
        pivot_lines.append(
            "**Pilot signal: M3-saturation (D6 row 5).** "
            "`weights_only F0 ≈ F1` — K_test is too large, M3 has "
            "ceiling'd, and there is no headroom for the substrate "
            "prior to demonstrate acceleration. The T3' smoke tripwire "
            "should have caught this BEFORE the pilot; firing here "
            "means the env shifted or T3' was bypassed.",
        )
        pivot_lines.append(
            "Pivot: drop K_test to 500 and rerun the pilot (+3 wall-h cap per design.md § D6).\n",
        )
    # Row 6: PPO destabilised by substrate. tei_weights F1 collapse
    # is unambiguous even if mean_delta is also negative.
    elif _detect_ppo_destabilised(survival_table, seeds):
        pivot_lines.append(
            "**Pilot signal: PPO destabilised (D6 row 6).** "
            "`tei_weights F1` collapses to ~0 — the substrate prior "
            "breaks PPO retraining. Either the clamp is too aggressive "
            "or the prior misdirects gradient updates beyond what M3 "
            "can recover.",
        )
        pivot_lines.append(
            "Pivot: STOP. Investigate substrate clamp / freeze-during-"
            "update as future work; document as useful negative result.\n",
        )
    # Row 4: substrate interferes with M3.
    elif _detect_substrate_interferes(cross_arm_results):
        pivot_lines.append(
            "**Pilot signal: substrate INTERFERES with M3 (D6 row 4).** "
            "`tei_weights < weights_only` by > 2pp — the substrate "
            "prior actively HURTS M3 retraining rather than "
            "accelerating it. Likely the prior misleads early F1+ "
            "exploration before inherited weights take over.",
        )
        pivot_lines.append(
            "Pivot: STOP. Recommends substrate-policy alignment as a "
            "future-work direction (the prior must be calibrated for "
            "the warm-start child's policy, not just the F0 elite's).\n",
        )
    # Row 2: clean GO (≥5pp positive delta).
    elif mean_delta >= PIVOT_DELTA_K_SENSITIVITY:
        pivot_lines.append(
            "**Pilot signal: clean GO at K_test (D6 row 2).** "
            "`tei_weights > weights_only` by ≥ 5pp — substrate prior "
            "accelerates M3 retraining at K_test. The M6.13 hypothesis "
            "is supported.",
        )
        pivot_lines.append(
            "Pivot: NONE. Proceed to full campaign at K_test only "
            "(no K sweep — the test-point selection was made at "
            "calibration time).\n",
        )
    # Row 3: K-sensitivity (positive but moderate delta, 2-5pp).
    elif mean_delta >= PIVOT_DELTA_INERT:
        pivot_lines.append(
            "**Pilot signal: K-sensitivity pivot (D6 row 3).** "
            "`tei_weights > weights_only` by 2-5pp at K_test — the "
            "substrate signal exists but is K-dependent. Mapping the "
            "dose-response curve is required to decide GO/STOP.",
        )
        pivot_lines.append(
            "Pivot: rerun pilot at K=500 AND K=1500 to map the "
            "dose-response. Decide GO/STOP at the K where Δ is "
            "largest. +6 wall-h cap per design.md § D6.\n",
        )
    # Row 1: substrate inert (|Δ| < 2pp).
    elif _detect_substrate_inert(cross_arm_results):
        pivot_lines.append(
            "**Pilot signal: substrate inert (D6 row 1).** "
            "`tei_weights ≈ weights_only` (|Δ| < 2pp) — the substrate "
            "prior carries no measurable signal under retraining. The "
            "composed mode collapses to pure M3; M6.13 hypothesis is "
            "falsified.",
        )
        pivot_lines.append(
            "Pivot: STOP. Logbook 020 documents the null finding.\n",
        )
    else:
        pivot_lines.append(
            "**Pilot signal: ambiguous.** Observed pattern does not "
            "match any of the 6 pre-declared design.md § D6 pivots. "
            "Inspect retention curves + cross-arm deltas manually.",
        )
        pivot_lines.append(
            "Pivot decision: USER REVIEW REQUIRED before proceeding.\n",
        )
    per_arm_gate_pass = bool(primary.get("per_arm_gate_pass", False))
    wilcoxon_pass = bool(primary.get("wilcoxon_pass", False))
    pivot_lines.append("\nObserved metrics:\n")
    pivot_lines.append(f"- `tei_weights` per-arm verdict: **{tei_weights_v}**")
    pivot_lines.append(
        f"- Cross-arm mean delta (tei_weights - weights_only): {mean_delta * 100:+.2f}pp",
    )
    pivot_lines.append(f"- Per-arm gate pass (tei_weights): {per_arm_gate_pass}")
    pivot_lines.append(f"- Wilcoxon p-threshold met: {wilcoxon_pass}")
    pivot_lines.append("\nThe full pivot table lives in design.md § D6.")
    path.write_text("\n".join(pivot_lines) + "\n", encoding="utf-8")


def _write_m614_frequency_prior_trigger(path: Path) -> None:
    """Emit m614_frequency_prior_trigger.md when the primary verdict is GO.

    M6.14 follow-up: the substrate-prior signal observed in M6.13 may
    or may not require the full bias-network MLP. M6.14 ablates the
    substrate down to a minimum-viable form (frequency prior — a 4-
    element action-frequency vector, no sensory conditioning) and
    re-runs the M6.13 three-arm campaign at the locked-in K_test. If
    the frequency prior recovers the bias-network's GO signal, the
    minimum-viable substrate is sufficient. If only the bias-network
    works, sensory conditioning is the load-bearing feature.
    """
    body = (
        "# M6.14 trigger — frequency-prior ablation\n\n"
        "The M6.13 primary verdict was **GO**: the composed mode\n"
        "(weights+transgenerational, tei_weights arm) outperforms M3\n"
        "alone (weights_only) by a statistically distinguishable\n"
        "margin (Wilcoxon p < 0.10 AND ≥ 5pp mean delta AND\n"
        "non-overlapping 80% bootstrap CIs).\n\n"
        "**Recommended next step**: scaffold the M6.14 OpenSpec change\n"
        "``add-frequency-prior-ablation``. The bias-network substrate\n"
        "is a 3-input by 8-hidden by 4-output sensory-conditional MLP\n"
        "(~60 parameters). M6.14 ablates it down to the minimum-viable\n"
        "substrate — a 4-element action-frequency vector (~4 parameters,\n"
        "no sensory conditioning) — and re-runs the M6.13 three-arm\n"
        "campaign at the locked-in K_test. Decision gate: if the\n"
        "frequency prior recovers the bias-network's positive delta\n"
        "vs weights_only, the minimum-viable substrate is sufficient\n"
        "(closer to the wet-lab Kaletsky 'low-bandwidth switch'\n"
        "framing). If only the bias-network produces the signal,\n"
        "sensory conditioning is the load-bearing feature.\n\n"
        "Use `/openspec:new-change add-frequency-prior-ablation`.\n"
    )
    path.write_text(body, encoding="utf-8")


def _write_m613_null_finding_note(path: Path) -> None:
    """Emit m613_null_finding_note.md when the primary verdict is STOP."""
    body = (
        "# M6.13 null finding\n\n"
        "The M6.13 primary verdict was **STOP**: the composed mode\n"
        "(weights+transgenerational, tei_weights arm) is not\n"
        "statistically distinguishable from M3 alone (weights_only)\n"
        "on F1+ retention at the chosen K_test.\n\n"
        "**Recommended next step**: M6.14 is **NOT** scaffolded —\n"
        "the frequency-prior ablation is only motivated when M6.13\n"
        "demonstrates a positive bias-network signal worth simplifying.\n"
        "Combined with PR-A's STOP on pure-TEI K=0, the M6.13 null\n"
        "closes the M6 thread: TEI does not transfer on this RL\n"
        "substrate in either form (pure-TEI floor OR substrate-on-top-\n"
        "of-M3 acceleration). Logbook 020 documents the result; future\n"
        "work (Phase 6 quantum substrates, M7 NEAT) takes a different\n"
        "direction.\n\n"
        "See design.md § D4 for the M6.14 trigger criterion.\n"
    )
    path.write_text(body, encoding="utf-8")


def main() -> int:  # noqa: C901 - linear orchestration; nested helpers would obscure flow
    """Entry point for the M6.13 three-arm aggregator."""
    parser = argparse.ArgumentParser(
        description="M6.13 TEI-as-prior-on-M3 three-arm pilot/full aggregator.",
    )
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
            "termination tracker enabled?",
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

    # Cross-arm pairwise stats (M6.13 reframe):
    # - PRIMARY pair: (tei_weights, weights_only). The cross-arm
    #   verdict tests whether the substrate prior accelerates M3
    #   retraining at K_test.
    # - SECONDARY pair 1: (weights_only, control) — M3 re-reproduction
    #   at K_test (should match PR-A's +17.5pp scaled to this K).
    # - SECONDARY pair 2: (tei_weights, control) — composed arm vs
    #   floor; sanity check that the composed arm is at least as good
    #   as TPE-fresh.
    all_seeds = sorted({s for seeds in seeds_per_arm.values() for s in seeds})
    cross_arm_pairs: list[tuple[str, str]] = [
        (ARM_TEI_WEIGHTS, ARM_WEIGHTS_ONLY),
        (ARM_WEIGHTS_ONLY, ARM_CONTROL),
        (ARM_TEI_WEIGHTS, ARM_CONTROL),
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

    # Primary verdict: tei_weights vs weights_only (the first pair).
    # Pass ``mode`` so the verdict reports INDETERMINATE rather than
    # STOP when the full campaign comes in under-powered (n<4 seeds
    # on the primary pair).
    primary_stats = cross_arm_results[0]
    primary_verdict = compute_cross_arm_primary_verdict(
        per_arm_evals[ARM_TEI_WEIGHTS],
        primary_stats,
        mode=args.mode,
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
                "tei_weights_per_arm_verdict": per_arm_verdicts.get(ARM_TEI_WEIGHTS, "UNKNOWN"),
                "primary_verdict_dict": primary_verdict,
                "survival_table": survival_table,
                "seeds": all_seeds,
                "cross_arm_results": cross_arm_results,
            },
            path=args.output_dir / "pilot_pivot_decision.md",
        )
    # M6.14 trigger and M6.13 null-finding note are FULL-campaign
    # decisions only. Pilot mode has n=1 (Wilcoxon p=0.5 always →
    # verdict=STOP), so emitting m613_null_finding_note.md after the
    # pilot would falsely claim the campaign is dead when the pilot is
    # just under-powered. Pilot's only verdict artefact is
    # pilot_pivot_decision.md above. Under INDETERMINATE the campaign
    # is structurally under-powered; neither M6.14 trigger nor M6.13
    # null-finding is appropriate — operator must re-run with missing
    # seeds first.
    if args.mode == "full":
        if primary_verdict["verdict"] == "GO":
            _write_m614_frequency_prior_trigger(
                args.output_dir / "m614_frequency_prior_trigger.md",
            )
        elif primary_verdict["verdict"] == "STOP":
            _write_m613_null_finding_note(args.output_dir / "m613_null_finding_note.md")

    logger.info("M6.13 aggregator output written to %s", args.output_dir)
    logger.info("Per-arm verdicts: %s", per_arm_verdicts)
    logger.info("Cross-arm primary verdict: %s", primary_verdict["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
