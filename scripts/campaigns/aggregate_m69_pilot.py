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
    aggregate_per_arm_verdict,
    build_survival_table,
    compute_cross_arm_delta_stats,
    evaluate_decision_gate_one_seed,
    load_f0_training_fitness_per_seed,
    read_per_gen_csv,
    require_complete_f0_override,
    write_cross_arm_verdict_csv,
)

# numpy/scipy are no longer imported here: every function that used them now
# lives in ``_common``, whose own guarded import raises the same actionable
# ImportError before this module's body runs.

logger = logging.getLogger(__name__)


# Cross-arm primary-verdict thresholds (M6.9+ noise-aware verdict).
CROSS_ARM_WILCOXON_P_THRESHOLD = 0.10
CROSS_ARM_MIN_DELTA_PP = 0.05  # 5 percentage points

# Full-campaign target n. When the primary pair (tei_on - control)
# has fewer than this many paired-seed deltas the cross-arm verdict
# is INDETERMINATE — the one-sided Wilcoxon at n=3 has minimum
# achievable p = 0.125, which can never satisfy
# CROSS_ARM_WILCOXON_P_THRESHOLD = 0.10. A null verdict in that
# regime cannot be distinguished from a real null, so we surface
# the under-powered state explicitly rather than silently labelling
# it STOP. M6 closed INCONCLUSIVE precisely because this distinction
# was not made.
CROSS_ARM_FULL_N_SEEDS = 4

# Three-arm campaign arms.
ARM_TEI_ON = "tei_on"
ARM_WEIGHTS_ONLY = "weights_only"
ARM_CONTROL = "control"
EXPECTED_ARMS = (ARM_TEI_ON, ARM_WEIGHTS_ONLY, ARM_CONTROL)


def compute_cross_arm_primary_verdict(
    tei_on_seed_evaluations: list[dict],
    cross_arm_stats: dict,
    *,
    mode: str = "full",
) -> dict:
    """Cross-arm primary verdict: GO iff per-arm gate passes AND noise-aware delta is positive.

    Per the evolution-framework spec § "Cross-Arm Statistical Verdict":
    GO requires ALL of:
      1. tei_on per-arm gate passes (≥ 2/4 seeds pass).
      2. Wilcoxon p < 0.10 (one-sided, tei_on > control).
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
    tei_on_arm_verdict = aggregate_per_arm_verdict(tei_on_seed_evaluations)
    per_arm_gate_pass = tei_on_arm_verdict == "GO"
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
        "tei_on_arm_verdict": tei_on_arm_verdict,
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


def _detect_chance_collapse(survival_table: dict, seeds: list[int]) -> bool:
    """Pilot pivot row 1: all 3 arms collapse to ≈ chance survival at F0.

    Heuristic: every arm's F0 survival across observed seeds is below
    the lower bound of the F0 envelope (0.30 — T1 tripwire). When the
    pilot lands here the reward-mode + env density combination is too
    punishing and the experiment cannot differentiate substrate
    signal from random-walk floor.
    """
    threshold = 0.30
    for arm in EXPECTED_ARMS:
        f0_means = [
            survival_table.get((arm, seed, 0))
            for seed in seeds
            if survival_table.get((arm, seed, 0)) is not None
        ]
        if not f0_means:
            return False
        if max(v for v in f0_means if v is not None) >= threshold:
            return False
    return True


def _detect_monotone_violated(
    survival_table: dict,
    arm: str,
    seeds: list[int],
) -> bool:
    """Pilot pivot row 4: F1 > F0 on the primary arm (monotone-decay broken).

    A substrate that *strengthens* with lineage depth is unstable
    by construction — the MLP fit is too sensitive to F0 elite
    idiosyncrasies and the F1+ cascade amplifies them rather than
    decays them.
    """
    for seed in seeds:
        f0 = survival_table.get((arm, seed, 0))
        f1 = survival_table.get((arm, seed, 1))
        if f0 is not None and f1 is not None and f1 > f0:
            return True
    return False


def _detect_matched_by_weights(cross_arm_results: list[dict]) -> bool:
    """Pilot pivot row 5: tei_on > control by ≥5pp BUT weights_only ≈ tei_on.

    Substrate signal is real (positive vs control) but matched by
    plain Lamarckian weight inheritance — substrate adds no value
    on top of trained weights. PR-B (TEI+weights) becomes the
    load-bearing question; the pure-TEI floor is necessary but
    insufficient for the strongest scientific claim.
    """
    tei_vs_control = next(
        (r for r in cross_arm_results if r["arm_a"] == "tei_on" and r["arm_b"] == "control"),
        None,
    )
    tei_vs_weights = next(
        (r for r in cross_arm_results if r["arm_a"] == "tei_on" and r["arm_b"] == "weights_only"),
        None,
    )
    if tei_vs_control is None or tei_vs_weights is None:
        return False
    return (
        tei_vs_control["mean_delta"] >= CROSS_ARM_MIN_DELTA_PP
        and abs(tei_vs_weights["mean_delta"]) < 0.02
    )


def _write_pilot_pivot_decision(
    pilot_observations: dict,
    path: Path,
) -> None:
    """Emit pilot_pivot_decision.md populated from design.md § D6 pivot table.

    Classifies the pilot observation against the six pre-declared
    pivots and writes a markdown summary the user reviews BEFORE
    unblocking the full campaign.

    Required keys in ``pilot_observations``:
        - ``tei_on_per_arm_verdict``: per-arm GO/PIVOT/STOP for tei_on
        - ``primary_verdict_dict``: cross-arm primary-verdict dict
        - ``survival_table``: ``{(arm, seed, gen): mean_survival}``
        - ``seeds``: observed seeds
        - ``cross_arm_results``: list of per-pair cross-arm stats
    """
    tei_on_v = pilot_observations.get("tei_on_per_arm_verdict", "UNKNOWN")
    primary = pilot_observations.get("primary_verdict_dict", {})
    mean_delta = float(primary.get("mean_delta", 0.0))
    per_arm_gate_pass = bool(primary.get("per_arm_gate_pass", False))
    wilcoxon_pass = bool(primary.get("wilcoxon_pass", False))
    survival_table = pilot_observations.get("survival_table", {})
    seeds = pilot_observations.get("seeds", [])
    cross_arm_results = pilot_observations.get("cross_arm_results", [])

    # Match against the six pre-declared pilot observations (design.md
    # § D6). Order matters — earlier branches are more specific.
    pivot_lines = ["# Pilot pivot decision\n"]
    pivot_lines.append(
        "Per design.md § D6, the pilot's outcome is classified against six pre-declared pivots:\n",
    )
    if _detect_chance_collapse(survival_table, seeds):
        pivot_lines.append(
            "**Pilot signal: chance-floor collapse (D6 row 1).** "
            "All 3 arms' F0 survival_rate < 0.30 — env+reward is too "
            "punishing to differentiate substrate signal from a random-walk floor.",
        )
        pivot_lines.append(
            "Pivot: widen lawn distribution OR retune "
            "`penalty_predator_contact` upward; re-run F0 calibration smoke before re-pilot.\n",
        )
    elif _detect_monotone_violated(survival_table, ARM_TEI_ON, seeds):
        pivot_lines.append(
            "**Pilot signal: monotone-decay violated (D6 row 4).** "
            "`tei_on` shows F1 > F0 on at least one seed — substrate is "
            "amplifying rather than decaying. The MLP fit is too "
            "sensitive to F0 elite idiosyncrasies.",
        )
        pivot_lines.append(
            "Pivot: reduce `bias_network.hidden_dim` 8→4 OR cap MLP "
            "fit epochs 50→20. Re-run pilot.\n",
        )
    elif mean_delta > 0.05 and per_arm_gate_pass and wilcoxon_pass:
        if _detect_matched_by_weights(cross_arm_results):
            pivot_lines.append(
                "**Pilot signal: substrate matched by weights (D6 row 5).** "
                "`tei_on > control` by ≥ 5pp AND Wilcoxon significant, BUT "
                "`weights_only ≈ tei_on` — substrate signal is real but "
                "is matched by plain Lamarckian weight-flow.",
            )
            pivot_lines.append(
                "Pivot: NONE for the full campaign — but PR-B "
                "(TEI+weights symmetric-compute control) becomes the "
                "load-bearing scientific question. The pure-TEI floor "
                "is necessary but not sufficient.\n",
            )
        else:
            pivot_lines.append(
                "**Pilot signal: clean differentiation (D6 row 6).** "
                "`tei_on > control` by ≥ 5pp AND Wilcoxon significant; "
                "monotone decay holds; no weights-matched signature.",
            )
            pivot_lines.append(
                "Pivot: NONE. Proceed to full campaign with no config changes.\n",
            )
    elif tei_on_v == "PIVOT":
        pivot_lines.append(
            "**Pilot signal: F0 diverse but cascade collapses (D6 row 3).** "
            "1/4 seed passes the tei_on per-arm gate — F0 substrate "
            "differs across seeds (T2 passed pre-flight) but F1+ "
            "retention is near-uniform.",
        )
        pivot_lines.append(
            "Pivot: decay shape too aggressive (geometric collapse); "
            "try `decay_shape: linear` or `decay_factor: 0.8`. Re-run pilot.\n",
        )
    elif mean_delta < 0.01 and tei_on_v == "STOP":
        pivot_lines.append(
            "**Pilot signal: substrate likely inert (D6 row 2).** "
            "`tei_on ≈ control` at F1+ — substrate carries no measurable signal.",
        )
        pivot_lines.append(
            "Pivot: widen `bias_network.hidden_dim` 8→16 OR add features "
            "to `input_features` (e.g. `stam_state_mean`). Re-run pilot.\n",
        )
    else:
        pivot_lines.append(
            "**Pilot signal: ambiguous.** Observed pattern does not match "
            "any of the 6 pre-declared design.md § D6 pivots. Inspect "
            "retention curves + cross-arm deltas manually.",
        )
        pivot_lines.append(
            "Pivot decision: USER REVIEW REQUIRED before proceeding.\n",
        )
    pivot_lines.append("\nObserved metrics:\n")
    pivot_lines.append(f"- `tei_on` per-arm verdict: **{tei_on_v}**")
    pivot_lines.append(f"- Cross-arm mean delta (tei_on - control): {mean_delta * 100:+.2f}pp")
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

    rows = read_per_gen_csv(args.per_gen_csv)
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

    # Fail closed before gating if the override does not cover every pair about
    # to be gated (#279) — a partial override would mix training-time and
    # post-hoc F0 baselines inside one arm verdict.
    require_complete_f0_override(
        f0_override,
        [(arm, seed) for arm in EXPECTED_ARMS for seed in seeds_per_arm.get(arm, [])],
    )

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

    # Primary verdict: tei_on vs control (the first pair). Pass
    # ``mode`` so the verdict reports INDETERMINATE rather than STOP
    # when the full campaign comes in under-powered (n<4 seeds on the
    # primary pair).
    primary_stats = cross_arm_results[0]
    primary_verdict = compute_cross_arm_primary_verdict(
        per_arm_evals[ARM_TEI_ON],
        primary_stats,
        mode=args.mode,
    )

    # Emit outputs.
    _write_retention_csv(survival_table, args.output_dir / "retention_table.csv")
    _write_decision_gate_csv(all_evals, args.output_dir / "decision_gate.csv")
    write_cross_arm_verdict_csv(cross_arm_results, args.output_dir / "cross_arm_verdict.csv")
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
                "survival_table": survival_table,
                "seeds": all_seeds,
                "cross_arm_results": cross_arm_results,
            },
            path=args.output_dir / "pilot_pivot_decision.md",
        )
    # PR-B trigger and M6.13 punt-note are FULL-campaign decisions only.
    # Pilot mode has n=1 (Wilcoxon p=0.5 always → verdict=STOP), so
    # emitting m6_13_punt_note.md after the pilot would falsely claim
    # the campaign is dead when the pilot is just under-powered.
    # Pilot's only verdict artefact is pilot_pivot_decision.md above.
    # Under INDETERMINATE the campaign is structurally under-powered;
    # neither PR-B trigger nor M6.13 punt is appropriate — the
    # operator needs to re-run with the missing seeds first.
    if args.mode == "full":
        if primary_verdict["verdict"] == "GO":
            _write_pr_b_trigger(args.output_dir / "pr_b_trigger.md")
        elif primary_verdict["verdict"] == "STOP":
            _write_m6_13_punt_note(args.output_dir / "m6_13_punt_note.md")

    logger.info("M6.9+ aggregator output written to %s", args.output_dir)
    logger.info("Per-arm verdicts: %s", per_arm_verdicts)
    logger.info("Cross-arm primary verdict: %s", primary_verdict["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
