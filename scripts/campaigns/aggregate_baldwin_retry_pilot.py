# pragma: no cover
r"""Aggregate the M4.5 Baldwin retry pilot's 4 arms + F1 + baseline.

Reads per-seed history.csv from THREE evolution arms (Baldwin,
Control, Lamarckian-rerun, each laid out under
``<root>/seed-<N>/[<session>/]history.csv``), the run_simulation.py-driven
hand-tuned baseline (per-seed log files), and the F1 learning-
acceleration CSV (one row per (seed, k_prime, episodes) triple,
written by ``baldwin_f1_postpilot_eval.py``).

Differences from the M4 aggregator (``aggregate_m4_pilot.py``):

1. **Pre-flight schema-equalisation check** (audit A1 closure per
   add-baldwin-retry design Decision 2).  Reads the first-evaluated
   generation's mean best_fitness from each arm (history.csv first
   data row, or equivalently lineage.csv generation == 0 — both
   refer to the same evaluations) and forces verdict to INCONCLUSIVE
   if |Δ| between Baldwin and Control means exceeds 0.05.  If A1 is
   not actually closed despite the schema-equalisation, no Baldwin-
   vs-Control claim can be made and the gates are skipped.

2. **F1 gate is recalibrated** to compare elite vs schema-prior
   baseline (paired K'-train per Decision 3) instead of M4's
   K=0-elite vs hand-tuned baseline.  Threshold tightens from
   +0.10pp to +0.05pp because both arms now include K' training
   (within-test comparison with symmetric noise).

3. **F1 CSV format** changed: reads
   ``f1_learning_acceleration.csv`` with columns
   ``seed, k_prime, episodes, elite_success_rate,
   baseline_success_rate, signal_delta`` (vs M4's
   ``f1_innate_only.csv`` which had ``seed, success_rate,
   elite_genome_id``).  Filters by --k-prime (default 10) to
   select the headline gate's measurement when multiple K' rows
   coexist.

4. **n = 8 default** (vs M4's n = 4).

5. **Baseline footnote**: the convergence plot's hand-tuned
   baseline horizontal line is annotated ``(n=4 seeds 42-45)`` —
   M2.11's existing artefacts cover only the original 4 seeds,
   while the M4.5 arms run n=8.  This asymmetry is documented for
   readers; the baseline isn't in any gate (only context for the
   plot) so the n-mismatch doesn't break verdict logic.

A new aggregator file (rather than extending ``aggregate_m4_pilot.py``)
keeps M4's audit trail clean — logbook 014 references ``aggregate_m4_pilot.py``
by name; modifying it would invalidate that documentation.

Usage:
    uv run python scripts/campaigns/aggregate_baldwin_retry_pilot.py \
        --baldwin-root evolution_results/baldwin_retry_baldwin_lstmppo_klinotaxis_predator \
        --lamarckian-root evolution_results/baldwin_retry_lamarckian_lstmppo_klinotaxis_predator \
        --control-root evolution_results/baldwin_retry_control_lstmppo_klinotaxis_predator \
        --baseline-root evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline \
        --f1-csv artifacts/logbooks/015/baldwin_retry_pilot/summary/f1_learning_acceleration.csv \
        --output-dir artifacts/logbooks/015/baldwin_retry_pilot/summary
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

# Best-fitness threshold for the Speed metric.  0.92 is calibrated to
# the predator-arm fitness landscape's documented saturation ceiling
# (per logbook 013).  Module-level so reviewers see the constant
# without grepping.
TARGET_FITNESS = 0.92

# Speed-gate tolerance: Baldwin GOs if it reaches TARGET_FITNESS at
# least this many generations earlier on average than the from-scratch
# control.  Set to 2 per add-baldwin-retry design Decision 5
# (~2-3sigma at n=8 vs M4's n=4 where +2 was roughly 1sigma).
SPEED_GAIN_GENERATIONS = 2

# F1 learning-acceleration gate threshold: Baldwin's elite genome
# (K'-trained for K' = 10 episodes per Decision 3) must beat the
# schema-prior baseline genome (also K'-trained for K' = 10) by at
# least this much.  Tightened from M4's +0.10 because both arms now
# include K' training — within-test comparison with symmetric noise
# (M4's was K=0 elite vs 100-episode-trained baseline, asymmetric).
# +0.05 ≈ 1.4sigma at n=8 with L=25 binomial noise — defensible per
# task 1.4's audit, with a documented type-I/type-II tradeoff.
F1_LEARNING_ACCELERATION_THRESHOLD = 0.05

# Comparative-gate tolerance: Baldwin's gen-to-092 must be within this
# many generations of Lamarckian's.  Sanity tripwire — at M3's
# published Lamarckian 4.5, this means Baldwin <= 8.5.
COMPARATIVE_GAP_GENERATIONS = 4

# Schema-equalisation pre-flight threshold: |Δ| between Baldwin and
# Control gen-0 mean fitness MUST be ≤ this for audit A1 to be
# considered closed.  Tighter than M4's measured |Δ| = 0.14.
# Per add-baldwin-retry design Decision 2.
SCHEMA_EQUALISATION_TOLERANCE = 0.05

# Default K' filter for the F1 gate.  Multiple K' values can coexist
# in the F1 CSV (per Decision 3 + task 8.3 fallback path); the gate
# uses K' = 10 unless --k-prime overrides.
DEFAULT_K_PRIME = 10


def _resolve_session(seed_dir: Path) -> Path:
    """Return the directory containing this seed's history.csv.

    Two layouts supported (matches the F1 evaluator's _resolve_session):

    1. Direct: ``seed_dir/history.csv``
    2. Nested: ``seed_dir/<session_id>/history.csv``

    Direct layout wins when both exist.
    """
    direct = seed_dir / "history.csv"
    if direct.exists():
        return seed_dir
    sessions = [p for p in seed_dir.iterdir() if p.is_dir()]
    if not sessions:
        msg = f"No history.csv at {direct} and no session subdirectories under {seed_dir}."
        raise FileNotFoundError(msg)
    return max(sessions, key=lambda p: p.stat().st_mtime)


def _read_history(seed_dir: Path) -> list[dict[str, float]]:
    """Read the latest session's history.csv as a list of float dicts."""
    history_path = _resolve_session(seed_dir) / "history.csv"
    if not history_path.exists():  # pragma: no cover - defensive
        msg = f"No history.csv at {history_path}"
        raise FileNotFoundError(msg)
    with history_path.open() as handle:
        reader = csv.DictReader(handle)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def _baseline_success_rates(baseline_root: Path) -> dict[int, float]:
    """Extract per-seed success rates from the run_simulation.py log files.

    Note: M2.11's baseline covers seeds 42-45 only (n=4).  M4.5's pilot
    arms run n=8 — the per-seed table will show "—" for the seeds
    without baseline data, and the convergence plot's baseline
    horizontal line is annotated to disclose the n-asymmetry.
    """
    rates: dict[int, float] = {}
    for log in sorted(baseline_root.glob("seed-*.log")):
        seed_match = re.search(r"seed-(\d+)\.log", log.name)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        for line in log.read_text().splitlines():
            m = re.match(r"^Success rate:\s+([\d.]+)%", line)
            if m:
                rates[seed] = float(m.group(1)) / 100.0
                break
    return rates


def _read_f1_csv(
    f1_csv: Path,
    *,
    k_prime: int,
) -> dict[int, tuple[float, float, float]]:
    """Read f1_learning_acceleration.csv and filter by K'.

    Returns ``{seed: (elite_success_rate, baseline_success_rate, signal_delta)}``
    for each row matching the requested K'.  Caller raises clearly
    if any pilot seed is missing for that K'.
    """
    if not f1_csv.exists():
        msg = (
            f"F1 CSV missing at {f1_csv}.  Run "
            f"scripts/campaigns/baldwin_f1_postpilot_eval.py first."
        )
        raise FileNotFoundError(msg)
    rows: dict[int, tuple[float, float, float]] = {}
    with f1_csv.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_kprime = int(row["k_prime"])
            if row_kprime != k_prime:
                continue
            seed = int(row["seed"])
            rows[seed] = (
                float(row["elite_success_rate"]),
                float(row["baseline_success_rate"]),
                float(row["signal_delta"]),
            )
    return rows


def _gen_first_reaches_target(
    history: list[dict[str, float]],
    target: float = TARGET_FITNESS,
) -> int | None:
    """Return the 1-indexed generation at which best_fitness first reaches ``target``.

    history.csv is 1-indexed at write time per the framework's
    existing convention; we return that label directly so per-seed
    tables match the file's own labelling.
    """
    for row in history:
        if row["best_fitness"] >= target:
            return int(row["generation"])
    return None


def _resolve_speed(g: int | None, fallback_gen: int) -> float:
    """Resolve a per-seed gen-to-target value to a float for averaging.

    Treats never-reached as the fallback (run's max generation + 1) so
    the metric is bounded; conservative for the GO check.
    """
    return float(g) if g is not None else float(fallback_gen)


def _first_gen_mean_fitness(
    histories: dict[int, list[dict[str, float]]],
    seeds: list[int],
) -> float:
    """Mean best_fitness of the first-evaluated generation across seeds.

    Reads ``history.csv`` first data row per seed (this is labelled
    ``generation = 1`` in the file because history.csv is 1-indexed,
    while the corresponding ``lineage.csv`` rows have ``generation == 0``
    because lineage.csv is 0-indexed — both refer to the same set of
    evaluations per the framework's existing convention).  Used for
    the audit A1 schema-equalisation pre-flight check.
    """
    values = [histories[s][0]["best_fitness"] for s in seeds]
    return float(np.mean(values))


def _check_schema_equalisation(
    baldwin_history: dict[int, list[dict[str, float]]],
    ctrl_history: dict[int, list[dict[str, float]]],
    seeds: list[int],
) -> tuple[bool, float, float, float]:
    """Audit A1 closure check.

    Returns ``(passes, baldwin_mean, ctrl_mean, abs_delta)``.  ``passes``
    is True iff |Δ| ≤ SCHEMA_EQUALISATION_TOLERANCE.  Failure forces
    the verdict to INCONCLUSIVE; gates are skipped.
    """
    baldwin_mean = _first_gen_mean_fitness(baldwin_history, seeds)
    ctrl_mean = _first_gen_mean_fitness(ctrl_history, seeds)
    abs_delta = abs(baldwin_mean - ctrl_mean)
    return abs_delta <= SCHEMA_EQUALISATION_TOLERANCE, baldwin_mean, ctrl_mean, abs_delta


def _compute_verdict(
    *,
    speed_gate_passes: bool,
    f1_gate_passes: bool,
    comparative_gate_passes: bool,
    schema_equalisation_passes: bool,
) -> tuple[str, str]:
    """Compute GO / PIVOT / STOP / INCONCLUSIVE verdict from the gates.

    Verdict logic:
      - INCONCLUSIVE if the schema-equalisation pre-flight check fails
        (audit A1 not actually closed; no Baldwin-vs-Control claim
        can be made regardless of the gates).
      - GO if all three gates pass.
      - PIVOT if the speed gate passes but at least one of the others
        fails.
      - STOP if the speed gate fails (regardless of the other two —
        without speed, Baldwin offers no improvement over control).
    """
    if not schema_equalisation_passes:
        return (
            "INCONCLUSIVE ⚠️",
            "Audit A1 (schema-shift confounder) is NOT closed: gen-0 "
            "mean best_fitness diverges between Baldwin and Control "
            f"by more than {SCHEMA_EQUALISATION_TOLERANCE} despite "
            "schema equalisation.  No Baldwin-vs-Control claim can be "
            "made from this pilot.  Investigate seed plumbing, encoder "
            "initialisation, and fitness-function determinism before "
            "re-running.",
        )
    if speed_gate_passes and f1_gate_passes and comparative_gate_passes:
        return (
            "GO ✅",
            "Baldwin inheritance accelerates convergence (speed gate), "
            "the elite genome learns faster than a schema-prior baseline "
            "at K' = 10 (F1 learning-acceleration gate), AND its "
            "convergence speed is competitive with Lamarckian "
            "(comparative gate).  Inheritance of learning bias is doing "
            "real work — proceed to the next experiment with Baldwin in "
            "the substrate.",
        )
    if speed_gate_passes:
        failed = []
        if not f1_gate_passes:
            failed.append("F1 learning-acceleration")
        if not comparative_gate_passes:
            failed.append("comparative")
        joined = " and ".join(failed)
        return (
            "PIVOT ⚠️",
            f"The speed gate passed but the {joined} gate(s) failed.  "
            "Baldwin shows partial signal — inheritance helps "
            "convergence but either the elite genome doesn't accelerate "
            "learning above a schema-prior baseline, or it underperforms "
            "Lamarckian by more than the comparative threshold.  Treat "
            "as inconclusive and review the per-seed trajectories before "
            "committing follow-up scope.",
        )
    return (
        "STOP ❌",
        "The speed gate failed: Baldwin does not accelerate convergence "
        "over the from-scratch control by the required margin.  Per the "
        "pre-registered STOP semantic (Decision 6): the Baldwin Effect "
        "is NOT exhibited on this testbed.  M5 (co-evolution) proceeds "
        "without Baldwin in its substrate; M6 (transgenerational memory) "
        "uses Lamarckian.  No further Baldwin pilot in this Phase.",
    )


def _format_summary(  # noqa: PLR0913
    seeds: list[int],
    baldwin_history: dict[int, list[dict[str, float]]],
    lam_history: dict[int, list[dict[str, float]]],
    ctrl_history: dict[int, list[dict[str, float]]],
    baseline_mean: float,
    speed_baldwin: dict[int, int | None],
    speed_lam: dict[int, int | None],
    speed_ctrl: dict[int, int | None],
    f1_rows: dict[int, tuple[float, float, float]],
    k_prime: int,
) -> str:
    """Render the schema-equalisation check + 3-gate verdict + per-seed details."""
    # Pre-flight: schema-equalisation check (audit A1 closure).
    schema_ok, baldwin_g0, ctrl_g0, abs_delta = _check_schema_equalisation(
        baldwin_history,
        ctrl_history,
        seeds,
    )
    schema_status = (
        "PASS" if schema_ok else f"FAIL (|Δ| = {abs_delta:.4f} > {SCHEMA_EQUALISATION_TOLERANCE})"
    )

    max_gens = max(
        len(h)
        for h in (
            *baldwin_history.values(),
            *lam_history.values(),
            *ctrl_history.values(),
        )
    )
    fallback_gen = max_gens + 1

    speed_baldwin_vals = [_resolve_speed(speed_baldwin[s], fallback_gen) for s in seeds]
    speed_lam_vals = [_resolve_speed(speed_lam[s], fallback_gen) for s in seeds]
    speed_ctrl_vals = [_resolve_speed(speed_ctrl[s], fallback_gen) for s in seeds]
    speed_mean_baldwin = float(np.mean(speed_baldwin_vals))
    speed_mean_lam = float(np.mean(speed_lam_vals))
    speed_mean_ctrl = float(np.mean(speed_ctrl_vals))

    elite_means = [f1_rows[s][0] for s in seeds]
    baseline_f1_means = [f1_rows[s][1] for s in seeds]
    delta_means = [f1_rows[s][2] for s in seeds]
    f1_elite_mean = float(np.mean(elite_means))
    f1_baseline_mean = float(np.mean(baseline_f1_means))
    f1_signal_mean = float(np.mean(delta_means))

    speed_gate_passes = (speed_mean_baldwin + SPEED_GAIN_GENERATIONS) <= speed_mean_ctrl
    f1_gate_passes = f1_signal_mean > F1_LEARNING_ACCELERATION_THRESHOLD
    comparative_gate_passes = speed_mean_baldwin <= (speed_mean_lam + COMPARATIVE_GAP_GENERATIONS)

    verdict, verdict_text = _compute_verdict(
        speed_gate_passes=speed_gate_passes,
        f1_gate_passes=f1_gate_passes,
        comparative_gate_passes=comparative_gate_passes,
        schema_equalisation_passes=schema_ok,
    )

    lines: list[str] = [
        "## Baldwin Retry Pilot — Summary (M4.5)",
        "",
        f"**Seeds**: {seeds} (n={len(seeds)})",
        f"**Hand-tuned baseline mean**: {baseline_mean:.3f} (n=4 seeds 42-45 from M2.11)",
        f"**F1 K' (training budget)**: {k_prime}",
        "",
        "### Schema-equalisation pre-flight check (audit A1 closure)",
        "",
        "| Arm | First-gen mean best_fitness |",
        "|-----|------------------------------|",
        f"| Baldwin | {baldwin_g0:.4f} |",
        f"| Control | {ctrl_g0:.4f} |",
        f"| **\\|Δ\\|** | **{abs_delta:.4f}** (tolerance: {SCHEMA_EQUALISATION_TOLERANCE}) |",
        f"| **Status** | **{schema_status}** |",
        "",
    ]

    if not schema_ok:
        lines.extend(
            [
                "⚠️ **Audit A1 NOT closed** — gates SKIPPED.  Investigate "
                "seed plumbing, encoder initialisation, or fitness-function "
                "determinism before re-running.  See verdict text below.",
                "",
                f"**Decision**: {verdict}",
                "",
                verdict_text,
            ],
        )
        return "\n".join(lines)

    lines.extend(
        [
            "### Decision Gates",
            "",
            f"- **Speed gate** (mean_gen_baldwin + {SPEED_GAIN_GENERATIONS} "
            f"<= mean_gen_control): {'PASS' if speed_gate_passes else 'FAIL'}",
            f"  - Baldwin mean gen-to-{TARGET_FITNESS}: {speed_mean_baldwin:.2f}",
            f"  - Control mean gen-to-{TARGET_FITNESS}: {speed_mean_ctrl:.2f}",
            f"  - Margin: {speed_mean_ctrl - speed_mean_baldwin:+.2f} "
            f"(need >= {SPEED_GAIN_GENERATIONS})",
            "",
            f"- **F1 learning-acceleration gate** (mean elite - mean baseline > "
            f"{F1_LEARNING_ACCELERATION_THRESHOLD}, K' = {k_prime}): "
            f"{'PASS' if f1_gate_passes else 'FAIL'}",
            f"  - Baldwin elite mean (K'={k_prime}, L=25):    {f1_elite_mean:.3f}",
            f"  - Schema-prior baseline mean (K'={k_prime}, L=25): {f1_baseline_mean:.3f}",
            f"  - Signal delta mean: {f1_signal_mean:+.3f} "
            f"(need > {F1_LEARNING_ACCELERATION_THRESHOLD})",
            "",
            f"- **Comparative gate** (mean_gen_baldwin <= "
            f"mean_gen_lamarckian + {COMPARATIVE_GAP_GENERATIONS}): "
            f"{'PASS' if comparative_gate_passes else 'FAIL'}",
            f"  - Baldwin mean gen-to-{TARGET_FITNESS}: {speed_mean_baldwin:.2f}",
            f"  - Lamarckian mean gen-to-{TARGET_FITNESS}: {speed_mean_lam:.2f}",
            f"  - Margin: {speed_mean_lam + COMPARATIVE_GAP_GENERATIONS - speed_mean_baldwin:+.2f} "
            f"(need >= 0)",
            "",
            f"**Decision**: {verdict}",
            "",
            verdict_text,
            "",
            "### Per-seed convergence speed (generations to first reach "
            f"best_fitness >= {TARGET_FITNESS}) + F1 learning-acceleration",
            "",
            "| Seed | Baldwin | Lamarckian | Control | F1 elite | F1 baseline | F1 signal |",
            "|------|---------|------------|---------|----------|-------------|-----------|",
        ],
    )

    def _fmt_gen(g: int | None) -> str:
        return str(g) if g is not None else "—"

    for s in seeds:
        elite, baseline_v, delta = f1_rows[s]
        lines.append(
            f"| {s} | {_fmt_gen(speed_baldwin[s])} | {_fmt_gen(speed_lam[s])} "
            f"| {_fmt_gen(speed_ctrl[s])} | {elite:.3f} | {baseline_v:.3f} | {delta:+.3f} |",
        )

    return "\n".join(lines)


def _write_speed_csv(  # noqa: PLR0913
    output_dir: Path,
    seeds: list[int],
    speed_baldwin: dict[int, int | None],
    speed_lam: dict[int, int | None],
    speed_ctrl: dict[int, int | None],
    f1_rows: dict[int, tuple[float, float, float]],
) -> Path:
    """Write the per-seed convergence-speed + F1 table."""
    path = output_dir / "convergence_speed.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "seed",
                "baldwin_gen_to_092",
                "lamarckian_gen_to_092",
                "control_gen_to_092",
                "f1_elite_success_rate",
                "f1_baseline_success_rate",
                "f1_signal_delta",
            ),
        )
        for s in seeds:
            elite, baseline_v, delta = f1_rows[s]
            writer.writerow(
                (
                    s,
                    speed_baldwin[s] if speed_baldwin[s] is not None else "",
                    speed_lam[s] if speed_lam[s] is not None else "",
                    speed_ctrl[s] if speed_ctrl[s] is not None else "",
                    f"{elite:.6f}",
                    f"{baseline_v:.6f}",
                    f"{delta:+.6f}",
                ),
            )
    return path


def _plot_convergence(  # noqa: PLR0913
    output_dir: Path,
    seeds: list[int],
    baldwin_history: dict[int, list[dict[str, float]]],
    lam_history: dict[int, list[dict[str, float]]],
    ctrl_history: dict[int, list[dict[str, float]]],
    baseline_mean: float,
    f1_rows: dict[int, tuple[float, float, float]],
) -> Path | None:
    """4-curve plot: Baldwin + Lamarckian + Control mean ± std + baseline + F1 markers.

    Baseline horizontal line is annotated with `(n=4)` since M2.11's
    artefacts only cover seeds 42-45 while the pilot arms run n=8.
    """
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; skipping plot)")
        return None

    def _stack(histories: dict[int, list[dict[str, float]]]) -> np.ndarray:
        # Per-seed histories may have unequal lengths if early-stop
        # fired at different generations across seeds.  Truncate every
        # seed to the shortest length so np.array sees a rectangular
        # nested list (within-arm truncation).
        min_len = min(len(histories[s]) for s in seeds)
        return np.array(
            [[row["best_fitness"] for row in histories[s][:min_len]] for s in seeds],
            dtype=np.float64,
        )

    baldwin = _stack(baldwin_history)
    lam = _stack(lam_history)
    ctrl = _stack(ctrl_history)

    # Cross-arm alignment: truncate all to the shortest arm.
    n = min(baldwin.shape[1], lam.shape[1], ctrl.shape[1])
    baldwin = baldwin[:, :n]
    lam = lam[:, :n]
    ctrl = ctrl[:, :n]
    gens = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label, data, colour in (
        ("Baldwin", baldwin, "tab:green"),
        ("Lamarckian", lam, "tab:blue"),
        ("Control (no-inheritance)", ctrl, "tab:orange"),
    ):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(gens, mean, color=colour, linewidth=2.5, label=f"{label} mean")
        ax.fill_between(
            gens,
            mean - std,
            mean + std,
            color=colour,
            alpha=0.18,
            label=f"{label} ±1 std",
        )

    ax.axhline(
        baseline_mean,
        linestyle="--",
        color="grey",
        label=f"Hand-tuned baseline ({baseline_mean:.2f}, n=4 seeds 42-45)",
    )
    ax.axhline(
        TARGET_FITNESS,
        linestyle=":",
        color="green",
        label=f"Target fitness ({TARGET_FITNESS})",
    )

    # F1 learning-acceleration: elite + baseline pair per seed at the
    # right margin.  Two marker styles let the reader distinguish.
    f1_x_elite = float(n) + 0.5
    f1_x_baseline = float(n) + 1.0
    elite_values = [f1_rows[s][0] for s in seeds]
    baseline_values = [f1_rows[s][1] for s in seeds]
    ax.scatter(
        [f1_x_elite] * len(elite_values),
        elite_values,
        marker="*",
        color="tab:green",
        s=120,
        zorder=5,
        label=f"F1 elite (mean {float(np.mean(elite_values)):.2f})",
    )
    ax.scatter(
        [f1_x_baseline] * len(baseline_values),
        baseline_values,
        marker="x",
        color="tab:red",
        s=80,
        zorder=5,
        label=f"F1 schema-prior baseline (mean {float(np.mean(baseline_values)):.2f})",
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness across population")
    ax.set_title(
        f"Baldwin retry pilot — convergence "
        f"({len(seeds)} seed{'s' if len(seeds) != 1 else ''}; M4.5)",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(visible=True, alpha=0.3)

    plot_path = output_dir / "convergence.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    return plot_path


def main() -> int:
    """Aggregate Baldwin + Lamarckian + Control + baseline + F1 → summary + plot + csv."""
    parser = argparse.ArgumentParser(
        description="Aggregate the M4.5 Baldwin retry pilot's results.",
    )
    parser.add_argument("--baldwin-root", type=Path, required=True)
    parser.add_argument("--lamarckian-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help=(
            "Path to the run_simulation.py-driven hand-tuned baseline tree.  "
            "Default canonical location: "
            "evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/  "
            "(M2.11's 4 seeds; n-asymmetry vs the M4.5 pilot's 8 seeds is "
            "footnoted on the convergence plot)."
        ),
    )
    parser.add_argument(
        "--f1-csv",
        type=Path,
        required=True,
        help=(
            "Path to f1_learning_acceleration.csv produced by "
            "scripts/campaigns/baldwin_f1_postpilot_eval.py.  Multiple K' "
            "rows can coexist; the gate uses --k-prime to filter."
        ),
    )
    parser.add_argument(
        "--k-prime",
        type=int,
        default=DEFAULT_K_PRIME,
        help=(
            f"K' value to use for the F1 gate (default {DEFAULT_K_PRIME}).  "
            "Filters f1_learning_acceleration.csv rows by k_prime column."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46, 47, 48, 49],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baldwin_history: dict[int, list[dict[str, float]]] = {}
    lam_history: dict[int, list[dict[str, float]]] = {}
    ctrl_history: dict[int, list[dict[str, float]]] = {}
    for seed in args.seeds:
        baldwin_history[seed] = _read_history(args.baldwin_root / f"seed-{seed}")
        lam_history[seed] = _read_history(args.lamarckian_root / f"seed-{seed}")
        ctrl_history[seed] = _read_history(args.control_root / f"seed-{seed}")

    # Baseline (run_simulation.py-driven, optimiser-independent, n=4 only).
    baseline_rates = _baseline_success_rates(args.baseline_root)
    baseline_seeds_present = sorted(baseline_rates)
    if not baseline_seeds_present:
        msg = (
            f"No baseline success rates found under {args.baseline_root}.  "
            f"Expected per-seed log files matching seed-N.log with a "
            f"'Success rate: NN.NN%' line.  Re-run "
            f"phase5_m2_hyperparam_lstmppo_klinotaxis_predator_baseline.sh "
            f"to populate {args.baseline_root}."
        )
        raise SystemExit(msg)
    # Use the seeds present in the baseline (subset of pilot seeds);
    # the convergence plot's annotation discloses the n-asymmetry.
    baseline_mean = float(np.mean([baseline_rates[s] for s in baseline_seeds_present]))

    # F1 learning-acceleration rates (from the post-pilot evaluator,
    # filtered by --k-prime).
    f1_rows = _read_f1_csv(args.f1_csv, k_prime=args.k_prime)
    missing_f1 = sorted(set(args.seeds) - set(f1_rows))
    if missing_f1:
        msg = (
            f"Missing F1 row(s) for seed(s) {missing_f1} at K' = {args.k_prime} "
            f"in {args.f1_csv}.  Re-run scripts/campaigns/baldwin_f1_postpilot_eval.py "
            f"--k-prime {args.k_prime} --seeds {' '.join(str(s) for s in args.seeds)} "
            f"to populate."
        )
        raise SystemExit(msg)

    # Convergence-speed metric per arm per seed.
    speed_baldwin = {s: _gen_first_reaches_target(baldwin_history[s]) for s in args.seeds}
    speed_lam = {s: _gen_first_reaches_target(lam_history[s]) for s in args.seeds}
    speed_ctrl = {s: _gen_first_reaches_target(ctrl_history[s]) for s in args.seeds}

    summary_md = _format_summary(
        args.seeds,
        baldwin_history,
        lam_history,
        ctrl_history,
        baseline_mean,
        speed_baldwin,
        speed_lam,
        speed_ctrl,
        f1_rows,
        args.k_prime,
    )
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text(summary_md + "\n")
    print(summary_md)
    print()
    print(f"Summary written to {summary_path}")

    csv_path = _write_speed_csv(
        args.output_dir,
        args.seeds,
        speed_baldwin,
        speed_lam,
        speed_ctrl,
        f1_rows,
    )
    print(f"Convergence-speed table written to {csv_path}")

    plot_path = _plot_convergence(
        args.output_dir,
        args.seeds,
        baldwin_history,
        lam_history,
        ctrl_history,
        baseline_mean,
        f1_rows,
    )
    if plot_path is not None:
        print(f"Plot written to {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
