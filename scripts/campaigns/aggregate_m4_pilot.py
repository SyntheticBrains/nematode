# pragma: no cover
r"""Aggregate Baldwin / Lamarckian / control / F1 evolution results across seeds.

Reads per-seed ``history.csv`` from THREE evolution arms (Baldwin,
Lamarckian-rerun, control-rerun, each laid out under
``<root>/seed-<N>/<session>/history.csv``), the run_simulation.py-driven
hand-tuned baseline (per-seed log files), and the F1 innate-only
forensic CSV (one row per seed, written by
``baldwin_f1_postpilot_eval.py``).  Produces:

- A 4-curve convergence plot: Baldwin + Lamarckian + control mean ± std
  vs generation (best_fitness across-seed mean as solid line, ±1 std
  as shaded band).  Hand-tuned baseline + 0.92 target as horizontal
  reference lines.  F1 innate-only success rates as separate per-seed
  markers in a small inset/marker set.
- A per-seed convergence-speed table written as ``convergence_speed.csv``:
  ``seed, baldwin_gen_to_092, lamarckian_gen_to_092,
  control_gen_to_092, f1_innate_only_success_rate``.  Empty cell
  when a seed never reached the 0.92 threshold within the run.
- A markdown summary with the GO/PIVOT/STOP verdict.  Three gates:
    Speed (Baldwin vs control):
      mean_gen_baldwin_to_092 + 2 <= mean_gen_control_to_092
    Genetic-assimilation (F1 vs hand-tuned baseline):
      mean_f1_baldwin > mean_baseline + 0.10
    Comparative (Baldwin vs Lamarckian):
      mean_gen_baldwin_to_092 <= mean_gen_lamarckian_to_092 + 4
    Verdict: GO if all three; PIVOT if speed only; STOP otherwise.

Usage:
    uv run python scripts/campaigns/aggregate_m4_pilot.py \
        --baldwin-root evolution_results/m4_baldwin_lstmppo_klinotaxis_predator \
        --lamarckian-root evolution_results/m4_lamarckian_lstmppo_klinotaxis_predator \
        --control-root evolution_results/m4_control_lstmppo_klinotaxis_predator \
        --baseline-root evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline \
        --f1-csv artifacts/logbooks/014/m4_baldwin_pilot/summary/f1_innate_only.csv \
        --output-dir artifacts/logbooks/014/m4_baldwin_pilot/summary
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

# Best-fitness threshold for the Speed metric.  0.92 is calibrated to
# the predator-arm fitness landscape's documented saturation ceiling
# (per M3 logbook 013).  Module-level so reviewers see the constant
# without grepping.
TARGET_FITNESS = 0.92

# Speed-gate tolerance: Baldwin GOs if it reaches TARGET_FITNESS at
# least this many generations earlier on average than the from-scratch
# control.  Set to 2 (vs M3's +4) because Baldwin's mechanism is
# subtler than Lamarckian's (hyperparameter evolution alone, no weight
# transfer).  Calibrated so Baldwin <= 7.75 mean generations passes
# against M3's published control mean of 9.75.
SPEED_GAIN_GENERATIONS = 2

# Genetic-assimilation gate threshold: Baldwin's elite genome (with
# K=0 frozen-eval) must beat the hand-tuned baseline by at least
# this much.  Calibrated against M3's published baseline of 0.17 →
# passing means F1 mean >= 0.27.  Documents "the genome alone produces
# noticeably better policies than hand-tuning, even without learning".
F1_OVER_BASELINE_THRESHOLD = 0.10

# Comparative-gate tolerance: Baldwin's gen-to-092 must be within this
# many generations of Lamarckian's.  Sanity tripwire — at M3's
# published Lamarckian 4.5, this means Baldwin <= 8.5 (essentially
# subsumed by the speed gate's <= 7.75 if Lamarckian rerun reproduces
# its M3 number).  Documented in design Decision 6.
COMPARATIVE_GAP_GENERATIONS = 4


def _latest_session(seed_dir: Path) -> Path:
    """Return the most recently modified subdirectory under ``seed_dir``."""
    sessions = [p for p in seed_dir.iterdir() if p.is_dir()]
    if not sessions:
        msg = f"No session directory under {seed_dir}"
        raise FileNotFoundError(msg)
    return max(sessions, key=lambda p: p.stat().st_mtime)


def _read_history(seed_dir: Path) -> list[dict[str, float]]:
    """Read the latest session's history.csv as a list of float dicts."""
    history_path = _latest_session(seed_dir) / "history.csv"
    if not history_path.exists():
        msg = f"No history.csv at {history_path}"
        raise FileNotFoundError(msg)
    with history_path.open() as handle:
        reader = csv.DictReader(handle)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def _baseline_success_rates(baseline_root: Path) -> dict[int, float]:
    """Extract per-seed success rates from the run_simulation.py log files."""
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


def _read_f1_csv(f1_csv: Path) -> dict[int, float]:
    """Read the F1 innate-only CSV produced by baldwin_f1_postpilot_eval.py.

    Returns ``{seed: success_rate}`` for every row.  Caller raises a
    clear error if any pilot seed is missing from the CSV.
    """
    if not f1_csv.exists():
        msg = (
            f"F1 CSV missing at {f1_csv}.  Run "
            f"scripts/campaigns/baldwin_f1_postpilot_eval.py first."
        )
        raise FileNotFoundError(msg)
    rows: dict[int, float] = {}
    with f1_csv.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[int(row["seed"])] = float(row["success_rate"])
    return rows


def _gen_first_reaches_target(
    history: list[dict[str, float]],
    target: float = TARGET_FITNESS,
) -> int | None:
    """Return the 1-indexed generation at which best_fitness first reaches ``target``.

    Returns ``None`` if the threshold is never crossed.  History rows
    are 0-indexed in the CSV; the returned generation is 1-indexed for
    human readability.
    """
    for row in history:
        if row["best_fitness"] >= target:
            return int(row["generation"]) + 1
    return None


def _resolve_speed(g: int | None, fallback_gen: int) -> float:
    """Resolve a per-seed gen-to-target value to a float for averaging.

    Treats never-reached as the fallback (run's max generation + 1) so
    the metric is bounded; conservative for the GO check (the arm
    doesn't get credit for "would have" reached eventually).
    """
    return float(g) if g is not None else float(fallback_gen)


def _compute_verdict(
    *,
    speed_gate_passes: bool,
    assimilation_gate_passes: bool,
    comparative_gate_passes: bool,
) -> tuple[str, str]:
    """Compute the GO/PIVOT/STOP verdict + summary text from the three gates.

    Verdict logic:
      - GO if all three gates pass.
      - PIVOT if the speed gate passes but at least one of the other
        two fails.
      - STOP if the speed gate fails (regardless of the other two —
        without speed, Baldwin offers no improvement over the
        from-scratch control).
    """
    if speed_gate_passes and assimilation_gate_passes and comparative_gate_passes:
        return (
            "GO ✅",
            "Baldwin inheritance accelerates convergence (speed gate), the "
            "elite genome alone produces useful priors without K-train "
            "(genetic-assimilation gate), AND its convergence speed is "
            "competitive with Lamarckian (comparative gate).  Inheritance "
            "of learning bias is doing real work — proceed to the next "
            "experiment.",
        )
    if speed_gate_passes:
        failed = []
        if not assimilation_gate_passes:
            failed.append("genetic-assimilation")
        if not comparative_gate_passes:
            failed.append("comparative")
        joined = " and ".join(failed)
        return (
            "PIVOT ⚠️",
            f"The speed gate passed but the {joined} gate(s) failed.  "
            "Baldwin shows partial signal — inheritance helps convergence "
            "but either the genome doesn't encode useful priors without "
            "learning, or it underperforms Lamarckian by more than the "
            "comparative threshold.  Treat as inconclusive and review the "
            "per-seed trajectories before committing follow-up scope.",
        )
    return (
        "STOP ❌",
        "The speed gate failed: Baldwin does not accelerate convergence "
        "over the from-scratch control by the required margin.  Either "
        "the richer learnability schema offers no exploitable signal, "
        "or the fitness landscape is the wrong testbed.  Re-evaluate "
        "before committing to follow-up scope.",
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
    f1_rates: dict[int, float],
) -> str:
    """Render the three-gate GO/PIVOT/STOP verdict + per-seed details."""
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

    f1_mean = float(np.mean([f1_rates[s] for s in seeds]))

    speed_gate_passes = (speed_mean_baldwin + SPEED_GAIN_GENERATIONS) <= speed_mean_ctrl
    assimilation_gate_passes = f1_mean > (baseline_mean + F1_OVER_BASELINE_THRESHOLD)
    comparative_gate_passes = speed_mean_baldwin <= (speed_mean_lam + COMPARATIVE_GAP_GENERATIONS)

    verdict, verdict_text = _compute_verdict(
        speed_gate_passes=speed_gate_passes,
        assimilation_gate_passes=assimilation_gate_passes,
        comparative_gate_passes=comparative_gate_passes,
    )

    lines: list[str] = [
        "## Baldwin Inheritance Pilot — Summary",
        "",
        f"**Seeds**: {seeds}",
        f"**Hand-tuned baseline mean**: {baseline_mean:.3f}",
        "",
        "### Decision Gates",
        "",
        f"- **Speed gate** (mean_gen_baldwin + {SPEED_GAIN_GENERATIONS} "
        f"<= mean_gen_control): {'PASS' if speed_gate_passes else 'FAIL'}",
        f"  - Baldwin mean gen-to-{TARGET_FITNESS}: {speed_mean_baldwin:.2f}",
        f"  - Control mean gen-to-{TARGET_FITNESS}: {speed_mean_ctrl:.2f}",
        f"  - Margin: {speed_mean_ctrl - speed_mean_baldwin:+.2f} "
        f"(need >= {SPEED_GAIN_GENERATIONS})",
        "",
        f"- **Genetic-assimilation gate** (mean_f1_baldwin > "
        f"mean_baseline + {F1_OVER_BASELINE_THRESHOLD}): "
        f"{'PASS' if assimilation_gate_passes else 'FAIL'}",
        f"  - Baldwin F1 innate-only mean: {f1_mean:.3f}",
        f"  - Hand-tuned baseline mean: {baseline_mean:.3f}",
        f"  - Margin: {f1_mean - baseline_mean:+.3f} (need > {F1_OVER_BASELINE_THRESHOLD})",
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
        f"best_fitness >= {TARGET_FITNESS}) + F1 innate-only success rate",
        "",
        "| Seed | Baldwin | Lamarckian | Control | F1 innate-only |",
        "|------|---------|------------|---------|----------------|",
    ]
    for s in seeds:

        def _fmt_gen(g: int | None) -> str:
            return str(g) if g is not None else "—"

        lines.append(
            f"| {s} | {_fmt_gen(speed_baldwin[s])} | {_fmt_gen(speed_lam[s])} "
            f"| {_fmt_gen(speed_ctrl[s])} | {f1_rates[s]:.3f} |",
        )

    return "\n".join(lines)


def _write_speed_csv(  # noqa: PLR0913
    output_dir: Path,
    seeds: list[int],
    speed_baldwin: dict[int, int | None],
    speed_lam: dict[int, int | None],
    speed_ctrl: dict[int, int | None],
    f1_rates: dict[int, float],
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
                "f1_innate_only_success_rate",
            ),
        )
        for s in seeds:
            writer.writerow(
                (
                    s,
                    speed_baldwin[s] if speed_baldwin[s] is not None else "",
                    speed_lam[s] if speed_lam[s] is not None else "",
                    speed_ctrl[s] if speed_ctrl[s] is not None else "",
                    f"{f1_rates[s]:.6f}",
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
    f1_rates: dict[int, float],
) -> Path | None:
    """4-curve plot: Baldwin + Lamarckian + control mean ± std + baseline + F1 markers."""
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
        label=f"Hand-tuned baseline ({baseline_mean:.2f})",
    )
    ax.axhline(
        TARGET_FITNESS,
        linestyle=":",
        color="green",
        label=f"Target fitness ({TARGET_FITNESS})",
    )

    # F1 innate-only success rates as separate markers off to the right
    # at x = max_gen + 0.5 so they don't overlap the trajectory curves.
    f1_x = float(n) + 0.5
    f1_values = [f1_rates[s] for s in seeds]
    ax.scatter(
        [f1_x] * len(f1_values),
        f1_values,
        marker="*",
        color="tab:red",
        s=100,
        zorder=5,
        label=f"Baldwin F1 innate-only (mean {float(np.mean(f1_values)):.2f})",
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness across population")
    ax.set_title(
        f"Baldwin inheritance pilot — convergence "
        f"({len(seeds)} seed{'s' if len(seeds) != 1 else ''})",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(visible=True, alpha=0.3)

    plot_path = output_dir / "convergence.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    return plot_path


def main() -> int:
    """Aggregate Baldwin + Lamarckian + control + baseline + F1 → summary + plot + csv."""
    parser = argparse.ArgumentParser(description="Aggregate Baldwin pilot results.")
    parser.add_argument("--baldwin-root", type=Path, required=True)
    parser.add_argument("--lamarckian-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help=(
            "Path to the run_simulation.py-driven hand-tuned baseline tree. "
            "Default canonical location: "
            "evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/"
        ),
    )
    parser.add_argument(
        "--f1-csv",
        type=Path,
        required=True,
        help=(
            "Path to the F1 innate-only CSV produced by "
            "scripts/campaigns/baldwin_f1_postpilot_eval.py."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baldwin_history: dict[int, list[dict[str, float]]] = {}
    lam_history: dict[int, list[dict[str, float]]] = {}
    ctrl_history: dict[int, list[dict[str, float]]] = {}
    for seed in args.seeds:
        baldwin_history[seed] = _read_history(args.baldwin_root / f"seed-{seed}")
        lam_history[seed] = _read_history(args.lamarckian_root / f"seed-{seed}")
        ctrl_history[seed] = _read_history(args.control_root / f"seed-{seed}")

    # Baseline (run_simulation.py-driven, optimiser-independent).
    baseline_rates = _baseline_success_rates(args.baseline_root)
    missing_baseline = sorted(set(args.seeds) - set(baseline_rates))
    if missing_baseline:
        msg = (
            f"Missing baseline success rate(s) for seed(s) {missing_baseline}.  Either "
            f"(a) the per-seed log files are absent — re-run "
            f"phase5_m2_hyperparam_lstmppo_klinotaxis_predator_baseline.sh to populate "
            f"{args.baseline_root}, or (b) the logs exist but lack a "
            f"'Success rate: NN.NN%' line — inspect "
            f"{args.baseline_root}/seed-N.log for crashes or format changes."
        )
        raise SystemExit(msg)
    baseline_mean = float(np.mean([baseline_rates[s] for s in args.seeds]))

    # F1 innate-only rates (from the post-pilot evaluator).
    f1_rates = _read_f1_csv(args.f1_csv)
    missing_f1 = sorted(set(args.seeds) - set(f1_rates))
    if missing_f1:
        msg = (
            f"Missing F1 innate-only rate(s) for seed(s) {missing_f1} in "
            f"{args.f1_csv}.  Re-run scripts/campaigns/baldwin_f1_postpilot_eval.py "
            f"with --seeds covering the full pilot seed set."
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
        f1_rates,
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
        f1_rates,
    )
    print(f"Convergence-speed table written to {csv_path}")

    plot_path = _plot_convergence(
        args.output_dir,
        args.seeds,
        baldwin_history,
        lam_history,
        ctrl_history,
        baseline_mean,
        f1_rates,
    )
    if plot_path is not None:
        print(f"Plot written to {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
