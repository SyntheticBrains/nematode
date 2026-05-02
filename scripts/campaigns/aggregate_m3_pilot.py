# pragma: no cover
r"""Aggregate M3 lamarckian-vs-control results across an arbitrary seed set.

Reads per-seed ``history.csv`` from BOTH the lamarckian and the
within-experiment from-scratch control arms (each laid out under
``<root>/seed-<N>/<session>/history.csv``), plus the run_simulation.py-
driven hand-tuned baseline (per-seed log files), and produces:

- A two-curve convergence plot: lamarckian mean ± std vs control mean
  ± std vs generation (best_fitness across-seed mean as solid line,
  ±1 std as shaded band).  Hand-tuned baseline mean as a horizontal
  dashed line.  Aligned with the GO/PIVOT/STOP gate which measures
  generations-to-0.92 on best_fitness, not mean.
- A per-seed convergence-speed table written as ``convergence_speed.csv``:
  ``seed, lamarckian_gen_to_092, control_gen_to_092``.  Empty cell
  when a seed never reached the 0.92 threshold within the run.
- A markdown summary with the M3 GO/PIVOT/STOP verdict.  Decision logic:
    GO if BOTH:
      (a) Speed: mean_gen_lamarckian_to_092 + 4 <= mean_gen_control_to_092
      (b) Floor: mean_gen1_lamarckian >= mean_gen3_control
    PIVOT if exactly one of (a)/(b) holds.
    STOP if neither.

Usage:
    uv run python scripts/campaigns/aggregate_m3_pilot.py \
        --lamarckian-root evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator \
        --control-root evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator_control \
        --baseline-root evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline \
        --output-dir artifacts/logbooks/013/m3_lamarckian_pilot/summary
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

# The 0.92 best-fitness threshold matches M2.12's documented saturation
# ceiling on the predator arm (3 of 4 seeds reached 0.92 by gen 6-12).
# Module-level so reviewers see the constant without grepping.
TARGET_FITNESS = 0.92

# Speed-metric tolerance: M3 GOs if it reaches TARGET_FITNESS at least
# this many generations earlier on average than the control.  4 = 20%
# of M2's 20-gen budget = the "10pp faster convergence" gate from the
# Phase 5 tracker, translated to the predator arm's saturation speed.
SPEED_GAIN_GENERATIONS = 4

# Floor-metric reference generation: M3 gen-1 mean must beat the
# control's gen-N mean (1-indexed in the table; 3 = "two generations
# of head start").
FLOOR_REFERENCE_GEN = 3


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


def _gen_first_reaches_target(
    history: list[dict[str, float]],
    target: float = TARGET_FITNESS,
) -> int | None:
    """Return the 1-indexed generation at which best_fitness first reaches ``target``.

    Returns ``None`` if the threshold is never crossed.  History is
    0-indexed in the CSV; the returned generation is 1-indexed for
    human readability (matches the convergence plot's x-axis).
    """
    for row in history:
        if row["best_fitness"] >= target:
            # CSV stores generation 0-indexed; report 1-indexed.
            return int(row["generation"]) + 1
    return None


def _gen_n_mean_best(
    histories: dict[int, list[dict[str, float]]],
    seeds: list[int],
    gen_index_1based: int,
) -> float:
    """Mean across-seed best_fitness at the 1-indexed generation."""
    gen_idx = gen_index_1based - 1
    values = [
        h[gen_idx]["best_fitness"]
        for h in (histories[s] for s in seeds)
        if gen_idx < len(h)
    ]
    return float(np.mean(values)) if values else 0.0


def _format_summary(  # noqa: PLR0913
    seeds: list[int],
    lam_history: dict[int, list[dict[str, float]]],
    ctrl_history: dict[int, list[dict[str, float]]],
    baseline_mean: float,
    speed_lam: dict[int, int | None],
    speed_ctrl: dict[int, int | None],
    floor_lam: float,
    floor_ctrl: float,
) -> str:
    """Render the GO/PIVOT/STOP verdict + per-seed details."""
    # Speed metric: across-seed mean of generations-to-0.92.  Treat
    # never-reached as the run's max generation + 1 so the metric is
    # bounded; this is conservative for the GO check (M3 doesn't get
    # credit for "would have" reached eventually).
    max_gens = max(len(h) for h in (*lam_history.values(), *ctrl_history.values()))
    fallback_gen = max_gens + 1

    def _resolve(g: int | None) -> float:
        return float(g) if g is not None else float(fallback_gen)

    speed_lam_vals = [_resolve(speed_lam[s]) for s in seeds]
    speed_ctrl_vals = [_resolve(speed_ctrl[s]) for s in seeds]
    speed_mean_lam = float(np.mean(speed_lam_vals))
    speed_mean_ctrl = float(np.mean(speed_ctrl_vals))

    speed_gate_passes = (speed_mean_lam + SPEED_GAIN_GENERATIONS) <= speed_mean_ctrl
    floor_gate_passes = floor_lam >= floor_ctrl

    if speed_gate_passes and floor_gate_passes:
        verdict = "GO ✅"
        verdict_text = (
            "Lamarckian inheritance both accelerates convergence (speed gate) "
            "AND lifts the gen-1 floor over the control's gen-3 baseline. "
            "M3 graduates; M4 (Baldwin Effect) starts on this configuration."
        )
    elif speed_gate_passes or floor_gate_passes:
        verdict = "PIVOT ⚠️"
        passed = "speed" if speed_gate_passes else "floor"
        verdict_text = (
            f"Only the {passed} gate passed.  Inheritance helps in one "
            "direction but not the other; treat M3 as inconclusive and "
            "review the per-seed trajectories before committing M4 scope."
        )
    else:
        verdict = "STOP ❌"
        verdict_text = (
            "Neither gate passed: inheritance does not accelerate convergence "
            "and does not lift the gen-1 floor over the control.  Either "
            "the schema is too narrow for inheritance to help, or the "
            "predator arm is the wrong testbed.  Re-evaluate before M4."
        )

    lines: list[str] = [
        "## M3 Lamarckian Inheritance Pilot — Summary",
        "",
        f"**Seeds**: {seeds}",
        f"**Hand-tuned baseline mean**: {baseline_mean:.3f}",
        "",
        "### Decision Gate",
        "",
        f"- **Speed gate** (mean_gen_lamarckian + {SPEED_GAIN_GENERATIONS} "
        f"<= mean_gen_control): "
        f"{'PASS' if speed_gate_passes else 'FAIL'}",
        f"  - Lamarckian mean gen-to-{TARGET_FITNESS}: {speed_mean_lam:.1f}",
        f"  - Control mean gen-to-{TARGET_FITNESS}: {speed_mean_ctrl:.1f}",
        f"  - Margin: {speed_mean_ctrl - speed_mean_lam:+.1f} "
        f"(need >= {SPEED_GAIN_GENERATIONS})",
        "",
        f"- **Floor gate** (mean_gen1_lamarckian >= mean_gen{FLOOR_REFERENCE_GEN}_control): "
        f"{'PASS' if floor_gate_passes else 'FAIL'}",
        f"  - Lamarckian gen-1 mean: {floor_lam:.3f}",
        f"  - Control gen-{FLOOR_REFERENCE_GEN} mean: {floor_ctrl:.3f}",
        f"  - Margin: {floor_lam - floor_ctrl:+.3f}",
        "",
        f"**Decision**: {verdict}",
        "",
        verdict_text,
        "",
        "### Per-seed convergence speed (generations to first reach "
        f"best_fitness >= {TARGET_FITNESS})",
        "",
        "| Seed | Lamarckian | Control |",
        "|------|------------|---------|",
    ]
    for s in seeds:
        lam_g = speed_lam[s]
        ctrl_g = speed_ctrl[s]
        lam_str = str(lam_g) if lam_g is not None else "—"
        ctrl_str = str(ctrl_g) if ctrl_g is not None else "—"
        lines.append(f"| {s} | {lam_str} | {ctrl_str} |")

    return "\n".join(lines)


def _write_speed_csv(
    output_dir: Path,
    seeds: list[int],
    speed_lam: dict[int, int | None],
    speed_ctrl: dict[int, int | None],
) -> Path:
    """Write the per-seed convergence-speed table."""
    path = output_dir / "convergence_speed.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "lamarckian_gen_to_092", "control_gen_to_092"))
        for s in seeds:
            writer.writerow(
                (
                    s,
                    speed_lam[s] if speed_lam[s] is not None else "",
                    speed_ctrl[s] if speed_ctrl[s] is not None else "",
                ),
            )
    return path


def _plot_convergence(  # noqa: PLR0913
    output_dir: Path,
    seeds: list[int],
    lam_history: dict[int, list[dict[str, float]]],
    ctrl_history: dict[int, list[dict[str, float]]],
    baseline_mean: float,
) -> Path | None:
    """Two-curve plot: lamarckian vs control mean ± std band of best_fitness."""
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; skipping plot)")
        return None

    def _stack(histories: dict[int, list[dict[str, float]]]) -> np.ndarray:
        return np.array(
            [[row["best_fitness"] for row in histories[s]] for s in seeds],
            dtype=np.float64,
        )

    lam = _stack(lam_history)
    ctrl = _stack(ctrl_history)
    if lam.shape[1] != ctrl.shape[1]:
        # Different generation counts → align to the shorter run for the plot.
        n = min(lam.shape[1], ctrl.shape[1])
        lam = lam[:, :n]
        ctrl = ctrl[:, :n]
    gens = np.arange(1, lam.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data, colour in (
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
            alpha=0.2,
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
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness across population")
    ax.set_title(
        f"M3 Lamarckian inheritance pilot — convergence "
        f"({len(seeds)} seed{'s' if len(seeds) != 1 else ''})",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(visible=True, alpha=0.3)

    plot_path = output_dir / "convergence.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    return plot_path


def main() -> int:
    """Aggregate lamarckian + control + baseline; emit summary + plot + speed table."""
    parser = argparse.ArgumentParser(description="Aggregate M3 lamarckian pilot results.")
    parser.add_argument("--lamarckian-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help=(
            "Path to the run_simulation.py-driven hand-tuned baseline tree, "
            "produced by phase5_m2_hyperparam_lstmppo_klinotaxis_predator_baseline.sh "
            "(re-run under the M3 revision per task 9.6).  Default canonical "
            "location: evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    lam_history: dict[int, list[dict[str, float]]] = {}
    ctrl_history: dict[int, list[dict[str, float]]] = {}
    for seed in args.seeds:
        lam_history[seed] = _read_history(args.lamarckian_root / f"seed-{seed}")
        ctrl_history[seed] = _read_history(args.control_root / f"seed-{seed}")

    # Baseline (run_simulation.py-driven, optimiser-independent).
    baseline_rates = _baseline_success_rates(args.baseline_root)
    missing = sorted(set(args.seeds) - set(baseline_rates))
    if missing:
        msg = (
            f"Missing baseline success rate(s) for seed(s) {missing}.  Re-run "
            f"phase5_m2_hyperparam_lstmppo_klinotaxis_predator_baseline.sh to "
            f"populate {args.baseline_root}."
        )
        raise SystemExit(msg)
    baseline_mean = float(np.mean([baseline_rates[s] for s in args.seeds]))

    # Convergence-speed metric per arm per seed.
    speed_lam = {s: _gen_first_reaches_target(lam_history[s]) for s in args.seeds}
    speed_ctrl = {s: _gen_first_reaches_target(ctrl_history[s]) for s in args.seeds}

    # Floor metric: gen-1 mean (lamarckian) vs gen-FLOOR_REFERENCE_GEN mean (control).
    floor_lam = _gen_n_mean_best(lam_history, args.seeds, 1)
    floor_ctrl = _gen_n_mean_best(ctrl_history, args.seeds, FLOOR_REFERENCE_GEN)

    summary_md = _format_summary(
        args.seeds,
        lam_history,
        ctrl_history,
        baseline_mean,
        speed_lam,
        speed_ctrl,
        floor_lam,
        floor_ctrl,
    )
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text(summary_md + "\n")
    print(summary_md)
    print()
    print(f"Summary written to {summary_path}")

    csv_path = _write_speed_csv(args.output_dir, args.seeds, speed_lam, speed_ctrl)
    print(f"Convergence-speed table written to {csv_path}")

    plot_path = _plot_convergence(
        args.output_dir,
        args.seeds,
        lam_history,
        ctrl_history,
        baseline_mean,
    )
    if plot_path is not None:
        print(f"Plot written to {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
