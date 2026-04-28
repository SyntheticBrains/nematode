# pragma: no cover
r"""Aggregate M2 hyperparameter-pilot results across 4 seeds.

Reads per-seed history.csv and best_params.json from each session
directory and produces:

- A summary table (per-seed best fitness, gen-1 fitness, gen-20 fitness)
- Mean / std across seeds at each generation
- A plot of the convergence curves (best fitness per generation per seed,
  plus the mean and the baseline threshold band)
- A markdown summary block ready to paste into the logbook

Usage:
    uv run python scripts/campaigns/aggregate_m2_pilot.py \
        --pilot-root tmp/evaluations/evolution/m2_pilot \
        --baseline-root tmp/evaluations/evolution/m2_baseline \
        --output-dir tmp/evaluations/evolution/m2_pilot_summary
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def _read_history(seed_dir: Path) -> list[dict[str, float]]:
    """Read the single session under seed_dir's history.csv into rows."""
    sessions = sorted(seed_dir.iterdir())
    if not sessions:
        msg = f"No session directory under {seed_dir}"
        raise FileNotFoundError(msg)
    # Take the latest session if multiple exist (e.g. resumed run)
    history_path = sessions[-1] / "history.csv"
    if not history_path.exists():
        msg = f"No history.csv at {history_path}"
        raise FileNotFoundError(msg)
    with history_path.open() as f:
        reader = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def _read_best(seed_dir: Path) -> dict[str, object]:
    """Read the single session under seed_dir's best_params.json."""
    sessions = sorted(seed_dir.iterdir())
    best_path = sessions[-1] / "best_params.json"
    return json.loads(best_path.read_text())


def _baseline_success_rates(baseline_root: Path) -> dict[int, float]:
    """Extract per-seed success rates from baseline run logs."""
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


def _format_summary(  # noqa: PLR0913
    pilot_seeds: list[int],
    pilot_history: dict[int, list[dict[str, float]]],
    pilot_best: dict[int, dict[str, object]],
    baseline_rates: dict[int, float],
    baseline_mean: float,
    go_threshold: float,
) -> str:
    """Build a markdown summary block for the logbook."""
    lines: list[str] = []
    lines.append("# M2 Hyperparameter-Evolution Pilot — Summary")
    lines.append("")
    lines.append("## Per-seed best fitness (eval-phase success rate, L=5)")
    lines.append("")
    lines.append(
        "| Seed | Gen 1 best | Gen 20 best | Mean across gens | Best params (lr, gamma, ...) |",
    )
    lines.append(
        "|------|-----------|-------------|------------------|------------------------------|",
    )
    pilot_finals: list[float] = []
    for seed in pilot_seeds:
        hist = pilot_history[seed]
        gen1 = hist[0]["best_fitness"]
        gen20 = hist[-1]["best_fitness"]
        mean_across = float(np.mean([row["best_fitness"] for row in hist]))
        pilot_finals.append(gen20)
        bp_raw = pilot_best[seed]["best_params"]
        # best_params.json stores best_params as a list[float]; the dict[str, object]
        # type loses that, so narrow explicitly.
        bp: list[float] = bp_raw if isinstance(bp_raw, list) else []
        bp_short = f"[{', '.join(f'{x:.2f}' for x in bp[:3])}, ...]"
        lines.append(
            f"| {seed} | {gen1:.3f} | {gen20:.3f} | {mean_across:.3f} | {bp_short} |",
        )
    lines.append("")
    pilot_mean = float(np.mean(pilot_finals))
    pilot_std = float(np.std(pilot_finals))
    lines.append(
        f"**Pilot mean (gen-20 best across seeds)**: {pilot_mean:.3f} ± {pilot_std:.3f}",
    )
    lines.append("")

    lines.append("## Baseline (hand-tuned MLPPPO, 100 episodes per seed)")
    lines.append("")
    lines.append("| Seed | Success rate |")
    lines.append("|------|--------------|")
    lines.extend(f"| {seed} | {baseline_rates[seed]:.3f} |" for seed in sorted(baseline_rates))
    lines.append("")
    lines.append(f"**Baseline mean**: {baseline_mean:.3f}")
    lines.append("")

    lines.append("## Decision gate")
    lines.append("")
    lines.append(f"- Baseline mean: **{baseline_mean:.3f}**")
    lines.append(f"- GO threshold (≥3pp over baseline): **{go_threshold:.3f}**")
    lines.append(f"- Pilot mean (gen-20 best): **{pilot_mean:.3f}**")
    lines.append(
        f"- Separation: {pilot_mean - baseline_mean:+.3f} ({(pilot_mean - baseline_mean) * 100:+.1f}pp)",
    )
    lines.append("")
    if pilot_mean >= go_threshold:
        lines.append("**Decision**: GO ✅")
        lines.append("")
        lines.append(
            "Hyperparameter evolution beats the hand-tuned baseline by "
            f"{(pilot_mean - baseline_mean) * 100:.1f}pp.  Mean across 4 "
            "seeds clears the 3pp gate threshold.",
        )
    elif pilot_mean >= baseline_mean + 0.01:
        lines.append("**Decision**: PIVOT 🟡")
        lines.append("")
        lines.append(
            "Hyperparameter evolution shows positive separation from "
            f"baseline ({(pilot_mean - baseline_mean) * 100:+.1f}pp) but "
            "doesn't clear the 3pp GO threshold.  Worth investigating "
            "schema/budget tweaks before either greenlighting or "
            "abandoning.",
        )
    else:
        lines.append("**Decision**: STOP ❌")
        lines.append("")
        lines.append(
            "Hyperparameter evolution does not separate from baseline "
            f"({(pilot_mean - baseline_mean) * 100:+.1f}pp).",
        )
    return "\n".join(lines)


def main() -> int:  # noqa: PLR0915 — sequential CLI driver; splitting hurts readability
    """Aggregate pilot + baseline results."""
    parser = argparse.ArgumentParser(description="Aggregate M2 pilot results.")
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45],
    )
    parser.add_argument(
        "--gate-pp",
        type=float,
        default=0.03,
        help="GO threshold above baseline mean (default 0.03 = 3pp).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pilot_history: dict[int, list[dict[str, float]]] = {}
    pilot_best: dict[int, dict[str, object]] = {}
    for seed in args.seeds:
        seed_dir = args.pilot_root / f"seed-{seed}"
        pilot_history[seed] = _read_history(seed_dir)
        pilot_best[seed] = _read_best(seed_dir)

    baseline_rates = _baseline_success_rates(args.baseline_root)
    baseline_values = list(baseline_rates.values())
    baseline_mean = float(np.mean(baseline_values))
    go_threshold = baseline_mean + args.gate_pp

    summary_md = _format_summary(
        args.seeds,
        pilot_history,
        pilot_best,
        baseline_rates,
        baseline_mean,
        go_threshold,
    )
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text(summary_md + "\n")
    print(summary_md)
    print()
    print(f"Summary written to {summary_path}")

    # Plot convergence curves
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_best, ax_mean) = plt.subplots(1, 2, figsize=(16, 6))
        best_curves: list[list[float]] = []
        mean_curves: list[list[float]] = []
        for seed in args.seeds:
            best = [row["best_fitness"] for row in pilot_history[seed]]
            mean = [row["mean_fitness"] for row in pilot_history[seed]]
            ax_best.plot(
                range(1, len(best) + 1),
                best,
                marker="o",
                alpha=0.7,
                label=f"Seed {seed}",
            )
            ax_mean.plot(
                range(1, len(mean) + 1),
                mean,
                marker="o",
                alpha=0.7,
                label=f"Seed {seed}",
            )
            best_curves.append(best)
            mean_curves.append(mean)

        # Mean across seeds (assumes equal length)
        if len({len(c) for c in best_curves}) == 1:
            mean_best = np.mean(best_curves, axis=0)
            ax_best.plot(
                range(1, len(mean_best) + 1),
                mean_best,
                color="black",
                linewidth=2.5,
                label="Mean (4 seeds)",
            )
        if len({len(c) for c in mean_curves}) == 1:
            mean_mean = np.mean(mean_curves, axis=0)
            ax_mean.plot(
                range(1, len(mean_mean) + 1),
                mean_mean,
                color="black",
                linewidth=2.5,
                label="Mean (4 seeds)",
            )

        for ax, title in [
            (ax_best, "Best fitness per generation"),
            (ax_mean, "Mean fitness across population per generation"),
        ]:
            ax.axhline(
                baseline_mean,
                linestyle="--",
                color="grey",
                label=f"Baseline mean ({baseline_mean:.2f})",
            )
            ax.axhline(
                go_threshold,
                linestyle=":",
                color="green",
                label=f"GO threshold ({go_threshold:.2f})",
            )
            ax.set_xlabel("Generation")
            ax.set_ylabel("Eval success rate")
            ax.set_title(title)
            ax.set_ylim(0, 1.05)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(visible=True, alpha=0.3)

        fig.suptitle("M2 Hyperparameter Pilot — convergence (4 seeds)", fontsize=14)
        plot_path = args.output_dir / "convergence.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        print(f"Plot written to {plot_path}")
    except ImportError:
        print("(matplotlib not available; skipping plot)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
