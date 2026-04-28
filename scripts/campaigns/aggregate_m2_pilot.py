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


def _latest_session(seed_dir: Path) -> Path:
    """Return the most recently modified subdirectory under ``seed_dir``.

    Filtering to directories (rather than relying on lexicographic order over
    ``iterdir()``) avoids stray files (``.DS_Store``, log tails, etc.) being
    mistaken for a session.  Selecting by ``stat().st_mtime`` instead of name
    means we don't depend on a particular session-id format ordering.
    """
    sessions = [p for p in seed_dir.iterdir() if p.is_dir()]
    if not sessions:
        msg = f"No session directory under {seed_dir}"
        raise FileNotFoundError(msg)
    return max(sessions, key=lambda p: p.stat().st_mtime)


def _read_history(seed_dir: Path) -> list[dict[str, float]]:
    """Read the single session under seed_dir's history.csv into rows."""
    history_path = _latest_session(seed_dir) / "history.csv"
    if not history_path.exists():
        msg = f"No history.csv at {history_path}"
        raise FileNotFoundError(msg)
    with history_path.open() as f:
        reader = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def _read_best(seed_dir: Path) -> dict[str, object]:
    """Read the single session under seed_dir's best_params.json."""
    best_path = _latest_session(seed_dir) / "best_params.json"
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


def _format_summary(  # noqa: PLR0913, PLR0915
    pilot_seeds: list[int],
    pilot_history: dict[int, list[dict[str, float]]],
    pilot_best: dict[int, dict[str, object]],
    baseline_rates: dict[int, float],
    baseline_mean: float,
    go_threshold: float,
    gate_pp: float,
) -> str:
    """Build a markdown summary block for the logbook.

    All quantitative labels (last generation, GO threshold percentage,
    seed count) are derived from the inputs rather than hard-coded so
    the summary remains accurate under truncated runs, non-default
    ``--gate-pp``, and arbitrary seed counts.
    """
    # PIVOT decision needs a positive-but-below-GO band.  1pp is a
    # reasonable floor for "this didn't separate from baseline at all";
    # below that we call STOP.  Lifted to a named constant so future
    # reviewers don't have to grep for the magic number.
    pivot_min_pp = 0.01
    seed_count = len(pilot_seeds)
    gate_pp_label = f"{gate_pp * 100:.1f}".rstrip("0").rstrip(".") + "pp"
    lines: list[str] = []
    lines.append("# M2 Hyperparameter-Evolution Pilot — Summary")
    lines.append("")
    lines.append("## Per-seed best fitness (eval-phase success rate, L=5)")
    lines.append("")
    # Detect a unified target generation rather than hard-coding "20" —
    # truncated runs (e.g. crash recovery, --generations override) and
    # other budgets would otherwise mislabel their results.  Use the
    # *minimum* last-generation across seeds so every per-seed row in
    # the table genuinely corresponds to the same generation; with
    # ``max`` a truncated seed would silently fall back to its own
    # earlier generation while the header read "Gen 20 best".
    last_gen = min(
        int(pilot_history[seed][-1].get("generation", len(pilot_history[seed])))
        for seed in pilot_seeds
    )
    lines.append(
        f"| Seed | Gen 1 best | Gen {last_gen} best | Mean across gens | Best params (lr, gamma, ...) |",
    )
    lines.append(
        "|------|-----------|-------------|------------------|------------------------------|",
    )
    # Build a {seed: {generation: row}} lookup so per-seed picks for the
    # "Gen N best" column always correspond to the same ``last_gen``.
    # The defensive ``.get(last_gen, hist[-1])`` below guards against
    # missing/non-monotonic ``generation`` fields; with the ``min``
    # selection above every seed should normally have the row.
    seed_rows_by_gen: dict[int, dict[int, dict[str, float]]] = {
        seed: {
            int(row.get("generation", idx + 1)): row for idx, row in enumerate(pilot_history[seed])
        }
        for seed in pilot_seeds
    }
    pilot_finals: list[float] = []
    for seed in pilot_seeds:
        hist = pilot_history[seed]
        gen1 = hist[0]["best_fitness"]
        # Pick the row matching the unified target generation; fall back to
        # the seed's actual final row if it was truncated below ``last_gen``.
        target_row = seed_rows_by_gen[seed].get(last_gen, hist[-1])
        last = target_row["best_fitness"]
        mean_across = float(np.mean([row["best_fitness"] for row in hist]))
        pilot_finals.append(last)
        bp_raw = pilot_best[seed]["best_params"]
        # ``best_params.json`` is supposed to store a list[float].  Failing
        # fast here surfaces upstream corruption (missing key, wrong type,
        # non-numeric element) instead of silently emitting "[, ...]" in the
        # markdown table.
        if not isinstance(bp_raw, list):
            msg = (
                f"best_params for seed {seed} is not a list (got "
                f"{type(bp_raw).__name__}); cannot format summary."
            )
            raise TypeError(msg)
        bp: list[float] = []
        for i, x in enumerate(bp_raw):
            try:
                bp.append(float(x))
            except (TypeError, ValueError) as exc:
                msg = (
                    f"best_params[{i}] for seed {seed} is not numeric "
                    f"(got {x!r}); cannot format summary."
                )
                raise ValueError(msg) from exc
        bp_short = f"[{', '.join(f'{x:.2f}' for x in bp[:3])}, ...]"
        lines.append(
            f"| {seed} | {gen1:.3f} | {last:.3f} | {mean_across:.3f} | {bp_short} |",
        )
    lines.append("")
    pilot_mean = float(np.mean(pilot_finals))
    pilot_std = float(np.std(pilot_finals))
    lines.append(
        f"**Pilot mean (gen-{last_gen} best across seeds)**: {pilot_mean:.3f} ± {pilot_std:.3f}",
    )
    lines.append("")

    lines.append("## Baseline (hand-tuned MLPPPO, 100 episodes per seed)")
    lines.append("")
    lines.append("| Seed | Success rate |")
    lines.append("|------|--------------|")
    # Filter the table to the requested seed list — stale logs under
    # baseline_root from prior runs would otherwise clutter the table
    # even though they don't enter baseline_mean.
    requested_seeds = set(pilot_seeds)
    lines.extend(
        f"| {seed} | {baseline_rates[seed]:.3f} |"
        for seed in sorted(requested_seeds & baseline_rates.keys())
    )
    lines.append("")
    lines.append(f"**Baseline mean**: {baseline_mean:.3f}")
    lines.append("")

    lines.append("## Decision gate")
    lines.append("")
    lines.append(f"- Baseline mean: **{baseline_mean:.3f}**")
    lines.append(f"- GO threshold (≥{gate_pp_label} over baseline): **{go_threshold:.3f}**")
    lines.append(f"- Pilot mean (gen-{last_gen} best): **{pilot_mean:.3f}**")
    lines.append(
        f"- Separation: {pilot_mean - baseline_mean:+.3f} ({(pilot_mean - baseline_mean) * 100:+.1f}pp)",
    )
    lines.append("")
    if pilot_mean >= go_threshold:
        lines.append("**Decision**: GO ✅")
        lines.append("")
        lines.append(
            "Hyperparameter evolution beats the hand-tuned baseline by "
            f"{(pilot_mean - baseline_mean) * 100:.1f}pp.  Mean across "
            f"{seed_count} seed{'s' if seed_count != 1 else ''} clears "
            f"the {gate_pp_label} gate threshold.",
        )
    elif pilot_mean >= baseline_mean + pivot_min_pp:
        lines.append("**Decision**: PIVOT 🟡")
        lines.append("")
        lines.append(
            "Hyperparameter evolution shows positive separation from "
            f"baseline ({(pilot_mean - baseline_mean) * 100:+.1f}pp) but "
            f"doesn't clear the {gate_pp_label} GO threshold.  Worth "
            "investigating schema/budget tweaks before either "
            "greenlighting or abandoning.",
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
    # Validate that we have a baseline rate for every requested seed.
    # Silently averaging over a subset would understate or overstate
    # the baseline mean and corrupt the GO threshold.
    missing_baseline = sorted(set(args.seeds) - set(baseline_rates))
    if missing_baseline:
        msg = (
            f"Missing baseline success rate(s) for seed(s) {missing_baseline}. "
            f"Expected logs at {args.baseline_root}/seed-<SEED>.log with a "
            "'Success rate: NN.NN%' line.  Found rates for: "
            f"{sorted(baseline_rates)}."
        )
        raise SystemExit(msg)
    baseline_values = [baseline_rates[seed] for seed in args.seeds]
    baseline_mean = float(np.mean(baseline_values))
    go_threshold = baseline_mean + args.gate_pp

    summary_md = _format_summary(
        args.seeds,
        pilot_history,
        pilot_best,
        baseline_rates,
        baseline_mean,
        go_threshold,
        args.gate_pp,
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

        # Mean across seeds (assumes equal length).  The label reflects
        # the actual seed count rather than a hard-coded "4" so custom
        # ``--seeds`` invocations don't mislabel.
        seed_count = len(args.seeds)
        seed_label = f"Mean ({seed_count} seed{'s' if seed_count != 1 else ''})"
        if len({len(c) for c in best_curves}) == 1:
            mean_best = np.mean(best_curves, axis=0)
            ax_best.plot(
                range(1, len(mean_best) + 1),
                mean_best,
                color="black",
                linewidth=2.5,
                label=seed_label,
            )
        if len({len(c) for c in mean_curves}) == 1:
            mean_mean = np.mean(mean_curves, axis=0)
            ax_mean.plot(
                range(1, len(mean_mean) + 1),
                mean_mean,
                color="black",
                linewidth=2.5,
                label=seed_label,
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

        fig.suptitle(
            f"M2 Hyperparameter Pilot — convergence ({seed_count} "
            f"seed{'s' if seed_count != 1 else ''})",
            fontsize=14,
        )
        plot_path = args.output_dir / "convergence.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        print(f"Plot written to {plot_path}")
    except ImportError:
        print("(matplotlib not available; skipping plot)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
