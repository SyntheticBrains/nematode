r"""R4 screen - re-aggregate pilot output with per-generation series.

The production aggregator (`scripts/campaigns/aggregate_m5_pilot.py`)
runs cycling + escalation on `champion_history.json`'s K-block-elite
series - 3 datapoints per side at the pilot's `generation_pairs=3`.
Both pilot arms returned `gate_fires=0` because:

1. The series is too short for the gate's lag/window thresholds, AND
2. Prey K-block-elites saturate at 1.0 (the best-of-K-block hits the
   ceiling whenever the population saturates), so even a longer
   K-block series would be flat for prey.

This script tests whether the co-evolution verdict gate fires when fed the
**per-generation mean** fitness from `lineage.csv` instead. With 30
generations per side at pilot scale, the cycling autocorrelation has
~10x more samples in its lag range, and the escalation regression has
~10x more samples in its window - comfortably above the metric's
internal `_MIN_SERIES_LENGTH` and t-test sample-count guards.

Reuses `redqueen_metrics.phenotypic_cycling` + `trait_escalation`
verbatim (no copy-paste of metric implementations).

Usage::

    uv run python scripts/campaigns/screen_r4_per_gen_reaggregate.py \
        --pilot-root tmp/evaluations/coevolution/pr6_pilot_<TS>/pilot_run \
        --output-dir tmp/evaluations/coevolution/pr6_pilot_<TS>/r4

Outputs a markdown summary + a comparison CSV (per-gen vs K-block-elite
gate firings, side-by-side).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.evolution.redqueen_metrics import (
    phenotypic_cycling,
    trait_escalation,
)

logger = logging.getLogger(__name__)


def _per_gen_mean_fitness(lineage_csv: Path) -> np.ndarray:
    """Compute per-generation mean fitness from a lineage CSV.

    Schema: `generation,child_id,parent_ids,fitness,brain_type,inherited_from`.
    Returns a 1-D array indexed by generation, with NaN at any gens
    that have no rows (defensive — shouldn't happen in well-formed
    pilot data).
    """
    fitnesses_by_gen: dict[int, list[float]] = defaultdict(list)
    with lineage_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                gen = int(row["generation"])
                fit = float(row["fitness"])
            except (KeyError, ValueError):
                continue
            fitnesses_by_gen[gen].append(fit)

    if not fitnesses_by_gen:
        return np.empty(0, dtype=np.float64)

    max_gen = max(fitnesses_by_gen)
    series = np.full(max_gen + 1, np.nan, dtype=np.float64)
    for gen, fits in fitnesses_by_gen.items():
        series[gen] = float(np.mean(fits))
    return series


def _block_elite_series(champion_history_path: Path, side: str) -> np.ndarray:
    """Mirror the production aggregator's K-block-elite series builder."""
    if not champion_history_path.is_file():
        return np.empty(0, dtype=np.float64)
    with champion_history_path.open() as fh:
        data = json.load(fh)
    records = data.get(side, [])
    if not records:
        return np.empty(0, dtype=np.float64)
    sorted_records = sorted(records, key=lambda r: r.get("k_block_index", 0))
    return np.asarray(
        [float(r.get("fitness", float("nan"))) for r in sorted_records],
        dtype=np.float64,
    )


def _format_series_stats(series: np.ndarray) -> str:
    """Compact stats line for a per-gen series; empty-safe.

    `np.nanmin` / `np.nanmax` / `np.nanstd` raise on zero-size arrays
    (and emit RuntimeWarning on all-NaN arrays). When a session crashed
    before producing any lineage rows, the per-gen series is empty and
    the aggregator should still produce a readable report instead of
    aborting.
    """
    n = series.size
    if n == 0:
        return "n=0, min=n/a, max=n/a, std=n/a"
    if np.all(np.isnan(series)):
        return f"n={n}, min=n/a, max=n/a, std=n/a (all NaN)"
    return (
        f"n={n}, "
        f"min={np.nanmin(series):.3f}, "
        f"max={np.nanmax(series):.3f}, "
        f"std={np.nanstd(series):.3f}"
    )


def _format_metric(name: str, result: dict[str, Any]) -> str:
    """Compact one-line summary of a metric result dict."""
    if name == "cycling":
        detected = result["cycling_detected"]
        period = result.get("dominant_period")
        p = result.get("p_value")
        return (
            f"cycling_detected={detected} period={period} p={p:.4f}"
            if p is not None and np.isfinite(p)
            else f"cycling_detected={detected} period={period} p=NaN"
        )
    if name == "escalation":
        detected = result["escalation_detected"]
        slope = result.get("slope")
        slope_sign = result.get("slope_sign")
        p = result.get("p_value")
        slope_str = f"{slope:+.4f}" if slope is not None and np.isfinite(slope) else "NaN"
        p_str = f"{p:.4f}" if p is not None and np.isfinite(p) else "NaN"
        return f"escalation_detected={detected} slope={slope_str} sign={slope_sign} p={p_str}"
    msg = f"unknown metric name: {name}"
    raise ValueError(msg)


def _run_session(
    session_dir: Path,
    *,
    cycling_lag_range: tuple[int, int],
    escalation_gen_window: tuple[int, int],
) -> dict[str, Any]:
    """Compute all four metric pairs for one session and return a result dict."""
    prey_lineage = session_dir / "prey" / "lineage.csv"
    predator_lineage = session_dir / "predator" / "lineage.csv"
    champion_history = session_dir / "champion_history.json"

    prey_per_gen = _per_gen_mean_fitness(prey_lineage)
    predator_per_gen = _per_gen_mean_fitness(predator_lineage)
    prey_block = _block_elite_series(champion_history, side="prey")
    predator_block = _block_elite_series(champion_history, side="predator")

    return {
        "session": session_dir,
        "prey_per_gen": prey_per_gen,
        "predator_per_gen": predator_per_gen,
        "prey_block": prey_block,
        "predator_block": predator_block,
        # Per-gen metrics (R4 hypothesis).
        "prey_cycling_pg": phenotypic_cycling(prey_per_gen, lag_range=cycling_lag_range),
        "predator_cycling_pg": phenotypic_cycling(predator_per_gen, lag_range=cycling_lag_range),
        "prey_escalation_pg": trait_escalation(prey_per_gen, gen_window=escalation_gen_window),
        "predator_escalation_pg": trait_escalation(
            predator_per_gen,
            gen_window=escalation_gen_window,
        ),
        # K-block-elite metrics (production gate, for side-by-side comparison).
        "prey_cycling_be": phenotypic_cycling(prey_block, lag_range=cycling_lag_range),
        "predator_cycling_be": phenotypic_cycling(predator_block, lag_range=cycling_lag_range),
        "prey_escalation_be": trait_escalation(prey_block, gen_window=escalation_gen_window),
        "predator_escalation_be": trait_escalation(
            predator_block,
            gen_window=escalation_gen_window,
        ),
    }


def _gate_fires(
    prey_cycling: dict,
    predator_cycling: dict,
    prey_escalation: dict,
    predator_escalation: dict,
) -> bool:
    """Co-evolution verdict gate: cycling OR escalation, either side."""
    return bool(
        prey_cycling.get("cycling_detected")
        or predator_cycling.get("cycling_detected")
        or prey_escalation.get("escalation_detected")
        or predator_escalation.get("escalation_detected"),
    )


def _format_summary(  # noqa: PLR0915 - linear formatter; splitting fragments output ordering
    results: list[dict[str, Any]],
    cycling_lag_range: tuple[int, int],
    escalation_gen_window: tuple[int, int],
) -> str:
    """Side-by-side markdown comparing per-gen gate vs K-block-elite gate."""
    lines = [
        "# R4 — Per-generation re-aggregation",
        "",
        "Re-runs the co-evolution verdict gate (cycling + escalation) on the per-generation",
        "mean-fitness series from each session's `prey/lineage.csv` and",
        "`predator/lineage.csv`, instead of the production aggregator's",
        "`champion_history.json` K-block-elite series. The hypothesis is that",
        "the gate didn't fire on pilot data because the K-block-elite series",
        "(only 3 points/side at `generation_pairs=3`) was too short for the",
        "metric's lag/window thresholds, AND because prey K-block-elites",
        "saturate at the population's success-rate ceiling. Per-gen mean",
        "is ~10x more samples and never saturates at 1.0 (population mean",
        "stays slightly below the elite max).",
        "",
        f"Cycling lag range: {cycling_lag_range[0]}-{cycling_lag_range[1]} gens",
        f"Escalation gen window: {escalation_gen_window[0]}-{escalation_gen_window[1]} gens",
        "",
    ]

    pg_fires = 0
    be_fires = 0
    for res in results:
        sess = res["session"]
        try:
            relpath: Path | str = sess.relative_to(Path.cwd())
        except ValueError:
            # Fallback: session lives outside cwd. Emit only the session
            # leaf (a timestamp+hash like `20260514_072921_e287c2df`)
            # so we never embed any parent-path component — those can
            # carry usernames (e.g. `/Users/<name>/...`, `/home/<name>/...`)
            # or other local-filesystem details. The leaf alone uniquely
            # identifies the run within a campaign.
            relpath = sess.name
        lines.append(f"## {relpath}")
        lines.append("")
        lines.append(f"- Prey per-gen series: {_format_series_stats(res['prey_per_gen'])}")
        lines.append(
            f"- Predator per-gen series: {_format_series_stats(res['predator_per_gen'])}",
        )
        lines.append(f"- Prey K-block-elite series: {res['prey_block'].tolist()}")
        lines.append(f"- Predator K-block-elite series: {res['predator_block'].tolist()}")
        lines.append("")
        lines.append("### R4 (per-generation series)")
        lines.append("")
        lines.append(f"- Prey {_format_metric('cycling', res['prey_cycling_pg'])}")
        lines.append(f"- Predator {_format_metric('cycling', res['predator_cycling_pg'])}")
        lines.append(f"- Prey {_format_metric('escalation', res['prey_escalation_pg'])}")
        lines.append(f"- Predator {_format_metric('escalation', res['predator_escalation_pg'])}")
        pg_gate = _gate_fires(
            res["prey_cycling_pg"],
            res["predator_cycling_pg"],
            res["prey_escalation_pg"],
            res["predator_escalation_pg"],
        )
        if pg_gate:
            pg_fires += 1
        lines.append(f"- **Gate fires (per-gen): {pg_gate}**")
        lines.append("")
        lines.append("### Production gate (K-block-elite series, for comparison)")
        lines.append("")
        lines.append(f"- Prey {_format_metric('cycling', res['prey_cycling_be'])}")
        lines.append(f"- Predator {_format_metric('cycling', res['predator_cycling_be'])}")
        lines.append(f"- Prey {_format_metric('escalation', res['prey_escalation_be'])}")
        lines.append(f"- Predator {_format_metric('escalation', res['predator_escalation_be'])}")
        be_gate = _gate_fires(
            res["prey_cycling_be"],
            res["predator_cycling_be"],
            res["prey_escalation_be"],
            res["predator_escalation_be"],
        )
        if be_gate:
            be_fires += 1
        lines.append(f"- **Gate fires (K-block-elite): {be_gate}**")
        lines.append("")

    n = len(results)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Sessions analysed: {n}")
    lines.append(f"- Per-gen gate fires: {pg_fires}/{n}")
    lines.append(f"- K-block-elite gate fires: {be_fires}/{n}")
    lines.append("")
    if pg_fires > be_fires:
        lines.append(
            "**R4 hypothesis supported**: per-generation series fires more often than "
            "K-block-elite series — the production gate was bottlenecked by series length, "
            "not signal absence.",
        )
    elif pg_fires == 0:
        lines.append(
            "**R4 hypothesis rejected**: per-generation series doesn't fire either. "
            "The gate isn't underpowered — the signal genuinely isn't present.",
        )
    else:
        lines.append(
            "**R4 hypothesis partially supported**: per-generation series fires at the "
            "same rate as K-block-elite. The bottleneck isn't pure series length — "
            "the actual fitness trajectory is flat regardless of granularity.",
        )
    return "\n".join(lines) + "\n"


def _write_comparison_csv(path: Path, results: list[dict[str, Any]]) -> None:
    """Compact CSV: one row per session x side x metric x series-source."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["session", "series_source", "side", "metric", "detected", "value", "p_value"],
        )
        for res in results:
            sess = str(res["session"].name)
            for source, suffix in (("per_gen", "pg"), ("kblock_elite", "be")):
                for side in ("prey", "predator"):
                    cyc = res[f"{side}_cycling_{suffix}"]
                    esc = res[f"{side}_escalation_{suffix}"]
                    writer.writerow(
                        [
                            sess,
                            source,
                            side,
                            "cycling",
                            cyc["cycling_detected"],
                            cyc.get("dominant_period", ""),
                            cyc.get("p_value", ""),
                        ],
                    )
                    writer.writerow(
                        [
                            sess,
                            source,
                            side,
                            "escalation",
                            esc["escalation_detected"],
                            esc.get("slope", ""),
                            esc.get("p_value", ""),
                        ],
                    )


def _discover_sessions(pilot_root: Path) -> list[Path]:
    """Find every session dir under pilot_root that has both lineage CSVs."""
    sessions: list[Path] = []
    for prey_csv in pilot_root.rglob("prey/lineage.csv"):
        session_dir = prey_csv.parent.parent
        predator_csv = session_dir / "predator" / "lineage.csv"
        if predator_csv.is_file():
            sessions.append(session_dir)
    return sorted(sessions)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the R4 per-gen re-aggregator."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument(
        "--pilot-root",
        type=Path,
        required=True,
        help="Root containing pilot session dirs (e.g., the `pilot_run/` dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination dir for summary.md + comparison.csv.",
    )
    parser.add_argument("--cycling-lag-low", type=int, default=3)
    parser.add_argument("--cycling-lag-high", type=int, default=15)
    parser.add_argument("--escalation-gen-low", type=int, default=5)
    parser.add_argument("--escalation-gen-high", type=int, default=30)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: discover sessions under `--pilot-root`, run metrics, write outputs."""
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    sessions = _discover_sessions(args.pilot_root)
    if not sessions:
        logger.error("No sessions found under %s", args.pilot_root)
        return 1

    logger.info("Discovered %d session(s)", len(sessions))
    cycling_lag_range = (args.cycling_lag_low, args.cycling_lag_high)
    escalation_gen_window = (args.escalation_gen_low, args.escalation_gen_high)

    results = [
        _run_session(
            s,
            cycling_lag_range=cycling_lag_range,
            escalation_gen_window=escalation_gen_window,
        )
        for s in sessions
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _format_summary(results, cycling_lag_range, escalation_gen_window)
    (args.output_dir / "summary.md").write_text(summary)
    _write_comparison_csv(args.output_dir / "comparison.csv", results)
    logger.info("Wrote %s/summary.md and comparison.csv", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
