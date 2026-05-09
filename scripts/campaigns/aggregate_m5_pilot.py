# pragma: no cover
r"""Aggregate the M5 co-evolution arms-race pilot/full output.

Reads per-seed `CoevolutionLoop` output dirs (per-side lineage CSVs +
champion_history JSON + generality_probe CSV) and emits:

- ``summary.md`` — overall verdict (GO / STOP / PIVOT) + per-seed
  cycling/escalation/generality results + plot references.
- ``cycling.png`` — per-seed prey + predator block-elite fitness
  trajectories with autocorrelation peak overlay.
- ``escalation.png`` — per-seed prey + predator block-elite fitness
  with linear-regression slope overlay.
- ``generality.png`` — per-seed held-out probe trajectory (one line
  per (side, opponent_index)).
- ``verdict.csv`` — one row per seed with the gate's metric values.

Verdict gate (per design.md D6):
- Per-seed firing = phenotypic cycling OR trait escalation in EITHER
  side's block-elite fitness series.
- Aggregate verdict = GO if firing in ≥2 of N seeds, STOP if 0,
  PIVOT if exactly 1.

Generality is reported alongside but is NOT a verdict input. When the
probe fitness column is NaN (deferred wiring), the generality scalar
is reported as NaN; the verdict gate is unaffected.

Usage::

    uv run python scripts/campaigns/aggregate_m5_pilot.py \
        --root evolution_results/m5_coevolution_pilot/arm_a \
        --output-dir artifacts/logbooks/017-coevolution-arms-race/pilot/arm_a

The ``--root`` directory is expected to contain one or more session
subdirs (the campaign driver mints a fresh session id per run). When
multiple sessions exist, the most recent is used by default; pass
``--session <id>`` to pin a specific one.

For pilot ablation comparison (arm A vs arm B), run the aggregator
separately on each arm's root. The summary.md from each is then
human-readable side-by-side; a future cross-arm aggregator could
overlay them in a single plot if needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.evolution.redqueen_metrics import (
    coupled_rate,
    fitness_lag,
    generality,
    phenotypic_cycling,
    trait_escalation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IO: per-seed loaders
# ---------------------------------------------------------------------------


def _resolve_session_dir(seed_root: Path, session: str | None) -> Path | None:
    """Pick the session subdir to read from a per-seed root.

    ``seed_root`` may itself be a session dir (contains ``prey/`` etc.)
    or a parent containing one or more ``<session_id>/`` subdirs. When
    parent: ``--session`` pins one explicitly; otherwise pick the most
    recently modified.
    """
    if (seed_root / "prey" / "lineage.csv").is_file():
        return seed_root
    candidates = sorted(
        (p for p in seed_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if session is not None:
        match = [p for p in candidates if p.name == session]
        if not match:
            logger.error(
                "--session %s not found under %s; available: %s",
                session,
                seed_root,
                [p.name for p in candidates],
            )
            return None
        return match[0]
    if not candidates:
        logger.error("No session subdir under %s", seed_root)
        return None
    return candidates[0]


def _load_champion_history(session_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the top-level champion_history.json into a {prey, predator} dict."""
    path = session_dir / "champion_history.json"
    if not path.is_file():
        return {"prey": [], "predator": []}
    with path.open() as fh:
        data = json.load(fh)
    return {
        "prey": list(data.get("prey", [])),
        "predator": list(data.get("predator", [])),
    }


def _load_probe_csv(session_dir: Path) -> list[dict[str, Any]]:
    """Read generality_probe.csv into a list of row dicts."""
    path = session_dir / "generality_probe.csv"
    if not path.is_file():
        return []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        return list(reader)


# ---------------------------------------------------------------------------
# Per-seed metric computation
# ---------------------------------------------------------------------------


def _block_elite_series(champion_history: list[dict[str, Any]]) -> np.ndarray:
    """Extract the per-K-block fitness series from one side's champion history.

    Returns a 1-D float array indexed by `k_block_index` order.
    """
    if not champion_history:
        return np.empty(0, dtype=np.float64)
    sorted_records = sorted(champion_history, key=lambda r: r.get("k_block_index", 0))
    return np.asarray(
        [float(r.get("fitness", float("nan"))) for r in sorted_records],
        dtype=np.float64,
    )


def _probe_matrix(  # noqa: C901 — collision-detection + dual-loop reshape; splitting fragments the keep-LAST contract
    probe_rows: list[dict[str, Any]],
    *,
    side_filter: str,
) -> np.ndarray:
    """Reshape a probe CSV's rows into a (num_probes x num_opponents) matrix.

    Filters by side and reshapes by `(generation, opponent_index)`. Rows
    with non-numeric fitness (NaN literal, empty string) are coerced to
    NaN so the matrix shape is preserved even when the probe body is
    deferred and writes NaN per row.

    Returns an empty array if no rows match.
    """
    filtered = [r for r in probe_rows if r.get("side") == side_filter]
    if not filtered:
        return np.empty((0, 0), dtype=np.float64)
    by_gen: dict[int, dict[int, float]] = {}
    # Track non-NaN collisions across the same `(gen, opp_idx)` pair so
    # we surface them with a logger.warning. The probe CSV under K-block
    # cadence less than `K_per_block` writes multiple rows at the same
    # `side.generation` — for the deferred-body case (NaN-filled) the
    # collision is harmless; once the probe body is wired, real fitness
    # values would be silently dropped without this warning. Keep-LAST
    # semantic matches the prior behaviour.
    collisions: list[tuple[int, int]] = []
    for row in filtered:
        try:
            gen = int(row["generation"])
            opp_idx = int(row["opponent_index"])
        except (KeyError, ValueError, TypeError):
            continue
        try:
            fitness = float(row["fitness"])
        except (KeyError, ValueError, TypeError):
            fitness = float("nan")
        cell = by_gen.setdefault(gen, {})
        prev = cell.get(opp_idx)
        if prev is not None and not (np.isnan(prev) and np.isnan(fitness)):
            # Both prev and incoming are non-NaN, OR one is non-NaN and the
            # other is NaN — either way it's a real collision worth flagging.
            collisions.append((gen, opp_idx))
        cell[opp_idx] = fitness
    if collisions:
        # De-dupe + cap the list so massive cadence-mismatches don't
        # spam logs.
        unique = sorted(set(collisions))
        sample = unique[:5]
        logger.warning(
            "%d duplicate (generation, opponent_index) pair(s) in probe CSV "
            "for side=%s; keeping the LAST fitness value per pair. "
            "Sample: %s%s",
            len(unique),
            side_filter,
            sample,
            " ..." if len(unique) > len(sample) else "",
        )
    if not by_gen:
        return np.empty((0, 0), dtype=np.float64)
    sorted_gens = sorted(by_gen.keys())
    all_opps = sorted({opp for opp_map in by_gen.values() for opp in opp_map})
    if not all_opps:
        return np.empty((0, 0), dtype=np.float64)
    matrix = np.full((len(sorted_gens), len(all_opps)), np.nan, dtype=np.float64)
    for i, gen in enumerate(sorted_gens):
        opp_map = by_gen[gen]
        for j, opp_idx in enumerate(all_opps):
            if opp_idx in opp_map:
                matrix[i, j] = opp_map[opp_idx]
    return matrix


def _seed_metrics(  # noqa: PLR0913 — kw-only knobs map 1:1 to redqueen_metrics defaults
    *,
    prey_series: np.ndarray,
    predator_series: np.ndarray,
    prey_probe: np.ndarray,
    predator_probe: np.ndarray,
    cycling_lag_range: tuple[int, int],
    escalation_gen_window: tuple[int, int],
    p_threshold: float,
) -> dict[str, Any]:
    """Compute the per-seed Red Queen metrics + verdict-gate firing.

    Per-seed gate fires when phenotypic cycling OR trait escalation
    fires on EITHER side's block-elite fitness series. Generality is
    reported alongside but is not in the gate.
    """
    # The redqueen_metrics functions handle short-series cases by
    # returning placeholder dicts (non-detected, NaN p-value). Wrap
    # each in a defensive try/except in case unexpected input shapes
    # surface in real-world data.
    prey_cycling = _safe_cycling(prey_series, lag_range=cycling_lag_range, p_threshold=p_threshold)
    predator_cycling = _safe_cycling(
        predator_series,
        lag_range=cycling_lag_range,
        p_threshold=p_threshold,
    )
    prey_escalation = _safe_escalation(
        prey_series,
        gen_window=escalation_gen_window,
        p_threshold=p_threshold,
    )
    predator_escalation = _safe_escalation(
        predator_series,
        gen_window=escalation_gen_window,
        p_threshold=p_threshold,
    )
    coupled = _safe_coupled(prey_series, predator_series)
    lag = _safe_lag(prey_series, predator_series)
    prey_generality = _safe_generality(prey_probe)
    predator_generality = _safe_generality(predator_probe)

    cycling_detected = bool(
        prey_cycling.get("cycling_detected", False)
        or predator_cycling.get("cycling_detected", False),
    )
    escalation_detected = bool(
        prey_escalation.get("escalation_detected", False)
        or predator_escalation.get("escalation_detected", False),
    )
    gate_fires = cycling_detected or escalation_detected

    return {
        "prey_cycling": prey_cycling,
        "predator_cycling": predator_cycling,
        "prey_escalation": prey_escalation,
        "predator_escalation": predator_escalation,
        "coupled_rate": coupled,
        "fitness_lag": lag,
        "prey_generality": prey_generality,
        "predator_generality": predator_generality,
        "cycling_detected": cycling_detected,
        "escalation_detected": escalation_detected,
        "gate_fires": gate_fires,
    }


def _safe_cycling(series: np.ndarray, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 — pass-through to phenotypic_cycling's typed kwargs
    if series.size == 0:
        return {"cycling_detected": False, "dominant_period": None, "p_value": float("nan")}
    try:
        return phenotypic_cycling(series, **kwargs)
    except (ValueError, RuntimeError) as exc:
        logger.warning("phenotypic_cycling failed: %s", exc)
        return {"cycling_detected": False, "dominant_period": None, "p_value": float("nan")}


def _safe_escalation(series: np.ndarray, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 — pass-through to trait_escalation's typed kwargs
    if series.size == 0:
        return {
            "escalation_detected": False,
            "slope": float("nan"),
            "slope_sign": 0,
            "slope_se": float("nan"),
            "p_value": float("nan"),
        }
    try:
        return trait_escalation(series, **kwargs)
    except (ValueError, RuntimeError) as exc:
        logger.warning("trait_escalation failed: %s", exc)
        return {
            "escalation_detected": False,
            "slope": float("nan"),
            "slope_sign": 0,
            "slope_se": float("nan"),
            "p_value": float("nan"),
        }


def _safe_coupled(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        return coupled_rate(a, b)
    except (ValueError, RuntimeError) as exc:
        logger.warning("coupled_rate failed: %s", exc)
        return float("nan")


def _safe_lag(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    try:
        return fitness_lag(a, b)
    except (ValueError, RuntimeError) as exc:
        logger.warning("fitness_lag failed: %s", exc)
        return float("nan")


def _safe_generality(probe_matrix: np.ndarray) -> float:
    if probe_matrix.size == 0:
        return float("nan")
    # The generality function expects 2D and >=2 rows. When the probe
    # body is deferred (NaN-filled), the metric still returns NaN —
    # that's the documented "deferred body" semantic.
    if probe_matrix.ndim != 2 or probe_matrix.shape[0] < 2:
        return float("nan")
    # All-NaN matrix → numpy regression on NaN propagates NaN, which is
    # the right semantic. Caller renders NaN as "N/A" in summary.md.
    try:
        return generality(probe_matrix)
    except (ValueError, RuntimeError) as exc:
        logger.warning("generality failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Aggregate verdict + outputs
# ---------------------------------------------------------------------------


def _aggregate_verdict(per_seed: list[dict[str, Any]]) -> tuple[str, int, int]:
    """Compute aggregate verdict across seeds.

    Returns ``(verdict, fires, total)``. Per design.md D6:
    GO if firing in ≥2 of N seeds; STOP if 0; PIVOT if exactly 1.
    """
    fires = sum(1 for m in per_seed if m["gate_fires"])
    total = len(per_seed)
    if total == 0:
        return "INCONCLUSIVE", 0, 0
    if fires == 0:
        verdict = "STOP"
    elif fires == 1:
        verdict = "PIVOT"
    else:
        verdict = "GO"
    return verdict, fires, total


def _write_verdict_csv(out_path: Path, per_seed_rows: list[dict[str, Any]]) -> None:
    """Emit one row per seed with the gate's metric values."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "cycling_detected",
        "escalation_detected",
        "escalation_slope",
        "escalation_p_value",
        "cycling_period",
        "cycling_p_value",
        "generality_scalar",
        "gate_fires",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_seed_rows:
            metrics = row["metrics"]
            # Pick the side whose values populate the cycling /
            # escalation columns. Order of preference:
            #   1. If exactly one side fires, prefer that side (so the
            #      row's `cycling_period` etc. come from the firing
            #      side, not from a non-firing side that happened to
            #      have a smaller p_value).
            #   2. If both sides fire, prefer the smaller-p-value side
            #      (more decisive).
            #   3. If neither fires, prefer the smaller-p-value side
            #      anyway — purely cosmetic in that case (the row's
            #      `*_detected=0` makes the slot's slope/period
            #      uninteresting).
            cyc = _pick_side(
                metrics["prey_cycling"],
                metrics["predator_cycling"],
                p_key="p_value",
                detected_key="cycling_detected",
            )
            esc = _pick_side(
                metrics["prey_escalation"],
                metrics["predator_escalation"],
                p_key="p_value",
                detected_key="escalation_detected",
            )
            # Generality: prefer the side with non-NaN value; if both
            # are NaN, write NaN.
            gen_scalar = metrics["prey_generality"]
            if np.isnan(gen_scalar):
                gen_scalar = metrics["predator_generality"]
            writer.writerow(
                {
                    "seed": row["seed"],
                    "cycling_detected": int(metrics["cycling_detected"]),
                    "escalation_detected": int(metrics["escalation_detected"]),
                    "escalation_slope": _fmt_float(esc.get("slope")),
                    "escalation_p_value": _fmt_float(esc.get("p_value")),
                    "cycling_period": _fmt_period(cyc.get("dominant_period")),
                    "cycling_p_value": _fmt_float(cyc.get("p_value")),
                    "generality_scalar": _fmt_float(gen_scalar),
                    "gate_fires": int(metrics["gate_fires"]),
                },
            )


def _pick_side(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    p_key: str,
    detected_key: str,
) -> dict[str, Any]:
    """Return whichever of `a` and `b` populates the verdict.csv slot.

    Selection rule (per spec scenario "Verdict CSV Schema"):

    1. If exactly one side has `detected_key=True`, prefer that side.
       This avoids the cross-side leak where the row's `cycling_period`
       comes from a non-firing side that happened to have a smaller
       `p_value` while `cycling_detected=1` (which is OR across sides).
    2. Otherwise (both fire, or neither fires), prefer the smaller-
       `p_key`-value side (more decisive). NaN-safe.
    3. Tie on equal p-values: side `a` wins. Convention: callers pass
       the prey side as `a`, so prey wins ties.
    """
    a_fires = bool(a.get(detected_key, False))
    b_fires = bool(b.get(detected_key, False))
    if a_fires != b_fires:
        return a if a_fires else b
    av = a.get(p_key, float("nan"))
    bv = b.get(p_key, float("nan"))
    av = float("nan") if av is None else av
    bv = float("nan") if bv is None else bv
    if np.isnan(av) and np.isnan(bv):
        return a
    if np.isnan(av):
        return b
    if np.isnan(bv):
        return a
    return a if av <= bv else b


def _fmt_float(v: Any) -> str:  # noqa: ANN401 — formatter accepts heterogeneous numeric/None
    """Format a float value to 6 decimal places; empty string for None / NaN."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if np.isnan(f):
        return ""
    return f"{f:.6f}"


def _fmt_period(v: Any) -> str:  # noqa: ANN401 — formatter accepts heterogeneous int/None
    """Format a cycling period as int; empty string for None / unparseable."""
    if v is None:
        return ""
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return ""


def _format_summary(  # noqa: PLR0913 — readability: each section is one parameter
    *,
    verdict: str,
    fires: int,
    total: int,
    per_seed_rows: list[dict[str, Any]],
    cycling_lag_range: tuple[int, int],
    escalation_gen_window: tuple[int, int],
    output_dir: Path,
) -> str:
    """Render the verdict + per-seed details into Markdown."""
    lines: list[str] = []
    lines.append("# M5 Co-evolution Arms Race — Aggregate")
    lines.append("")
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append(
        f"**Seeds firing the gate:** {fires} of {total} (GO if ≥2; PIVOT if 1; STOP if 0).",
    )
    lines.append("")
    lines.append(
        f"Gate criterion: phenotypic cycling (lag ∈ {cycling_lag_range}) "
        f"OR trait escalation (gen window {escalation_gen_window}) on "
        "either side's block-elite fitness series. Generality is "
        "reported alongside but is not a gate input.",
    )
    lines.append("")
    lines.append("## Per-seed results")
    lines.append("")
    if not per_seed_rows:
        lines.append(
            "**No seeds resolved** — every seed dir under `--root` was "
            "missing or unreadable. The aggregator emits this summary + "
            "an empty-body verdict.csv anyway so downstream tooling can "
            "detect the INCONCLUSIVE case without grepping logs.",
        )
        lines.append("")
    for row in per_seed_rows:
        seed = row["seed"]
        metrics = row["metrics"]
        lines.append(f"### Seed {seed}")
        lines.append("")
        lines.append(f"- **Gate fires:** {'YES' if metrics['gate_fires'] else 'no'}")
        lines.append(
            f"- Cycling: prey={_fmt_cycling(metrics['prey_cycling'])}, "
            f"predator={_fmt_cycling(metrics['predator_cycling'])}",
        )
        lines.append(
            f"- Escalation: prey={_fmt_escalation(metrics['prey_escalation'])}, "
            f"predator={_fmt_escalation(metrics['predator_escalation'])}",
        )
        lines.append(
            f"- Coupled rate (prey ↔ predator): {_fmt_float_or_na(metrics['coupled_rate'])}",
        )
        lines.append(
            f"- Fitness lag (prey ↔ predator): {_fmt_float_or_na(metrics['fitness_lag'])}",
        )
        lines.append(
            f"- Generality: prey={_fmt_float_or_na(metrics['prey_generality'])}, "
            f"predator={_fmt_float_or_na(metrics['predator_generality'])}",
        )
        lines.append("")
    lines.append("## Plots")
    lines.append("")
    for plot_name in ("cycling.png", "escalation.png", "generality.png"):
        plot_path = output_dir / plot_name
        if plot_path.is_file():
            lines.append(f"- [{plot_name}]({plot_name})")
        else:
            lines.append(f"- {plot_name} (not generated)")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Per-opponent generality probe evaluation is currently a no-op "
        "(`_probe_one_opponent` returns NaN); the schema and cadence "
        "are normative but the body is deferred. Generality columns "
        "show NaN until the body is wired.",
    )
    lines.append(
        "- Verdict is computed from cycling + escalation only; "
        "generality is interpretation guide per design.md D6.",
    )
    lines.append("")
    return "\n".join(lines)


def _fmt_cycling(d: dict[str, Any]) -> str:
    if d.get("cycling_detected"):
        period = d.get("dominant_period")
        p = _fmt_float_or_na(d.get("p_value"))
        return f"YES (period={period}, p={p})"
    p = _fmt_float_or_na(d.get("p_value"))
    return f"no (p={p})"


def _fmt_escalation(d: dict[str, Any]) -> str:
    if d.get("escalation_detected"):
        slope = _fmt_float_or_na(d.get("slope"))
        p = _fmt_float_or_na(d.get("p_value"))
        return f"YES (slope={slope}, p={p})"
    p = _fmt_float_or_na(d.get("p_value"))
    return f"no (p={p})"


def _fmt_float_or_na(v: Any) -> str:  # noqa: ANN401 — formatter accepts heterogeneous numeric/None
    """Format a float to 4 decimal places; "N/A" for None / NaN."""
    if v is None:
        return "N/A"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(f):
        return "N/A"
    return f"{f:.4f}"


# ---------------------------------------------------------------------------
# Plots (matplotlib lazy import)
# ---------------------------------------------------------------------------


def _plot_cycling(per_seed_rows: list[dict[str, Any]], out_path: Path) -> None:
    """Per-seed prey + predator block-elite trajectories."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    n_seeds = len(per_seed_rows)
    if n_seeds == 0:
        logger.warning("No seeds to plot for cycling; skipping %s", out_path)
        return
    fig, axes = plt.subplots(n_seeds, 1, figsize=(10, 3 * max(1, n_seeds)), squeeze=False)
    for i, row in enumerate(per_seed_rows):
        ax = axes[i, 0]
        prey = row["prey_series"]
        predator = row["predator_series"]
        if prey.size:
            ax.plot(np.arange(prey.size), prey, "o-", label="prey", color="tab:blue")
        if predator.size:
            ax.plot(np.arange(predator.size), predator, "s-", label="predator", color="tab:red")
        ax.set_title(f"Seed {row['seed']} — block-elite fitness over K-blocks")
        ax.set_xlabel("K-block index")
        ax.set_ylabel("Fitness")
        ax.legend(loc="best")
        ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_escalation(per_seed_rows: list[dict[str, Any]], out_path: Path) -> None:
    """Per-seed block-elite series + linear regression overlay."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    n_seeds = len(per_seed_rows)
    if n_seeds == 0:
        logger.warning("No seeds to plot for escalation; skipping %s", out_path)
        return
    fig, axes = plt.subplots(n_seeds, 1, figsize=(10, 3 * max(1, n_seeds)), squeeze=False)
    for i, row in enumerate(per_seed_rows):
        ax = axes[i, 0]
        for series, esc, label, color in (
            (row["prey_series"], row["metrics"]["prey_escalation"], "prey", "tab:blue"),
            (
                row["predator_series"],
                row["metrics"]["predator_escalation"],
                "predator",
                "tab:red",
            ),
        ):
            if not series.size:
                continue
            x = np.arange(series.size)
            ax.plot(x, series, "o-", label=label, color=color, alpha=0.7)
            slope = esc.get("slope")
            if slope is not None and not np.isnan(slope):
                # Line through (0, series[0]) with the fitted slope.
                fit = series[0] + slope * x
                ax.plot(x, fit, "--", color=color, alpha=0.5, label=f"{label} fit")
        ax.set_title(f"Seed {row['seed']} — escalation regression")
        ax.set_xlabel("K-block index")
        ax.set_ylabel("Fitness")
        ax.legend(loc="best")
        ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_generality(per_seed_rows: list[dict[str, Any]], out_path: Path) -> None:
    """Per-seed held-out probe trajectories (one line per opponent)."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    n_seeds = len(per_seed_rows)
    if n_seeds == 0:
        logger.warning("No seeds to plot for generality; skipping %s", out_path)
        return
    fig, axes = plt.subplots(n_seeds, 1, figsize=(10, 3 * max(1, n_seeds)), squeeze=False)
    for i, row in enumerate(per_seed_rows):
        ax = axes[i, 0]
        for label, matrix, base_color in (
            ("prey", row["prey_probe"], "tab:blue"),
            ("predator", row["predator_probe"], "tab:red"),
        ):
            if matrix.size == 0 or np.all(np.isnan(matrix)):
                continue
            for j in range(matrix.shape[1]):
                ax.plot(
                    np.arange(matrix.shape[0]),
                    matrix[:, j],
                    "o-",
                    color=base_color,
                    alpha=0.4,
                    label=f"{label} opp {j}" if j == 0 else None,
                )
        ax.set_title(f"Seed {row['seed']} — generality probe")
        ax.set_xlabel("Probe index (every `generality_probe_every` gens)")
        ax.set_ylabel("Probe fitness")
        ax.legend(loc="best")
        ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the aggregator."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate co-evolution arms-race output into summary.md + verdict.csv + plots."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Campaign root containing per-seed subdirs "
            "(`seed-42/`, `seed-43/`, ...). The pilot wrapper writes "
            "to `evolution_results/m5_coevolution_pilot/{arm_a,arm_b}/`; "
            "pass each arm's path separately."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to aggregate (default: discover from `--root` subdirs).",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help=(
            "Pin a specific session id under each seed's dir. When "
            "omitted, picks the most recently modified session."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination dir for summary.md + verdict.csv + plots.",
    )
    parser.add_argument(
        "--cycling-lag-low",
        type=int,
        default=3,
        help="Low end of the autocorrelation lag-range scan (default 3).",
    )
    parser.add_argument(
        "--cycling-lag-high",
        type=int,
        default=15,
        help="High end of the autocorrelation lag-range scan (default 15).",
    )
    parser.add_argument(
        "--escalation-gen-low",
        type=int,
        default=5,
        help="Low generation index for the escalation regression window (default 5).",
    )
    parser.add_argument(
        "--escalation-gen-high",
        type=int,
        default=30,
        help="High generation index for the escalation regression window (default 30).",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="Significance threshold for cycling + escalation gates (default 0.05).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _discover_seeds(root: Path) -> list[int]:
    """Find `seed-<N>/` subdirs under `root` and return the integer N values."""
    out: list[int] = []
    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith("seed-"):
            continue
        try:
            out.append(int(p.name.removeprefix("seed-")))
        except ValueError:
            continue
    return sorted(out)


def main() -> int:
    """Entry point: load per-seed inputs, compute verdict, write artefacts."""
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.root.is_dir():
        logger.error("--root is not a directory: %s", args.root)
        return 1

    seeds = args.seeds if args.seeds is not None else _discover_seeds(args.root)
    if not seeds:
        # Two single-seed layouts to recognise as a fallback before
        # failing:
        #   (a) `--root` IS the session dir (contains prey/, predator/,
        #       champion_history.json directly).
        #   (b) `--root` is a per-seed parent (no seed-<N>/) containing
        #       one or more session subdirs. The smoke driver writes
        #       this layout when called without `seed-<N>/` wrapping.
        # In both cases, treat the data as one synthetic seed (-1) so
        # the rest of the pipeline still runs.
        single_session_direct = (args.root / "prey" / "lineage.csv").is_file() or (
            args.root / "champion_history.json"
        ).is_file()
        single_session_parent = any(
            (p / "prey" / "lineage.csv").is_file() for p in args.root.iterdir() if p.is_dir()
        )
        if single_session_direct or single_session_parent:
            seeds = [-1]
            logger.info("--root looks like a single-session layout; aggregating as seed -1")
        else:
            logger.error(
                "No `seed-<N>/` subdirs found under %s and the root itself "
                "doesn't look like a session dir or a session-parent dir. "
                "Pass --seeds explicitly or check the layout.",
                args.root,
            )
            return 1

    cycling_lag_range = (args.cycling_lag_low, args.cycling_lag_high)
    escalation_gen_window = (args.escalation_gen_low, args.escalation_gen_high)

    per_seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_root = args.root if seed == -1 else args.root / f"seed-{seed}"
        if not seed_root.is_dir():
            logger.warning("Seed %d dir missing, skipping: %s", seed, seed_root)
            continue
        session_dir = _resolve_session_dir(seed_root, args.session)
        if session_dir is None:
            continue
        champion_history = _load_champion_history(session_dir)
        prey_series = _block_elite_series(champion_history["prey"])
        predator_series = _block_elite_series(champion_history["predator"])
        probe_rows = _load_probe_csv(session_dir)
        prey_probe = _probe_matrix(probe_rows, side_filter="prey")
        predator_probe = _probe_matrix(probe_rows, side_filter="predator")
        metrics = _seed_metrics(
            prey_series=prey_series,
            predator_series=predator_series,
            prey_probe=prey_probe,
            predator_probe=predator_probe,
            cycling_lag_range=cycling_lag_range,
            escalation_gen_window=escalation_gen_window,
            p_threshold=args.p_threshold,
        )
        per_seed_rows.append(
            {
                "seed": seed,
                "session_dir": session_dir,
                "prey_series": prey_series,
                "predator_series": predator_series,
                "prey_probe": prey_probe,
                "predator_probe": predator_probe,
                "metrics": metrics,
            },
        )

    # Zero-resolved-seeds case: emit an INCONCLUSIVE artefact set per
    # the co-evolution capability's "Verdict Aggregation Across Seeds"
    # scenario. Distinct from STOP (which is a substantive null result
    # over ≥1 resolved seeds). Skip plots (no series to plot) but
    # still write summary.md + an empty-body verdict.csv so downstream
    # tooling can detect the case without grepping logs.
    verdict, fires, total = _aggregate_verdict([r["metrics"] for r in per_seed_rows])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_verdict_csv(args.output_dir / "verdict.csv", per_seed_rows)

    # Plots are optional (matplotlib import happens lazily inside each
    # plot function). Don't fail the run if matplotlib is unavailable.
    # Run plots BEFORE writing summary.md so the summary can reference
    # which plot files exist on disk.
    try:
        _plot_cycling(per_seed_rows, args.output_dir / "cycling.png")
        _plot_escalation(per_seed_rows, args.output_dir / "escalation.png")
        _plot_generality(per_seed_rows, args.output_dir / "generality.png")
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots.")
    except Exception:
        logger.exception("Plot generation failed (continuing).")

    summary = _format_summary(
        verdict=verdict,
        fires=fires,
        total=total,
        per_seed_rows=per_seed_rows,
        cycling_lag_range=cycling_lag_range,
        escalation_gen_window=escalation_gen_window,
        output_dir=args.output_dir,
    )
    (args.output_dir / "summary.md").write_text(summary)

    logger.info(
        "Aggregate written to %s (verdict=%s, %d/%d seeds firing)",
        args.output_dir,
        verdict,
        fires,
        total,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
