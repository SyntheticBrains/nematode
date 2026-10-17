#!/usr/bin/env python
"""M.1(a): what happens inside a body learner in the 1,500 episodes before it collapses.

Logbook 092's thermal body panel lost eight learners late in training: the wild type on seeds 2313
and 2328, the gap-only null on 2304, 2305, 2307, 2308, 2319 and 2330. Each is re-run with per-update
telemetry, and so is the other wiring on the same seed, which did not collapse: the pair shares a
seed, a task draw and every initial chemical weight, so each collapse is read against its own
intact twin. The MLP control's eight retry seeds are re-run too, for its incidence.

**Identity.** A re-run licenses the diagnosis only if its final-quarter plateau equals the committed
per-seed value to the precision it was printed at. The step-level logs are archived off-repo, so the
plateau is the field compared; nothing else is.

**The envelope.** Each intact twin is aligned on its collapsed twin's onset, so the eight intact runs
give, for every statistic and every episode before onset, a mean and a standard deviation; the
envelope is mean +/- 2 sd. A run **departs** on a statistic when the statistic, smoothed over 100
episodes, sits outside the envelope in the predicted direction for at least 200 consecutive episodes
inside the 1,500 before onset. Each intact twin is held to the same rule against the envelope of the
other seven.

**The causes**, fixed before any log is read, each a conjunction of departures:

1. weight growth: the weight norm up, the saturated fraction up and ``srank`` down;
2. a vanishing signal: the raw advantages' spread down, with the gradient norm not departing;
3. stale second moments: the gradient norm up, then KL up, the gradient's departure first;
4. critic failure: explained variance down and value loss up;
5. variance collapse: ``log_std`` down;
6. mean-head saturation: the ``|u| > 2`` fraction up, with ``log_std`` not departing.

A cause is **indicated** when it departs in at least three quarters of the collapsed runs (6 of 8)
and in at most one eighth of the intact twins (1 of 8); **not indicated** when it departs in at
most a quarter of the collapsed runs (2 of 8); otherwise **mixed**.

Usage::

    uv run python scripts/analysis/collapse_diagnosis.py identity --logs <dir> --out identity.json
    uv run python scripts/analysis/collapse_diagnosis.py diagnose --logs <dir> --out diagnosis.json
    uv run python scripts/analysis/collapse_diagnosis.py mlp --logs <dir> --out mlp.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import collapse as cl  # noqa: E402  # pyright: ignore[reportMissingImports]
import t7_continuous_ranking as t7  # noqa: E402  # pyright: ignore[reportMissingImports]
from quantumnematode.brain.arch._telemetry import SRANK_ROWS  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"
SUPPORTING = REPO / "docs" / "experiments" / "logbooks" / "supporting" / "092-thermal-body"

_P = "connectomeppo_small_continuous2d_thermal_klinotaxis_body"
WT, GAP_ONLY = f"{_P}_wt_ppo_t20", f"{_P}_gaponly_ppo_t20"
MLP = "mlpppo_small_continuous2d_thermal_klinotaxis_body_ppo_t20"
COLLAPSED: dict[int, str] = {
    2313: WT,
    2328: WT,
    **dict.fromkeys((2304, 2305, 2307, 2308, 2319, 2330), GAP_ONLY),
}
MLP_SEEDS: tuple[int, ...] = tuple(range(9721, 9729))
EPISODES = 6000


@dataclass(frozen=True)
class Pair:
    """One seed: the run that collapsed on the committed panel, and its intact twin."""

    seed: int
    collapsed: str
    intact: str


PAIRS: tuple[Pair, ...] = tuple(
    Pair(seed, stem, GAP_ONLY if stem == WT else WT) for seed, stem in sorted(COLLAPSED.items())
)
RUNS: tuple[tuple[str, int], ...] = (
    *((p.collapsed, p.seed) for p in PAIRS),
    *((p.intact, p.seed) for p in PAIRS),
    *((MLP, s) for s in MLP_SEEDS),
)

LEAD = 1500
# The window ends this many episodes before onset: the trailing-100 success rate crosses half the
# peak some way into the fall, and the statistics are smoothed over 100 episodes, so the last
# hundred episodes before onset already carry the fall itself.
GAP = 100
# Each run's own baseline: its smoothed mean over the 1,500 episodes before the window. Departures
# are read on the change from it, so a level that differs between wirings from the first episode is
# not read as a cause.
BASELINE = 1500
SMOOTH = 100
MIN_RUN = 200
SDS = 2.0
# The envelope's half-width is never below this fraction of the envelope runs' mean baseline, nor
# below the statistic's absolute floor: a statistic that sits at one value on every intact run has
# no spread, and any movement at all would otherwise depart.
FLOOR_FRACTION = 0.10
ABS_FLOOR: dict[str, float] = {
    "saturated_fraction": 0.01,
    "action_mean_large_fraction": 0.01,
    "clip_fraction": 0.01,
    "approx_kl": 1e-4,
    "explained_variance": 0.05,
    "srank": 1.0,
}
# The transient form of stale second moments: a gradient-norm spike followed within this many
# updates by a KL spike, a spike being above the run's own baseline median by this many MADs.
SPIKE_MADS = 5.0
SPIKE_FOLLOW = 20
# The stacks written out: every this many episodes, through this many after onset.
STACK_STEP = 25
AFTER = 300
CONTEXT = ("entropy", "clip_fraction", "approx_kl", "critic_over_actor_grad")
INDICATED_COLLAPSED = 0.75
INDICATED_INTACT = 1 / 8
NOT_INDICATED = 0.25


class DiagnosisError(ValueError):
    """The runs on disk are not the re-runs this module reads."""


# ── Identity ─────────────────────────────────────────────────────────────────────────────────
def committed_plateaus() -> dict[tuple[str, int], tuple[float, int]]:
    """``(stem, seed) -> (committed plateau %, printed decimals)`` from 092's per-seed CSVs."""
    out: dict[tuple[str, int], tuple[float, int]] = {}
    with (SUPPORTING / "per-seed.csv").open() as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            for stem, column in ((WT, "gap_only_wt_plateau"), (GAP_ONLY, "gap_only_rn_plateau")):
                text = row[column]
                out[stem, seed] = (float(text), len(text.partition(".")[2]))
    with (SUPPORTING / "retry" / "per_seed.csv").open() as handle:
        for row in csv.DictReader(handle):
            text = row["mlp_plateau_pct"]
            out[MLP, int(row["seed"])] = (float(text), len(text.partition(".")[2]))
    return out


def _find(log_dirs: list[Path], stem: str, seed: int) -> Path | None:
    name = f"{stem}-seed{seed}.log"
    return next((d / name for d in log_dirs if (d / name).is_file()), None)


def identity(log_dirs: list[Path]) -> dict[str, Any]:
    """Compare each re-run's plateau with the committed value, to the committed precision."""
    committed = committed_plateaus()
    runs: dict[str, Any] = {}
    for stem, seed in RUNS:
        log = _find(log_dirs, stem, seed)
        value, decimals = committed[stem, seed]
        tail = t7.plateau_tail(log) if log is not None else None
        rerun = None if tail is None else round(tail[0], decimals)
        runs[f"{stem}-seed{seed}"] = {
            "committed": value,
            "rerun": rerun,
            "identical": rerun is not None and rerun == round(value, decimals),
        }
    return {
        "compared": "final-quarter plateau success %, to the committed CSV's printed precision",
        "uncompared": "every other field: the step-level logs are archived off-repo",
        "runs": runs,
        "all_identical": all(r["identical"] for r in runs.values()),
    }


# ── Telemetry to per-episode statistics ──────────────────────────────────────────────────────
def _weight_norm(line: dict[str, Any]) -> float:
    """Return the learner's weight norm, over every parameter but the critic's."""
    norms = line["weight_norms"]
    return float(np.sqrt(sum(v * v for k, v in norms.items() if not k.startswith("critic."))))


STATISTICS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "weight_norm": _weight_norm,
    "saturated_fraction": lambda line: line["saturated_fraction"],
    # A batch shorter than the fixed row count gives a smaller rank for that reason alone.
    "srank": lambda line: (
        line["srank"] if line.get("srank_rows", SRANK_ROWS) >= SRANK_ROWS else None
    ),
    "adv_std": lambda line: line["adv_std"],
    "grad_norm": lambda line: line["grad_norm"],
    "approx_kl": lambda line: line["approx_kl"],
    "clip_fraction": lambda line: line["clip_fraction"],
    "explained_variance": lambda line: line["explained_variance"],
    "value_loss": lambda line: line["value_loss"],
    "log_std": lambda line: line["log_std"],
    "action_mean_large_fraction": lambda line: line.get("action_mean_large_fraction"),
    "entropy": lambda line: line["entropy"],
    "critic_over_actor_grad": lambda line: (
        line["critic_grad"] / line["actor_grad"] if line["actor_grad"] else None
    ),
}


def episode_series(lines: list[dict[str, Any]], horizon: int = EPISODES) -> dict[str, np.ndarray]:
    """Each statistic per episode: the mean of that episode's updates, carried forward between."""
    out: dict[str, np.ndarray] = {}
    for name, read in STATISTICS.items():
        sums, counts = np.zeros(horizon), np.zeros(horizon)
        for line in lines:
            value = read(line)
            episode = int(line["episode"])
            if value is not None and 1 <= episode <= horizon:
                sums[episode - 1] += float(value)
                counts[episode - 1] += 1
        series = np.full(horizon, np.nan)
        seen = counts > 0
        series[seen] = sums[seen] / counts[seen]
        out[name] = _carry_forward(series)
    return out


def _carry_forward(series: np.ndarray) -> np.ndarray:
    out = series.copy()
    for t in range(1, len(out)):
        if np.isnan(out[t]):
            out[t] = out[t - 1]
    return out


def smoothed(series: np.ndarray, window: int = SMOOTH) -> np.ndarray:
    """Return the trailing mean over ``window`` episodes, ignoring episodes not yet seen."""
    seen = ~np.isnan(series)
    values = np.cumsum(np.insert(np.where(seen, series, 0.0), 0, 0.0))
    counts = np.cumsum(np.insert(seen.astype(float), 0, 0.0))
    lo = np.maximum(np.arange(len(series)) - window + 1, 0)
    hi = np.arange(1, len(series) + 1)
    n = counts[hi] - counts[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (values[hi] - values[lo]) / n, np.nan)


def window_bounds(onset: int) -> tuple[int, int]:
    """Return the window's ``[start, end)`` indices: the ``LEAD`` episodes ending ``GAP`` before onset."""
    end = onset - 1 - GAP
    return end - LEAD, end


def aligned(series: np.ndarray, onset: int) -> np.ndarray:
    """Return the window's values, NaN-padded where the run did not reach that far back."""
    start, end = window_bounds(onset)
    out = np.full(LEAD, np.nan)
    lo = max(0, start)
    if end > lo:
        out[lo - start : lo - start + (end - lo)] = series[lo:end]
    return out


def baseline(series: np.ndarray, onset: int) -> float:
    """Return a run's mean over the ``BASELINE`` episodes before its window (what exists of them)."""
    start, _ = window_bounds(onset)
    chunk = series[max(0, start - BASELINE) : max(0, start)]
    seen = chunk[~np.isnan(chunk)]
    return float(seen.mean()) if seen.size else float("nan")


# ── The envelope and the departure rule ──────────────────────────────────────────────────────
def envelope(
    stack: list[np.ndarray],
    baselines: list[float],
    statistic: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the envelope's centre and half-width at each aligned episode.

    The centre is the stacked runs' mean; the half-width is two sample standard deviations
    (``ddof=1``), never below ``FLOOR_FRACTION`` of the runs' mean absolute baseline nor below the
    statistic's absolute floor.
    """
    array = np.vstack(stack)
    with warnings.catch_warnings():
        # An aligned episode no run reached is NaN in every row, and stays NaN.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean, sd = np.nanmean(array, axis=0), np.nanstd(array, axis=0, ddof=1)
        level = float(np.nanmean(np.abs(baselines))) if baselines else 0.0
    floor = max(FLOOR_FRACTION * (0.0 if np.isnan(level) else level), ABS_FLOOR.get(statistic, 0.0))
    return mean, np.fmax(SDS * np.nan_to_num(sd, nan=0.0), floor)


def departure(
    series: np.ndarray,
    centre: np.ndarray,
    half_width: np.ndarray,
    direction: int,
    min_run: int = MIN_RUN,
) -> int | None:
    """Return how far before onset a ``min_run``-long stretch outside the envelope begins.

    ``direction`` is +1 for above the envelope, -1 for below. Returns the stretch's lead before
    onset, in episodes (``GAP + LEAD`` is the earliest), or ``None`` when there is no such stretch.
    """
    with np.errstate(invalid="ignore"):
        outside = (
            series > centre + half_width if direction > 0 else series < centre - half_width
        ) & ~np.isnan(series)
    run = 0
    for t, flag in enumerate(outside):
        run = run + 1 if flag else 0
        if run == min_run:
            return GAP + len(series) - (t - min_run + 1)
    return None


@dataclass(frozen=True)
class Component:
    """One statistic of a cause's pattern, and the way it must move (0: it must not depart)."""

    statistic: str
    direction: int


CAUSES: dict[str, tuple[Component, ...]] = {
    "weight_growth": (
        Component("weight_norm", +1),
        Component("saturated_fraction", +1),
        Component("srank", -1),
    ),
    "vanishing_signal": (Component("adv_std", -1), Component("grad_norm", 0)),
    "stale_second_moments": (Component("grad_norm", +1), Component("approx_kl", +1)),
    "critic_failure": (Component("explained_variance", -1), Component("value_loss", +1)),
    "variance_collapse": (Component("log_std", -1),),
    "mean_head_saturation": (
        Component("action_mean_large_fraction", +1),
        Component("log_std", 0),
    ),
}


def cause_departs(
    run: dict[str, np.ndarray],
    reference: dict[str, tuple[np.ndarray, np.ndarray]],
    components: tuple[Component, ...],
) -> tuple[bool, dict[str, int | None]]:
    """Whether every component of a cause departs as it must, and each component's lead."""
    leads: dict[str, int | None] = {}
    ok = True
    for c in components:
        mean, sd = reference[c.statistic]
        if c.direction == 0:
            up = departure(run[c.statistic], mean, sd, +1)
            down = departure(run[c.statistic], mean, sd, -1)
            leads[c.statistic] = up if up is not None else down
            ok = ok and up is None and down is None
        else:
            leads[c.statistic] = departure(run[c.statistic], mean, sd, c.direction)
            ok = ok and leads[c.statistic] is not None
    return ok, leads


def _ordered(name: str, leads: dict[str, int | None]) -> bool:
    """Stale second moments need the gradient's departure to come no later than KL's."""
    if name != "stale_second_moments":
        return True
    grad, kl = leads.get("grad_norm"), leads.get("approx_kl")
    return grad is not None and kl is not None and grad >= kl


def classify(collapsed_hits: int, n_collapsed: int, intact_hits: int, n_intact: int) -> str:
    """Read a cause as indicated, not indicated or mixed, by the registered fractions."""
    if n_collapsed == 0:
        return "unreadable"
    if collapsed_hits >= INDICATED_COLLAPSED * n_collapsed and (
        intact_hits <= INDICATED_INTACT * n_intact
    ):
        return "indicated"
    if collapsed_hits <= NOT_INDICATED * n_collapsed:
        return "not_indicated"
    return "mixed"


def _view(
    series: dict[str, np.ndarray],
    onset: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return a run's windowed change from its own baseline, and the baselines, per statistic."""
    smooth = {k: smoothed(v) for k, v in series.items()}
    base = {k: baseline(v, onset) for k, v in smooth.items()}
    return {k: aligned(v, onset) - np.nan_to_num(base[k]) for k, v in smooth.items()}, base


def _reference(
    views: dict[int, dict[str, np.ndarray]],
    bases: dict[int, dict[str, float]],
    seeds: list[int],
    statistics: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        k: envelope([views[s][k] for s in seeds], [bases[s][k] for s in seeds], k)
        for k in statistics
    }


def diagnose_series(
    collapsed: dict[int, dict[str, np.ndarray]],
    intact: dict[int, dict[str, np.ndarray]],
    onsets: dict[int, int],
) -> dict[str, Any]:
    """Read every cause on windowed, smoothed, baseline-relative series; one pair per seed."""
    seeds = sorted(onsets)
    views: dict[str, dict[int, dict[str, np.ndarray]]] = {"collapsed": {}, "intact": {}}
    bases: dict[str, dict[int, dict[str, float]]] = {"collapsed": {}, "intact": {}}
    for side, runs in (("collapsed", collapsed), ("intact", intact)):
        for s in seeds:
            views[side][s], bases[side][s] = _view(runs[s], onsets[s])
    statistics = sorted({c.statistic for comps in CAUSES.values() for c in comps})
    full = _reference(views["intact"], bases["intact"], seeds, statistics)
    out: dict[str, Any] = {"seeds": seeds, "onsets": onsets, "causes": {}, "baselines": bases}
    for name, components in CAUSES.items():
        per_run: dict[str, Any] = {"collapsed": {}, "intact": {}}
        for s in seeds:
            hit, leads = cause_departs(views["collapsed"][s], full, components)
            per_run["collapsed"][s] = {"departs": hit and _ordered(name, leads), "leads": leads}
            others = [t for t in seeds if t != s]
            loo = _reference(views["intact"], bases["intact"], others, statistics)
            hit, leads = cause_departs(views["intact"][s], loo, components)
            per_run["intact"][s] = {"departs": hit and _ordered(name, leads), "leads": leads}
        hits_c = sum(r["departs"] for r in per_run["collapsed"].values())
        hits_i = sum(r["departs"] for r in per_run["intact"].values())
        out["causes"][name] = {
            "collapsed_departures": hits_c,
            "intact_departures": hits_i,
            "n": len(seeds),
            "reading": classify(hits_c, len(seeds), hits_i, len(seeds)),
            "per_run": per_run,
        }
    out["indicated"] = [n for n, c in out["causes"].items() if c["reading"] == "indicated"]
    out["context_at_onset"] = {
        side: {s: _context(runs[s], onsets[s]) for s in seeds}
        for side, runs in (("collapsed", collapsed), ("intact", intact))
    }
    out["stacks"] = [
        row
        for side, runs in (("collapsed", collapsed), ("intact", intact))
        for s in seeds
        for row in _stack_rows(side, s, runs[s], onsets[s])
    ]
    return out


def _context(series: dict[str, np.ndarray], onset: int) -> dict[str, float | None]:
    """Return the context statistics' smoothed mean over the 100 episodes around onset."""
    out: dict[str, float | None] = {}
    for k in CONTEXT:
        chunk = smoothed(series[k])[max(0, onset - 51) : onset + 49]
        seen = chunk[~np.isnan(chunk)]
        out[k] = float(seen.mean()) if seen.size else None
    return out


def _stack_rows(
    side: str,
    seed: int,
    series: dict[str, np.ndarray],
    onset: int,
) -> list[dict[str, Any]]:
    """Every statistic, smoothed, every ``STACK_STEP`` episodes from the baseline to ``AFTER``."""
    start, _ = window_bounds(onset)
    lo = max(0, start - BASELINE)
    rows: list[dict[str, Any]] = []
    for k, v in series.items():
        smooth = smoothed(v)
        rows.extend(
            {
                "side": side,
                "seed": seed,
                "statistic": k,
                "episode": t + 1,
                "relative_to_onset": t + 1 - onset,
                "value": float(smooth[t]),
            }
            for t in range(lo, min(len(v), onset - 1 + AFTER), STACK_STEP)
            if not np.isnan(smooth[t])
        )
    return rows


def spike_events(lines: list[dict[str, Any]], onset: int) -> int:
    """Count gradient-norm spikes followed within ``SPIKE_FOLLOW`` updates by a KL spike.

    Read per update, unsmoothed, inside the window; a spike is above the run's own baseline
    median by ``SPIKE_MADS`` scaled MADs. The transient form of stale second moments, which the
    smoothed departure rule cannot see; reported beside the causes, never as one.
    """
    start, end = window_bounds(onset)
    base = [x for x in lines if max(0, start - BASELINE) < x["episode"] <= max(0, start)]
    window = [x for x in lines if start < x["episode"] <= end]
    if len(base) < 2 or not window:
        return 0

    def threshold(key: str) -> float:
        values = np.array([float(x[key]) for x in base])
        median = float(np.median(values))
        return median + SPIKE_MADS * 1.4826 * float(np.median(np.abs(values - median)))

    grad_bar, kl_bar = threshold("grad_norm"), threshold("approx_kl")
    events = 0
    for i, line in enumerate(window):
        if float(line["grad_norm"]) > grad_bar and any(
            float(later["approx_kl"]) > kl_bar for later in window[i + 1 : i + 1 + SPIKE_FOLLOW]
        ):
            events += 1
    return events


# ── Reading the re-runs ──────────────────────────────────────────────────────────────────────
def _telemetry(log: Path, experiments: Path = EXPERIMENTS) -> list[dict[str, Any]]:
    """Return a run's telemetry lines, found through its tracked-experiment record."""
    from l4_panel import _experiment_json  # pyright: ignore[reportMissingImports]

    experiment = _experiment_json(log.read_text(), experiments)
    exports = experiment.get("exports_path") if experiment else None
    path = REPO / exports / "telemetry" / "updates.jsonl" if exports else None
    if path is None or not path.is_file():
        msg = f"{log.name}: no telemetry found through its experiment record"
        raise DiagnosisError(msg)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def diagnose(log_dirs: list[Path]) -> dict[str, Any]:
    """Find each collapsed re-run's onset, build both sides' series, and read the causes.

    A pair leaves the stacks, named, when its collapsed run does not collapse under the registered
    definition, when its intact twin collapses under it, or when either run's telemetry stops
    before the window ends.
    """
    onsets: dict[int, int] = {}
    collapsed: dict[int, dict[str, np.ndarray]] = {}
    intact: dict[int, dict[str, np.ndarray]] = {}
    excluded: dict[int, str] = {}
    spikes: dict[str, dict[int, int]] = {"collapsed": {}, "intact": {}}
    for pair in PAIRS:
        logs = {stem: _find(log_dirs, stem, pair.seed) for stem in (pair.collapsed, pair.intact)}
        missing = [stem for stem, log in logs.items() if log is None]
        if missing:
            msg = f"seed {pair.seed}: no re-run for {missing}"
            raise DiagnosisError(msg)
        collapsed_log, intact_log = (
            cast("Path", logs[pair.collapsed]),
            cast("Path", logs[pair.intact]),
        )
        reading = cl.detect(cl.success_series(collapsed_log))
        if reading.onset is None:
            excluded[pair.seed] = "the collapsed run does not collapse under the definition"
            continue
        if cl.detect(cl.success_series(intact_log)).onset is not None:
            excluded[pair.seed] = "the intact twin collapses under the definition"
            continue
        lines = {"collapsed": _telemetry(collapsed_log), "intact": _telemetry(intact_log)}
        _, end = window_bounds(reading.onset)
        short = [side for side, x in lines.items() if not x or max(y["episode"] for y in x) < end]
        if short:
            excluded[pair.seed] = f"telemetry stops before the window ends: {short}"
            continue
        onsets[pair.seed] = reading.onset
        collapsed[pair.seed] = episode_series(lines["collapsed"])
        intact[pair.seed] = episode_series(lines["intact"])
        for side, x in lines.items():
            spikes[side][pair.seed] = spike_events(x, reading.onset)
    out = diagnose_series(collapsed, intact, onsets)
    out["excluded"] = excluded
    out["transient_stale_moments"] = spikes
    return out


def write_stacks(result: dict[str, Any], path: Path) -> Path:
    """Write the aligned, smoothed stacks wide: one row per run and sampled episode.

    A column per statistic, values to six significant figures, so the file stays small enough to
    commit beside the analysis.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[tuple[str, int, int, int], dict[str, str]] = {}
    statistics: list[str] = []
    for r in result["stacks"]:
        key = (r["side"], int(r["seed"]), int(r["episode"]), int(r["relative_to_onset"]))
        rows.setdefault(key, {})[r["statistic"]] = f"{r['value']:.6g}"
        if r["statistic"] not in statistics:
            statistics.append(r["statistic"])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["side", "seed", "episode", "relative_to_onset", *statistics])
        for key in sorted(rows):
            writer.writerow([*key, *(rows[key].get(k, "") for k in statistics)])
    return path


def mlp(log_dirs: list[Path]) -> dict[str, Any]:
    """Report the MLP control's collapse incidence at 6,000 episodes, beside the connectome's."""
    series = {}
    for seed in MLP_SEEDS:
        log = _find(log_dirs, MLP, seed)
        if log is None:
            msg = f"MLP seed {seed} has no re-run"
            raise DiagnosisError(msg)
        series[f"{MLP}-seed{seed}"] = cl.success_series(log)
    connectome = {
        f"{stem}-seed{seed}": cl.success_series(log)
        for stem, seed in RUNS
        if stem != MLP and (log := _find(log_dirs, stem, seed)) is not None
    }
    committed = json.loads((SUPPORTING / "sensitivity.json").read_text())["collapse"]
    return {
        "mlp": cl.panel(series),
        "connectome_panel_092": {
            arm: {
                "collapsed": len(committed[key]["collapsed_seeds"]),
                "n": 32,
                "rule": "092 blocks",
            }
            for arm, key in (("wild_type", "wt_ppo"), ("gap_only_null", "gaponly_ppo"))
        },
        "connectome_reruns_selected": {
            "note": (
                "the 16 re-runs were chosen half on collapse, so this is a check on the "
                "definition against 092's rule, not an incidence"
            ),
            **cl.incidence([cl.detect(s) for s in connectome.values()]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    parsers = {
        name: sub.add_parser(name, help=text)
        for name, text in (
            ("identity", "compare each re-run's plateau with the committed one"),
            ("diagnose", "the event-aligned stacks and the cause readout"),
            ("mlp", "the MLP control's collapse incidence"),
        )
    }
    for parser in parsers.values():
        parser.add_argument("--logs", type=Path, action="append", required=True)
        parser.add_argument("--out", type=Path)
    parsers["diagnose"].add_argument("--csv", type=Path, help="write the aligned stacks here")
    args = ap.parse_args(argv)
    ok = True
    if args.command == "identity":
        result = identity(args.logs)
        ok = result["all_identical"]
    elif args.command == "diagnose":
        result = diagnose(args.logs)
        if args.csv:
            write_stacks(result, args.csv)
        result = {k: v for k, v in result.items() if k != "stacks"}
    else:
        result = mlp(args.logs)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
