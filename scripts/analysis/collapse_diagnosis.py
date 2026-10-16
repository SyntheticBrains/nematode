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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import collapse as cl  # noqa: E402  # pyright: ignore[reportMissingImports]
import t7_continuous_ranking as t7  # noqa: E402  # pyright: ignore[reportMissingImports]

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
SMOOTH = 100
MIN_RUN = 200
SDS = 2.0
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
    "srank": lambda line: line["srank"],
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


def aligned(series: np.ndarray, onset: int, lead: int = LEAD) -> np.ndarray:
    """Return the ``lead`` episodes before ``onset`` (1-indexed), NaN-padded before the start."""
    end = onset - 1
    start = end - lead
    out = np.full(lead, np.nan)
    lo = max(0, start)
    out[lo - start :] = series[lo:end]
    return out


# ── The envelope and the departure rule ──────────────────────────────────────────────────────
def envelope(stack: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and standard deviation at each aligned episode across the stacked runs."""
    array = np.vstack(stack)
    with warnings.catch_warnings():
        # An aligned episode no run reached is NaN in every row, and stays NaN.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(array, axis=0), np.nanstd(array, axis=0)


def departure(
    series: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    direction: int,
    min_run: int = MIN_RUN,
) -> int | None:
    """Return how far before onset a ``min_run``-long stretch outside the envelope begins.

    ``direction`` is +1 for above the envelope, -1 for below. Returns the stretch's lead before
    onset, in episodes (``LEAD`` is the earliest), or ``None`` when there is no such stretch.
    """
    with np.errstate(invalid="ignore"):
        outside = (
            series > mean + SDS * sd if direction > 0 else series < mean - SDS * sd
        ) & ~np.isnan(series)
    run = 0
    for t, flag in enumerate(outside):
        run = run + 1 if flag else 0
        if run == min_run:
            return len(series) - (t - min_run + 1)
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


def diagnose_series(
    collapsed: dict[int, dict[str, np.ndarray]],
    intact: dict[int, dict[str, np.ndarray]],
    onsets: dict[int, int],
) -> dict[str, Any]:
    """Read every cause on aligned, smoothed series; keys are seeds, one pair per seed."""
    seeds = sorted(onsets)
    view = {
        side: {s: {k: aligned(smoothed(v), onsets[s]) for k, v in runs[s].items()} for s in seeds}
        for side, runs in (("collapsed", collapsed), ("intact", intact))
    }
    statistics = sorted({c.statistic for comps in CAUSES.values() for c in comps})
    full = {k: envelope([view["intact"][s][k] for s in seeds]) for k in statistics}
    out: dict[str, Any] = {"seeds": seeds, "onsets": onsets, "causes": {}}
    for name, components in CAUSES.items():
        per_run: dict[str, Any] = {"collapsed": {}, "intact": {}}
        for s in seeds:
            hit, leads = cause_departs(view["collapsed"][s], full, components)
            per_run["collapsed"][s] = {"departs": hit and _ordered(name, leads), "leads": leads}
            others = [t for t in seeds if t != s]
            loo = {k: envelope([view["intact"][t][k] for t in others]) for k in statistics}
            hit, leads = cause_departs(view["intact"][s], loo, components)
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
    return out


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
    """Find each collapsed re-run's onset, build both sides' series, and read the causes."""
    onsets: dict[int, int] = {}
    collapsed: dict[int, dict[str, np.ndarray]] = {}
    intact: dict[int, dict[str, np.ndarray]] = {}
    not_collapsed: list[int] = []
    for pair in PAIRS:
        logs = {stem: _find(log_dirs, stem, pair.seed) for stem in (pair.collapsed, pair.intact)}
        missing = [stem for stem, log in logs.items() if log is None]
        if missing:
            msg = f"seed {pair.seed}: no re-run for {missing}"
            raise DiagnosisError(msg)
        reading = cl.detect(cl.success_series(logs[pair.collapsed]))  # type: ignore[arg-type]
        if reading.onset is None:
            not_collapsed.append(pair.seed)
            continue
        onsets[pair.seed] = reading.onset
        collapsed[pair.seed] = episode_series(_telemetry(logs[pair.collapsed]))  # type: ignore[arg-type]
        intact[pair.seed] = episode_series(_telemetry(logs[pair.intact]))  # type: ignore[arg-type]
    out = diagnose_series(collapsed, intact, onsets)
    out["not_collapsed_under_the_definition"] = not_collapsed
    return out


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
    return {
        "mlp": cl.panel(series),
        "connectome_reruns": cl.incidence(
            [cl.detect(s) for s in connectome.values()],
        ),
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
    args = ap.parse_args(argv)
    ok = True
    if args.command == "identity":
        result = identity(args.logs)
        ok = result["all_identical"]
    elif args.command == "diagnose":
        result = diagnose(args.logs)
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
