#!/usr/bin/env python
"""M.1(b): read a collapse remedy against a placebo re-roll of the same seeds, paired by seed.

M.1(a) found late collapse to be a sudden event (Logbook 096): any setting that changes a run's
trajectory re-rolls its collapse, so a remedy cannot be read against the runs it was chosen to fix. Each
remedy round re-runs M.9's 16 seeds (the eight collapsers and their eight intact twins) under the remedy,
beside a placebo round that changes only the minibatch shuffle stream, and the two are compared run for
run.

**The statistic**: a run's drawdown, its highest trailing-200-episode success rate minus its mean success
over its last 500 episodes.

**A remedy passes** when, over the 16 runs paired with the placebo by the instrument's paired test (80%
interval): ``placebo - remedy`` drawdown has a mean of at least 0.05 and an interval above zero; and
``remedy - placebo`` success to episode 3,000 has an interval whose lower end is above -0.05.

Beside it, never deciding: collapse counts under D25, rescues and harms against M.1(a)'s original
re-runs, the KL stop's firing rate, and the MLP preflight (its plateau and collapses under the remedy
against M.1(a)'s MLP re-runs). A run that exits non-zero after one re-run counts against the remedy: its
drawdown is its peak and its early success zero.

Usage::

    uv run python scripts/analysis/collapse_remedies.py read --round klstop \
        --logs campaigns/m1b-placebo/logs --logs campaigns/m1b-klstop/logs \
        --original campaigns/m1-diagnosis/logs --original campaigns/m1-mlp/logs --out klstop.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import collapse as cl  # noqa: E402  # pyright: ignore[reportMissingImports]
import collapse_diagnosis as cd  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_collapse_remedy_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import t7_continuous_ranking as t7  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

REMEDIES: tuple[str, ...] = ("klstop", "wclip", "adam")
PEAK_WINDOW = 200
TAIL = 500
EARLY = 3000
MINIMUM_REDUCTION = 0.05
GUARD = -0.05
# The 16 connectome runs: each of M.1(a)'s eight seeds on both wirings.
RUNS: tuple[tuple[str, int], ...] = tuple(
    (parent, pair.seed) for pair in cd.PAIRS for parent in (gen.WILD_TYPE, gen.GAP_ONLY)
)


class RemedyError(ValueError):
    """The runs on disk are not the rounds this module reads."""


def drawdown(series: np.ndarray) -> float:
    """Return the highest trailing-200 success rate minus the mean over the last 500 episodes."""
    peak = cl.trailing_mean(series, PEAK_WINDOW)
    top = float(np.nanmax(peak)) if np.any(~np.isnan(peak)) else float(series.mean())
    return top - float(series[-TAIL:].mean())


def _find(log_dirs: list[Path], stem: str, seed: int) -> Path | None:
    name = f"{stem}-seed{seed}.log"
    return next((d / name for d in log_dirs if (d / name).is_file()), None)


def _require(log_dirs: list[Path], stem: str, seed: int) -> Path:
    """Return a run's log, or refuse: a missing log is a wrong path, never a failed run."""
    log = _find(log_dirs, stem, seed)
    if log is None:
        msg = f"no log for {stem} seed {seed} in {[str(d) for d in log_dirs]}"
        raise RemedyError(msg)
    return log


def _exited_cleanly(log: Path) -> bool:
    status = log.with_suffix(".exit")
    return status.is_file() and status.read_text().strip() == "0"


def run_reading(log: Path) -> dict[str, Any]:
    """One run's drawdown, early success and collapse; a failed run counts against the remedy."""
    series = cl.success_series(log)
    if not _exited_cleanly(log) or len(series) == 0:
        peak = cl.trailing_mean(series, PEAK_WINDOW) if len(series) else np.zeros(0)
        top = float(np.nanmax(peak)) if len(peak) and np.any(~np.isnan(peak)) else 1.0
        return {"failed": True, "drawdown": top, "early": 0.0, "collapsed": None}
    reading = cl.detect(series)
    return {
        "failed": False,
        "drawdown": drawdown(series),
        "early": float(series[:EARLY].mean()),
        "collapsed": reading.collapsed,
        "onset": reading.onset,
    }


def paired(deltas: list[float]) -> dict[str, Any]:
    """Run the instrument's paired test (80% bootstrap interval) on per-run differences."""
    test = wp.paired_seed_wilcoxon_bootstrap(deltas)
    return {
        "mean": float(np.mean(deltas)),
        "ci": [test["ci_lo"], test["ci_hi"]],
        "wilcoxon_p_one_sided": test["wilcoxon_p"],
        "n": len(deltas),
    }


def passes(reduction: dict[str, Any], guard: dict[str, Any]) -> bool:
    """Apply the registered rule: drawdown down by the minimum, learning not slowed past the guard."""
    return (
        reduction["mean"] >= MINIMUM_REDUCTION
        and reduction["ci"][0] > 0.0
        and guard["ci"][0] > GUARD
    )


def _kl_stop_rate(log: Path) -> float | None:
    """Return the fraction of a run's updates a KL stop ended, after its window has filled.

    The stop cannot fire before its 100-update window fills, so those updates are left out.
    """
    try:
        lines = cd._telemetry(log)
    except cd.DiagnosisError:
        return None
    flags = [bool(x.get("kl_stopped")) for x in lines[100:] if "kl_stopped" in x]
    return float(np.mean(flags)) if flags else None


def read_round(round_name: str, log_dirs: list[Path], originals: list[Path]) -> dict[str, Any]:
    """Read one remedy round against the placebo, run for run, with the beside readings."""
    if round_name not in REMEDIES:
        msg = f"{round_name!r} is not a remedy round; the rounds are {REMEDIES}"
        raise RemedyError(msg)
    runs: dict[str, Any] = {}
    reduction, early = [], []
    for parent, seed in RUNS:
        remedy_log = _require(log_dirs, gen.stem(parent, round_name), seed)
        remedy = run_reading(remedy_log)
        placebo = run_reading(_require(log_dirs, gen.stem(parent, "placebo"), seed))
        original = run_reading(_require(originals, parent, seed))
        if placebo["failed"]:
            msg = f"the placebo run {parent} seed {seed} failed; the round cannot be read"
            raise RemedyError(msg)
        reduction.append(placebo["drawdown"] - remedy["drawdown"])
        early.append(remedy["early"] - placebo["early"])
        runs[f"{parent}-seed{seed}"] = {
            "remedy": remedy,
            "placebo": placebo,
            "original_collapsed": original["collapsed"],
            "kl_stop_rate": _kl_stop_rate(remedy_log),
        }
    reduction_test, guard_test = paired(reduction), paired(early)
    collapsed = {
        side: sum(bool(r[side]["collapsed"]) for r in runs.values())
        for side in ("remedy", "placebo")
    }
    # A failed remedy run (collapse unknown) is never a rescue.
    rescued = sum(
        1 for r in runs.values() if r["original_collapsed"] and r["remedy"]["collapsed"] is False
    )
    harmed = sum(
        1 for r in runs.values() if r["original_collapsed"] is False and r["remedy"]["collapsed"]
    )
    return {
        "round": round_name,
        "drawdown_reduction": reduction_test,
        "early_success_guard": guard_test,
        "passes": passes(reduction_test, guard_test),
        "beside": {
            "collapsed": collapsed,
            "rescued_against_original": rescued,
            "harmed_against_original": harmed,
            "kl_stop_rate": [r["kl_stop_rate"] for r in runs.values()],
        },
        "mlp_preflight": mlp_preflight(round_name, log_dirs, originals),
        "runs": runs,
    }


def mlp_preflight(round_name: str, log_dirs: list[Path], originals: list[Path]) -> dict[str, Any]:
    """Compare the MLP's plateau and collapses under the remedy with its M.1(a) re-runs.

    A preflight for M.1(c)'s MLP hold, never part of the selection: it carries no MLP placebo, so a
    difference here mixes the remedy with a re-roll.
    """
    rows = []
    for seed in cd.MLP_SEEDS:
        remedy_log = _require(log_dirs, gen.stem(gen.MLP, round_name), seed)
        original_log = _require(originals, gen.MLP, seed)
        remedy = t7.plateau_tail(remedy_log)
        original = t7.plateau_tail(original_log)
        rows.append(
            {
                "seed": seed,
                "remedy_plateau": None if remedy is None else remedy[0],
                "original_plateau": None if original is None else original[0],
                "remedy_collapsed": run_reading(remedy_log)["collapsed"],
            },
        )
    deltas = [
        r["remedy_plateau"] - r["original_plateau"]
        for r in rows
        if r["remedy_plateau"] is not None and r["original_plateau"] is not None
    ]
    return {
        "plateau_change": paired(deltas) if deltas else None,
        "collapsed": sum(bool(r["remedy_collapsed"]) for r in rows),
        "per_seed": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    rd = sub.add_parser("read", help="read one remedy round against the placebo")
    rd.add_argument("--round", required=True, choices=REMEDIES)
    rd.add_argument("--logs", type=Path, action="append", required=True)
    rd.add_argument("--original", type=Path, action="append", required=True)
    rd.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    result = read_round(args.round, args.logs, args.original)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
