"""Connectome-structure control, efficiency axis - does the wiring help it *learn faster*.

Companion to ``connectome_structure_controls.py``. That harness answered the **peak** question
(final-quarter plateau success) and found DEGREE-STATISTICS: the wild-type *C. elegans* connectome is
indistinguishable from its degree-preserving rewired-null. This harness asks the natural follow-up -
does the specific wiring confer a **learning-efficiency / inductive-bias** advantage even when the
peak is the same? - by reading the *whole* per-episode series (not just the plateau tail) from the
same paired ``.out`` panel and comparing wild-type vs rewired-null on:

- ``auc_success`` - mean full-clear success over the common horizon (the learning integral).
- ``auc_foods`` - mean foods/episode over the horizon (a finer, less-binary learning signal).
- ``episodes_to_30pct_success`` - episodes to first cross a 30% rolling full-clear rate.
- ``episodes_to_90pct_foods_plateau`` - episodes to reach 90% of the run's own foods plateau.

Deltas are oriented so **positive = wild-type better/faster**, then run through the *same* committed
paired-seed Wilcoxon + 80% bootstrap CI layer as the peak control, with BH-FDR across the four
metrics. Verdict mirrors the peak control: **SPECIFIC-WIRING (efficiency)** if any metric shows the
wild-type significantly faster/higher (BH-FDR ``q < 0.05``, positive delta); **DEGREE-STATISTICS**
if none does (the efficiency axis agrees with the peak axis).

Usage::

    uv run python scripts/analysis/connectome_structure_efficiency.py \
        --manifest <run-dir>/_manifest.txt --out <run-dir>/efficiency.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path

import numpy as np

# Reuse the committed statistics layer verbatim (identical methodology to the peak control + 029).
from weight_search_architecture_ranking import bh_fdr, paired_seed_wilcoxon_bootstrap

REPO = Path(__file__).resolve().parents[2]

_WILD = "wild_type"
_REWIRED = "rewired_null"
_ROLLING_WINDOW = 200  # episodes; smoothing for the threshold-crossing "learning speed" metrics
_SUCCESS_THRESHOLD = 0.30  # rolling full-clear rate for episodes_to_30pct_success
_FOODS_PLATEAU_FRAC = 0.90  # fraction of own foods plateau for episodes_to_90pct_foods_plateau
_PLATEAU_TAIL_FRAC = 0.25  # final-quarter window defining a run's own plateau (matches the ranking)
_SIG_Q = 0.05

# Same per-episode line shape the 029 ranking parses: "Run: N Status: S ... Eaten: X/10".
_RUN_RE = re.compile(r"Run:\s+\d+\s+Status:\s+(\S+).*?Eaten:\s+(\d+)/")

# Per metric: (higher-is-better?) - orients the paired delta so positive = wild-type better/faster.
_METRICS = {
    "auc_success": True,
    "auc_foods": True,
    "episodes_to_30pct_success": False,
    "episodes_to_90pct_foods_plateau": False,
}


def _parse_series(out_path: Path) -> tuple[list[float], list[float]]:
    """Full per-episode (success 0/1, foods 0-10) series from one run's ``.out``."""
    succ: list[float] = []
    foods: list[float] = []
    for line in out_path.read_text(errors="ignore").splitlines():
        m = _RUN_RE.match(line)
        if m:
            succ.append(1.0 if m.group(1) == "SUCCESS" else 0.0)
            foods.append(float(m.group(2)))
    return succ, foods


def _rolling(xs: list[float], window: int) -> list[float]:
    out: list[float] = []
    total = 0.0
    dq: deque[float] = deque()
    for x in xs:
        dq.append(x)
        total += x
        if len(dq) > window:
            total -= dq.popleft()
        out.append(total / len(dq))
    return out


def _eps_to_threshold(series: list[float], tau: float, horizon: int) -> float:
    """Episodes to first reach ``tau`` on the rolling window; right-censored at ``horizon``."""
    for t, v in enumerate(_rolling(series, _ROLLING_WINDOW)):
        if v >= tau:
            return float(t + 1)
    return float(horizon)


def _plateau(series: list[float]) -> float:
    tail = max(1, int(len(series) * _PLATEAU_TAIL_FRAC))
    return float(np.mean(series[-tail:]))


def _run_metrics(out_path: Path, horizon: int) -> dict[str, float]:
    succ, foods = _parse_series(out_path)
    succ_h, foods_h = succ[:horizon], foods[:horizon]
    foods_plateau = _plateau(foods)
    return {
        "auc_success": float(np.mean(succ_h)),
        "auc_foods": float(np.mean(foods_h)),
        "episodes_to_30pct_success": _eps_to_threshold(succ, _SUCCESS_THRESHOLD, horizon),
        "episodes_to_90pct_foods_plateau": (
            _eps_to_threshold(foods, _FOODS_PLATEAU_FRAC * foods_plateau, horizon)
            if foods_plateau > 0
            else float(horizon)
        ),
    }


def _load_runs(manifest: Path) -> dict[str, dict[int, Path]]:
    arms: dict[str, dict[int, Path]] = {}
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        arm, seed, out = line.split()
        arms.setdefault(arm, {})[int(seed)] = REPO / out
    return arms


def analyse(manifest: Path) -> dict:
    """Wild-type vs rewired-null efficiency comparison from a paired ``.out`` manifest."""
    runs = _load_runs(manifest)
    if _WILD not in runs or _REWIRED not in runs:
        msg = f"manifest must contain both {_WILD!r} and {_REWIRED!r} arms"
        raise ValueError(msg)

    horizon = min(len(_parse_series(p)[0]) for arm in (_WILD, _REWIRED) for p in runs[arm].values())
    per_seed = {
        arm: {seed: _run_metrics(path, horizon) for seed, path in sorted(runs[arm].items())}
        for arm in (_WILD, _REWIRED)
    }
    seeds = sorted(set(per_seed[_WILD]) & set(per_seed[_REWIRED]))

    metrics_out: dict[str, dict] = {}
    pvals: list[float] = []
    for name, higher_better in _METRICS.items():
        deltas = []
        for s in seeds:
            w = per_seed[_WILD][s][name]
            r = per_seed[_REWIRED][s][name]
            deltas.append((w - r) if higher_better else (r - w))  # positive = wild better/faster
        stats = paired_seed_wilcoxon_bootstrap(deltas)
        metrics_out[name] = {
            "higher_is_better": higher_better,
            "wild_mean": float(np.mean([per_seed[_WILD][s][name] for s in seeds])),
            "rewired_mean": float(np.mean([per_seed[_REWIRED][s][name] for s in seeds])),
            "wild_minus_rewired_oriented": stats["mean_delta"],
            "wild_better_seeds": sum(1 for d in stats["per_seed_deltas"] if d > 0),
            **stats,
        }
        pvals.append(stats["wilcoxon_p"])

    for (_name, entry), q in zip(metrics_out.items(), bh_fdr(pvals), strict=True):
        entry["bh_fdr_q"] = q

    specific = any(e["bh_fdr_q"] < _SIG_Q and e["mean_delta"] > 0 for e in metrics_out.values())
    return {
        "verdict": "specific_wiring_efficiency" if specific else "degree_statistics",
        "horizon_episodes": horizon,
        "rolling_window": _ROLLING_WINDOW,
        "n_paired_seeds": len(seeds),
        "metrics": metrics_out,
        "per_seed": {arm: {str(s): m for s, m in d.items()} for arm, d in per_seed.items()},
    }


def _print_summary(report: dict) -> None:
    print(
        f"VERDICT: {report['verdict'].upper()}  "
        f"(n={report['n_paired_seeds']}, horizon={report['horizon_episodes']} episodes)"
    )
    print(f"{'metric':32} {'wild':>9} {'rewired':>9} {'delta(+=wild)':>13} {'q':>7}")
    for name, e in report["metrics"].items():
        print(
            f"{name:32} {e['wild_mean']:>9.2f} {e['rewired_mean']:>9.2f} "
            f"{e['mean_delta']:>+13.3f} {e['bh_fdr_q']:>7.3f}  "
            f"wild-better {e['wild_better_seeds']}/{report['n_paired_seeds']}"
        )


def main() -> None:
    """CLI entry point: analyse the paired panel and optionally write the JSON report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "tmp/evaluations/t7-connectome-controls/panel_n8/_manifest.txt",
    )
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report to this path")
    args = ap.parse_args()

    report = analyse(args.manifest)
    _print_summary(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
