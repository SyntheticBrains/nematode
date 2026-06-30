"""Reactive-cell stability A/B: minGRU / minLSTM vs the plain LSTM on the 029 C3 cell.

The memory-independent prong of the minimal-RNN bring-up. Reads each run's plateau-tail
full-clear success (the 029 primary metric: final-quarter fraction of episodes that collect all
target foods) from the per-run ``.out`` and reports, per arm, the plateau level + per-seed spread
(the stability signal), then the paired-seed delta of each minimal arm vs ``lstmppo`` (reusing the
committed paired-seed Wilcoxon + 80% bootstrap CI + BH-FDR layer).

Hypothesis: the minimal RNNs (no saturating hidden-to-hidden matrix) train at least as stably as
the 029 LSTM laggard. NB: the minimal arms carry the memory-friendly hold-bias init, so a delta
here conflates the minimal architecture with that init (see logbook 031 Analysis).

Usage::

    uv run python scripts/analysis/minimal_rnn_reactive_ab.py --dir <run-dir> --out <run-dir>/reactive_ab.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Reuse the committed statistics layer verbatim (identical methodology to the arch ranking).
from weight_search_architecture_ranking import bh_fdr, paired_seed_wilcoxon_bootstrap

REPO = Path(__file__).resolve().parents[2]

_FINAL_WINDOW_FRAC = 0.25  # plateau-tail window: the final quarter of the run
_CONVERGED_BAR = 30.0  # a seed "converged" if its plateau-tail full-clear success clears this %
BASELINE = "lstmppo"
NEW_ARMS = ("mingruppo", "minlstmppo")
_SEEDS = range(1, 9)
# 029 C3 metric: Status: SUCCESS == full clear; Eaten count is the foraging sub-metric.
_RUN_LINE = re.compile(r"Run:\s+\d+\s+Status:\s+(\S+).*?Eaten:\s+(\d+)/")


def _plateau_tail_success(out_path: Path) -> float | None:
    """Final-quarter full-clear success % from a run's ``.out`` (the 029 primary), or None."""
    if not out_path.exists():
        return None
    succ = [
        1.0 if (m := _RUN_LINE.match(ln)) and m.group(1) == "SUCCESS" else 0.0
        for ln in out_path.read_text().splitlines()
        if _RUN_LINE.match(ln)
    ]
    if not succ:
        return None
    tail = max(1, int(len(succ) * _FINAL_WINDOW_FRAC))
    return 100.0 * float(np.mean(succ[-tail:]))


def load(run_dir: Path) -> dict[str, dict[int, float]]:
    """Return {arm: {seed: plateau-tail success %}} from ``c3_<arm>_s<seed>.out`` files."""
    table: dict[str, dict[int, float]] = {}
    for arm in (BASELINE, *NEW_ARMS):
        for seed in _SEEDS:
            val = _plateau_tail_success(run_dir / f"c3_{arm}_s{seed}.out")
            if val is not None:
                table.setdefault(arm, {})[seed] = val
            else:
                print(f"  WARN {arm} seed {seed}: no parseable episodes — dropped")
    return table


def analyse(table: dict[str, dict[int, float]], out: dict) -> None:
    """Print + record per-arm plateau success, the stability spread, and the vs-lstmppo deltas."""
    print("\n" + "=" * 72)
    print("MINIMAL-RNN REACTIVE A/B — plateau-tail full-clear success % (029 C3 cell)")
    print("=" * 72)
    out["per_arm"] = {}
    for arm in (BASELINE, *NEW_ARMS):
        vals = table.get(arm, {})
        if not vals:
            print(f"  {arm:12} (no data)")
            continue
        v = list(vals.values())
        converged = sum(x >= _CONVERGED_BAR for x in v)
        print(
            f"  {arm:12} mean={np.mean(v):5.1f}  min={min(v):5.1f}  std={np.std(v):4.1f}  "
            f"converged({_CONVERGED_BAR:.0f}%+)={converged}/{len(v)}  "
            f"per-seed={[round(x, 1) for x in v]}",
        )
        out["per_arm"][arm] = {
            "mean": float(np.mean(v)),
            "min": float(min(v)),
            "std": float(np.std(v)),
            "converged": converged,
            "n": len(v),
            "per_seed": {s: vals[s] for s in sorted(vals)},
        }

    print("\n  Paired-seed delta vs lstmppo (one-sided Wilcoxon a>b, 80% bootstrap CI, BH-FDR):")
    pairs, stats = [], []
    base = table.get(BASELINE, {})
    for arm in NEW_ARMS:
        common = sorted(set(table.get(arm, {})) & set(base))
        if len(common) < 2:
            print(f"    {arm:12} vs lstmppo  SKIPPED (only {len(common)} common seed(s))")
            continue
        deltas = [table[arm][s] - base[s] for s in common]
        pairs.append(arm)
        stats.append(paired_seed_wilcoxon_bootstrap(deltas))
    qs = bh_fdr([s["wilcoxon_p"] for s in stats]) if stats else []
    out["vs_lstmppo"] = []
    for arm, st, q in zip(pairs, stats, qs, strict=True):
        sig = "***" if q < 0.05 else ("." if q < 0.10 else "ns")
        print(
            f"    {arm:12} vs lstmppo  d={st['mean_delta']:+6.1f}  "
            f"CI[{st['ci_lo']:+.1f},{st['ci_hi']:+.1f}]  q={q:.3f}  {sig}",
        )
        out["vs_lstmppo"].append({"arm": arm, **st, "bh_q": q})


def main() -> None:
    """Load the run dir, compute + print + write the reactive-A/B summary."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True, help="dir with c3_<arm>_s<seed>.out files")
    ap.add_argument("--out", type=Path, default=None, help="write the A/B summary JSON here")
    args = ap.parse_args()

    table = load(args.dir)
    out: dict = {}
    analyse(table, out)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
