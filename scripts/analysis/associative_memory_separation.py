"""Associative-memory separation analysis: does the comparison separate working-memory *update*.

Reads each run's per-episode response accuracy from the ``run_simulation`` ``.out`` and reports,
per arm, the plateau-tail (final-quarter) mean accuracy across paired seeds, plus the pairwise
deltas (reusing the committed paired-seed Wilcoxon + 80% bootstrap CI + BH-FDR layer). The verdict
is **separation** when the recurrent/attention arms clear both chance and the memoryless arm, and
**null** otherwise.

Because a reversal fires on ~half the trials and flips the rewarded cue, at ``reversal_prob = 0.5``
a *hold-only* policy (retains the initial association, never updates) scores at chance — like the
memoryless MLP. So overall accuracy above chance requires genuine working-memory **update**. The
per-arm **reversal / non-reversal split** (parsed from the printed ``AssocMemory:`` line) makes the
hold-vs-update breakdown directly visible: a hold-only arm reads ~1.0 non-reversal but ~chance on
reversal trials.

Overall-accuracy source: with the canonical reward (``reward_correct=1`` / ``penalty_wrong=0``) the
per-episode reward equals the correct-response count, so accuracy = ``reward / num_responses``
(``num_responses = trials_per_episode * response_steps``, default 20). Pass ``--num-responses`` if
the configs differ.

Usage::

    uv run python scripts/analysis/associative_memory_separation.py \
        --manifest <run-dir>/_manifest.txt --num-responses 20 --out <run-dir>/separation.json
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

# Reuse the committed statistics layer verbatim (identical methodology to the arch ranking).
from weight_search_architecture_ranking import bh_fdr, paired_seed_wilcoxon_bootstrap

REPO = Path(__file__).resolve().parents[2]

_FINAL_WINDOW_FRAC = 0.25  # plateau-tail window: the final quarter of the training run
_CHANCE = 0.5  # binary readout -> chance accuracy
_CHANCE_BAND = 0.15  # a memoryless arm reads "at chance" within this of _CHANCE
_SEPARATION_THRESHOLD = 0.80  # an arm must clear this (genuine update) to count as "solving"
_MIN_PAIRED_SEEDS = 2  # a pairwise Wilcoxon needs >= 2 common seeds

MEMORY_ARMS = ("lstmppo", "cfcppo", "transformerppo", "mingruppo", "minlstmppo")
MEMORYLESS_ARM = "mlpppo"  # the designated memoryless baseline the memory arms must beat
MEMORYLESS_ARMS = ("mlpppo",)  # expected at chance (the at-chance anchor)
_RUN_LINE = re.compile(r"Run:\s+\d+.*?Reward:\s+([-0-9.]+)")
_SPLIT_LINE = re.compile(r"AssocMemory:.*?reversal=([-0-9.]+)\s+non_reversal=([-0-9.]+)")


def _plateau_tail(values: list[float]) -> float | None:
    if not values:
        return None
    tail = max(1, int(len(values) * _FINAL_WINDOW_FRAC))
    return float(np.mean(values[-tail:]))


def _overall_accuracy(out_path: Path, num_responses: int) -> float | None:
    """Final-quarter mean response accuracy from a run's per-episode rewards, or None."""
    if not out_path.exists():
        return None
    rates = [
        float(m.group(1)) / num_responses
        for line in out_path.read_text().splitlines()
        if (m := _RUN_LINE.search(line))
    ]
    return _plateau_tail(rates)


def _split_accuracy(out_path: Path) -> tuple[float, float] | None:
    """Final-quarter mean (reversal, non_reversal) accuracy from the printed AssocMemory lines."""
    if not out_path.exists():
        return None
    rev, non = [], []
    for line in out_path.read_text().splitlines():
        m = _SPLIT_LINE.search(line)
        if m:
            rev.append(float(m.group(1)))
            non.append(float(m.group(2)))
    r, n = _plateau_tail(rev), _plateau_tail(non)
    return None if r is None or n is None else (r, n)


def load(manifest: Path, num_responses: int) -> tuple[dict, dict]:
    """Return ({arch:{seed:accuracy}}, {arch:{seed:(rev,non_rev)}}) from an <arch> <seed> <out> manifest."""
    overall: dict[str, dict[int, float]] = {}
    split: dict[str, dict[int, tuple[float, float]]] = {}
    for line in manifest.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        arch, seed, out_path = parts[0], int(parts[1]), Path(parts[2])
        acc = _overall_accuracy(REPO / out_path, num_responses)
        if acc is None:
            print(f"  WARN {arch} seed {seed}: no parseable episodes in {out_path} — dropped")
            continue
        overall.setdefault(arch, {})[seed] = acc
        sp = _split_accuracy(REPO / out_path)
        if sp is not None:
            split.setdefault(arch, {})[seed] = sp
    return overall, split


def analyse(overall: dict, split: dict, out: dict) -> None:
    """Print + record the per-arm accuracy, the reversal split, the pairwise deltas, and the verdict."""
    archs = sorted(overall, key=lambda a: -float(np.mean(list(overall[a].values()))))
    print("\n" + "=" * 72)
    print("ASSOCIATIVE-MEMORY SEPARATION — plateau-tail response accuracy (chance = 0.50)")
    print("  (at reversal_prob 0.5 a hold-only policy is at chance; above-chance needs *update*)")
    print("=" * 72)
    means: dict[str, float] = {}
    out["per_arm"] = {}
    for a in archs:
        vals = list(overall[a].values())
        means[a] = float(np.mean(vals))
        sp = split.get(a, {})
        rev = float(np.mean([v[0] for v in sp.values()])) if sp else float("nan")
        non = float(np.mean([v[1] for v in sp.values()])) if sp else float("nan")
        print(
            f"  {a:16} {means[a]:.3f}   n={len(vals)}   "
            f"reversal={rev:.3f} non_reversal={non:.3f}   per-seed={[round(v, 3) for v in vals]}",
        )
        out["per_arm"][a] = {
            "mean": means[a],
            "n": len(vals),
            "per_seed": vals,
            "reversal_mean": rev,
            "non_reversal_mean": non,
        }

    print("\n  Pairwise paired-seed deltas (one-sided Wilcoxon a>b, 80% bootstrap CI, BH-FDR):")
    pairs, stats = [], []
    for a, b in combinations(archs, 2):
        common = sorted(set(overall[a]) & set(overall[b]))
        if len(common) < _MIN_PAIRED_SEEDS:
            print(f"    {a:14} vs {b:14}  SKIPPED (only {len(common)} common seed(s))")
            continue
        deltas = [overall[a][s] - overall[b][s] for s in common]
        pairs.append((a, b))
        stats.append(paired_seed_wilcoxon_bootstrap(deltas))
    qs = bh_fdr([s["wilcoxon_p"] for s in stats])
    out["pairwise"] = []
    for (a, b), st, q in zip(pairs, stats, qs, strict=True):
        sig = "***" if q < 0.05 else ("." if q < 0.10 else "ns")
        print(
            f"    {a:14} vs {b:14}  d={st['mean_delta']:+.3f}  "
            f"CI[{st['ci_lo']:+.3f},{st['ci_hi']:+.3f}]  q={q:.3f}  {sig}",
        )
        out["pairwise"].append({"arm_a": a, "arm_b": b, **st, "bh_q": q})

    q_by_pair: dict[tuple[str, str], float] = {}
    for (a, b), q in zip(pairs, qs, strict=True):
        q_by_pair[(a, b)] = q
        q_by_pair[(b, a)] = q
    mlp_mean = means.get(MEMORYLESS_ARM)
    separators = []
    for a in MEMORY_ARMS:
        if a not in means or mlp_mean is None:
            continue
        q = q_by_pair.get((a, MEMORYLESS_ARM))
        beats_mlp = means[a] > mlp_mean and q is not None and q < 0.05
        if means[a] >= _SEPARATION_THRESHOLD and beats_mlp:
            separators.append(a)
    memoryless_means = {a: means[a] for a in MEMORYLESS_ARMS if a in means}
    memoryless_at_chance = bool(memoryless_means) and all(
        abs(m - _CHANCE) < _CHANCE_BAND for m in memoryless_means.values()
    )
    separated = bool(separators) and memoryless_at_chance
    out["verdict"] = {
        "separated": separated,
        "separating_arms": separators,
        "memoryless_means": memoryless_means,
        "memoryless_at_chance": memoryless_at_chance,
    }
    print("\n" + "-" * 72)
    if separated:
        print(
            f"  VERDICT: SEPARATION — {', '.join(separators)} clear the update threshold + beat the "
            f"memoryless baseline ({memoryless_means}); the reversal split shows which arms update.",
        )
    else:
        print(
            "  VERDICT: NULL — no arm cleared working-memory update above the memoryless baseline "
            f"({memoryless_means}, separating arms {separators}); the null is the finding.",
        )


def main() -> None:
    """Load the manifest, compute + print + write the separation analysis."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="<arch> <seed> <out_path> per line",
    )
    ap.add_argument(
        "--num-responses",
        type=int,
        default=20,
        help="responses per episode = trials_per_episode * response_steps (default 20)",
    )
    ap.add_argument("--out", type=Path, default=None, help="write the separation summary JSON here")
    args = ap.parse_args()
    if args.num_responses <= 0:
        ap.error("--num-responses must be a positive integer")

    overall, split = load(args.manifest, args.num_responses)
    out: dict = {}
    analyse(overall, split, out)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
