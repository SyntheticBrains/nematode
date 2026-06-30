"""Bit-memory separation analysis: does the comparison separate working memory.

The decision-gate analysis for ``T7.separation.bit_memory_control``. Reads each run's
per-episode cue-match rate from the ``run_simulation`` ``.out`` and reports, per arm, the
plateau-tail (final-quarter) mean cue-match across paired seeds, plus the pairwise deltas
(reusing the committed paired-seed Wilcoxon + 80% bootstrap CI + BH-FDR layer). The verdict
is **separation** when the recurrent/attention arms clear both chance and the memoryless MLP,
and **null** otherwise (the comparison cannot resolve working memory — itself a finding).

Cue-match source: with the canonical bit-memory reward (``reward_correct=1``,
``penalty_wrong=0``) the per-episode reward equals the number of correct responses, so the
cue-match rate is ``reward / num_responses`` where ``num_responses = trials_per_episode *
response_steps`` (default 20). Pass ``--num-responses`` if the configs differ.

Usage::

    uv run python scripts/analysis/bit_memory_separation.py \
        --manifest <run-dir>/_manifest.txt --num-responses 20 --out <run-dir>/separation.json
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

# Reuse the committed statistics layer verbatim (identical methodology to the T4/T7 ranking).
from weight_search_architecture_ranking import bh_fdr, paired_seed_wilcoxon_bootstrap

REPO = Path(__file__).resolve().parents[2]

_FINAL_WINDOW_FRAC = 0.25  # plateau-tail window: the final quarter of the training run
_CHANCE = 0.5  # binary cue -> chance accuracy
_SEPARATION_THRESHOLD = 0.80  # a memory arm must clear this to count as "solving" the task

MEMORY_ARMS = ("lstmppo", "cfcppo", "transformerppo")
MEMORYLESS_ARM = "mlpppo"
_RUN_LINE = re.compile(r"Run:\s+\d+.*?Reward:\s+([-0-9.]+)")


def _plateau_tail_cue_match(out_path: Path, num_responses: int) -> float | None:
    """Final-quarter mean cue-match rate from a run's per-episode rewards, or None."""
    if not out_path.exists():
        return None
    rates = [
        float(m.group(1)) / num_responses
        for line in out_path.read_text().splitlines()
        if (m := _RUN_LINE.search(line))
    ]
    if not rates:
        return None
    tail = max(1, int(len(rates) * _FINAL_WINDOW_FRAC))
    return float(np.mean(rates[-tail:]))


def load(manifest: Path, num_responses: int) -> dict[str, dict[int, float]]:
    """Return {arch: {seed: plateau-tail cue-match}} from a ``<arch> <seed> <out_path>`` manifest."""
    table: dict[str, dict[int, float]] = {}
    for line in manifest.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        arch, seed, out_path = parts[0], int(parts[1]), Path(parts[2])
        cue_match = _plateau_tail_cue_match(REPO / out_path, num_responses)
        if cue_match is not None:
            table.setdefault(arch, {})[seed] = cue_match
    return table


def analyse(table: dict[str, dict[int, float]], out: dict) -> None:
    """Print + record the per-arm cue-match, the pairwise deltas, and the separation verdict."""
    archs = sorted(table, key=lambda a: -float(np.mean(list(table[a].values()))))
    print("\n" + "=" * 72)
    print("BIT-MEMORY SEPARATION — plateau-tail cue-match (chance = 0.50)")
    print("=" * 72)
    means: dict[str, float] = {}
    out["per_arm"] = {}
    for a in archs:
        vals = list(table[a].values())
        means[a] = float(np.mean(vals))
        print(f"  {a:16} {means[a]:.3f}   n={len(vals)}   per-seed={[round(v, 3) for v in vals]}")
        out["per_arm"][a] = {"mean": means[a], "n": len(vals), "per_seed": vals}

    print("\n  Pairwise paired-seed deltas (one-sided Wilcoxon a>b, 80% bootstrap CI, BH-FDR):")
    pairs, stats = [], []
    for a, b in combinations(archs, 2):
        common = sorted(set(table[a]) & set(table[b]))
        deltas = [table[a][s] - table[b][s] for s in common]
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

    # Verdict: a memory arm "separates" when it clears the threshold AND significantly beats
    # the memoryless MLP, while the MLP itself sits near chance.
    q_by_pair = {(p[0], p[1]): q for p, q in zip(pairs, qs, strict=True)}
    mlp_mean = means.get(MEMORYLESS_ARM)
    separators = []
    for a in MEMORY_ARMS:
        if a not in means or mlp_mean is None:
            continue
        q = q_by_pair.get((a, MEMORYLESS_ARM)) or q_by_pair.get((MEMORYLESS_ARM, a))
        beats_mlp = means[a] > mlp_mean and q is not None and q < 0.05
        if means[a] >= _SEPARATION_THRESHOLD and beats_mlp:
            separators.append(a)
    mlp_at_chance = mlp_mean is not None and abs(mlp_mean - _CHANCE) < 0.15
    separated = bool(separators) and mlp_at_chance
    out["verdict"] = {
        "separated": separated,
        "separating_arms": separators,
        "mlp_mean": mlp_mean,
        "mlp_at_chance": mlp_at_chance,
    }
    print("\n" + "-" * 72)
    if separated:
        print(
            f"  VERDICT: SEPARATION CONFIRMED — {', '.join(separators)} clear chance + beat "
            f"MLP (at {mlp_mean:.3f}). The memory-axis follow-ons (new_arch_candidates, "
            "ars_depletion) are unblocked.",
        )
    else:
        print(
            "  VERDICT: NULL — the comparison did not separate working memory "
            f"(MLP {mlp_mean}, separating arms {separators}). The memory-axis follow-ons "
            "stay deferred; the null is the finding.",
        )


def main() -> None:
    """Load the manifest, compute + print + write the separation analysis."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest", type=Path, required=True, help="<arch> <seed> <out_path> per line",
    )
    ap.add_argument(
        "--num-responses",
        type=int,
        default=20,
        help="responses per episode = trials_per_episode * response_steps (default 20)",
    )
    ap.add_argument("--out", type=Path, default=None, help="write the separation summary JSON here")
    args = ap.parse_args()

    table = load(args.manifest, args.num_responses)
    out: dict = {}
    analyse(table, out)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
