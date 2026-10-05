#!/usr/bin/env python
"""A.3's registered structural predictor: a null's sensor-to-motor shortcuts against block V's gap.

A degree-preserving rewiring routes some food sensors straight onto readout motor neurons. The wild
type has none one hop away; across rewirings the count varies. A.2 found the wiring advantage
depth-critical, and its hop probe tied that to these shortcuts, after the fact. This test registers
the claim the mechanism makes across seeds: **a null with more one-hop shortcuts should learn faster
relative to the wild type, so its seed's wild-type lead should be smaller.**

Statistic: Spearman's rho between each seed's current-null one-hop count (readout motor neurons one
hop from a food sensor, over the propagating graph of chemical synapses and gap junctions) and the
same seed's wild-type minus current-null ``auc_success`` gap, on committed hard350 PPO panels at
block V's point (depth 4). Predicted direction negative; minimum |rho| 0.3; two-sided permutation p;
an 80% bootstrap interval. Nothing here trains anything.

Usage::

    uv run python scripts/analysis/hop_predictor.py --out <hop-predictor.json>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sensory_motor_hops as hops  # noqa: E402  # pyright: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parents[2]
SUPPORTING = REPO / "docs" / "experiments" / "logbooks" / "supporting"
A6_CSV = SUPPORTING / "074-null-strength-control" / "per-seed.csv"
A2_CSV = SUPPORTING / "071-operating-point-surface" / "ppo-per-seed.csv"
METRIC = "auc_success"
NULL_WIRING = "rewired_degree_preserving"
MINIMUM_RHO = 0.3
ALPHA = 0.05
N_PERMUTATIONS = 20_000
N_BOOTSTRAP = 20_000
CI_LEVEL = 0.80
RNG_SEED = 20261006


def gaps_a6(path: Path = A6_CSV) -> dict[int, float]:
    """A.6's PPO wild-type minus current-null gap per seed (its ``full`` level)."""
    rows = csv.DictReader(path.open())
    return {int(r["seed"]): float(r[f"full_gap_{METRIC}"]) for r in rows if r["half"] == "ppo"}


def gaps_a2_centre(path: Path = A2_CSV) -> dict[int, float]:
    """A.2's PPO centre gap per seed: each level row's gap minus its interaction with the centre.

    Every level row of a seed must give the same centre gap, or the file is not what this assumes.
    """
    out: dict[int, float] = {}
    for r in csv.DictReader(path.open()):
        if r["half"] != "ppo" or r["metric"] != METRIC:
            continue
        seed = int(r["seed"])
        centre = float(r["wiring_gap"]) - float(r["interaction"])
        if seed in out and abs(out[seed] - centre) > 1e-5:
            msg = f"A.2 seed {seed}: centre gap {centre} disagrees with {out[seed]}"
            raise ValueError(msg)
        out[seed] = centre
    return out


def one_hop_count(seed: int) -> int:
    """Readout motor neurons one hop from a food sensor in the seed's current null."""
    reach = hops.motor_reach(hops._topology(NULL_WIRING, seed))
    return sum(1 for h in reach["hops"] if h == 1)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        tied = x == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho: Pearson's r on average ranks."""
    rx, ry = _ranks(x), _ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def classify(rho: float, p: float, ci: tuple[float, float]) -> str:
    """Map the registered statistic to its verdict."""
    if p < ALPHA:
        if rho <= -MINIMUM_RHO:
            return "predicts"
        if rho >= MINIMUM_RHO:
            return "opposite"
        return "below_minimum"
    if ci[0] > -MINIMUM_RHO and ci[1] < MINIMUM_RHO:
        return "no_prediction"
    return "unresolved"


def registered_test(
    predictor: np.ndarray,
    outcome: np.ndarray,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Spearman's rho, its two-sided permutation p and its bootstrap interval."""
    rng = np.random.default_rng(seed)
    rho = spearman(predictor, outcome)
    perms = np.array([spearman(predictor, rng.permutation(outcome)) for _ in range(N_PERMUTATIONS)])
    p = float((np.sum(np.abs(perms) >= abs(rho)) + 1) / (N_PERMUTATIONS + 1))
    n = len(predictor)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if np.unique(predictor[idx]).size > 1:
            boots.append(spearman(predictor[idx], outcome[idx]))
    tail = (1 - CI_LEVEL) / 2
    ci = (float(np.quantile(boots, tail)), float(np.quantile(boots, 1 - tail)))
    return {"n": n, "rho": rho, "p_two_sided": p, "ci": ci, "verdict": classify(rho, p, ci)}


def run() -> dict[str, Any]:
    """Assemble every seed's predictor and outcome, then the registered test."""
    gaps = {**gaps_a2_centre(), **gaps_a6()}
    seeds = sorted(gaps)
    counts = {s: one_hop_count(s) for s in seeds}
    predictor = np.array([counts[s] for s in seeds], dtype=float)
    outcome = np.array([gaps[s] for s in seeds], dtype=float)
    return {
        "statistic": "spearman(current-null one-hop count, wild-type minus null auc_success gap)",
        "predicted_direction": "negative",
        "minimum_abs_rho": MINIMUM_RHO,
        "sources": {"a2_centre": sorted(gaps_a2_centre()), "a6_full": sorted(gaps_a6())},
        "per_seed": {str(s): {"one_hop": counts[s], "gap": gaps[s]} for s in seeds},
        "result": registered_test(predictor, outcome),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, help="write the result JSON here")
    args = ap.parse_args(argv)
    result = run()
    payload = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload)
    print(json.dumps(result["result"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
