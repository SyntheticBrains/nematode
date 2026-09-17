#!/usr/bin/env python
"""The structural probe on L.1's open puzzle, why the per-neuron readout hurt the null.

[L.1](../../docs/experiments/logbooks/066-l4-readout-width.md) found the per-neuron readout helped
the wild type (+0.130 on `auc_success`) and **hurt** the degree-preserving null (-0.152). The
registered design predicted an interaction, not that direction of it.

**Hypothesis, registered before the correlation was computed.** Degree-preserving rewiring
**decorrelates the inputs within each motor class**. The four class means average out seed-specific
input noise that 39 per-neuron weights fit instead; on the wild type, neurons within a class share
inputs, so per-neuron weights find structure rather than noise.

**Statistic.** For a wiring, the mean pairwise Jaccard of the presynaptic sets of the motor neurons
within a class (``m_chem[:, j]`` for pool neuron ``j``), averaged over the four classes.

**Registered test.** Across the 96 rewirings, Spearman rho between a seed's within-class Jaccard and
that seed's ``rn_wide - rn_pooled`` from L.1's committed ``per-seed.csv``, **one-sided positive** at
q = 0.05, with a registered minimum of **rho >= 0.3**. A positive licenses the hypothesis for a
follow-up that manipulates within-class correlation directly; **it is not a mechanism claim**.

Feasibility only had been looked at when this was registered: the wild type reads 0.07-0.23 across
VB/DB/VA/DA and three rewirings read 0.01-0.04. No runs: 97 topology builds and one CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import l4_readout_width as rw  # noqa: E402  # pyright: ignore[reportMissingImports]

SEEDS = tuple(range(1, 97))
MIN_RHO = 0.3
SIG = 0.05
CONFIG = (
    rw.EXPERIMENTS.parent
    / "configs"
    / "scenarios"
    / "foraging"
    / f"{rw._STEM}_readout_only_wide.yml"
)

READINGS = {
    "probe_supported": (
        "within-class input correlation predicts how much the per-neuron readout hurt a rewiring, "
        "at the registered minimum. Licenses a follow-up that manipulates it directly; not a "
        "mechanism claim"
    ),
    "probe_below_minimum": "significant but under rho = 0.3: named, licenses nothing on its own",
    "probe_null": "the puzzle is not this, and stays a puzzle",
}


def within_class_jaccard(
    mask: np.ndarray,
    flat: np.ndarray,
    slices: list[tuple[int, int]],
) -> list[float]:
    """Mean pairwise Jaccard of presynaptic sets within each motor class.

    ``mask`` is ``[pre, post]``; the presynaptic set of pool neuron ``j`` is the column ``mask[:, j]``.
    """
    out: list[float] = []
    for start, stop in slices:
        members = flat[start:stop]
        pres = [set(np.nonzero(mask[:, j])[0].tolist()) for j in members]
        pairs = [len(a & b) / max(1, len(a | b)) for i, a in enumerate(pres) for b in pres[i + 1 :]]
        out.append(float(np.mean(pairs)) if pairs else 0.0)
    return out


def topology_jaccard(seed: int, wiring: str, config: Path = CONFIG) -> list[float]:
    """Build the wiring a run at ``seed`` would have used and return its per-class Jaccard."""
    from quantumnematode.brain.arch.connectome_ppo import (
        ConnectomePPOBrain,
        ConnectomePPOBrainConfig,
    )
    from quantumnematode.brain.arch.dtypes import DeviceType
    from quantumnematode.utils.config_loader import load_simulation_config

    cfg = load_simulation_config(str(config)).brain
    if cfg is None or not isinstance(cfg.config, ConnectomePPOBrainConfig):
        msg = f"{config} does not describe a connectome brain"
        raise TypeError(msg)
    brain_cfg = cfg.config.model_copy(update={"seed": seed, "wiring": wiring})
    torch.manual_seed(seed)
    topo = ConnectomePPOBrain(config=brain_cfg, device=DeviceType.CPU).topology
    mask = topo.m_chem.detach().cpu().numpy().astype(bool)
    flat = topo._motor_flat_indices.cpu().numpy()
    return within_class_jaccard(mask, flat, list(topo._motor_class_slices))


def per_seed_drop(per_seed_csv: Path, metric: str = "auc_success") -> dict[int, float]:
    """Return ``rn_wide - rn_pooled`` per seed from L.1's committed per-seed file."""
    rows: dict[tuple[str, str, int], float] = {}
    with per_seed_csv.open() as handle:
        for row in csv.DictReader(handle):
            rows[(row["wiring"], row["width"], int(row["seed"]))] = float(row[metric])
    seeds = sorted({s for (w, _, s) in rows if w == "rn"})
    out: dict[int, float] = {}
    for s in seeds:
        if ("rn", "wide", s) in rows and ("rn", "pooled", s) in rows:
            out[s] = rows[("rn", "wide", s)] - rows[("rn", "pooled", s)]
    return out


def test_probe(jaccard: dict[int, float], drop: dict[int, float]) -> dict[str, Any]:
    """The registered one-sided Spearman test, with the minimum applied."""
    common = sorted(set(jaccard) & set(drop))
    missing = sorted((set(jaccard) | set(drop)) - set(common))
    if missing:
        msg = f"seeds present on one side only: {missing}"
        raise ValueError(msg)
    x = [jaccard[s] for s in common]
    y = [drop[s] for s in common]
    res = spearmanr(x, y)
    rho = float(res.statistic)
    p_two = float(res.pvalue)
    p_one = p_two / 2.0 if rho > 0 else 1.0 - p_two / 2.0
    significant = p_one <= SIG
    if significant and rho >= MIN_RHO:
        name = "probe_supported"
    elif significant:
        name = "probe_below_minimum"
    else:
        name = "probe_null"
    return {
        "n": len(common),
        "rho": rho,
        "p_one_sided_positive": p_one,
        "minimum_rho": MIN_RHO,
        "reading": name,
        "why": READINGS[name],
        "registered_direction": "positive: higher within-class Jaccard, smaller loss from widening",
    }


def analyse(
    per_seed_csv: Path,
    seeds: tuple[int, ...] = SEEDS,
    config: Path = CONFIG,
) -> dict[str, Any]:
    """Run the registered probe and the descriptive companion."""
    drop = per_seed_drop(per_seed_csv)
    jaccard = {
        s: float(np.mean(topology_jaccard(s, "rewired_degree_preserving", config))) for s in seeds
    }
    wild = topology_jaccard(1, "wild_type", config)
    wild_mean = float(np.mean(wild))
    rewired = np.array([jaccard[s] for s in seeds])
    result = test_probe(jaccard, {s: drop[s] for s in seeds if s in drop})
    return {
        "probe": result,
        "descriptive": {
            "wild_type_per_class": wild,
            "wild_type_mean": wild_mean,
            "rewired_mean": float(rewired.mean()),
            "rewired_max": float(rewired.max()),
            "wild_exceeds_every_rewiring": bool(wild_mean > rewired.max()),
        },
        "per_seed": {
            str(s): {"jaccard": jaccard[s], "rn_wide_minus_rn_pooled": drop.get(s)} for s in seeds
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the probe against L.1's committed per-seed file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default=f"{SEEDS[0]}-{SEEDS[-1]}")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    low, _, high = args.seeds.partition("-")
    seeds = tuple(range(int(low), int(high or low) + 1))
    result = analyse(args.per_seed, seeds)
    p, d = result["probe"], result["descriptive"]
    print(
        f"\nSTRUCTURAL PROBE  n={p['n']}  rho={p['rho']:+.3f}  p(one-sided +)={p['p_one_sided_positive']:.4f}  min rho {MIN_RHO}",
    )
    print(
        f"  wild-type mean Jaccard {d['wild_type_mean']:.3f} vs rewired mean {d['rewired_mean']:.3f} (max {d['rewired_max']:.3f})",
    )
    print(f"  READING: {p['reading'].upper().replace('_', '-')}\n    {p['why']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
