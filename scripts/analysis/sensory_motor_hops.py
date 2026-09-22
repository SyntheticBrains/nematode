#!/usr/bin/env python
"""How many hops a food signal needs to reach a motor neuron, wild type against rewired nulls.

A.2's PPO surface found the wiring advantage is **depth-critical**: replicated at the committed
settling depth of 4 and at 6, abolished at 3, reversed at 2, with the rewired null reaching 62%
full-clear against the wild type's 10% at depth 2. The obvious mechanism is a graph property. A
degree-preserving rewiring is a randomisation, and random graphs have shorter characteristic path
lengths than structured ones, so if the real connectome needs more hops to carry sensation to motor
neurons than its null does, a settling budget too short for the real topology would favour the null
for a reason that has nothing to do with the null being better wired.

This probe measures that directly rather than leaving it as a story. It walks the **same graph the
simulation propagates through** -- the brain's own masked chemical matrix plus its gap junctions,
read off a constructed topology rather than rebuilt from the data files -- and reports the hop
distance from the six food sensors to each of the 39 motor neurons.

**What a result here can and cannot say.** A shorter path in the null is *consistent* with the depth
finding and would make the mechanism worth testing; it is not itself evidence that path length
causes the effect, because the rewiring changes many things at once. The registered structural-
predictor discipline applies: a statistic, its direction and its minimum are named before the
correlation is computed, and this probe names none, so it is reported as a description of the graphs
and not as a mechanism confirmed.

Usage::

    uv run python scripts/analysis/sensory_motor_hops.py --seeds 1-8
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

REPO = Path(__file__).resolve().parents[2]
ARM = (
    REPO
    / "configs"
    / "scenarios"
    / "foraging"
    / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350.yml"
)
UNREACHED = -1


def _topology(wiring: str, seed: int) -> Any:
    """Construct one arm and hand back its topology, so the graph measured is the graph run."""
    brain_config = load_simulation_config(str(ARM)).brain
    assert brain_config is not None
    assert isinstance(brain_config.config, ConnectomePPOBrainConfig)
    update: dict[str, Any] = {"seed": seed}
    if wiring != "wild_type":
        update["wiring"] = "rewired_degree_preserving"
    torch.manual_seed(seed)
    return ConnectomePPOBrain(
        config=brain_config.config.model_copy(update=update),
        device=DeviceType.CPU,
    ).topology


def hop_distances(topology: Any) -> dict[int, int]:
    """Breadth-first hops from the food sensors to every neuron, over the propagating graph.

    Chemical synapses are directed pre to post; gap junctions conduct both ways. Both carry signal
    in the settling loop, so both are edges here -- measuring the chemical graph alone would
    describe a network the simulation does not have.
    """
    mask = (topology.m_chem != 0).bool()
    gap = (topology.g_gap != 0).bool()
    adjacency = mask | gap | gap.T

    sources = topology._food_neuron_indices.tolist()  # noqa: SLF001
    distance = dict.fromkeys(sources, 0)
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        for nxt in torch.nonzero(adjacency[node]).flatten().tolist():
            if nxt not in distance:
                distance[nxt] = distance[node] + 1
                queue.append(nxt)
    return distance


def motor_reach(topology: Any) -> dict[str, Any]:
    """Hops to each motor neuron, and how many are inside each candidate settling budget."""
    distance = hop_distances(topology)
    motors = topology._motor_flat_indices.tolist()  # noqa: SLF001
    hops = [distance.get(m, UNREACHED) for m in motors]
    reached = [h for h in hops if h != UNREACHED]
    return {
        "n_motor": len(motors),
        "n_reached": len(reached),
        "min_hops": min(reached) if reached else None,
        "median_hops": st.median(reached) if reached else None,
        "max_hops": max(reached) if reached else None,
        # The settling loop runs `forward_pass_depth` iterations, so a motor neuron more than that
        # many hops from a sensor cannot have been reached by the food signal at all.
        "within_depth": {d: sum(1 for h in reached if h <= d) for d in (2, 3, 4, 6)},
        "hops": hops,
    }


def compare(seeds: tuple[int, ...]) -> dict[str, Any]:
    """Measure the wild type once and a fresh rewired null per seed."""
    wild = motor_reach(_topology("wild_type", seeds[0]))
    nulls = [motor_reach(_topology("rewired", s)) for s in seeds]
    out: dict[str, Any] = {"wild_type": wild, "rewired_nulls": nulls, "seeds": list(seeds)}
    for depth in (2, 3, 4, 6):
        null_counts = [n["within_depth"][depth] for n in nulls]
        out.setdefault("within_depth_summary", {})[depth] = {
            "wild_type": wild["within_depth"][depth],
            "rewired_mean": st.mean(null_counts),
            "rewired_min": min(null_counts),
            "rewired_max": max(null_counts),
        }
    return out


def _print(result: dict[str, Any]) -> None:
    wild = result["wild_type"]
    nulls = result["rewired_nulls"]
    print(f"food sensors -> {wild['n_motor']} motor neurons, over chemical + gap-junction edges\n")
    print(f"  wild type      min {wild['min_hops']}  median {wild['median_hops']}  max {wild['max_hops']}")
    med = st.mean([n["median_hops"] for n in nulls])
    print(
        f"  rewired nulls  min {st.mean([n['min_hops'] for n in nulls]):.2f}  "
        f"median {med:.2f}  max {st.mean([n['max_hops'] for n in nulls]):.2f}   "
        f"(mean over {len(nulls)} rewirings)",
    )
    print("\n  motor neurons reachable within a settling budget of N hops:")
    print(f"    {'N':>3}  {'wild type':>10}  {'rewired null (mean, range)':>30}")
    for depth, row in result["within_depth_summary"].items():
        rng = f"{row['rewired_mean']:.1f}  [{row['rewired_min']}-{row['rewired_max']}]"
        print(f"    {depth:>3}  {row['wild_type']:>10}  {rng:>30}")


def main(argv: list[str] | None = None) -> int:
    """CLI: measure and print, optionally writing the JSON."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seeds", default="1-8", help="rewiring seeds, e.g. 1-8")
    ap.add_argument("--out", type=Path, help="write the measurement JSON here")
    args = ap.parse_args(argv)

    lo, _, hi = args.seeds.partition("-")
    seeds = tuple(range(int(lo), int(hi) + 1)) if hi else (int(lo),)
    result = compare(seeds)
    _print(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
