"""V.2 probe - graph properties of each seed's rewiring against how slowly that seed learned.

Logbook 057 found the wild-type wiring reaching competence ~35% sooner than its degree-preserving
rewired null, heterogeneously: 44 of 64 seeds favour the wild type and the rewired arm's times span
32 to 2275 episodes. Each seed's rewiring is a **different graph** (``rewire_seed`` derives from the
run seed) while the wild type is one graph with varying weight initialisation, so the rewired arm
carries a variance source the wild type does not.

The substrate settles ``forward_pass_depth`` steps per environment step, so a sensory signal needs a
directed path of at most that many hops to reach a motor-readout neuron within one forward pass. A
rewiring that pushes the thermosensory route beyond it cannot deliver temperature to the action.

**Exploratory.** Four properties, fixed before looking (see `probe-v2.md`), Spearman against the
rewired arm's ``episodes_to_30pct_success``, BH-FDR across the four. It proposes a mechanism; it
cannot strengthen 057's claim.

Usage::

    uv run python scripts/analysis/wiring_premise_graph_probe.py \
        --panel <panel1.json> --panel <panel2.json> --panel <panel3.json> --out <probe.json>
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.brain.arch.connectome_ppo import (
    _MOTOR_CLASSES,
    _SENSOR_NEURONS_FOOD,
    _SENSOR_NEURONS_THERMOTAXIS,
)
from quantumnematode.connectome import load_cook_2019_hermaphrodite
from quantumnematode.connectome.rewiring import rewire_degree_preserving
from scipy import stats

# Both fixed by the committed configs the panel ran: `forward_pass_depth: 4`.
SETTLING_DEPTH = 4
SEEDS = range(1, 65)
METRIC = "episodes_to_30pct_success"

# The four registered properties, in the order the record states them.
PROPERTIES = ("P1_thermo_path", "P2_thermo_reach_4hop", "P3_food_reach_4hop", "P4_char_path")


def _adjacency(connectome) -> dict[str, list[str]]:  # noqa: ANN001
    """Directed chemical adjacency, pre -> [post]."""
    out: dict[str, list[str]] = {}
    for synapse in connectome.chemical_synapses:
        out.setdefault(synapse.pre, []).append(synapse.post)
    return out


def _motor_neurons(connectome) -> set[str]:  # noqa: ANN001
    """Return the readout's neurons: the VB/DB/VA/DA classes, as the brain defines them."""
    return {name for name in connectome.neurons if name.startswith(_MOTOR_CLASSES)}


def _bfs_depths(adjacency: dict[str, list[str]], sources: set[str]) -> dict[str, int]:
    """Hop count from the nearest source to every reachable neuron."""
    depth = dict.fromkeys(sources, 0)
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        for nxt in adjacency.get(node, ()):
            if nxt not in depth:
                depth[nxt] = depth[node] + 1
                queue.append(nxt)
    return depth


def _char_path_length(adjacency: dict[str, list[str]], names: list[str]) -> float:
    """Mean directed shortest-path length over reachable ordered pairs."""
    total, pairs = 0, 0
    for name in names:
        depths = _bfs_depths(adjacency, {name})
        for target, d in depths.items():
            if target != name:
                total += d
                pairs += 1
    return total / pairs if pairs else float("nan")


def graph_properties(connectome) -> dict[str, float]:  # noqa: ANN001
    """Compute the four registered properties of one graph."""
    adjacency = _adjacency(connectome)
    names = list(connectome.neurons)
    motor = _motor_neurons(connectome)

    thermo = _bfs_depths(adjacency, set(_SENSOR_NEURONS_THERMOTAXIS))
    thermo_to_motor = [d for name, d in thermo.items() if name in motor and d > 0]
    food = _bfs_depths(adjacency, set(_SENSOR_NEURONS_FOOD))

    return {
        "P1_thermo_path": float(min(thermo_to_motor)) if thermo_to_motor else float("inf"),
        "P2_thermo_reach_4hop": float(
            sum(1 for name, d in thermo.items() if name in motor and 0 < d <= SETTLING_DEPTH),
        ),
        "P3_food_reach_4hop": float(
            sum(1 for name, d in food.items() if name in motor and 0 < d <= SETTLING_DEPTH),
        ),
        "P4_char_path": _char_path_length(adjacency, names),
    }


def _bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up, matching the layer the panels were scored with."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n = len(pvals)
    qs = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, pvals[i] * n / (n - rank + 1))
        qs[i] = prev
    return qs


def load_times(panels: list[Path]) -> dict[int, float]:
    """Rewired-arm time-to-competence per seed, from the committed panel JSONs."""
    times: dict[int, float] = {}
    for path in panels:
        per_seed = json.loads(path.read_text())["efficiency"]["thermal"]["per_seed"]["rewired_null"]
        for seed, metrics in per_seed.items():
            key = int(seed)
            if key in times:
                msg = f"seed {key} appears in more than one panel"
                raise ValueError(msg)
            times[key] = float(metrics[METRIC])
    return times


def analyse(panels: list[Path]) -> dict[str, Any]:
    """Score every seed's rewiring and correlate each property against its learning time."""
    times = load_times(panels)
    missing = [s for s in SEEDS if s not in times]
    if missing:
        msg = f"panels are missing seeds {missing}"
        raise ValueError(msg)

    wild = load_cook_2019_hermaphrodite()
    wild_props = graph_properties(wild)

    per_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        rewired = rewire_degree_preserving(wild, np.random.default_rng(seed))
        per_seed[seed] = graph_properties(rewired)

    correlations: dict[str, dict[str, Any]] = {}
    pvals = []
    for prop in PROPERTIES:
        xs = np.array([per_seed[s][prop] for s in SEEDS])
        ys = np.array([times[s] for s in SEEDS])
        finite = np.isfinite(xs)
        distinct = sorted(set(xs[finite]))
        if finite.sum() < 3 or len(distinct) < 2:
            # A property that does not vary across rewirings cannot predict anything, and saying so
            # is the informative output - reporting a failed correlation would hide the reason.
            correlations[prop] = {
                "rho": float("nan"),
                "p": 1.0,
                "n": int(finite.sum()),
                "constant_at": distinct[0] if distinct else float("nan"),
                "wild_type": wild_props[prop],
                "note": "constant across all rewirings - cannot discriminate",
            }
            pvals.append(1.0)
            continue
        # scipy returns a result object that pyright cannot narrow; take both fields as floats.
        result: Any = stats.spearmanr(xs[finite], ys[finite])
        rho, pvalue = float(result.statistic), float(result.pvalue)
        correlations[prop] = {
            "rho": rho,
            "p": pvalue,
            "n": int(finite.sum()),
            "mean": float(np.mean(xs[finite])),
            "min": float(np.min(xs[finite])),
            "max": float(np.max(xs[finite])),
            "wild_type": wild_props[prop],
        }
        pvals.append(pvalue)

    for prop, q in zip(PROPERTIES, _bh_fdr(pvals), strict=True):
        correlations[prop]["q"] = q

    return {
        "settling_depth": SETTLING_DEPTH,
        "metric": METRIC,
        "n_seeds": len(times),
        "wild_type": wild_props,
        "correlations": correlations,
        "per_seed": {str(s): {**per_seed[s], "rewired_time": times[s]} for s in SEEDS},
    }


def print_report(report: dict[str, Any]) -> None:
    """Print the probe's table and the reading its record registered."""
    print("=" * 88)
    print(f"V.2 PROBE - graph property vs rewired time-to-competence (n={report['n_seeds']} seeds)")
    print(f"settling depth {report['settling_depth']} hops; EXPLORATORY, generates hypotheses only")
    print("=" * 88)
    print(f"  {'property':24} {'wild':>8} {'mean':>8} {'range':>13}  {'rho':>7} {'q':>7}")
    for prop in PROPERTIES:
        c = report["correlations"][prop]
        if "constant_at" in c:
            print(
                f"  {prop:24} {c['wild_type']:8.2f} {c['constant_at']:8.2f} {'constant':>13}"
                f"        --      --  {c['note']}",
            )
            continue
        rng = f"{c['min']:.2f}-{c['max']:.2f}"
        print(
            f"  {prop:24} {c['wild_type']:8.2f} {c['mean']:8.2f}"
            f" {rng:>13}  {c['rho']:+7.3f} {c['q']:7.3f}",
        )


def main() -> None:
    """Load the panels, score the rewirings, print and write the probe."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, action="append", required=True, help="panel JSON; repeat")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report = analyse(args.panel)
    print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
