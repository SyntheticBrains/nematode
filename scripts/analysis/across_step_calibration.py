#!/usr/bin/env python
"""Calibrate the leaky substrate's input gain on untrained wild-type brains; nothing is trained.

The pilot at unit input gain found the leaky wild type learning neither cell. Its cause is a gain:
the policy mean's steady-state sensitivity to the sensory input is hundreds of times below the
settling substrate's. This script measures, per cell and on calibration seeds disjoint from every
band and pilot:

* the settling wild type's sensitivity: the largest absolute derivative of the policy mean with
  respect to the cell's sensory features (food on hard350, temperature on thermal), median over
  seeds and feature vectors;
* the leaky wild type's steady-state sensitivity at each input gain on a doubling grid, the same
  way, after 60 steps at a held input, with gradients through every step;
* the critical recurrent gain of each untrained leaky wild type: the factor on the chemical
  weights at which the linearised rest state loses stability. The registered recurrent gain is 1,
  so it must sit below every seed's critical gain.

**The rule, fixed with the recalibration:** the input gain whose sensitivity ratio to settling is
closest to 1 on a log scale on the worse of the two cells.

Usage::

    uv run python scripts/analysis/across_step_calibration.py --out calibration.json
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import functools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomeTopology
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

REPO = Path(__file__).resolve().parents[2]
CONFIGS = {
    "hard350": REPO / "configs/scenarios/foraging/"
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350.yml",
    "thermal": REPO / "configs/scenarios/thermal_foraging/"
    "connectomeppo_small_continuous2d_thermal_klinotaxis_t35.yml",
}
INPUT_GAINS: tuple[float, ...] = (64.0, 128.0, 256.0, 512.0, 1024.0)
SENSITIVITY_SEEDS: tuple[int, ...] = tuple(range(2001, 2009))
CRITICAL_SEEDS: tuple[int, ...] = tuple(range(2001, 2065))
N_FEATURE_VECTORS = 4
FEATURE_SCALE = 0.5
FEATURE_SEED = 7
STEADY_STEPS = 60


def _topology(cell: str, seed: int, **overrides: object) -> ConnectomeTopology:
    container = load_simulation_config(str(CONFIGS[cell])).brain
    if container is None:
        msg = f"{CONFIGS[cell]} has no brain section"
        raise ValueError(msg)
    config = container.config.model_copy(update={"seed": seed, **overrides})
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU).topology  # type: ignore[arg-type]


def _inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return fixed ``(food, temperature)`` feature vectors, the same for every seed and setting."""
    gen = torch.Generator().manual_seed(FEATURE_SEED)
    return [
        (
            torch.randn(3, generator=gen) * FEATURE_SCALE,
            torch.randn(3, generator=gen) * FEATURE_SCALE,
        )
        for _ in range(N_FEATURE_VECTORS)
    ]


def _probe(cell: str, food: torch.Tensor, thermo: torch.Tensor) -> torch.Tensor:
    """Return the features the sensitivity is taken against: food, or temperature on thermal."""
    return food if cell == "hard350" else thermo


def _settling_mean(
    topo: ConnectomeTopology,
    cell: str,
    food: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    if cell == "hard350":
        return topo.forward_with_hidden(x)[0]
    return topo.forward_with_hidden(food, thermotaxis_features=x)[0]


def _leaky_mean(
    topo: ConnectomeTopology,
    cell: str,
    food: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    f, th = (x, None) if cell == "hard350" else (food, x)
    current = topo.input_gain * topo._sensor_current(f, None, None, None, th)
    v = torch.zeros(topo.n_neurons)
    for _ in range(STEADY_STEPS):
        v = topo._leaky_substeps(v, current)
    return topo.readout @ topo._pool_motor(torch.tanh(v))


def settling_sensitivity(cell: str) -> float:
    """Median largest |d policy mean / d features| of the settling wild type."""
    values: list[float] = []
    for seed in SENSITIVITY_SEEDS:
        topo = _topology(cell, seed)
        for food, thermo in _inputs():
            mean = functools.partial(_settling_mean, topo, cell, food)
            jac = torch.autograd.functional.jacobian(mean, _probe(cell, food, thermo))
            values.append(float(jac.abs().max()))
    return statistics.median(values)


def leaky_sensitivity(cell: str, input_gain: float) -> float:
    """Median largest |d steady-state policy mean / d features| of the leaky wild type."""
    values: list[float] = []
    for seed in SENSITIVITY_SEEDS:
        topo = _topology(cell, seed, dynamics="leaky", input_gain=input_gain)
        for food, thermo in _inputs():
            mean = functools.partial(_leaky_mean, topo, cell, food)
            jac = torch.autograd.functional.jacobian(mean, _probe(cell, food, thermo))
            values.append(float(jac.abs().max()))
    return statistics.median(values)


def critical_gain(topo: ConnectomeTopology) -> float:
    """Return the chemical-weight factor at which -(I + L) + g W^T first gains Re >= 0."""
    gap = topo.g_gap.double()
    laplacian = torch.diag(gap.sum(dim=1)) - gap
    chem = (topo.w_chem * topo.m_chem).detach().double()
    eye = torch.eye(topo.n_neurons, dtype=torch.float64)

    def growth(g: float) -> float:
        return float(torch.linalg.eigvals(-(eye + laplacian) + g * chem.T).real.max())

    lo, hi = 0.3, 10.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if growth(mid) < 0 else (lo, mid)
    return lo


def select(ratios: dict[float, dict[str, float]]) -> dict[str, Any]:
    """Choose the input gain whose worse cell's ratio is closest to 1 on a log scale."""
    distance = {
        gain: max(abs(math.log(r)) for r in by_cell.values()) for gain, by_cell in ratios.items()
    }
    chosen = min(distance, key=lambda gain: (distance[gain], gain))
    return {"input_gain": chosen, "log_distance": distance}


def run() -> dict[str, Any]:
    """Measure both cells, the critical gains, and select."""
    settling = {cell: settling_sensitivity(cell) for cell in CONFIGS}
    ratios = {
        gain: {cell: leaky_sensitivity(cell, gain) / settling[cell] for cell in CONFIGS}
        for gain in INPUT_GAINS
    }
    # The chemical weights and gap junctions do not depend on the cell, so one cell suffices.
    critical = np.array(
        [critical_gain(_topology("hard350", s, dynamics="leaky")) for s in CRITICAL_SEEDS],
    )
    return {
        "settling_sensitivity": settling,
        "ratio_to_settling": {f"{g:g}": r for g, r in ratios.items()},
        "critical_recurrent_gain": {
            "seeds": [CRITICAL_SEEDS[0], CRITICAL_SEEDS[-1]],
            "min": float(critical.min()),
            "p5": float(np.percentile(critical, 5)),
            "median": float(np.median(critical)),
            "max": float(critical.max()),
            "unit_gain_subcritical_on_every_seed": bool(critical.min() > 1.0),
        },
        "selection": select(ratios),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, help="write the calibration JSON here")
    args = ap.parse_args(argv)
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
