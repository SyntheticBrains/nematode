#!/usr/bin/env python
"""Calibrate each grounded sign level's chemical weight scale to random signs' operating point.

Grounded synapse signs are mostly excitatory, add coherently, and saturate much of the settled
network at the draw's scale, which suits random signs. For each grounded level this measures, at
initialisation and with no training, the mean fraction of units with ``|h| > 0.99`` after settling
-- over a fixed set of inputs, the calibration seeds and both wirings -- at each scale on a grid,
and picks the scale whose fraction is closest to the random-sign level's at scale 1.0, ties to the
larger scale. Each scale's action-mean spread across inputs is reported beside.

Usage::

    uv run python scripts/analysis/sign_scale_calibration.py --out calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import generate_sign_rung_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
from quantumnematode.brain.arch.connectome_ppo import (  # noqa: E402
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType  # noqa: E402
from quantumnematode.utils.config_loader import load_simulation_config  # noqa: E402

SEEDS: tuple[int, ...] = tuple(range(9601, 9617))
SCALES: tuple[float, ...] = (1.0, 0.85, 0.7, 0.6, 0.5, 0.42, 0.35, 0.3, 0.25, 0.2)
N_INPUTS = 64
INPUT_SEED = 20261012
SATURATED = 0.99
WIRINGS: tuple[str, ...] = ("wild_type", "rewired_chemical_only")


def base_config() -> ConnectomePPOBrainConfig:
    """Return the random-sign wild type's brain config on the thermal cell at target 35."""
    path = gen.THERMAL / f"{gen.stem('random', null=False, frozen=False)}.yml"
    container = load_simulation_config(str(path)).brain
    assert container is not None  # noqa: S101 - a committed config
    assert isinstance(container.config, ConnectomePPOBrainConfig)  # noqa: S101
    return container.config


def inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return the fixed (food, thermal) input vectors."""
    rng = np.random.default_rng(INPUT_SEED)
    return [
        (
            torch.tensor(rng.uniform(-1.0, 1.0, 3), dtype=torch.float32),
            torch.tensor(rng.uniform(-1.0, 1.0, 3), dtype=torch.float32),
        )
        for _ in range(N_INPUTS)
    ]


def measure(
    config: ConnectomePPOBrainConfig,
    probe: list[tuple[torch.Tensor, ...]],
) -> tuple[float, float]:
    """Return one brain's saturated fraction and its action-mean spread across the probe inputs."""
    topology = ConnectomePPOBrain(config=config, device=DeviceType.CPU).topology
    saturated, outputs = [], []
    with torch.no_grad():
        for food, thermal in probe:
            out, hidden = topology.forward_with_hidden(food, thermotaxis_features=thermal)
            saturated.append(float((hidden.abs() > SATURATED).float().mean()))
            outputs.append(out.numpy())
    return float(np.mean(saturated)), float(np.array(outputs).std(axis=0).mean())


def level_statistic(signs: str, scale: float, probe: list[Any]) -> dict[str, float]:
    """Average the saturated fraction and action spread over the seeds and both wirings."""
    base = base_config()
    changes: dict[str, Any] = {"chemical_weight_scale": scale}
    if signs != "random":
        changes |= {"synapse_signs": signs, "enforce_synapse_signs": True}
    values = [
        measure(base.model_copy(update={**changes, "seed": seed, "wiring": wiring}), probe)
        for seed in SEEDS
        for wiring in WIRINGS
    ]
    return {
        "saturated": float(np.mean([v[0] for v in values])),
        "action_spread": float(np.mean([v[1] for v in values])),
    }


def calibrate() -> dict[str, Any]:
    """Measure the target, then each grounded level's grid, and choose each level's scale."""
    probe = inputs()
    target = level_statistic("random", 1.0, probe)
    out: dict[str, Any] = {
        "seeds": list(SEEDS),
        "grid": list(SCALES),
        "threshold": SATURATED,
        "target": target,
        "levels": {},
    }
    for signs in ("atlas", "per_connection"):
        grid = {scale: level_statistic(signs, scale, probe) for scale in SCALES}
        chosen = min(SCALES, key=lambda s: (abs(grid[s]["saturated"] - target["saturated"]), -s))
        out["levels"][signs] = {
            "grid": {str(s): v for s, v in grid.items()},
            "chosen_scale": chosen,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: run the calibration and write it."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = calibrate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["chosen_scale"] for k, v in result["levels"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
