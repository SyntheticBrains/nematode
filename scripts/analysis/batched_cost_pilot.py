#!/usr/bin/env python
r"""Time the batched evaluator against the per-step runner, and project the fitting budget.

For each controller on each assay, the evaluator is timed on seeds 5001-5032, repeated to reach N, at
N = 32, 128, 200 and 256. The per-step runner is timed on seeds 5001-5008. All timings are per core: before
torch or numpy load, every thread pool is pinned to one thread, as the campaign runner pins them.
``--default-threads`` re-times N = 256 in a child process with the machine's own thread settings, for
comparison.

The record gives:

- microseconds per animal-step at each N, and the speedup per core over the per-step runner;
- whether the target was met: at most 25 us per animal-step at N = 256;
- the wall time of one 200-animal evaluation per assay, measured at N = 200;
- the projected fitting budget, as a formula with an illustration.

The budget formula is::

    one fit  = population x generations x (sum over assays of a 200-animal evaluation)
    the panel = (1 + null arms x draws per arm) x one fit

Usage::

    uv run python scripts/analysis/batched_cost_pilot.py \
        --out docs/experiments/logbooks/supporting/104-batched-evaluator/cost_pilot.json
"""

from __future__ import annotations

import argparse
import os
import sys

_PINNED = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
# The thread settings as the shell set them, before pinning, for the default-thread child.
_ORIGINAL = {name: os.environ.get(name) for name in _PINNED}
if "--child-default-threads" not in sys.argv:
    for _name in _PINNED:
        os.environ[_name] = "1"

import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import quantumnematode.brain  # noqa: E402, F401
import torch  # noqa: E402
from quantumnematode.env.hazard_controllers import ConstantHazards, PirouetteHazards  # noqa: E402
from quantumnematode.evaluation.batched import BatchedEvaluator  # noqa: E402
from quantumnematode.evaluation.controllers import (  # noqa: E402
    BatchedConnectome,
    BatchedConstantHazards,
    BatchedMLP,
    BatchedPirouetteHazards,
)
from quantumnematode.evaluation.reference import run_reference  # noqa: E402
from quantumnematode.utils.config_loader import load_simulation_config  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import body_kinematics_eval as harness  # noqa: E402

FORAGING = REPO / "configs" / "scenarios" / "foraging"
STEMS = {
    "mlp": "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives",
    "connectome": (
        "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_modulated_s1_primitives"
    ),
}
PROTOCOLS = (
    "scheer_lawn",
    "food_removal_lopezcruz",
    "food_removal_calhoun",
    "uniform_lawn_flavell",
)
CONTROLLERS = ("mlp", "connectome", "constant", "pirouette")
BAND = list(range(5001, 5033))
REFERENCE_SEEDS = list(range(5001, 5009))
SIZES = (32, 128, 200, 256)
TARGET_US = 25.0
EVALUATION_ANIMALS = 200
HAZARDS = (0.02, 0.01, 0.01)
# The illustration's settings, which the fits' own registrations replace.
ILLUSTRATION = {"population": 20, "generations": 200, "null_arms": 5, "draws_per_arm": (8, 16)}


def _path(controller: str, protocol: str) -> Path:
    stem = STEMS["connectome" if controller == "connectome" else "mlp"]
    return FORAGING / f"{stem}_assay_{protocol}.yml"


def _controllers(controller: str, protocol: str) -> tuple[Any, Any]:
    if controller == "constant":
        return ConstantHazards(0.15, HAZARDS), BatchedConstantHazards(0.15, HAZARDS)
    if controller == "pirouette":
        return (
            PirouetteHazards(0.15, HAZARDS, beta=3.0),
            BatchedPirouetteHazards(0.15, HAZARDS, beta=3.0),
        )
    brain, _, _ = harness.build_brain(_path(controller, protocol), BAND[0])
    if controller == "mlp":
        return brain, BatchedMLP(brain)  # type: ignore[arg-type]
    return brain, BatchedConnectome(brain)  # type: ignore[arg-type]


def _time_batched(config: Any, batched: Any, n: int) -> float:  # noqa: ANN401
    seeds = [BAND[i % len(BAND)] for i in range(n)]
    start = time.perf_counter()
    BatchedEvaluator(config, batched, seeds).run()
    return time.perf_counter() - start


def measure(controller: str, protocol: str, sizes: tuple[int, ...]) -> dict[str, Any]:
    """Time one controller on one assay: the per-step runner, then the evaluator at each size."""
    config = load_simulation_config(str(_path(controller, protocol)))
    steps = int(config.max_steps)  # type: ignore[arg-type]
    single, batched = _controllers(controller, protocol)
    start = time.perf_counter()
    for seed in REFERENCE_SEEDS:
        run_reference(config, single, seed)
    reference_us = (time.perf_counter() - start) / (len(REFERENCE_SEEDS) * steps) * 1e6
    batched_us = {n: _time_batched(config, batched, n) / (n * steps) * 1e6 for n in sizes}
    at_largest = batched_us[max(sizes)]
    return {
        "steps": steps,
        "reference_us_per_animal_step": reference_us,
        "batched_us_per_animal_step": {str(n): v for n, v in batched_us.items()},
        "speedup_per_core": reference_us / at_largest,
        "target_met": at_largest <= TARGET_US,
        "evaluation_200_s": batched_us[EVALUATION_ANIMALS] * EVALUATION_ANIMALS * steps / 1e6,
    }


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def budget(results: dict[str, dict[str, Any]], controller: str) -> dict[str, Any]:
    """Return one controller's projected fitting budget, from its 200-animal evaluation times."""
    evaluation_s = sum(results[f"{controller}/{p}"]["evaluation_200_s"] for p in PROTOCOLS)
    fit_h = ILLUSTRATION["population"] * ILLUSTRATION["generations"] * evaluation_s / 3600.0
    return {
        "evaluation_all_assays_s": evaluation_s,
        "one_fit_core_hours": fit_h,
        "panel_core_hours": {
            str(draws): (1 + ILLUSTRATION["null_arms"] * draws) * fit_h
            for draws in ILLUSTRATION["draws_per_arm"]
        },
    }


def main() -> int:
    """CLI: run the pilot and write its record."""
    parser = argparse.ArgumentParser(description="Time the batched evaluator.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--default-threads", action="store_true")
    parser.add_argument("--child-default-threads", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_default_threads:
        timings = {
            f"{c}/{p}": measure(c, p, (256,))["batched_us_per_animal_step"]["256"]
            for c in ("mlp", "connectome")
            for p in PROTOCOLS
        }
        args.out.write_text(json.dumps(timings, indent=2) + "\n")
        return 0
    torch.set_num_threads(1)
    results = {f"{c}/{p}": measure(c, p, SIZES) for c in CONTROLLERS for p in PROTOCOLS}
    record: dict[str, Any] = {
        "description": "Batched evaluator against the per-step runner, per core.",
        "commit": _commit(),
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "threads": {name: os.environ.get(name) for name in _PINNED} | {"torch": 1},
        "default_threads": {name: value or "unset" for name, value in _ORIGINAL.items()},
        "seeds": {"batched": "5001-5032, repeated to N", "reference": "5001-5008"},
        "target_us_per_animal_step": TARGET_US,
        "results": results,
        "budget_formula": (
            "one fit = population x generations x sum over assays of a 200-animal evaluation; "
            "panel = (1 + null arms x draws per arm) x one fit"
        ),
        "budget_illustration": {"settings": ILLUSTRATION}
        | {c: budget(results, c) for c in CONTROLLERS},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    if args.default_threads:
        child = args.out.with_name(args.out.stem + "_default_threads.json")
        environment = {k: v for k, v in os.environ.items() if k not in _PINNED}
        environment |= {k: v for k, v in _ORIGINAL.items() if v is not None}
        subprocess.run(  # noqa: S603 — fixed argv
            [sys.executable, __file__, "--out", str(child), "--child-default-threads"],
            check=True,
            env=environment,
        )
        record["default_threads_us_per_animal_step_n256"] = json.loads(child.read_text())
        child.unlink()
        args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v["speedup_per_core"] for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
