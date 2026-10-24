"""Shared machinery for the batched evaluator's parity tests: build, run both ways, compare."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from quantumnematode.env.hazard_controllers import ConstantHazards, PirouetteHazards
from quantumnematode.env.primitives import PRIMITIVE_KINDS
from quantumnematode.evaluation.batched import BatchedEvaluator, BatchedSenses
from quantumnematode.evaluation.controllers import (
    BatchedConnectome,
    BatchedConstantHazards,
    BatchedMLP,
    BatchedPirouetteHazards,
)
from quantumnematode.evaluation.reference import ReferenceEpisode, run_reference

__all__ = ["BatchedEvaluator", "run_reference"]
from quantumnematode.utils.config_loader import load_simulation_config

REPO = Path(__file__).resolve().parents[5]
FORAGING = REPO / "configs" / "scenarios" / "foraging"
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import body_kinematics_eval as harness  # noqa: E402  # pyright: ignore[reportMissingImports]

MLP_STEM = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives"
CONNECTOME_STEM = (
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_modulated_s1_primitives"
)
PROTOCOLS = (
    "scheer_lawn",
    "food_removal_lopezcruz",
    "food_removal_calhoun",
    "uniform_lawn_flavell",
)
CONTROLLERS = ("mlp", "connectome", "constant", "pirouette")
PARITY_SEEDS = (5001, 5002, 5003, 5004)
HAZARDS = (0.02, 0.01, 0.01)

FEATURE_TOLERANCE = 1e-5
# Float32 rounding grows around the closed sensory loop. On Scheer's lawn edge, where the odour
# gradient is steepest, the MLP's difference from the per-step runner grew about tenfold every 40
# steps: 1e-7 at step 0, 1e-5 by 180, 1e-3 by 228. So positions and headings are held to 1e-3 over
# the first 150 steps (worst case measured 1.6e-5), and bouts to exact identity over all 300.
POSITION_TOLERANCE = 1e-3
POSITION_HORIZON = 150
SATIETY_TOLERANCE = 1e-6


def config_path(controller: str, protocol: str) -> Path:
    """Return the assay config a controller runs on: the connectome's, or the MLP's for the rest."""
    stem = CONNECTOME_STEM if controller == "connectome" else MLP_STEM
    return FORAGING / f"{stem}_assay_{protocol}.yml"


def build(controller: str, protocol: str) -> tuple[Any, Any, Any]:
    """Return the config, the per-step controller and its batched form."""
    path = config_path(controller, protocol)
    config = load_simulation_config(str(path))
    if controller == "constant":
        return config, ConstantHazards(0.15, HAZARDS), BatchedConstantHazards(0.15, HAZARDS)
    if controller == "pirouette":
        return (
            config,
            PirouetteHazards(0.15, HAZARDS, beta=3.0),
            BatchedPirouetteHazards(0.15, HAZARDS, beta=3.0),
        )
    brain, _, _ = harness.build_brain(path, PARITY_SEEDS[0])
    batched = BatchedMLP(brain) if controller == "mlp" else BatchedConnectome(brain)
    return config, brain, batched


@dataclass
class BatchedLog:
    """What the batched run saw each step, for the state-level comparison."""

    features: list[np.ndarray]
    means: list[np.ndarray]
    derivatives: list[np.ndarray]
    densities: list[np.ndarray]


def _squash(mean: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> np.ndarray:
    return ((high + low) / 2.0 + (high - low) / 2.0 * torch.tanh(mean)).double().numpy()


def run_batched(
    config: Any,
    batched: Any,
    seeds: tuple[int, ...] | list[int],
    steps: int,
    *,
    tamper: Any = None,
) -> tuple[Any, BatchedLog]:
    """Run the evaluator for ``steps``, logging features, squashed means and densities per step."""
    log = BatchedLog([], [], [], [])
    evaluator = BatchedEvaluator(config.model_copy(update={"max_steps": steps}), batched, seeds)
    if tamper is not None:
        build = evaluator._build_animals

        def build_then_tamper() -> None:
            build()
            tamper(evaluator)

        evaluator._build_animals = build_then_tamper  # type: ignore[method-assign]

    def observe(_t: int, ev: BatchedEvaluator, senses: BatchedSenses) -> None:
        log.derivatives.append(senses.derivative.copy())
        log.densities.append(ev.density.copy() if ev.lawn_on.any() else np.zeros((len(seeds), 0)))
        if isinstance(batched, BatchedMLP):
            features = batched.features(senses)
            with torch.no_grad():
                mean = batched.brain.actor(torch.from_numpy(features))
            brain = batched.brain
        elif isinstance(batched, BatchedConnectome):
            features = batched.state(senses)
            brain = batched.brain
            states = torch.from_numpy(features)
            with torch.no_grad():
                mean, _ = brain.topology.forward_with_hidden_batched(
                    states[:, : brain.topology.n_food_features],
                    modulators=brain._unpack_modulators(states),
                )
        else:
            log.means.append(batched.act(senses, None))
            return
        log.features.append(features.astype(np.float64))
        log.means.append(_squash(mean, brain._action_low, brain._action_high))

    trajectories = evaluator.run(observer=observe)
    return trajectories, log


def references(config: Any, controller: Any, steps: int) -> list[ReferenceEpisode]:
    """Run the per-step reference for every parity seed."""
    return [
        run_reference(config, controller, seed, steps, record_density=True) for seed in PARITY_SEEDS
    ]


def state_mismatches(refs: list[ReferenceEpisode], log: BatchedLog, steps: int) -> list[str]:
    """Return every state-level value that differs by more than the feature tolerance."""
    problems: list[str] = []
    for i, ref in enumerate(refs):
        for t in range(steps):
            if ref.features:
                diff = np.max(np.abs(ref.features[t] - log.features[t][i]))
                if diff > FEATURE_TOLERANCE:
                    problems.append(f"animal {i} step {t}: features differ by {diff:.2e}")
            diff = np.max(np.abs(ref.means[t] - log.means[t][i]))
            if diff > FEATURE_TOLERANCE:
                problems.append(f"animal {i} step {t}: action means differ by {diff:.2e}")
            diff = abs(ref.derivatives[t] - float(log.derivatives[t][i]))
            if diff > FEATURE_TOLERANCE:
                problems.append(f"animal {i} step {t}: derivative differs by {diff:.2e}")
    return problems


def trajectory_mismatches(
    refs: list[ReferenceEpisode],
    trajectories: Any,
    log: BatchedLog,
) -> list[str]:
    """Return every trajectory value outside its tolerance, and every bout that differs."""
    problems: list[str] = []
    for i, ref in enumerate(refs):
        for t, step in enumerate(ref.behaviour):
            position = math.hypot(step.x - trajectories.x[i, t], step.y - trajectories.y[i, t])
            heading = abs(math.remainder(step.heading_rad - trajectories.heading[i, t], math.tau))
            kind = PRIMITIVE_KINDS[int(trajectories.primitive[i, t])]
            if step.primitive != kind:
                problems.append(f"animal {i} step {t}: primitive {step.primitive} against {kind}")
                break
            if t < POSITION_HORIZON and (
                position > POSITION_TOLERANCE or heading > POSITION_TOLERANCE
            ):
                problems.append(
                    f"animal {i} step {t}: position off {position:.2e}, heading {heading:.2e}",
                )
                break
            if t >= POSITION_HORIZON:
                continue
            if abs((step.satiety or 0.0) - trajectories.satiety[i, t]) > SATIETY_TOLERANCE:
                problems.append(f"animal {i} step {t}: satiety differs")
                break
            if ref.densities[t].size and log.densities[t].size:
                diff = np.max(np.abs(ref.densities[t] - log.densities[t][i]))
                if diff > SATIETY_TOLERANCE:
                    problems.append(f"animal {i} step {t}: lawn density differs by {diff:.2e}")
                    break
    return problems
