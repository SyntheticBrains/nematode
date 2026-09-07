#!/usr/bin/env python
r"""Behavioural cloning: fit a connectome student's action mean to a recorded teacher.

The student is built from a config at a seed exactly as the simulation entry point builds
it, so its initial weights are the arm's at that seed. Each recorded observation is
reconstructed as ``BrainParams``, preprocessed through the student's own feature path,
unpacked into the batched feature tensors the PPO update uses, and run through the
topology's batched forward, whose first return in continuous mode is already the
readout-mapped Gaussian mean. That mean is squashed and rescaled exactly as sampling does,
and the loss is the mean squared error against the teacher's recorded action mean in the
action space -- bounded and scale-consistent, so a target the teacher saturated is matched
at the bound rather than chased to infinity.

Two parameter sets::

    plastic   the chemical weights alone, their update masked to the wiring, behind
              whatever readout the config built (anatomical under the plastic configs)
    full      every parameter PPO trains: chemical weights, sensory gains, readout, noise

A seeded fraction of episodes is held out. The trainer reports the initial, final and
held-out losses and the set's norm change, refuses to save when the final loss is not
below the initial, and otherwise saves through ``save_weights`` with a ``clone.json``
beside the file. It never runs the environment: a clone's behaviour is measured by
running it.

Usage::

    uv run python scripts/campaigns/l4_behavioural_clone.py \\
        --config configs/scenarios/foraging_predator_thermal/<arm>.yml --seed 1 \\
        --rollouts campaigns/teacher/rollouts.jsonl --parameter-set plastic \\
        --out campaigns/clones/<arm>-seed1.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._policy import continuous_deterministic_action
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain
from quantumnematode.brain.arch.dtypes import DEFAULT_QUBITS, DEFAULT_SHOTS, BrainType, DeviceType
from quantumnematode.brain.rollouts import read_rollouts
from quantumnematode.brain.weights import save_weights
from quantumnematode.optimizers.gradient_methods import GradientCalculationMethod
from quantumnematode.optimizers.learning_rate import DynamicLearningRate
from quantumnematode.utils.brain_factory import setup_brain_model
from quantumnematode.utils.config_loader import (
    ParameterInitializerConfig,
    configure_brain,
    load_simulation_config,
)

PARAMETER_SETS: tuple[str, ...] = ("plastic", "full")
SAVED_COMPONENTS: set[str] = {"topology", "training_state"}


def build_student(
    config_path: Path,
    seed: int,
    device: DeviceType = DeviceType.CPU,
) -> ConnectomePPOBrain:
    """Build the brain the way the simulation entry point does, at ``seed``."""
    config = load_simulation_config(str(config_path))
    if config.brain is None:
        msg = f"{config_path} configures no brain"
        raise ValueError(msg)
    brain_config = configure_brain(config).model_copy(update={"seed": seed})
    brain = setup_brain_model(
        brain_type=BrainType(config.brain.name),
        brain_config=brain_config,
        shots=config.shots if config.shots is not None else DEFAULT_SHOTS,
        qubits=config.qubits if config.qubits is not None else DEFAULT_QUBITS,
        device=device,
        learning_rate=DynamicLearningRate(),
        gradient_method=GradientCalculationMethod.RAW,
        gradient_max_norm=None,
        parameter_initializer_config=ParameterInitializerConfig(),
    )
    if not isinstance(brain, ConnectomePPOBrain):
        msg = f"behavioural cloning targets the connectome brain; {config_path} builds {type(brain).__name__}"
        raise TypeError(msg)
    if not brain.continuous:
        msg = "behavioural cloning needs a continuous-action student"
        raise ValueError(msg)
    return brain


def load_dataset(
    rows: list[dict[str, Any]],
    brain: ConnectomePPOBrain,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """States through the student's preprocessing, teacher action means, episode ids."""
    states, targets, episodes = [], [], []
    for row in rows:
        if row.get("action_mean") is None:
            msg = "a recorded step carries no action mean; record the teacher with a brain that reports one"
            raise ValueError(msg)
        params = BrainParams.model_validate(row["params"])
        states.append(brain.preprocess(params))
        targets.append(row["action_mean"])
        episodes.append(int(row["episode"]))
    return (
        torch.from_numpy(np.stack(states).astype(np.float32)).to(brain.device),
        torch.tensor(targets, dtype=torch.float32, device=brain.device),
        np.asarray(episodes),
    )


def split_episodes(
    episodes: np.ndarray,
    holdout: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded held-out split by episode; the held-out set is empty when there is one episode."""
    ids = np.unique(episodes)
    n_held = round(len(ids) * holdout) if len(ids) > 1 else 0
    held = set(rng.choice(ids, size=n_held, replace=False).tolist()) if n_held else set()
    mask = np.array([e in held for e in episodes])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def student_mean(brain: ConnectomePPOBrain, states: torch.Tensor) -> torch.Tensor:
    """Compute the student's squashed, rescaled action mean for a batch of preprocessed states."""
    food, distal, mechano, zone_onehot, thermo = brain._unpack_state_batched(states)
    mean, _hidden = brain.topology.forward_with_hidden_batched(
        food,
        predator_distal_features=distal,
        predator_mechano_features=mechano,
        contact_zone_onehot=zone_onehot,
        thermotaxis_features=thermo,
    )
    return continuous_deterministic_action(mean, brain._action_low, brain._action_high)


def parameter_set(brain: ConnectomePPOBrain, name: str) -> list[torch.nn.Parameter]:
    """Return the tensors a parameter set trains."""
    if name == "plastic":
        return [brain.topology.w_chem]
    if name == "full":
        return list(brain.topology.learnable_parameters)
    msg = f"unknown parameter set {name!r}; choose from {PARAMETER_SETS}"
    raise ValueError(msg)


def _norm(params: list[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for p in params:
        total = total + torch.sum(p.detach().to(torch.float64) ** 2)
    return float(torch.sqrt(total).item())


def _loss(
    brain: ConnectomePPOBrain,
    states: torch.Tensor,
    targets: torch.Tensor,
    index: np.ndarray,
    batch: int,
) -> float:
    if len(index) == 0:
        return math.nan
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(index), batch):
            sel = torch.as_tensor(index[start : start + batch], device=states.device)
            total += torch.sum((student_mean(brain, states[sel]) - targets[sel]) ** 2).item()
    return total / (len(index) * targets.shape[1])


def clone(  # noqa: PLR0913
    brain: ConnectomePPOBrain,
    states: torch.Tensor,
    targets: torch.Tensor,
    train_index: np.ndarray,
    held_index: np.ndarray,
    *,
    parameter_set_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Fit the chosen parameter set; return the losses and the norm change."""
    params = parameter_set(brain, parameter_set_name)
    before = _norm(params)
    optimiser = torch.optim.Adam(params, lr=lr)
    mask = brain.topology.m_chem.to(brain.topology.w_chem.dtype)
    initial = _loss(brain, states, targets, train_index, batch_size)
    initial_held = _loss(brain, states, targets, held_index, batch_size)
    for _ in range(epochs):
        order = rng.permutation(train_index)
        for start in range(0, len(order), batch_size):
            sel = torch.as_tensor(order[start : start + batch_size], device=states.device)
            optimiser.zero_grad(set_to_none=True)
            loss = torch.mean((student_mean(brain, states[sel]) - targets[sel]) ** 2)
            loss.backward()
            if parameter_set_name == "plastic" and brain.topology.w_chem.grad is not None:
                brain.topology.w_chem.grad.mul_(mask)
            optimiser.step()
    after = _norm(params)
    return {
        "parameter_set": parameter_set_name,
        "initial_loss": initial,
        "final_loss": _loss(brain, states, targets, train_index, batch_size),
        "initial_held_out_loss": initial_held,
        "held_out_loss": _loss(brain, states, targets, held_index, batch_size),
        "norm_before": before,
        "norm_after": after,
        "n_train": len(train_index),
        "n_held_out": len(held_index),
    }


def main(argv: list[str] | None = None) -> int:
    """Clone from the command line; return the exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--parameter-set", choices=PARAMETER_SETS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--holdout", type=float, default=0.2)
    args = ap.parse_args(argv)

    rows = read_rollouts(args.rollouts)
    if not rows:
        print(f"error: {args.rollouts} holds no steps", file=sys.stderr)
        return 2
    brain = build_student(args.config, args.seed)
    states, targets, episodes = load_dataset(rows, brain)
    rng = np.random.default_rng(args.seed)
    train_index, held_index = split_episodes(episodes, args.holdout, rng)
    torch.manual_seed(args.seed)
    report = clone(
        brain,
        states,
        targets,
        train_index,
        held_index,
        parameter_set_name=args.parameter_set,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        rng=rng,
    )
    for key, value in report.items():
        print(f"  {key}: {value}")
    if not report["final_loss"] < report["initial_loss"]:
        print(
            "error: the clone did not improve on its initial loss; nothing saved",
            file=sys.stderr,
        )
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_weights(brain, args.out, components=SAVED_COMPONENTS)
    record = {
        "config": str(args.config),
        "seed": args.seed,
        "rollouts": str(args.rollouts),
        "rollouts_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
        "n_steps": len(rows),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "holdout": args.holdout,
        **report,
    }
    args.out.with_name(
        "clone.json" if args.out.name == "clone.pt" else f"{args.out.stem}.clone.json",
    ).write_text(
        json.dumps(record, indent=2),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
