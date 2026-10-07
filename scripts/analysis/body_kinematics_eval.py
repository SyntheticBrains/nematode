r"""Run a trained body-drive policy with posture capture and read the kinematic instruments.

The agent is rebuilt the way the simulation entry point builds it, from the run's config at its
seed, with the run's final weights loaded (or none, for the untrained policy) and learning frozen. Each evaluation episode is seeded
away from the training runs' seeds, captures the body's posture at every sub-step, and the
instruments read the captured episodes together. ``--substeps`` overrides the body's sub-step
count, which the half-step convergence check uses.

Usage::

    uv run python scripts/analysis/body_kinematics_eval.py --config <cfg.yml> --seed 1505 \\
        --weights exports/<session>/weights/final.pt --episodes 10 [--substeps 40] --out k.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quantumnematode.agent import DEFAULT_AGENT_BODY_LENGTH, DEFAULT_MAX_STEPS, QuantumNematodeAgent
from quantumnematode.brain.arch.dtypes import DEFAULT_QUBITS, DEFAULT_SHOTS, BrainType, DeviceType
from quantumnematode.brain.modules import ModuleName
from quantumnematode.brain.weights import load_weights
from quantumnematode.env.continuous_2d import Continuous2DEnvironment
from quantumnematode.env.theme import Theme
from quantumnematode.optimizers.gradient_methods import GradientCalculationMethod
from quantumnematode.optimizers.learning_rate import DynamicLearningRate
from quantumnematode.utils.brain_factory import setup_brain_model
from quantumnematode.utils.config_loader import (
    ParameterInitializerConfig,
    SensingConfig,
    SimulationConfig,
    apply_sensing_mode,
    configure_brain,
    configure_environment,
    configure_reward,
    configure_satiety,
    create_env_from_config,
    load_simulation_config,
    validate_sensing_config,
)
from quantumnematode.utils.seeding import derive_run_seed, get_rng, set_global_seed
from quantumnematode.validation.body_kinematics import Kinematics, in_bands, measure

if TYPE_CHECKING:
    from quantumnematode.brain.arch import Brain

# Evaluation episodes take run indices from here on, past any training campaign's run count, so
# their arenas are never ones the policy trained in.
EVALUATION_RUN_OFFSET = 1_000_000


def build_brain(config_path: Path, seed: int) -> tuple[Brain, SimulationConfig, SensingConfig]:
    """Build a run's brain, learning frozen, the way the simulation entry point builds it."""
    config = load_simulation_config(str(config_path))
    if config.brain is None:
        msg = f"{config_path} configures no brain"
        raise ValueError(msg)
    sensing_config = validate_sensing_config(configure_environment(config).get_sensing_config())
    brain_config = configure_brain(config).model_copy(update={"seed": seed, "freeze_updates": True})
    # The sensing mode rewrites the brain's sensory modules, as the simulation entry point does, so
    # the rebuilt network has the input width the weights were trained at.
    modules = getattr(brain_config, "sensory_modules", None)
    if modules is not None:
        translated = apply_sensing_mode([m.value for m in modules], sensing_config)
        brain_config = brain_config.model_copy(
            update={"sensory_modules": [ModuleName(m) for m in translated]},
        )
    brain = setup_brain_model(
        brain_type=BrainType(config.brain.name),
        brain_config=brain_config,
        shots=config.shots if config.shots is not None else DEFAULT_SHOTS,
        qubits=config.qubits if config.qubits is not None else DEFAULT_QUBITS,
        device=DeviceType.CPU,
        learning_rate=DynamicLearningRate(),
        gradient_method=GradientCalculationMethod.RAW,
        gradient_max_norm=None,
        parameter_initializer_config=ParameterInitializerConfig(),
    )
    return brain, config, sensing_config


def evaluate(  # noqa: PLR0913 - a run's identity and the evaluation's settings
    config_path: Path,
    seed: int,
    weights: Path | None,
    *,
    episodes: int = 10,
    substeps: int | None = None,
    wall_margin_mm: float = 1.0,
) -> Kinematics:
    """Return the instruments' readings for one run over ``episodes`` frozen episodes.

    ``weights`` of ``None`` reads the seed's untrained policy.
    """
    brain, config, sensing_config = build_brain(config_path, seed)
    environment_config = configure_environment(config)
    if weights is not None:
        load_weights(brain, weights)

    continuous = environment_config.continuous
    if continuous is None or continuous.body_model != "kinematic":
        msg = f"{config_path} does not run the kinematic body"
        raise ValueError(msg)
    if substeps is not None:
        continuous = continuous.model_copy(update={"body_substeps": substeps})
        environment_config = environment_config.model_copy(update={"continuous": continuous})
    body_length = config.body_length or DEFAULT_AGENT_BODY_LENGTH
    max_steps = config.max_steps or DEFAULT_MAX_STEPS
    reward_config = configure_reward(config)

    agent = QuantumNematodeAgent(
        brain=brain,
        env=create_env_from_config(
            environment_config,
            seed=derive_run_seed(seed, EVALUATION_RUN_OFFSET),
            max_body_length=body_length,
            theme=Theme.HEADLESS,
        ),
        max_body_length=body_length,
        theme=Theme.HEADLESS,
        satiety_config=configure_satiety(config),
        sensing_config=sensing_config,
    )
    captured: list[list[dict[str, Any]]] = []
    for episode in range(episodes):
        run_seed = derive_run_seed(seed, EVALUATION_RUN_OFFSET + episode)
        set_global_seed(run_seed)
        if episode:
            agent.env.seed = run_seed
            agent.env.rng = get_rng(run_seed)
            agent.reset_environment()
            agent.reset_brain()
        env = _body_env(agent)
        log: list[dict[str, Any]] = []
        env.posture_log = log  # type: ignore[assignment]
        agent.run_episode(reward_config, max_steps=max_steps)
        captured.append(log)

    env = _body_env(agent)
    body = env._body.params
    return measure(
        captured,
        world_size_mm=env.continuous.world_size_mm,
        body_length_mm=body.body_length_mm,
        step_seconds=body.step_seconds,
        reversal_threshold=body.reversal_threshold,
        wall_margin_mm=wall_margin_mm,
    )


def _body_env(agent: QuantumNematodeAgent) -> Continuous2DEnvironment:
    if not isinstance(agent.env, Continuous2DEnvironment):
        msg = "the kinematic body needs the continuous-2D environment"
        raise TypeError(msg)
    return agent.env


def main(argv: list[str] | None = None) -> int:
    """Evaluate one run and print, and optionally write, its readings as JSON."""
    parser = argparse.ArgumentParser(
        description="Read the kinematic instruments on a trained body-drive run.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="omit to read the untrained policy",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument("--wall-margin-mm", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    kinematics = evaluate(
        args.config,
        args.seed,
        args.weights,
        episodes=args.episodes,
        substeps=args.substeps,
        wall_margin_mm=args.wall_margin_mm,
    )
    summary = {"kinematics": asdict(kinematics), "in_bands": in_bands(kinematics)}
    text = json.dumps(summary, indent=2)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
