#!/usr/bin/env python3
"""Quantum Plasticity Test — Sequential multi-objective training evaluation.

Tests whether PQC (Parameterized Quantum Circuit) unitarity prevents catastrophic
forgetting during sequential multi-objective training compared to classical networks.

Protocol: Train on A (foraging) → B (pursuit predators) → C (thermotaxis+pursuit) → A'
(foraging return), measuring backward forgetting, forward transfer, and plasticity
retention at each transition.

Single architecture per invocation. Cross-architecture comparison via
scripts/compare_plasticity_results.py.

Usage:
    uv run python scripts/run_plasticity_test.py --config artifacts/logbooks/008/plasticity/qrh_plasticity.yml
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from quantumnematode.agent import QuantumNematodeAgent, SatietyConfig
from quantumnematode.brain.arch.dtypes import BrainType
from quantumnematode.env.theme import Theme
from quantumnematode.logging_config import configure_file_logging, logger
from quantumnematode.plasticity import (
    EvalResult,
    PhaseTrainingResult,
    SeedResult,
    compute_seed_metrics,
    restore_brain_state,
    snapshot_brain_state,
)
from quantumnematode.plasticity.metrics import _get_eval_score
from quantumnematode.plasticity.snapshot import save_checkpoint
from quantumnematode.utils.brain_factory import setup_brain_model
from quantumnematode.utils.config_loader import (
    PlasticityConfig,
    configure_brain,
    configure_gradient_method,
    configure_learning_rate,
    create_env_from_config,
    load_plasticity_config,
)
from quantumnematode.utils.seeding import derive_run_seed, set_global_seed
from quantumnematode.utils.session import generate_session_id
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from quantumnematode.agent import RewardConfig
    from quantumnematode.brain.arch._brain import Brain
    from quantumnematode.utils.config_loader import PlasticityPhaseConfig

console = Console()


# ---------------------------------------------------------------------------
# Agent / environment construction
# ---------------------------------------------------------------------------


def build_agent_for_phase(
    brain: Brain,
    phase_config: PlasticityPhaseConfig,
    seed: int,
    body_length: int = 2,
) -> tuple[QuantumNematodeAgent, RewardConfig, int]:
    """Construct a fresh agent for a phase, reusing the existing brain."""
    env = create_env_from_config(
        phase_config.environment,
        seed=seed,
        max_body_length=body_length,
        theme=Theme.ASCII,
    )
    satiety_config = (
        SatietyConfig(**phase_config.satiety.model_dump())
        if phase_config.satiety
        else SatietyConfig()
    )
    agent = QuantumNematodeAgent(
        brain=brain,
        env=env,
        max_body_length=body_length,
        theme=Theme.ASCII,
        satiety_config=satiety_config,
    )
    reward_config = phase_config.reward
    max_steps = phase_config.max_steps
    return agent, reward_config, max_steps


# ---------------------------------------------------------------------------
# Training phase execution
# ---------------------------------------------------------------------------


def run_training_phase(
    brain: Brain,
    phase_config: PlasticityPhaseConfig,
    num_episodes: int,
    seed: int,
    body_length: int = 2,
) -> PhaseTrainingResult:
    """Run training episodes for a single phase."""
    from quantumnematode.report.dtypes import TerminationReason

    result = PhaseTrainingResult(phase_name=phase_config.name)

    for ep in range(num_episodes):
        run_seed = derive_run_seed(seed, ep)
        set_global_seed(run_seed)

        agent, reward_config, max_steps = build_agent_for_phase(
            brain,
            phase_config,
            run_seed,
            body_length,
        )

        step_result = agent.run_episode(
            reward_config=reward_config,
            max_steps=max_steps,
            render_text="",
            show_last_frame_only=True,
        )

        success = step_result.termination_reason in (
            TerminationReason.GOAL_REACHED,
            TerminationReason.COMPLETED_ALL_FOOD,
        )
        result.episode_successes.append(success)
        result.episode_rewards.append(agent._episode_tracker.rewards)  # noqa: SLF001
        result.episode_steps.append(agent._episode_tracker.steps)  # noqa: SLF001

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            recent_sr = np.mean(result.episode_successes[max(0, ep - 19) : ep + 1])
            logger.info(
                f"  [{phase_config.name}] Episode {ep + 1}/{num_episodes}: "
                f"SR(last20)={recent_sr:.1%}",
            )

    return result


# ---------------------------------------------------------------------------
# Evaluation block
# ---------------------------------------------------------------------------


def run_eval_block(  # noqa: PLR0913
    brain: Brain,
    phase_config: PlasticityPhaseConfig,
    num_episodes: int,
    seed: int,
    transition_point: str,
    body_length: int = 2,
) -> EvalResult:
    """Run evaluation episodes with state snapshot/restore.

    Brain state is saved before eval and restored after, so any learning
    that occurs during eval episodes leaves no trace.
    """
    from quantumnematode.report.dtypes import TerminationReason

    snapshot = snapshot_brain_state(brain)

    successes: list[bool] = []
    rewards: list[float] = []
    steps: list[int] = []

    try:
        for ep in range(num_episodes):
            run_seed = derive_run_seed(seed + 10000, ep)  # Offset to avoid seed collision
            set_global_seed(run_seed)

            agent, reward_config, max_steps = build_agent_for_phase(
                brain,
                phase_config,
                run_seed,
                body_length,
            )

            step_result = agent.run_episode(
                reward_config=reward_config,
                max_steps=max_steps,
                render_text="",
                show_last_frame_only=True,
            )

            success = step_result.termination_reason in (
                TerminationReason.GOAL_REACHED,
                TerminationReason.COMPLETED_ALL_FOOD,
            )
            successes.append(success)
            rewards.append(agent._episode_tracker.rewards)  # noqa: SLF001
            steps.append(agent._episode_tracker.steps)  # noqa: SLF001
    finally:
        # Always restore brain to pre-eval state, even on exception
        restore_brain_state(brain, snapshot)

    return EvalResult(
        objective_name=phase_config.name,
        transition_point=transition_point,
        mean_success_rate=float(np.mean(successes)),
        mean_reward=float(np.mean(rewards)),
        mean_steps=float(np.mean(steps)),
    )


def run_eval_matrix(  # noqa: PLR0913
    brain: Brain,
    phase_configs: dict[str, PlasticityPhaseConfig],
    num_episodes: int,
    seed: int,
    transition_point: str,
    body_length: int = 2,
) -> list[EvalResult]:
    """Run the evaluation matrix for a given transition point.

    Matrix:
        Pre-training:  A, B
        Post-A:        A, B
        Post-B:        A, B
        Post-C:        A, C
        Post-A':       A
    """
    objectives_by_transition: dict[str, list[str]] = {
        "pre_training": ["foraging", "pursuit_predators"],
        "post_A": ["foraging", "pursuit_predators"],
        "post_B": ["foraging", "pursuit_predators"],
        "post_C": ["foraging", "thermotaxis_pursuit"],
        "post_A_prime": ["foraging"],
    }

    objectives = objectives_by_transition.get(transition_point, ["foraging"])
    results: list[EvalResult] = []

    for obj_name in objectives:
        if obj_name not in phase_configs:
            continue
        result = run_eval_block(
            brain=brain,
            phase_config=phase_configs[obj_name],
            num_episodes=num_episodes,
            seed=seed,
            transition_point=transition_point,
            body_length=body_length,
        )
        results.append(result)
        logger.info(
            f"  Eval [{transition_point}] on {obj_name}: "
            f"SR={result.mean_success_rate:.1%}, R={result.mean_reward:.1f}",
        )

    return results


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def write_seed_csv(
    seed_result: SeedResult,
    output_dir: Path,
) -> None:
    """Write per-seed phase results CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase_results.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Training metrics
        writer.writerow(["type", "phase", "episode", "success", "reward", "steps"])
        for phase in seed_result.training_results:
            for i, (s, r, st) in enumerate(
                zip(
                    phase.episode_successes,
                    phase.episode_rewards,
                    phase.episode_steps,
                    strict=True,
                ),
            ):
                writer.writerow(["training", phase.phase_name, i + 1, s, f"{r:.4f}", st])

        # Eval metrics
        writer.writerow([])
        writer.writerow(
            [
                "type",
                "transition",
                "objective",
                "mean_success_rate",
                "mean_reward",
                "mean_steps",
            ],
        )
        for ev in seed_result.eval_results:
            writer.writerow(
                [
                    "eval",
                    ev.transition_point,
                    ev.objective_name,
                    f"{ev.mean_success_rate:.4f}",
                    f"{ev.mean_reward:.4f}",
                    f"{ev.mean_steps:.1f}",
                ],
            )

    logger.info(f"Wrote seed CSV: {csv_path}")


def write_aggregate_csv(
    seed_results: list[SeedResult],
    brain_name: str,
    output_path: Path,
) -> None:
    """Write aggregate metrics CSV across seeds."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bfs = [s.backward_forgetting for s in seed_results if s.backward_forgetting is not None]
    fts = [s.forward_transfer for s in seed_results if s.forward_transfer is not None]
    prs = [s.plasticity_retention for s in seed_results if s.plasticity_retention is not None]

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["architecture", "metric", "mean", "std", "n_seeds", "values"])
        if bfs:
            writer.writerow(
                [
                    brain_name,
                    "backward_forgetting",
                    f"{np.mean(bfs):.4f}",
                    f"{np.std(bfs):.4f}",
                    len(bfs),
                    ";".join(f"{v:.4f}" for v in bfs),
                ],
            )
        if fts:
            writer.writerow(
                [
                    brain_name,
                    "forward_transfer",
                    f"{np.mean(fts):.4f}",
                    f"{np.std(fts):.4f}",
                    len(fts),
                    ";".join(f"{v:.4f}" for v in fts),
                ],
            )
        if prs:
            writer.writerow(
                [
                    brain_name,
                    "plasticity_retention",
                    f"{np.mean(prs):.4f}",
                    f"{np.std(prs):.4f}",
                    len(prs),
                    ";".join(f"{v:.4f}" for v in prs),
                ],
            )

        # Per-phase eval scores at key transitions
        for transition in ("post_A", "post_B", "post_C", "post_A_prime"):
            for objective in ("foraging", "pursuit_predators", "thermotaxis_pursuit"):
                scores = []
                for sr in seed_results:
                    score = _get_eval_score(sr.eval_results, objective, transition)
                    if score is not None:
                        scores.append(score)
                if scores:
                    writer.writerow(
                        [
                            brain_name,
                            f"eval_{transition}_{objective}",
                            f"{np.mean(scores):.4f}",
                            f"{np.std(scores):.4f}",
                            len(scores),
                            ";".join(f"{v:.4f}" for v in scores),
                        ],
                    )

    logger.info(f"Wrote aggregate CSV: {output_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(
    seed_results: list[SeedResult],
    brain_name: str,
) -> None:
    """Print Rich summary table to console."""
    bfs = [s.backward_forgetting for s in seed_results if s.backward_forgetting is not None]
    fts = [s.forward_transfer for s in seed_results if s.forward_transfer is not None]
    prs = [s.plasticity_retention for s in seed_results if s.plasticity_retention is not None]

    table = Table(title=f"Plasticity Results: {brain_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("N", justify="right")

    if bfs:
        table.add_row(
            "Backward Forgetting",
            f"{np.mean(bfs):.4f}",
            f"{np.std(bfs):.4f}",
            str(len(bfs)),
        )
    if fts:
        table.add_row(
            "Forward Transfer",
            f"{np.mean(fts):.4f}",
            f"{np.std(fts):.4f}",
            str(len(fts)),
        )
    if prs:
        table.add_row(
            "Plasticity Retention",
            f"{np.mean(prs):.4f}",
            f"{np.std(prs):.4f}",
            str(len(prs)),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------


def run_plasticity_protocol(config: PlasticityConfig) -> list[SeedResult]:
    """Execute the full plasticity evaluation protocol."""
    protocol = config.plasticity

    # Build phase config lookup (by name)
    phase_configs: dict[str, PlasticityPhaseConfig] = {p.name: p for p in protocol.phases}

    # Resolve brain config
    from quantumnematode.utils.config_loader import SimulationConfig

    sim_config = SimulationConfig(
        brain=config.brain,
        learning_rate=config.learning_rate,
        gradient=config.gradient,
        parameter_initializer=config.parameter_initializer,
        shots=config.shots,
        qubits=config.qubits,
        modules=config.modules,
    )
    brain_config = configure_brain(sim_config)
    from quantumnematode.brain.arch.dtypes import BRAIN_NAME_ALIASES

    canonical_name = BRAIN_NAME_ALIASES.get(config.brain.name, config.brain.name)
    brain_name = canonical_name  # Use canonical name for all outputs
    brain_type = BrainType(canonical_name)

    # Resolve learning rate / gradient config
    learning_rate = configure_learning_rate(sim_config)
    from quantumnematode.optimizers.gradient_methods import GradientCalculationMethod

    gradient_method, gradient_max_norm = configure_gradient_method(
        GradientCalculationMethod.RAW,
        sim_config,
    )

    # Parameter initializer
    from quantumnematode.brain.arch.dtypes import DeviceType
    from quantumnematode.utils.config_loader import ParameterInitializerConfig

    parameter_initializer_config = config.parameter_initializer or ParameterInitializerConfig()

    session_id = generate_session_id()
    configure_file_logging(session_id)
    export_base = Path.cwd() / "exports" / session_id / "plasticity"

    all_seed_results: list[SeedResult] = []

    for seed_idx, seed in enumerate(protocol.seeds):
        console.print(
            f"\n[bold]Seed {seed_idx + 1}/{len(protocol.seeds)}: {seed}[/bold]",
        )

        # Construct fresh brain for this seed (deepcopy config to avoid mutation)
        seed_brain_config = deepcopy(brain_config)
        seed_brain_config.seed = seed  # type: ignore[attr-defined]
        set_global_seed(seed)

        brain = setup_brain_model(
            brain_type=brain_type,
            brain_config=seed_brain_config,
            shots=config.shots or 1024,
            qubits=config.qubits or 8,
            device=DeviceType.CPU,
            learning_rate=learning_rate,
            gradient_method=gradient_method,
            gradient_max_norm=gradient_max_norm,
            parameter_initializer_config=parameter_initializer_config,
        )

        seed_result = SeedResult(seed=seed)
        seed_dir = export_base / f"seed_{seed}"
        checkpoint_dir = seed_dir

        # --- Pre-training eval ---
        console.print("  [dim]Eval: pre_training[/dim]")
        evals = run_eval_matrix(
            brain,
            phase_configs,
            protocol.eval_episodes,
            seed,
            "pre_training",
            config.body_length,
        )
        seed_result.eval_results.extend(evals)

        # --- Phase loop ---
        phase_order = ["foraging", "pursuit_predators", "thermotaxis_pursuit", "foraging_return"]
        transition_names = ["post_A", "post_B", "post_C", "post_A_prime"]

        for phase_name, transition_name in zip(phase_order, transition_names, strict=True):
            phase_cfg = phase_configs[phase_name]
            console.print(f"  [bold cyan]Training: {phase_name}[/bold cyan]")

            # Train
            training_result = run_training_phase(
                brain=brain,
                phase_config=phase_cfg,
                num_episodes=protocol.training_episodes_per_phase,
                seed=seed,
                body_length=config.body_length,
            )
            seed_result.training_results.append(training_result)

            # Checkpoint
            save_checkpoint(brain, checkpoint_dir, phase_name)

            # Eval
            console.print(f"  [dim]Eval: {transition_name}[/dim]")
            evals = run_eval_matrix(
                brain,
                phase_configs,
                protocol.eval_episodes,
                seed,
                transition_name,
                config.body_length,
            )
            seed_result.eval_results.extend(evals)

        # --- Compute metrics ---
        compute_seed_metrics(seed_result, protocol.convergence_threshold)
        console.print(
            f"  BF={seed_result.backward_forgetting}, "
            f"FT={seed_result.forward_transfer}, "
            f"PR={seed_result.plasticity_retention}",
        )

        # --- Write per-seed CSV ---
        write_seed_csv(seed_result, seed_dir)

        all_seed_results.append(seed_result)

    # --- Aggregate ---
    aggregate_path = export_base / "aggregate_metrics.csv"
    write_aggregate_csv(all_seed_results, brain_name, aggregate_path)
    print_summary(all_seed_results, brain_name)

    console.print(f"\n[green]Results saved to: {export_base}[/green]")
    return all_seed_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the plasticity test CLI."""
    parser = argparse.ArgumentParser(
        description="Quantum Plasticity Test — sequential multi-objective training evaluation",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to plasticity config YAML",
    )
    args = parser.parse_args()

    config = load_plasticity_config(args.config)

    console.print("[bold]Quantum Plasticity Test[/bold]")
    console.print(f"Architecture: {config.brain.name}")
    console.print(f"Phases: {[p.name for p in config.plasticity.phases]}")
    console.print(f"Training episodes/phase: {config.plasticity.training_episodes_per_phase}")
    console.print(f"Eval episodes: {config.plasticity.eval_episodes}")
    console.print(f"Seeds: {config.plasticity.seeds}")
    console.print(f"Convergence threshold: {config.plasticity.convergence_threshold}")

    run_plasticity_protocol(config)


if __name__ == "__main__":
    main()
