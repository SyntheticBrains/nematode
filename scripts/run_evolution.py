"""Run evolutionary optimization for quantum brain parameters.

This script uses population-based search (CMA-ES or GA) to find optimal
quantum circuit parameters, bypassing noisy gradient-based learning.

Example usage:
    python scripts/run_evolution.py --config configs/evolution.yml --generations 50
    python scripts/run_evolution.py --config configs/evolution.yml --algorithm ga --population 50
"""

import argparse
import json
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.modular import ModularBrain, ModularBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import DynamicForagingEnvironment
from quantumnematode.logging_config import logger
from quantumnematode.optimizers.evolutionary import (
    CMAESOptimizer,
    EvolutionaryOptimizer,
    EvolutionResult,
    GeneticAlgorithmOptimizer,
)
from quantumnematode.utils.config_loader import load_simulation_config

DEFAULT_GENERATIONS = 50
DEFAULT_POPULATION_SIZE = 20
DEFAULT_EPISODES_PER_EVAL = 15
DEFAULT_SIGMA0 = 0.5
DEFAULT_PARALLEL_WORKERS = 1


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evolutionary optimization for quantum brain parameters.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help=f"Number of generations to evolve (default: {DEFAULT_GENERATIONS}).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help=f"Population size (default: {DEFAULT_POPULATION_SIZE}).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES_PER_EVAL,
        help=f"Episodes per fitness evaluation (default: {DEFAULT_EPISODES_PER_EVAL}).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="cmaes",
        choices=["cmaes", "ga"],
        help="Evolutionary algorithm to use (default: cmaes).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA0,
        help=f"Initial step size / mutation std (default: {DEFAULT_SIGMA0}).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_PARALLEL_WORKERS}).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint file to resume from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evolution_results",
        help="Directory to save results (default: evolution_results).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    return parser.parse_args()


def create_brain_from_config(
    config_path: str,
    param_array: list[float] | None = None,
) -> ModularBrain:
    """Create a ModularBrain from config, optionally with specific parameters.

    Args:
        config_path: Path to YAML config file.
        param_array: Optional flat array of parameter values.

    Returns
    -------
        Configured ModularBrain instance.
    """
    config = load_simulation_config(config_path)

    # Extract brain config
    brain_config = ModularBrainConfig()
    if config.brain and config.brain.config:
        brain_dict = config.brain.config
        if "modules" in brain_dict:
            brain_config.modules = {ModuleName(k): v for k, v in brain_dict["modules"].items()}
        if "num_layers" in brain_dict:
            brain_config.num_layers = brain_dict["num_layers"]

    # Create brain with no learning (we're using evolution)
    brain = ModularBrain(
        config=brain_config,
        device=DeviceType.CPU,
        shots=config.shots or 1024,
    )

    # Set parameters if provided
    if param_array is not None:
        param_keys = list(brain.parameter_values.keys())
        if len(param_array) != len(param_keys):
            msg = f"Parameter array length {len(param_array)} != expected {len(param_keys)}"
            raise ValueError(msg)
        for i, key in enumerate(param_keys):
            brain.parameter_values[key] = param_array[i]

    return brain


def create_env_from_config(config_path: str) -> DynamicForagingEnvironment:
    """Create environment from config.

    Args:
        config_path: Path to YAML config file.

    Returns
    -------
        Configured DynamicForagingEnvironment instance.
    """
    config = load_simulation_config(config_path)

    # Extract environment config
    env_config = config.environment
    if env_config is None or env_config.type != "dynamic":
        msg = "Evolution requires a dynamic foraging environment"
        raise ValueError(msg)

    dynamic_config = env_config.dynamic
    if dynamic_config is None:
        msg = "Dynamic environment config is required"
        raise ValueError(msg)

    foraging_config = dynamic_config.get_foraging_config()
    predator_config = dynamic_config.get_predator_config()

    return DynamicForagingEnvironment(
        grid_size=dynamic_config.grid_size,
        foods_on_grid=foraging_config.foods_on_grid,
        target_foods_to_collect=foraging_config.target_foods_to_collect,
        min_food_distance=foraging_config.min_food_distance,
        agent_exclusion_radius=foraging_config.agent_exclusion_radius,
        gradient_decay_constant=foraging_config.gradient_decay_constant,
        gradient_strength=foraging_config.gradient_strength,
        viewport_size=dynamic_config.viewport_size,
        predators_enabled=predator_config.enabled,
        num_predators=predator_config.count,
        predator_speed=predator_config.speed,
        predator_detection_radius=predator_config.detection_radius,
        predator_kill_radius=predator_config.kill_radius,
        predator_gradient_decay=predator_config.gradient_decay_constant,
        predator_gradient_strength=predator_config.gradient_strength,
    )


def evaluate_fitness(
    param_array: list[float],
    config_path: str,
    episodes: int,
    max_steps: int = 200,
) -> float:
    """Evaluate fitness of a parameter set.

    Runs episodes and returns negative success rate (for minimization).

    Args:
        param_array: Flat array of parameter values.
        config_path: Path to YAML config file.
        episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.

    Returns
    -------
        Negative success rate (lower is better).
    """
    from quantumnematode.brain.arch import BrainParams

    brain = create_brain_from_config(config_path, param_array)
    successes = 0

    for _ in range(episodes):
        env = create_env_from_config(config_path)
        state = env.reset()

        for _ in range(max_steps):
            # Convert state to BrainParams
            params = BrainParams(
                gradient_strength=state.get("gradient_strength"),
                gradient_direction=state.get("gradient_direction"),
                agent_direction=state.get("agent_direction"),
                agent_position=state.get("agent_position"),
                food_gradient_strength=state.get("food_gradient_strength"),
                food_gradient_direction=state.get("food_gradient_direction"),
                predator_gradient_strength=state.get("predator_gradient_strength"),
                predator_gradient_direction=state.get("predator_gradient_direction"),
                satiety=state.get("satiety"),
            )

            # Get action (no reward, no learning)
            actions = brain.run_brain(params, reward=None)
            if not actions:
                break

            action = actions[0].action
            state, _reward, done, info = env.step(action)

            if done:
                reason = info.get("termination_reason", "")
                if reason == "target_reached" or info.get("success", False):
                    successes += 1
                break

    success_rate = successes / episodes
    return -success_rate  # Negate for minimization


def run_evolution(  # noqa: PLR0913
    optimizer: EvolutionaryOptimizer,
    config_path: str,
    generations: int,
    episodes: int,
    output_dir: Path,
    parallel_workers: int = 1,
) -> EvolutionResult:
    """Run evolutionary optimization loop.

    Args:
        optimizer: Evolutionary optimizer instance.
        config_path: Path to YAML config file.
        generations: Number of generations to run.
        episodes: Episodes per fitness evaluation.
        output_dir: Directory to save results.
        parallel_workers: Number of parallel workers.

    Returns
    -------
        EvolutionResult with best parameters.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evolution for {generations} generations")
    logger.info(f"Population size: {optimizer.population_size}")
    logger.info(f"Episodes per evaluation: {episodes}")
    logger.info(f"Parallel workers: {parallel_workers}")

    start_time = time.time()

    for gen in range(generations):
        gen_start = time.time()

        # Get candidate solutions
        solutions = optimizer.ask()

        # Evaluate fitness
        if parallel_workers > 1:
            from multiprocessing import Pool

            eval_args = [(sol, config_path, episodes) for sol in solutions]
            with Pool(processes=parallel_workers) as pool:
                fitnesses = pool.starmap(evaluate_fitness, eval_args)
        else:
            fitnesses = [evaluate_fitness(sol, config_path, episodes) for sol in solutions]

        # Report fitness to optimizer
        optimizer.tell(solutions, fitnesses)

        # Log progress
        gen_time = time.time() - gen_start
        best_fitness = min(fitnesses)
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)

        # Convert to success rate for readability
        best_success = -best_fitness
        mean_success = -mean_fitness

        logger.info(
            f"Gen {gen + 1}/{generations}: "
            f"best={best_success:.1%}, mean={mean_success:.1%}, "
            f"std={std_fitness:.3f}, time={gen_time:.1f}s",
        )

        # Save checkpoint every 10 generations
        if (gen + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_gen{gen + 1}.pkl"
            save_checkpoint(optimizer, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Check for convergence (CMA-ES only)
        if optimizer.stop():
            logger.info(f"Optimizer converged at generation {gen + 1}")
            break

    total_time = time.time() - start_time
    result = optimizer.result

    logger.info(f"Evolution complete in {total_time:.1f}s")
    logger.info(f"Best success rate: {-result.best_fitness:.1%}")
    logger.info(f"Best parameters: {result.best_params}")

    return result


def save_checkpoint(optimizer: EvolutionaryOptimizer, path: Path) -> None:
    """Save optimizer state to checkpoint file."""
    with path.open("wb") as f:
        pickle.dump(optimizer, f)


def load_checkpoint(path: Path) -> EvolutionaryOptimizer:
    """Load optimizer state from checkpoint file."""
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def save_results(
    result: EvolutionResult,
    config_path: str,
    output_dir: Path,
    timestamp: str,
) -> None:
    """Save evolution results to files."""
    # Save best parameters as JSON
    brain = create_brain_from_config(config_path)
    param_keys = list(brain.parameter_values.keys())
    best_params_dict = dict(zip(param_keys, result.best_params, strict=False))

    results_file = output_dir / f"best_params_{timestamp}.json"
    with Path.open(results_file, "w") as f:
        json.dump(
            {
                "best_params": best_params_dict,
                "best_success_rate": -result.best_fitness,
                "generations": result.generations,
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved best parameters: {results_file}")

    # Save history as CSV
    history_file = output_dir / f"history_{timestamp}.csv"
    with Path.open(history_file, "w") as f:
        f.write("generation,best_fitness,mean_fitness,std_fitness\n")
        f.writelines(
            f"{entry['generation']},"
            f"{entry['best_fitness']},"
            f"{entry['mean_fitness']},"
            f"{entry['std_fitness']}\n"
            for entry in result.history
        )
    logger.info(f"Saved history: {history_file}")


def main() -> None:
    """Run evolutionary optimization."""
    args = parse_arguments()

    # Configure logging
    logger.setLevel(args.log_level)

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Create output directory
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Session ID: {timestamp}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {output_dir}")

    # Get number of parameters from brain
    brain = create_brain_from_config(args.config)
    num_params = len(brain.parameter_values)
    logger.info(f"Number of parameters: {num_params}")

    # Get initial parameters (current values from brain)
    x0 = list(brain.parameter_values.values())

    # Create or load optimizer
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        optimizer = load_checkpoint(Path(args.resume))
    elif args.algorithm == "cmaes":
        optimizer = CMAESOptimizer(
            num_params=num_params,
            x0=x0,
            population_size=args.population,
            sigma0=args.sigma,
        )
    else:
        optimizer = GeneticAlgorithmOptimizer(
            num_params=num_params,
            population_size=args.population,
            sigma0=args.sigma,
            seed=args.seed,
        )

    logger.info(f"Using algorithm: {args.algorithm.upper()}")

    # Run evolution
    result = run_evolution(
        optimizer=optimizer,
        config_path=args.config,
        generations=args.generations,
        episodes=args.episodes,
        output_dir=output_dir,
        parallel_workers=args.parallel,
    )

    # Save results
    save_results(result, args.config, output_dir, timestamp)

    # Print final summary
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best success rate: {-result.best_fitness:.1%}")
    print(f"Generations: {result.generations}")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
