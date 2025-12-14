# pragma: no cover

"""Run evolutionary optimization for quantum brain parameters.

This script uses population-based search (CMA-ES or GA) to find optimal
quantum circuit parameters, bypassing noisy gradient-based learning.

Example usage:
    python scripts/run_evolution.py --config configs/evolution.yml --generations 50
    python scripts/run_evolution.py --config configs/evolution.yml --algorithm ga --population 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import random
import signal
import time
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.modular import ModularBrain, ModularBrainConfig
from quantumnematode.env import DynamicForagingEnvironment
from quantumnematode.logging_config import logger
from quantumnematode.optimizers.evolutionary import (
    CMAESOptimizer,
    EvolutionResult,
    GeneticAlgorithmOptimizer,
)
from quantumnematode.utils.config_loader import (
    configure_brain,
    configure_satiety,
    load_simulation_config,
)

if TYPE_CHECKING:
    from quantumnematode.optimizers.evolutionary import EvolutionaryOptimizer

import math

DEFAULT_GENERATIONS = 50
DEFAULT_POPULATION_SIZE = 20
DEFAULT_EPISODES_PER_EVAL = 15
# Default sigma covers [-π, π] range for quantum circuit parameters
# π/2 ≈ 1.57 means ~95% of initial samples fall within [-π, π]
DEFAULT_SIGMA0 = math.pi / 2
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
        help=f"Initial step size / mutation std (default: π/2 ≈ {DEFAULT_SIGMA0:.2f}, "
        "covers [-π, π] range for quantum params).",
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
        help="Path to checkpoint file to resume from. WARNING: Only use trusted files "
        "(uses pickle which can execute arbitrary code).",
    )
    parser.add_argument(
        "--init-params",
        type=str,
        help="Path to JSON file with initial parameters (e.g., best_params_*.json).",
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


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments for sensible values.

    Args:
        args: Parsed command-line arguments.

    Raises
    ------
        ValueError: If any argument has an invalid value.
    """
    if args.episodes <= 0:
        msg = "--episodes must be > 0"
        raise ValueError(msg)
    if args.generations <= 0:
        msg = "--generations must be > 0"
        raise ValueError(msg)
    if args.population <= 0:
        msg = "--population must be > 0"
        raise ValueError(msg)
    if args.parallel <= 0:
        msg = "--parallel must be > 0"
        raise ValueError(msg)
    if args.sigma <= 0:
        msg = "--sigma must be > 0"
        raise ValueError(msg)


def load_init_params(init_params_path: str, param_keys: list[str]) -> list[float]:
    """Load initial parameters from a JSON file.

    Supports two formats:
    1. best_params_*.json format: {"best_params": {"θ_rx1_0": 0.1, ...}, ...}
    2. Simple dict format: {"θ_rx1_0": 0.1, ...}

    Args:
        init_params_path: Path to JSON file with parameters.
        param_keys: Expected parameter names in order (from brain.parameter_values.keys()).

    Returns
    -------
        List of parameter values in the correct order.
    """
    # Load the JSON file
    with Path(init_params_path).open() as f:
        data = json.load(f)

    # Handle both formats
    params_dict = data.get("best_params", data)

    # Map loaded params to the expected order
    param_array = []
    for key in param_keys:
        if key in params_dict:
            param_array.append(float(params_dict[key]))
        else:
            msg = f"Parameter '{key}' not found in {init_params_path}"
            raise KeyError(msg)

    return param_array


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

    # Use configure_brain to get the properly typed brain config
    brain_config = configure_brain(config)

    # Ensure we have a ModularBrainConfig
    if not isinstance(brain_config, ModularBrainConfig):
        msg = f"Evolution requires ModularBrain, got {type(brain_config).__name__}"
        raise TypeError(msg)

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


def run_episode(  # noqa: PLR0913
    brain: ModularBrain,
    env: DynamicForagingEnvironment,
    max_steps: int,
    initial_satiety: float = 200.0,
    satiety_decay_rate: float = 1.0,
    satiety_gain_per_food: float = 0.2,
    *,
    use_separated_gradients: bool = False,
) -> bool:
    """Run a single episode and return whether it was successful.

    Args:
        brain: Brain instance to use for decisions.
        env: Environment instance.
        max_steps: Maximum steps per episode.
        initial_satiety: Starting satiety level.
        satiety_decay_rate: Satiety decay per step.
        satiety_gain_per_food: Fraction of initial_satiety restored per food.
        use_separated_gradients: Whether to compute separated food/predator gradients.

    Returns
    -------
        True if episode was successful (collected target foods).
    """
    brain.prepare_episode()
    success = False
    foods_collected = 0
    satiety = initial_satiety

    try:
        for _ in range(max_steps):
            # Get state from environment
            position = (env.agent_pos[0], env.agent_pos[1])
            gradient_strength, gradient_direction = env.get_state(position, disable_log=True)

            # Only compute separated gradients if needed (avoids duplicate gradient computation)
            if use_separated_gradients:
                separated_grads = env.get_separated_gradients(position, disable_log=True)
                food_gradient_strength = separated_grads.get("food_gradient_strength")
                food_gradient_direction = separated_grads.get("food_gradient_direction")
                predator_gradient_strength = separated_grads.get("predator_gradient_strength")
                predator_gradient_direction = separated_grads.get("predator_gradient_direction")
            else:
                food_gradient_strength = None
                food_gradient_direction = None
                predator_gradient_strength = None
                predator_gradient_direction = None

            # Build BrainParams
            params = BrainParams(
                gradient_strength=gradient_strength,
                gradient_direction=gradient_direction,
                food_gradient_strength=food_gradient_strength,
                food_gradient_direction=food_gradient_direction,
                predator_gradient_strength=predator_gradient_strength,
                predator_gradient_direction=predator_gradient_direction,
                satiety=satiety,
                agent_direction=env.current_direction,
                agent_position=(float(position[0]), float(position[1])),
            )

            # Get action from brain (no reward, no learning)
            actions = brain.run_brain(params, reward=None)
            if not actions:
                break

            action = actions[0].action

            # Move agent
            env.move_agent(action)

            # Check for food collection
            agent_pos_tuple = (env.agent_pos[0], env.agent_pos[1])
            if agent_pos_tuple in env.foods:
                env.foods.remove(agent_pos_tuple)
                foods_collected += 1
                satiety_gain = initial_satiety * satiety_gain_per_food
                satiety = min(satiety + satiety_gain, initial_satiety)
                env.spawn_food()  # Spawn new food

                # Check for success (collected target foods)
                if foods_collected >= env.target_foods_to_collect:
                    success = True
                    return True

            # Move predators and check for death
            if env.predators_enabled:
                env.update_predators()
                if env.check_predator_collision():
                    return False  # Died to predator

            # Decay satiety
            satiety -= satiety_decay_rate
            if satiety <= 0:
                return False  # Starved

        return False  # Max steps reached without success
    finally:
        brain.post_process_episode(episode_success=success)


def _derive_episode_seed(base_seed: int, gen: int, candidate_idx: int, episode: int) -> int:
    """Derive a deterministic seed for a specific episode.

    Uses BLAKE2b hash of (base_seed, gen, candidate_idx, episode) to produce
    independent, reproducible seeds for each episode evaluation.

    Note: We use hashlib.blake2b instead of Python's hash() because hash()
    is salted (randomized per process since Python 3.3), which would break
    reproducibility across multiprocessing workers and separate runs.

    Args:
        base_seed: Base seed from --seed argument.
        gen: Generation number.
        candidate_idx: Index of candidate in population.
        episode: Episode number within evaluation.

    Returns
    -------
        Deterministic seed in valid range for numpy.
    """
    # Stable hash independent of PYTHONHASHSEED and process
    payload = f"{base_seed}:{gen}:{candidate_idx}:{episode}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") & 0xFFFF_FFFF


def evaluate_fitness(  # noqa: PLR0913
    param_array: list[float],
    config_path: str,
    episodes: int,
    base_seed: int | None = None,
    gen: int = 0,
    candidate_idx: int = 0,
) -> float:
    """Evaluate fitness of a parameter set.

    Runs episodes and returns negative success rate (for minimization).

    Args:
        param_array: Flat array of parameter values.
        config_path: Path to YAML config file.
        episodes: Number of episodes to run.
        base_seed: Optional base seed for reproducibility. When provided,
            each episode gets a deterministic seed derived from
            (base_seed, gen, candidate_idx, episode). This seeds numpy RNG
            to ensure independent randomness across parallel workers. Note:
            environment uses secrets module for food/predator placement,
            which is not seedable by design.
        gen: Generation number (used for seed derivation).
        candidate_idx: Index of candidate in population (used for seed derivation).

    Returns
    -------
        Negative success rate (lower is better).
    """
    brain = create_brain_from_config(config_path, param_array)
    config = load_simulation_config(config_path)
    satiety_config = configure_satiety(config)

    # Get max_steps from config
    max_steps = config.max_steps or 500

    # Check if separated gradients are needed (for appetitive/aversive modules)
    use_separated_gradients = False
    if config.environment and config.environment.dynamic:
        use_separated_gradients = config.environment.dynamic.use_separated_gradients

    successes = 0

    for ep in range(episodes):
        # Seed RNGs for reproducibility when base_seed is provided
        if base_seed is not None:
            episode_seed = _derive_episode_seed(base_seed, gen, candidate_idx, ep)
            np.random.seed(episode_seed)  # noqa: NPY002
            random.seed(episode_seed)

        env = create_env_from_config(config_path)
        if run_episode(
            brain,
            env,
            max_steps,
            initial_satiety=satiety_config.initial_satiety,
            satiety_decay_rate=satiety_config.satiety_decay_rate,
            satiety_gain_per_food=satiety_config.satiety_gain_per_food,
            use_separated_gradients=use_separated_gradients,
        ):
            successes += 1

    success_rate = successes / episodes
    return -success_rate  # Negate for minimization


def _init_worker(log_level: int) -> None:
    """Initialize logging in worker processes and ignore SIGINT.

    Workers ignore SIGINT so the parent process handles Ctrl+C gracefully.
    This prevents worker processes from crashing with KeyboardInterrupt
    and spewing stack traces when the user interrupts.

    Args:
        log_level: Logging level to use in worker.
    """
    # Ignore SIGINT in workers - parent will handle it and terminate pool
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Import logger in worker to configure it
    from quantumnematode.logging_config import logger as worker_logger

    worker_logger.setLevel(log_level)
    for handler in worker_logger.handlers:
        handler.setLevel(log_level)


def _prompt_interrupt() -> str:
    """Prompt user for action after keyboard interrupt.

    Returns
    -------
        User choice: 'y' (save & exit), 'n' (exit without saving), 'c' (continue).
    """
    print("\n" + "=" * 60)
    print("INTERRUPTED - What would you like to do?")
    print("  [y] Save checkpoint and exit (default)")
    print("  [n] Exit without saving")
    print("  [c] Continue running")
    print("=" * 60)

    try:
        choice = input("Choice [y/n/c]: ").strip().lower()
        if choice in ("y", "n", "c", ""):
            return choice if choice else "y"
        print(f"Invalid choice '{choice}', defaulting to save & exit")
        return "y"  # noqa: TRY300
    except (EOFError, KeyboardInterrupt):
        # Second interrupt or EOF - force exit without saving
        print("\nForce exit requested")
        return "n"


def run_evolution(  # noqa: C901, PLR0912, PLR0913, PLR0915
    optimizer: EvolutionaryOptimizer,
    config_path: str,
    generations: int,
    episodes: int,
    output_dir: Path,
    parallel_workers: int = 1,
    log_level: int = logging.WARNING,
    base_seed: int | None = None,
) -> tuple[EvolutionResult, bool]:
    """Run evolutionary optimization loop.

    Args:
        optimizer: Evolutionary optimizer instance.
        config_path: Path to YAML config file.
        generations: Number of generations to run.
        episodes: Episodes per fitness evaluation.
        output_dir: Directory to save results.
        parallel_workers: Number of parallel workers.
        log_level: Logging level for worker processes.
        base_seed: Optional base seed for reproducible episode evaluation.
            When provided, each episode gets a deterministic seed.

    Returns
    -------
        Tuple of (EvolutionResult with best parameters, should_save flag).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evolution for {generations} generations")
    logger.info(f"Population size: {optimizer.population_size}")
    logger.info(f"Episodes per evaluation: {episodes}")
    logger.info(f"Parallel workers: {parallel_workers}")

    start_time = time.time()
    should_save = True
    interrupted = False
    session_best_fitness = float("inf")  # Track best fitness across all generations

    # Create pool once outside the loop to avoid process spawn overhead per generation
    pool = None
    if parallel_workers > 1:
        pool = Pool(
            processes=parallel_workers,
            initializer=_init_worker,
            initargs=(log_level,),
        )

    try:
        for gen in range(generations):
            gen_start = time.time()

            try:
                # Get candidate solutions
                solutions = optimizer.ask()

                # Evaluate fitness
                if pool is not None:
                    eval_args = [
                        (sol, config_path, episodes, base_seed, gen, idx)
                        for idx, sol in enumerate(solutions)
                    ]
                    fitnesses = pool.starmap(evaluate_fitness, eval_args)
                else:
                    fitnesses = [
                        evaluate_fitness(
                            sol,
                            config_path,
                            episodes,
                            base_seed=base_seed,
                            gen=gen,
                            candidate_idx=idx,
                        )
                        for idx, sol in enumerate(solutions)
                    ]

                # Report fitness to optimizer
                optimizer.tell(solutions, fitnesses)

            except KeyboardInterrupt:
                # Terminate pool immediately to stop workers
                if pool is not None:
                    pool.terminate()
                    pool.join()
                    pool = None  # Mark as terminated so finally block doesn't try again

                logger.warning(f"Interrupted during generation {gen + 1}")
                choice = _prompt_interrupt()

                if choice == "c":
                    # Recreate pool for continuing
                    if parallel_workers > 1:
                        pool = Pool(
                            processes=parallel_workers,
                            initializer=_init_worker,
                            initargs=(log_level,),
                        )
                    logger.info("Continuing evolution...")
                    continue
                if choice == "n":
                    should_save = False

                interrupted = True
                # Save checkpoint at interruption point
                if should_save:
                    checkpoint_path = output_dir / f"checkpoint_gen{gen}_interrupted.pkl"
                    save_checkpoint(optimizer, checkpoint_path)
                    logger.info(f"Saved interrupt checkpoint: {checkpoint_path}")
                break

            # Log progress
            gen_time = time.time() - gen_start
            best_fitness = min(fitnesses)
            best_idx = fitnesses.index(best_fitness)
            best_params_this_gen = solutions[best_idx]
            mean_fitness = float(np.mean(fitnesses))
            std_fitness = float(np.std(fitnesses))

            # Convert to success rate for readability
            best_success = -best_fitness
            mean_success = -mean_fitness

            logger.info(
                f"Gen {gen + 1:3d}/{generations}: "
                f"best={best_success:5.1%}, mean={mean_success:5.1%}, "
                f"std={std_fitness:.3f}, time={gen_time:5.1f}s",
            )

            # Log best parameters for this generation at debug level
            logger.debug(f"Gen {gen + 1} best params: {best_params_this_gen}")

            # Check if this is a new session best
            if best_fitness < session_best_fitness:
                session_best_fitness = best_fitness
                logger.info(
                    f"New session best: {-session_best_fitness:.1%} - params: {best_params_this_gen}",
                )

            # Save checkpoint every 10 generations
            if (gen + 1) % 10 == 0:
                checkpoint_path = output_dir / f"checkpoint_gen{gen + 1}.pkl"
                save_checkpoint(optimizer, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Check for convergence (CMA-ES only)
            # Only stop early if we've run at least 10 generations and have non-zero fitness
            if gen >= 10 and best_fitness < 0 and optimizer.stop():
                logger.info(f"Optimizer converged at generation {gen + 1}")
                break
    finally:
        # Ensure pool is properly closed (if not already terminated by interrupt handler)
        if pool is not None:
            pool.close()
            pool.join()

    total_time = time.time() - start_time
    result = optimizer.result

    if interrupted:
        logger.info(f"Evolution interrupted after {total_time:.1f}s")
    else:
        logger.info(f"Evolution complete in {total_time:.1f}s")

    logger.info(f"Best success rate: {-result.best_fitness:.1%}")
    logger.info(f"Best parameters: {result.best_params}")

    return result, should_save


def save_checkpoint(optimizer: EvolutionaryOptimizer, path: Path) -> None:
    """Save optimizer state to checkpoint file."""
    with path.open("wb") as f:
        pickle.dump(optimizer, f)


def load_checkpoint(path: Path) -> EvolutionaryOptimizer:
    """Load optimizer state from checkpoint file.

    SECURITY WARNING: This uses pickle.load which can execute arbitrary code.
    Only load checkpoint files that you created yourself or from trusted sources.
    Do not load checkpoints from untrusted or unknown origins.
    """
    logger.warning(
        f"Loading checkpoint from {path} using pickle. "
        "Only load checkpoints from trusted sources (pickle can execute arbitrary code).",
    )
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def save_results(
    result: EvolutionResult,
    config_path: str,
    output_dir: Path,
    timestamp: str,
) -> None:
    """Save evolution results to files."""
    # Ensure output directory exists (defensive - should already exist from main/run_evolution)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters as JSON
    brain = create_brain_from_config(config_path)
    param_keys = list(brain.parameter_values.keys())
    best_params_dict = dict(zip(param_keys, result.best_params, strict=False))

    results_file = output_dir / f"best_params_{timestamp}.json"
    with results_file.open("w") as f:
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
    with history_file.open("w") as f:
        f.write("generation,best_fitness,mean_fitness,std_fitness\n")
        f.writelines(
            f"{entry['generation']},"
            f"{entry['best_fitness']},"
            f"{entry['mean_fitness']},"
            f"{entry['std_fitness']}\n"
            for entry in result.history
        )
    logger.info(f"Saved history: {history_file}")


def main() -> None:  # noqa: PLR0915
    """Run evolutionary optimization."""
    args = parse_arguments()
    _validate_args(args)

    # Configure logging with console output
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)

    # Update existing file handlers to use the specified log level
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(log_level)

    # Add console handler if not already present
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"),
        )
        logger.addHandler(console_handler)

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)  # noqa: NPY002

    # Create output directory
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Session ID: {timestamp}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {output_dir}")

    # Log all run arguments for reproducibility
    logger.info(f"Algorithm: {args.algorithm.upper()}")
    logger.info(f"Generations: {args.generations}")
    logger.info(f"Population size: {args.population}")
    logger.info(f"Episodes per eval: {args.episodes}")
    logger.info(f"Sigma (step size): {args.sigma}")
    logger.info(f"Parallel workers: {args.parallel}")
    logger.info(f"Random seed: {args.seed}")

    # Get number of parameters from brain
    brain = create_brain_from_config(args.config)
    param_keys = list(brain.parameter_values.keys())
    num_params = len(param_keys)
    logger.info(f"Number of parameters: {num_params}")

    # Get initial parameters
    if args.init_params:
        logger.info(f"Loading initial parameters from: {args.init_params}")
        x0 = load_init_params(args.init_params, param_keys)
        logger.info(f"Initial parameters: {x0}")
    else:
        # Use current values from brain (typically random small init)
        x0 = list(brain.parameter_values.values())

    # Create or load optimizer
    optimizer: EvolutionaryOptimizer
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        optimizer = load_checkpoint(Path(args.resume))
    elif args.algorithm == "cmaes":
        optimizer = CMAESOptimizer(
            num_params=num_params,
            x0=x0,
            population_size=args.population,
            sigma0=args.sigma,
            seed=args.seed,
        )
    else:
        optimizer = GeneticAlgorithmOptimizer(
            num_params=num_params,
            x0=x0,
            population_size=args.population,
            sigma0=args.sigma,
            seed=args.seed,
        )

    logger.info(f"Using algorithm: {args.algorithm.upper()}")

    # Run evolution
    result, should_save = run_evolution(
        optimizer=optimizer,
        config_path=args.config,
        generations=args.generations,
        episodes=args.episodes,
        output_dir=output_dir,
        parallel_workers=args.parallel,
        log_level=log_level,
        base_seed=args.seed,
    )

    # Save results (unless user chose not to on interrupt)
    if should_save:
        save_results(result, args.config, output_dir, timestamp)

    # Print final summary
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE" if should_save else "EVOLUTION ABORTED")
    print("=" * 60)
    print(f"Best success rate: {-result.best_fitness:.1%}")
    print(f"Generations: {result.generations}")
    if should_save:
        print(f"Results saved to: {output_dir}")
    else:
        print("Results NOT saved (user requested no save)")
    print("=" * 60)


if __name__ == "__main__":
    main()
