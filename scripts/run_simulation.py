"""Run the Quantum Nematode simulation."""

import argparse
import logging
from datetime import datetime, UTC
from pathlib import Path

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from quantumnematode.agent import (  # pyright: ignore[reportMissingImports]
    QuantumNematodeAgent,
)
from quantumnematode.constants import MIN_GRID_SIZE  # pyright: ignore[reportMissingImports]
from quantumnematode.logging_config import (  # pyright: ignore[reportMissingImports]
    logger,
)
from quantumnematode.report import summary  # pyright: ignore[reportMissingImports]


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the Quantum Nematode simulation."""
    parser = argparse.ArgumentParser(description="Run the Quantum Nematode simulation.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of steps for the simulation (default: 100)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
        help="Set the logging level (default: INFO). Use 'NONE' to disable logging.",
    )
    parser.add_argument(
        "--maze-grid-size",
        type=int,
        default=5,
        help="Size of the maze grid (minimum: 5, default: 5)",
    )
    parser.add_argument(
        "--show-last-frame-only",
        action="store_true",
        help="Only display the last frame in the CLI output.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of simulation runs to perform (default: 1)",
    )
    parser.add_argument(
        "--brain",
        type=str,
        choices=["simple", "complex", "reduced", "memory"],
        default="simple",
        help="Choose the quantum brain architecture to use (default: simple)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to use for AerSimulator",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100,
        help="Number of shots for the AerSimulator",
    )
    parser.add_argument(
        "--body-length",
        type=int,
        default=0,
        help="Length of the agent's body, excluding head (default: 0)",
    )

    args = parser.parse_args()

    if args.maze_grid_size < MIN_GRID_SIZE:
        error_message = (
            f"Grid size must be at least {MIN_GRID_SIZE}. "
            f"Provided grid size: {args.maze_grid_size}."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Configure logging level
    if args.log_level == "NONE":
        logger.disabled = True
    else:
        logger.setLevel(args.log_level)
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(args.log_level)

    # Pass the device and shots arguments to the brain classes
    device = args.device.upper()
    shots = args.shots

    # Select the brain architecture
    if args.brain == "simple":
        from quantumnematode.brain.simple import (  # pyright: ignore[reportMissingImports]
            SimpleBrain,
        )

        brain = SimpleBrain(device=device, shots=shots)
    elif args.brain == "complex":
        from quantumnematode.brain.complex import (  # pyright: ignore[reportMissingImports]
            ComplexBrain,
        )

        if device != "CPU":
            logger.warning(
                "ComplexBrain is not optimized for GPU. Using CPU instead.",
            )
        brain = ComplexBrain(device=device, shots=shots)
    elif args.brain == "reduced":
        from quantumnematode.brain.reduced import (  # pyright: ignore[reportMissingImports]
            ReducedBrain,
        )

        brain = ReducedBrain(device=device, shots=shots)
    elif args.brain == "memory":
        from quantumnematode.brain.memory import (  # pyright: ignore[reportMissingImports]
            MemoryBrain,
        )

        brain = MemoryBrain(device=device, shots=shots)
    else:
        error_message = f"Unknown brain architecture: {args.brain}"
        raise ValueError(error_message)

    # Update the agent to use the selected brain architecture
    agent = QuantumNematodeAgent(
        maze_grid_size=args.maze_grid_size,
        brain=brain,
        max_body_length=args.body_length,
    )

    all_results = []

    for run in range(args.runs):
        logger.info(f"Starting run {run + 1} of {args.runs}")
        path = agent.run_episode(
            max_steps=args.max_steps,
            show_last_frame_only=args.show_last_frame_only,
        )

        steps = len(path)
        total_reward = sum(
            agent.env.get_state(pos, disable_log=True)[0] for pos in path
        )  # Calculate total reward for the run
        all_results.append((run + 1, steps, path, total_reward))  # Include total reward in results

        logger.info(f"Run {run + 1}/{args.runs} completed in {steps} steps.")

        if run < args.runs - 1:
            agent.reset_environment()

    # Calculate and log performance metrics
    metrics = agent.calculate_metrics(total_runs=args.runs)
    logger.info("\nPerformance Metrics:")
    logger.info(f"Success Rate: {metrics['success_rate']:.2f}")
    logger.info(f"Average Steps: {metrics['average_steps']:.2f}")
    logger.info(f"Average Reward: {metrics['average_reward']:.2f}")

    # Final summary of all runs.
    summary(args.runs, args.max_steps, all_results)

    # Generate plots after the simulation
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    plot_results(all_results, metrics, timestamp)


def plot_results(
    all_results: list[tuple[int, int, list[tuple[int, int]], float]],
    metrics: dict[str, float],
    timestamp: str,
) -> None:
    """Generate and save plots for the simulation results."""
    runs: list[int] = [result[0] for result in all_results]
    steps: list[int] = [result[1] for result in all_results]

    plot_dir: Path = Path.cwd() / "plots" / timestamp
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot: Steps per Run
    plt.figure(figsize=(10, 6))
    plt.plot(runs, steps, marker="o", label="Steps per Run")
    plt.axhline(y=metrics["average_steps"], color="r", linestyle="--", label="Average Steps")
    plt.title("Steps per Run")
    plt.xlabel("Run")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / "steps_per_run.png")
    plt.close()

    # Plot: Cumulative Reward per Run
    cumulative_rewards: list[float] = [
        result[3] for result in all_results
    ]  # Assuming rewards are stored as result[3]
    plt.figure(figsize=(10, 6))
    plt.plot(runs, cumulative_rewards, marker="o", label="Cumulative Reward per Run")
    plt.title("Cumulative Reward per Run")
    plt.xlabel("Run")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / "cumulative_reward_per_run.png")
    plt.close()


if __name__ == "__main__":
    main()
