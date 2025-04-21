"""Run the Quantum Nematode simulation."""

import argparse
import logging

from quantumnematode.agent import (  # pyright: ignore[reportMissingImports]
    QuantumNematodeAgent,
)
from quantumnematode.logging_config import (  # pyright: ignore[reportMissingImports]
    logger,
)
from quantumnematode.report import summary  # pyright: ignore[reportMissingImports]


def main() -> None:  # noqa: C901, PLR0915
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
        help="Size of the maze grid (default: 5)",
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

    args = parser.parse_args()

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
    agent = QuantumNematodeAgent(maze_grid_size=args.maze_grid_size, brain=brain)

    all_results = []

    for run in range(args.runs):
        logger.info(f"Starting run {run + 1} of {args.runs}")
        path = agent.run_episode(
            max_steps=args.max_steps,
            show_last_frame_only=args.show_last_frame_only,
        )

        steps = len(path)
        all_results.append((run + 1, steps, path))

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


if __name__ == "__main__":
    main()
