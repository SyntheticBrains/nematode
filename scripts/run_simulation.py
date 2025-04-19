"""Run the Quantum Nematode simulation."""

import argparse
import logging

from quantumnematode.agent import (  # pyright: ignore[reportMissingImports]
    QuantumNematodeAgent,
)
from quantumnematode.logging_config import (  # pyright: ignore[reportMissingImports]
    logger,
)

# Suppress logs from external libraries like Qiskit
logging.getLogger("qiskit").setLevel(logging.WARNING)


def main() -> None:
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

    args = parser.parse_args()

    # Configure logging level
    if args.log_level == "NONE":
        logger.disabled = True
    else:
        logger.setLevel(args.log_level)

    agent = QuantumNematodeAgent(maze_grid_size=args.maze_grid_size)

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

    # Final summary of all runs.
    average_steps = sum(steps for _, steps, _ in all_results) / args.runs
    improvement_rate = (all_results[0][1] - all_results[-1][1]) / all_results[0][1] * 100

    if logger.disabled:
        print("All runs completed:")  # noqa: T201
        for run, steps, path in all_results:
            print(f"Run {run}: {steps} steps, Path: {path}")  # noqa: T201

        print("Summary of all runs:")  # noqa: T201
        for run, steps, _path in all_results:
            print(f"Run {run}: {steps} steps")  # noqa: T201

        print(f"Total runs: {len(all_results)}")  # noqa: T201

        print(f"Average steps per run: {average_steps:.2f}")  # noqa: T201
        print(f"Improvement metric (steps): {improvement_rate:.2f}%")  # noqa: T201
    else:
        logger.info("All runs completed:")
        for run, steps, path in all_results:
            logger.info(f"Run {run}: {steps} steps, Path: {path}")

        logger.info("Summary of all runs:")
        for run, steps, _path in all_results:
            logger.info(f"Run {run}: {steps} steps")
        logger.info(f"Total runs: {len(all_results)}")
        logger.info(f"Average steps per run: {average_steps:.2f}")
        logger.info(f"Improvement metric (steps): {improvement_rate:.2f}%")
        logger.info("Simulation completed.")


if __name__ == "__main__":
    main()
