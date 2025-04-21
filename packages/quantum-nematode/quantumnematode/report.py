"""Reporting module for Quantum Nematode simulation results."""

from quantumnematode.logging_config import (  # pyright: ignore[reportMissingImports]
    logger,
)


def summary(num_runs: int, max_steps: int, all_results: list[tuple[int, int, list[tuple]]]) -> None:
    """
    Print a summary of the simulation results.

    Parameters
    ----------
    num_runs : int
        The number of simulation runs.
    all_results : list[tuple[int, int, list[tuple]]]
        A list of tuples containing the run number, number of steps taken, and path taken.
    """
    average_steps = sum(steps for _, steps, _ in all_results) / num_runs
    improvement_rate = (all_results[0][1] - all_results[-1][1]) / all_results[0][1] * 100
    success_rate = sum(1 for _, steps, _ in all_results if steps < max_steps) / num_runs * 100

    print("All runs completed:")  # noqa: T201
    for run, steps, _path in all_results:
        print(f"Run {run}: {steps} steps")  # noqa: T201

    print(f"Average steps per run: {average_steps:.2f}")  # noqa: T201
    print(f"Improvement metric (steps): {improvement_rate:.2f}%")  # noqa: T201
    print(f"Success rate: {success_rate:.2f}%")  # noqa: T201

    if not logger.disabled:
        logger.info("All runs completed:")
        for run, steps, path in all_results:
            logger.info(f"Run {run}: {steps} steps, Path: {path}")

        logger.info(f"Average steps per run: {average_steps:.2f}")
        logger.info(f"Improvement metric (steps): {improvement_rate:.2f}%")
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info("Simulation completed.")
