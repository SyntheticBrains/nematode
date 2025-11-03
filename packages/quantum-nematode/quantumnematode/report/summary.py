"""Reporting module for Quantum Nematode simulation results."""

from quantumnematode.logging_config import (
    logger,
)
from quantumnematode.report.dtypes import SimulationResult


def summary(
    num_runs: int,
    max_steps: int,
    all_results: list[SimulationResult],
) -> None:
    """
    Print a summary of the simulation results.

    Parameters
    ----------
    num_runs : int
        The number of simulation runs.
    all_results : list[SimulationResult]
        A list of simulation results.
    """
    average_steps = sum(result.steps for result in all_results) / num_runs
    average_efficiency_score = sum(result.efficiency_score for result in all_results) / num_runs
    # Calculate improvement metric (percentage of steps reduced)
    improvement_rate = (all_results[0].steps - all_results[-1].steps) / all_results[0].steps * 100
    success_rate = sum(1 for result in all_results if result.steps < max_steps) / num_runs * 100

    print("All runs completed:")  # noqa: T201
    for result in all_results:
        print(f"Run {result.run}: {result.steps} steps")  # noqa: T201

    print(f"Average steps per run: {average_steps:.2f}")  # noqa: T201
    print(f"Average efficiency score: {average_efficiency_score:.2f}")  # noqa: T201
    print(f"Improvement metric (steps): {improvement_rate:.2f}%")  # noqa: T201
    print(f"Success rate: {success_rate:.2f}%")  # noqa: T201

    if not logger.disabled:
        logger.info("All runs completed:")
        for result in all_results:
            logger.info(f"Run {result.run}: {result.steps} steps")

        # Verbose run results logging for debug level
        for result in all_results:
            logger.debug(f"Run {result.run}: {result.steps} steps, Path: {result.path}")

        logger.info(f"Average steps per run: {average_steps:.2f}")
        logger.info(f"Average efficiency score: {average_efficiency_score:.2f}")
        logger.info(f"Improvement metric (steps): {improvement_rate:.2f}%")
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info("Simulation completed.")
