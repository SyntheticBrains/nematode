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

    # Build output lines once - use fixed-width formatting for alignment
    output_lines = ["All runs completed:"]

    for result in all_results:
        final_status = "SUCCESS" if result.success else "FAILED"
        
        # Handle both maze and dynamic environments (since maze does not track foods_collected)
        foods_collected = (
            result.foods_collected
            if result.foods_collected is not None
            else 1
            if result.success
            else 0
        )

        # Use fixed-width fields: Run(3), Status(7), Reason(20), Steps(6), ...
        # ... Eaten(6), Reward(8), Efficiency(11)
        output_lines.append(
            f"Run: {result.run:<3} "
            f"Status: {final_status:<7} "
            f"Reason: {result.termination_reason.value:<20} "
            f"Steps: {result.steps:<6} "
            f"Eaten: {foods_collected!s:<6} "
            f"Reward: {result.total_reward:>7.2f} "
            f"Efficiency: {result.efficiency_score:>10.4f}",
        )

    output_lines.append("")
    output_lines.append(f"Average steps per run: {average_steps:.2f}")
    output_lines.append(f"Average efficiency score: {average_efficiency_score:.2f}")
    output_lines.append(f"Improvement metric (steps): {improvement_rate:.2f}%")
    output_lines.append(f"Success rate: {success_rate:.2f}%")

    # Print to console
    for line in output_lines:
        print(line)  # noqa: T201

    # Log if logger is enabled
    if not logger.disabled:
        for line in output_lines:
            logger.info(line)

        # Verbose run results logging for debug level
        for result in all_results:
            logger.debug(f"Run: {result.run:<3} Path: {result.path}")

        logger.info("Simulation completed.")
