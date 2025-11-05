"""Reporting module for Quantum Nematode simulation results."""

from quantumnematode.env import DynamicForagingEnvironment, EnvironmentType, MazeEnvironment
from quantumnematode.logging_config import (
    logger,
)
from quantumnematode.report.dtypes import SimulationResult


def summary(  # noqa: C901, PLR0912
    num_runs: int,
    max_steps: int,
    all_results: list[SimulationResult],
    env_type: EnvironmentType,
) -> None:
    """
    Print a summary of the simulation results.

    Parameters
    ----------
    num_runs : int
        The number of simulation runs.
    max_steps : int
        The maximum number of steps allowed per run.
    all_results : list[SimulationResult]
        A list of simulation results.
    env_type : EnvironmentType
        The type of environment used in the simulation.
    """
    average_steps = sum(result.steps for result in all_results) / num_runs

    average_efficiency_score = None
    improvement_rate = None

    if isinstance(env_type, MazeEnvironment):
        success_rate = sum(1 for result in all_results if result.steps < max_steps) / num_runs * 100

        # Calculate average efficiency score for maze environment, filtering out None values
        efficiency_scores = [
            result.efficiency_score for result in all_results if result.efficiency_score is not None
        ]
        if efficiency_scores:
            average_efficiency_score = sum(efficiency_scores) / len(efficiency_scores)

        # Calculate improvement metric (percentage of steps reduced)
        if len(all_results) >= 2:  # noqa: PLR2004
            improvement_rate = (
                (all_results[0].steps - all_results[-1].steps) / all_results[0].steps * 100
            )
    elif isinstance(env_type, DynamicForagingEnvironment):
        success_rate = sum(result.success for result in all_results) / num_runs * 100

    # Build output lines once - use fixed-width formatting for alignment
    output_lines = ["All runs completed:"]

    for result in all_results:
        final_status = "SUCCESS" if result.success else "FAILED"

        # Add dynamic environment specific data
        additional_info = " "
        if result.satiety_remaining is not None:
            additional_info += f"Satiety: {result.satiety_remaining:<6} "
        if result.foods_collected is not None and result.foods_available is not None:
            foods_info = f"Eaten: {result.foods_collected}/{result.foods_available}"
            additional_info += foods_info
        if result.efficiency_score is not None:
            additional_info += f"Efficiency: {result.efficiency_score:<10.4f}"

        # Use fixed-width fields
        output_lines.append(
            f"Run: {result.run:<3} "
            f"Status: {final_status:<7} "
            f"Reason: {result.termination_reason.value:<20} "
            f"Steps: {result.steps:<6} "
            f"Reward: {result.total_reward:>7.2f} "
            f"{additional_info}",
        )

    output_lines.append("")
    output_lines.append(f"Average steps per run: {average_steps:.2f}")
    output_lines.append(f"Success rate: {success_rate:.2f}%")

    if average_efficiency_score is not None:
        output_lines.append(f"Average efficiency score: {average_efficiency_score:.2f}")
    if improvement_rate is not None:
        output_lines.append(f"Improvement metric (steps): {improvement_rate:.2f}%")

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
