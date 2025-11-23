"""Reporting module for Quantum Nematode simulation results."""

from quantumnematode.env import DynamicForagingEnvironment, EnvironmentType, StaticEnvironment
from quantumnematode.logging_config import (
    logger,
)
from quantumnematode.report.dtypes import PerformanceMetrics, SimulationResult


def summary(  # noqa: C901, PLR0912, PLR0913, PLR0915
    metrics: PerformanceMetrics,
    session_id: str,
    num_runs: int,
    max_steps: int,
    all_results: list[SimulationResult],
    env_type: EnvironmentType,
) -> None:
    """
    Print a summary of the simulation results.

    Parameters
    ----------
    metrics : PerformanceMetrics
        The performance metrics calculated from the simulation.
    session_id : str
        The unique identifier for the simulation session.
    num_runs : int
        The number of simulation runs.
    max_steps : int
        The maximum number of steps allowed per run.
    all_results : list[SimulationResult]
        A list of simulation results.
    env_type : EnvironmentType
        The type of environment used in the simulation.
    """
    total_runs_done = len(all_results)

    if total_runs_done == 0:
        logger.warning("No simulation results to summarize.")
        return

    average_efficiency_score = None
    improvement_rate = None

    if isinstance(env_type, StaticEnvironment):
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
    output_lines = [""]
    output_lines.append("All runs completed:")
    output_lines.append(f"Session ID: {session_id}")
    output_lines.append("")

    total_foods_collected: int | None = None
    total_foods_available: int | None = None
    for result in all_results:
        final_status = "SUCCESS" if result.success else "FAILED"

        # Add environment specific data
        additional_info = " "
        if result.satiety_remaining is not None:
            additional_info += f"Satiety: {result.satiety_remaining:<6} "
        if result.foods_collected is not None and result.foods_available is not None:
            foods_info = f"Eaten: {result.foods_collected}/{result.foods_available:<6} "
            additional_info += foods_info
        if result.efficiency_score is not None:
            additional_info += f"Efficiency: {result.efficiency_score:<10.4f} "
        if result.average_distance_efficiency is not None:
            additional_info += f"Avg Dist Eff: {result.average_distance_efficiency:<10.4f} "

        # Set other environment totals
        if result.foods_collected is not None:
            if total_foods_collected is None:
                total_foods_collected = 0
            total_foods_collected += result.foods_collected
        if result.foods_available is not None:
            if total_foods_available is None:
                total_foods_available = 0
            total_foods_available += result.foods_available

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
    output_lines.append(f"Total runs completed: {total_runs_done}")
    output_lines.append(
        f"Successful runs: {metrics.total_successes} "
        f"({metrics.total_successes / total_runs_done * 100:.1f}%)",
    )

    if metrics.total_starved is not None:
        output_lines.append(
            f"Failed runs - Starved: {metrics.total_starved} "
            f"({metrics.total_starved / total_runs_done * 100:.1f}%)",
        )
    if metrics.total_predator_deaths is not None:
        output_lines.append(
            f"Failed runs - Eaten by Predator: {metrics.total_predator_deaths} "
            f"({metrics.total_predator_deaths / total_runs_done * 100:.1f}%)",
        )
    if metrics.total_max_steps is not None:
        output_lines.append(
            f"Failed runs - Max Steps: {metrics.total_max_steps} "
            f"({metrics.total_max_steps / total_runs_done * 100:.1f}%)",
        )
    if metrics.total_interrupted > 0:
        output_lines.append(
            f"Failed runs - Interrupted: {metrics.total_interrupted} "
            f"({metrics.total_interrupted / total_runs_done * 100:.1f}%)",
        )

    if metrics.average_foods_collected is not None:
        output_lines.append(
            f"Average foods collected per run: {metrics.average_foods_collected:.2f}",
        )
    output_lines.append(f"Average steps per run: {metrics.average_steps:.2f}")
    output_lines.append(f"Average reward per run: {metrics.average_reward:.2f}")
    if metrics.average_distance_efficiency is not None:
        output_lines.append(
            f"Average distance efficiency per run: {metrics.average_distance_efficiency:.2f}",
        )
    if metrics.foraging_efficiency is not None:
        output_lines.append(f"Foraging efficiency per run: {metrics.foraging_efficiency:.2f}")
    if total_foods_collected is not None and total_foods_available is not None:
        output_lines.append(
            f"Total foods collected: {total_foods_collected}/{total_foods_available}",
        )

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
