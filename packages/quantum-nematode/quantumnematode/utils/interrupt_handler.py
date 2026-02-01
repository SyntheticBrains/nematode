"""Shared interrupt handling for simulation scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quantumnematode.brain.arch import QuantumBrain
from quantumnematode.brain.arch.dtypes import BrainType  # noqa: TC001
from quantumnematode.logging_config import logger
from quantumnematode.report.csv_export import (
    export_performance_metrics_to_csv,
    export_simulation_results_to_csv,
    export_tracking_data_to_csv,
)
from quantumnematode.report.plots import (
    plot_tracking_data_by_session,
)
from quantumnematode.report.summary import summary

if TYPE_CHECKING:
    from collections.abc import Callable

    from quantumnematode.agent import QuantumNematodeAgent
    from quantumnematode.report.dtypes import (
        SimulationResult,
        TrackingData,
    )


def manage_simulation_halt(  # noqa: PLR0913
    max_steps: int,
    brain_type: BrainType,
    qubits: int,
    timestamp: str,
    agent: QuantumNematodeAgent,
    all_results: list[SimulationResult],
    total_runs_done: int,
    tracking_data: TrackingData,
    plot_dir: Path,
    plot_results_fn: Callable[..., Any] | None = None,
) -> None:
    """Handle simulation halt triggered by a KeyboardInterrupt.

    Provides options to exit, output partial results and plots,
    or print circuit details.

    Args:
        max_steps: Maximum number of steps for the simulation.
        brain_type: Type of brain architecture used.
        qubits: Number of qubits used.
        timestamp: Timestamp for the current session.
        agent: The simulation agent.
        all_results: List of results for each run.
        total_runs_done: Total runs completed so far.
        tracking_data: Data tracked during the simulation.
        plot_dir: Directory where plots will be saved.
        plot_results_fn: Optional callable for plotting results.
    """
    data_dir = Path.cwd() / "exports" / timestamp / "session" / "data"
    while True:
        prompt_intro_message = (
            "KeyboardInterrupt detected. The simulation has halted. "
            "You can choose to exit or output the results up to this point."
        )
        logger.warning(prompt_intro_message)
        print(prompt_intro_message)
        print("0. Exit")
        print("1. Output the summary, plots, and tracking until this point in time.")
        print("2. Print the circuit's details.")

        try:
            choice = int(input("Enter your choice (0-2): "))
        except ValueError:
            logger.error("Invalid input. Please enter a number between 0 and 2.")
            continue
        except KeyboardInterrupt:
            continue

        if choice == 0:
            logger.info("Exiting the session.")
            sys.exit(0)
        elif choice == 1:
            logger.info("Generating partial results and plots.")
            metrics = agent.calculate_metrics(total_runs=total_runs_done)

            summary(
                metrics=metrics,
                session_id=timestamp,
                num_runs=total_runs_done,
                max_steps=max_steps,
                all_results=all_results,
                env_type=agent.env,
            )

            file_prefix = f"{total_runs_done}_"
            if plot_results_fn is not None:
                plot_results_fn(
                    all_results=all_results,
                    metrics=metrics,
                    file_prefix=file_prefix,
                    plot_dir=plot_dir,
                )
            plot_tracking_data_by_session(
                tracking_data=tracking_data,
                plot_dir=plot_dir,
                brain_type=brain_type,
                qubits=qubits,
                file_prefix=file_prefix,
            )

            export_simulation_results_to_csv(
                all_results=all_results,
                data_dir=data_dir,
                file_prefix=file_prefix,
            )
            export_performance_metrics_to_csv(
                metrics=metrics,
                data_dir=data_dir,
                file_prefix=file_prefix,
            )
            export_tracking_data_to_csv(
                tracking_data=tracking_data,
                brain_type=brain_type,
                data_dir=data_dir,
                qubits=qubits,
                file_prefix=file_prefix,
            )
        elif choice == 2:
            logger.info("Printing circuit details.")
            if isinstance(agent.brain, QuantumBrain):
                circuit = agent.brain.inspect_circuit()
                logger.info(f"Circuit details:\n{circuit}")
                print(circuit)
            else:
                logger.error(
                    "Circuit details are only available for QuantumBrain architectures.",
                )
                print("Circuit details are only available for QuantumBrain architectures.")
        else:
            logger.error("Invalid choice. Please enter a number between 1 and 4.")


def prompt_interrupt() -> str:
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
        print("\nForce exit requested")
        return "n"
