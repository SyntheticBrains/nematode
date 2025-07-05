"""Plotting functions for Quantum Nematode."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from quantumnematode.brain.actions import ActionData
from quantumnematode.logging_config import (
    logger,
)
from quantumnematode.report.dtypes import TrackingData


def plot_efficiency_score_over_time(
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    efficiency_scores: list[float],
) -> None:
    """
    Plot efficiency scores over time and save the plot.

    Args:
        file_prefix (str): Prefix for the output file name.
        runs (list[int]): List of run indices.
        plot_dir (Path): Directory to save the plot.
        efficiency_scores (list[float]): List of efficiency scores.
    """
    average_efficiency_score = sum(efficiency_scores) / len(efficiency_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(runs, efficiency_scores, marker="o", label="Efficiency Score Over Time")
    plt.axhline(
        y=average_efficiency_score,
        color="r",
        linestyle="--",
        label="Average Efficiency Score",
    )
    plt.title("Efficiency Score Over Time")
    plt.xlabel("Run")
    plt.ylabel("Efficiency Score")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}efficiency_score_over_time.png")
    plt.close()


def plot_success_rate_over_time(
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    success_rates: list[float],
) -> None:
    """
    Plot success rates over time and save the plot.

    Args:
        file_prefix (str): Prefix for the output file name.
        runs (list[int]): List of run indices.
        plot_dir (Path): Directory to save the plot.
        success_rates (list[float]): List of success rates.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(runs, success_rates, marker="o", label="Success Rate Over Time")
    plt.title("Success Rate Over Time")
    plt.xlabel("Run")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}success_rate_over_time.png")
    plt.close()


def plot_last_cumulative_rewards(
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    last_cumulative_rewards: list[float],
) -> None:
    """
    Plot last cumulative rewards over time and save the plot.

    Args:
        file_prefix (str): Prefix for the output file name.
        runs (list[int]): List of run indices.
        plot_dir (Path): Directory to save the plot.
        last_cumulative_rewards (list[float]): List of last cumulative rewards.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(runs, last_cumulative_rewards, marker="o", label="Last Cumulative Rewards")
    plt.title("Last Cumulative Reward Over Time")
    plt.xlabel("Run")
    plt.ylabel("Last Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}cumulative_last_reward_over_time.png")
    plt.close()


def plot_cumulative_reward_per_run(
    file_prefix: str,
    runs: list[int],
    plot_dir: Path,
    cumulative_rewards: list[float],
) -> None:
    """
    Plot cumulative rewards per run and save the plot.

    Args:
        file_prefix (str): Prefix for the output file name.
        runs (list[int): List of run indices.
        plot_dir (Path): Directory to save the plot.
        cumulative_rewards (list[float]): List of cumulative rewards.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(runs, cumulative_rewards, marker="o", label="Cumulative Reward per Run")
    plt.title("Cumulative Reward per Run")
    plt.xlabel("Run")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}cumulative_reward_per_run.png")
    plt.close()


def plot_steps_per_run(
    metrics: dict[str, int | float],
    file_prefix: str,
    runs: list[int],
    steps: list[int],
    plot_dir: Path,
) -> None:
    """
    Plot steps per run and save the plot.

    Args:
        metrics (dict[str, int | float]): Dictionary containing metrics like average steps.
        file_prefix (str): Prefix for the output file name.
        runs (list[int]): List of run indices.
        steps (list[int]): List of steps per run.
        plot_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(runs, steps, marker="o", label="Steps per Run")
    plt.axhline(y=metrics["average_steps"], color="r", linestyle="--", label="Average Steps")
    plt.title("Steps per Run")
    plt.xlabel("Run")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid()
    plt.savefig(plot_dir / f"{file_prefix}steps_per_run.png")
    plt.close()


def plot_tracking_data_by_session(  # noqa: C901, PLR0912, PLR0915
    tracking_data: TrackingData,
    timestamp: str,
    brain_type: str,
    qubits: int | None = None,
    file_prefix: str = "",
) -> None:
    """
    Generate and save plots for the last data point of each tracked variable per run.

    Args:
        tracking_data (TrackingData): Tracking data containing brain history.
        timestamp (str): Timestamp for the plot directory.
        brain_type (str): Type of the brain (e.g., "quantum", "classical").
        qubits (int | None): Number of qubits if applicable, otherwise None.
        file_prefix (str): Prefix for the output file names.
    """
    plot_dir: Path = Path.cwd() / "plots" / timestamp
    plot_dir.mkdir(parents=True, exist_ok=True)

    title_postfix: str = (
        f" [{brain_type} {qubits}Q]" if isinstance(qubits, int) else f" [{brain_type}]"
    )

    runs = sorted(tracking_data.data.keys())
    if not runs:
        logger.warning("No runs found in tracking data. Skipping plots.")
        return
    first_run_data = tracking_data.data[runs[0]]
    keys = list(first_run_data.__dict__.keys())

    for key in keys:
        # Gather the last value for this key in each run
        last_values = []
        for run in runs:
            v = getattr(tracking_data.data[run], key, None)
            if isinstance(v, list) and v:
                last_values.append(v[-1])
            else:
                last_values.append(v)
        # Skip if all values are None or empty
        if all(val is None for val in last_values):
            logger.warning(f"No data available for {key} across runs. Skipping plot.")
            continue
        # Handle last value is dict (e.g., parameter dicts)
        if isinstance(last_values[0], dict):
            param_keys = set()
            for d in last_values:
                if isinstance(d, dict):
                    param_keys.update(d.keys())
            plt.figure(figsize=(10, 6))
            for param in param_keys:
                param_values = [
                    d.get(param, np.nan) if isinstance(d, dict) else np.nan for d in last_values
                ]
                plt.plot(runs, param_values, marker="o", label=param)
            plt.title(f"{key}{title_postfix}")
            plt.xlabel("Run")
            plt.ylabel(key)
            plt.legend()
            plt.grid()
            plt.savefig(plot_dir / f"{file_prefix}track_{key}_over_runs.png")
            plt.close()
        # Handle last value is ActionData
        elif isinstance(last_values[0], ActionData):
            probs = [a.probability if isinstance(a, ActionData) else np.nan for a in last_values]
            actions = [a.action if isinstance(a, ActionData) else "" for a in last_values]
            plt.figure(figsize=(10, 6))
            plt.plot(runs, probs, marker="o", color="orange", label="Probability (last step)")
            for run, prob, action in zip(runs, probs, actions, strict=False):
                plt.text(
                    run,
                    prob + 0.02,
                    action,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            plt.title(f"{key} (last step per run){title_postfix}")
            plt.xlabel("Run")
            plt.ylabel("Probability")
            plt.legend()
            plt.grid()
            plt.savefig(plot_dir / f"{file_prefix}track_{key}_over_runs.png")
            plt.close()
        # Handle last value is scalar
        elif isinstance(last_values[0], (int, float)) or all(
            (v is None or isinstance(v, (int, float))) for v in last_values
        ):
            plot_values = [v if v is not None else np.nan for v in last_values]
            plt.figure(figsize=(10, 6))
            plt.plot(runs, plot_values, marker="o", label=key)
            plt.title(f"{key}{title_postfix}")
            plt.xlabel("Run")
            plt.ylabel(key)
            plt.legend()
            plt.grid()
            plt.savefig(plot_dir / f"{file_prefix}track_{key}_over_runs.png")
            plt.close()
        else:
            logger.warning(f"Unrecognized data type for {key}. Skipping plot.")


def plot_tracking_data_by_latest_run(
    tracking_data: TrackingData,
    timestamp: str,
    run: int,
) -> None:
    """
    Generate and save plots for tracked agent data for a single run.

    Args:
        timestamp (str): Timestamp for the plot directory.
        agent (QuantumNematodeAgent): Agent containing brain with tracking histories.
        run (int): Index of the run.
    """
    run_dir = Path.cwd() / "plots" / timestamp / f"run_{run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    current_run_data = tracking_data.data.get(run, None)

    if current_run_data is None:
        logger.warning(f"No tracking data available for run {run}. Skipping plots.")
        return

    for key, values in current_run_data.__dict__.items():
        if values is None or len(values) == 0:
            logger.warning(f"No data available for {key} in run {run}. Skipping plot.")
            continue

        plt.figure(figsize=(8, 4))
        # values is always a list of the type
        first = values[0]
        if isinstance(first, ActionData):
            # Plot probability as a line, annotate actions at each step
            actions = [a.action for a in values]
            probs = [a.probability for a in values]
            steps = list(range(len(values)))
            plt.plot(steps, probs, color="orange", marker="o", label="Probability")
            for _i, (step, prob, action) in enumerate(zip(steps, probs, actions, strict=False)):
                plt.text(
                    step,
                    prob + 0.02,
                    action,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            plt.title(f"Action Probabilities with Actions (run {run})")
            plt.ylabel("probability")
            plt.legend()
        elif isinstance(first, dict):
            # Plot each parameter in the dict as a line over steps
            param_keys = list(first.keys())
            steps = list(range(len(values)))
            for param in param_keys:
                param_values = [v.get(param, np.nan) for v in values]
                plt.plot(steps, param_values, marker="o", label=param)
            plt.title(f"{key} (run {run})")
            plt.ylabel(key)
            plt.legend()
        else:
            # Scalar or list of floats/ints
            steps = list(range(len(values)))
            plt.plot(steps, values, marker="o")
            plt.ylabel(key)
        plt.title(f"{key} (run {run})")
        plt.xlabel("step")
        plt.tight_layout()
        plt.savefig(run_dir / f"{key}.png")
        plt.close()
