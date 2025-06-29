"""Plotting functions for Quantum Nematode."""

from pathlib import Path

from matplotlib import pyplot as plt  # pyright: ignore[reportMissingImports]

from quantumnematode.agent import QuantumNematodeAgent
from quantumnematode.logging_config import (
    logger,
)


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
        runs (list[int]): List of run indices.
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


def plot_tracking_data_per_session(
    tracking_data: dict[str, list[int | float | dict]],
    timestamp: str,
    brain_type: str,
    qubits: int,
    file_prefix: str = "",
) -> None:
    """
    Generate and save plots for tracking data.

    Args:
        tracking_data (dict[str, list[int | float | dict[str, float]]]): Dictionary
            containing tracking data.
        timestamp (str): Timestamp for the plot directory.
        brain_type (str): Type of brain (e.g., "modular").
        qubits (int): Number of qubits.
        file_prefix (str, optional): Prefix for the output file name. Defaults to "".
    """
    plot_dir: Path = Path.cwd() / "plots" / timestamp
    plot_dir.mkdir(parents=True, exist_ok=True)

    title_postfix: str = (
        f" [{brain_type} {qubits}Q]" if brain_type == "dynamic" else f" [{brain_type}]"
    )

    # Plot each tracked variable
    for key, values in tracking_data.items():
        if key == "run":
            continue

        logger.debug(f"Tracking data for {key}: {str(values).replace('θ', 'theta_')}")

        if isinstance(values, list) and all(isinstance(v, dict) for v in values):
            # Flatten dictionaries into lists of values for plotting
            title = key.replace("_", " ").title()
            label = next(
                list(param_dict.keys()) for param_dict in values if isinstance(param_dict, dict)
            )
            values = [  # noqa: PLW2901
                list(param_dict.values()) for param_dict in values if isinstance(param_dict, dict)
            ]
        elif key == "computed_gradients":
            title = "Computed Gradients"
            label = (
                [str(n + 1) for n in range(len(values[0]))] if isinstance(values[0], list) else []
            )
        else:
            label = key.replace("_", " ").title()
            title = label

        title += title_postfix

        plt.figure(figsize=(10, 6))
        plt.plot(tracking_data["run"], values, marker="o", label=label)
        plt.title(title)
        plt.xlabel("Run")
        plt.ylabel(str(label))
        plt.legend()
        plt.grid()
        plt.savefig(plot_dir / f"{file_prefix}track_{key}_over_runs.png")
        plt.close()


def plot_tracking_data_per_run(
    timestamp: str,
    agent: QuantumNematodeAgent,
    run: int,
) -> None:
    """
    Generate and save plots for tracked agent data for a single run.

    Args:
        timestamp (str): Timestamp for the plot directory.
        agent (QuantumNematodeAgent): Agent containing brain with tracking histories.
        run (int): Index of the run.
    """
    run_dir = Path.cwd() / "plots" / timestamp / f"run_{run + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tracked: dict[str, list[dict[str, float]] | list[float] | None] = {
        "input_parameters": getattr(agent.brain, "history_input_parameters", None),
        "updated_parameters": getattr(agent.brain, "history_updated_parameters", None),
        "gradients": getattr(agent.brain, "history_gradients", None),
        "gradient_strengths": getattr(agent.brain, "history_gradient_strengths", None),
        "gradient_directions": getattr(agent.brain, "history_gradient_directions", None),
        "rewards": getattr(agent.brain, "history_rewards", None),
        "rewards_norm": getattr(agent.brain, "history_rewards_norm", None),
        "learning_rates": getattr(agent.brain, "history_learning_rates", None),
        "temperatures": getattr(agent.brain, "history_temperatures", None),
    }
    for key, values in tracked.items():
        if values is not None and len(values) > 0:
            plt.figure(figsize=(8, 4))
            if isinstance(values[0], dict):
                # Plot each parameter in the dict
                for param in values[0]:
                    plt.plot(
                        [v[param] for v in values if isinstance(v, dict) and param in v],
                        label=param,
                    )
                plt.legend()
            else:
                plt.plot(values)
            plt.title(f"{key} (run {run + 1})")
            plt.xlabel("Step")
            plt.ylabel(key)
            plt.tight_layout()
            plt.savefig(run_dir / f"{key}.png")
            plt.close()
