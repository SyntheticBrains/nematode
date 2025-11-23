#!/usr/bin/env python
"""CLI tool for querying and comparing experiments."""

import argparse
import json
import sys
from datetime import UTC, datetime

from quantumnematode.experiment import compare_experiments, list_experiments, load_experiment
from quantumnematode.experiment.metadata import ExperimentMetadata


def format_table_row(cells: list[str], widths: list[int]) -> str:
    """Format table row with aligned columns.

    Parameters
    ----------
    cells : list[str]
        Cell contents.
    widths : list[int]
        Column widths.

    Returns
    -------
    str
        Formatted row.
    """
    formatted = []
    for cell, width in zip(cells, widths, strict=False):
        formatted.append(cell.ljust(width))
    return "| " + " | ".join(formatted) + " |"


def print_experiments_table(experiments: list[ExperimentMetadata]) -> None:
    """Print experiments in a formatted table.

    Parameters
    ----------
    experiments : list[ExperimentMetadata]
        List of experiments to display.
    """
    if not experiments:
        print("No experiments found.")
        return

    # Define columns
    headers = ["ID", "Config", "Env", "Brain", "Success", "Avg Steps", "Date"]
    widths = [17, 30, 15, 10, 8, 10, 10]

    # Print header
    print(format_table_row(headers, widths))
    print(format_table_row(["-" * w for w in widths], widths))

    # Print rows
    for exp in experiments:
        config_name = exp.config_file.split("/")[-1] if "/" in exp.config_file else exp.config_file
        if len(config_name) > 30:
            config_name = config_name[:27] + "..."

        cells = [
            exp.experiment_id,
            config_name,
            f"{exp.environment.type}/{exp.environment.grid_size}",
            exp.brain.type,
            f"{exp.results.success_rate:.0%}",
            f"{exp.results.avg_steps:.0f}",
            exp.timestamp.strftime("%Y-%m-%d"),
        ]
        print(format_table_row(cells, widths))

    print(f"\nTotal: {len(experiments)} experiments")


def print_experiment_details(metadata: ExperimentMetadata) -> None:  # noqa: C901, PLR0912, PLR0915
    """Print detailed experiment information.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata to display.
    """
    print(f"\n{'=' * 80}")
    print(f"Experiment ID: {metadata.experiment_id}")
    print(f"Timestamp: {metadata.timestamp.isoformat()}")
    print(f"{'=' * 80}\n")

    print("Configuration:")
    print(f"  Config File: {metadata.config_file}")
    print(f"  Config Hash: {metadata.config_hash[:16]}...")

    print("\nGit Context:")
    print(f"  Commit: {metadata.git_commit or 'N/A'}")
    print(f"  Branch: {metadata.git_branch or 'N/A'}")
    print(f"  Dirty: {'Yes' if metadata.git_dirty else 'No'}")

    print("\nEnvironment:")
    print(f"  Type: {metadata.environment.type}")
    print(f"  Grid Size: {metadata.environment.grid_size}")
    if metadata.environment.num_foods:
        print(f"  Foods Visible: {metadata.environment.num_foods}")
        print(f"  Foods to Collect: {metadata.environment.target_foods_to_collect}")
        print(f"  Initial Satiety: {metadata.environment.initial_satiety}")
    if metadata.environment.predators_enabled:
        print("  Predators Enabled: Yes")
        print(f"    Count: {metadata.environment.num_predators}")
        print(f"    Speed: {metadata.environment.predator_speed}")
        print(f"    Detection Radius: {metadata.environment.predator_detection_radius}")
        print(f"    Kill Radius: {metadata.environment.predator_kill_radius}")
        if metadata.environment.predator_gradient_decay:
            print(f"    Gradient Decay: {metadata.environment.predator_gradient_decay}")
        if metadata.environment.predator_gradient_strength:
            print(f"    Gradient Strength: {metadata.environment.predator_gradient_strength}")
        if metadata.environment.predator_proximity_penalty:
            print(f"    Proximity Penalty: {metadata.environment.predator_proximity_penalty}")

    print("\nBrain:")
    print(f"  Type: {metadata.brain.type}")
    print(f"  Learning Rate: {metadata.brain.learning_rate}")
    if metadata.brain.qubits:
        print(f"  Qubits: {metadata.brain.qubits}")
        print(f"  Shots: {metadata.brain.shots}")
    if metadata.brain.hidden_dim:
        print(f"  Hidden Dim: {metadata.brain.hidden_dim}")
    if metadata.brain.parameter_initializer:
        print(f"  Parameter Initializer: {metadata.brain.parameter_initializer.type}")
        if metadata.brain.parameter_initializer.manual_parameter_values:
            num_params = len(metadata.brain.parameter_initializer.manual_parameter_values)
            print(f"    Manual Parameters: {num_params} values")

    print("\nResults:")
    print(f"  Total Runs: {metadata.results.total_runs}")
    print(f"  Success Rate: {metadata.results.success_rate:.1%}")
    print(f"  Avg Steps: {metadata.results.avg_steps:.1f}")
    print(f"  Avg Reward: {metadata.results.avg_reward:.2f}")
    if metadata.results.avg_foods_collected:
        print(f"  Avg Foods: {metadata.results.avg_foods_collected:.1f}")
    if metadata.results.avg_distance_efficiency:
        print(f"  Avg Dist Efficiency: {metadata.results.avg_distance_efficiency:.3f}")

    print("\nTermination Breakdown:")
    print(f"  Goal Reached: {metadata.results.goal_reached}")
    print(f"  All Foods: {metadata.results.completed_all_food}")
    print(f"  Starved: {metadata.results.starved}")
    print(f"  Max Steps: {metadata.results.max_steps_reached}")
    if metadata.results.predator_deaths > 0:
        print(f"  Predator Deaths: {metadata.results.predator_deaths}")

    if metadata.results.avg_predator_encounters is not None:
        print("\nPredator Metrics:")
        print(f"  Avg Encounters: {metadata.results.avg_predator_encounters:.2f}")
        if metadata.results.avg_successful_evasions is not None:
            print(f"  Avg Evasions: {metadata.results.avg_successful_evasions:.2f}")

    print("\nSystem:")
    print(f"  Python: {metadata.system.python_version}")
    print(f"  Qiskit: {metadata.system.qiskit_version}")
    print(f"  Device: {metadata.system.device_type}")

    if metadata.exports_path:
        print(f"\nDetailed Exports: {metadata.exports_path}")

    print()


def print_comparison(comparison: dict) -> None:
    """Print experiment comparison.

    Parameters
    ----------
    comparison : dict
        Comparison data from compare_experiments.
    """
    print(f"\n{'=' * 80}")
    print("Experiment Comparison")
    print(f"{'=' * 80}\n")

    print(f"Experiment 1: {comparison['experiment_ids']['exp1']}")
    print(f"Experiment 2: {comparison['experiment_ids']['exp2']}")

    if comparison["config_diff"]:
        print("\nConfiguration Differences:")
        for key, values in comparison["config_diff"].items():
            print(f"  {key}:")
            print(f"    Exp1: {values['exp1']}")
            print(f"    Exp2: {values['exp2']}")
    else:
        print("\nNo configuration differences")

    print("\nResults Comparison:")
    for metric, values in comparison["results_comparison"].items():
        print(f"  {metric}:")
        print(f"    Exp1: {values['exp1']}")
        print(f"    Exp2: {values['exp2']}")

    print("\nPerformance Delta:")
    delta = comparison["performance_delta"]
    print(
        f"  Success Rate: {delta['success_rate_diff']:+.1%} (Better: {delta['better_success_rate']})",
    )
    print(f"  Avg Steps: {delta['avg_steps_diff']:+.1f} (Better: {delta['better_avg_steps']})")
    print(f"  Avg Reward: {delta['avg_reward_diff']:+.2f} (Better: {delta['better_avg_reward']})")
    print()


def cmd_list(args: argparse.Namespace) -> None:
    """Handle list command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments.
    """
    # Parse since date if provided
    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)

    experiments = list_experiments(
        environment_type=args.env_type,
        brain_type=args.brain_type,
        min_success_rate=args.min_success_rate,
        since=since,
        limit=args.limit,
    )

    if args.json:
        output = [exp.to_dict() for exp in experiments]
        print(json.dumps(output, indent=2, default=str))
    else:
        print_experiments_table(experiments)


def cmd_show(args: argparse.Namespace) -> None:
    """Handle show command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments.
    """
    try:
        metadata = load_experiment(args.experiment_id)
        if args.json:
            print(json.dumps(metadata.to_dict(), indent=2, default=str))
        else:
            print_experiment_details(metadata)
    except FileNotFoundError:
        print(f"Error: Experiment '{args.experiment_id}' not found", file=sys.stderr)
        sys.exit(1)


def cmd_compare(args: argparse.Namespace) -> None:
    """Handle compare command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments.
    """
    try:
        comparison = compare_experiments(args.exp_id_1, args.exp_id_2)
        if args.json:
            print(json.dumps(comparison, indent=2, default=str))
        else:
            print_comparison(comparison)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Query experiments entrypoint."""
    parser = argparse.ArgumentParser(
        description="Query and compare Quantum Nematode experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments with optional filtering")
    list_parser.add_argument("--env-type", help="Filter by environment type (static/dynamic)")
    list_parser.add_argument("--brain-type", help="Filter by brain type")
    list_parser.add_argument(
        "--min-success-rate",
        type=float,
        help="Minimum success rate (0.0-1.0)",
    )
    list_parser.add_argument("--since", help="Only show experiments after date (ISO format)")
    list_parser.add_argument("--limit", type=int, help="Maximum number of results")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show detailed experiment information")
    show_parser.add_argument("experiment_id", help="Experiment ID to display")
    show_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two experiments")
    compare_parser.add_argument("exp_id_1", help="First experiment ID")
    compare_parser.add_argument("exp_id_2", help="Second experiment ID")
    compare_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "list":
        cmd_list(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
