#!/usr/bin/env python
"""CLI tool for managing NematodeBench benchmark submissions.

NematodeBench requires 10+ independent experiment sessions for official
benchmark submissions. Use this tool to aggregate experiments, validate
seeds, and submit to the leaderboard.
"""

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from quantumnematode.benchmark import (
    generate_leaderboards,
    generate_readme_section,
    list_benchmarks,
    update_leaderboard,
    update_readme,
)
from quantumnematode.brain.arch.dtypes import QUANTUM_BRAIN_TYPES, BrainType
from quantumnematode.experiment import (
    MIN_SESSIONS_REQUIRED,
    AggregateMetrics,
    ExperimentMetadata,
    NematodeBenchSubmission,
    SessionReference,
    StatValue,
    validate_submission,
)


def load_experiment_from_folder(experiment_dir: Path) -> ExperimentMetadata:
    """Load experiment metadata from an experiment folder.

    Parameters
    ----------
    experiment_dir : Path
        Path to experiment folder containing <id>.json and config files.

    Returns
    -------
    ExperimentMetadata
        Loaded experiment metadata.

    Raises
    ------
    FileNotFoundError
        If no JSON file found in the folder.
    """
    # Find JSON file in the folder
    json_files = list(experiment_dir.glob("*.json"))
    if not json_files:
        msg = f"No JSON file found in {experiment_dir}"
        raise FileNotFoundError(msg)

    if len(json_files) > 1:
        print(
            f"Warning: Multiple JSON files in {experiment_dir}, using {json_files[0].name}",
            file=sys.stderr,
        )

    json_path = json_files[0]
    with json_path.open() as f:
        data = json.load(f)

    return ExperimentMetadata.from_dict(data)


# Valid benchmark categories
VALID_CATEGORIES = [
    # Static maze
    "static_maze/quantum",
    "static_maze/classical",
    # Foraging (dynamic without predators)
    "foraging_small/quantum",
    "foraging_small/classical",
    "foraging_medium/quantum",
    "foraging_medium/classical",
    "foraging_large/quantum",
    "foraging_large/classical",
    # Predator evasion (dynamic with predators)
    "predator_small/quantum",
    "predator_small/classical",
    "predator_medium/quantum",
    "predator_medium/classical",
    "predator_large/quantum",
    "predator_large/classical",
]


def validate_category(category: str) -> tuple[bool, str]:
    """Validate that a category string is valid.

    Parameters
    ----------
    category : str
        Category string to validate.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message if invalid)
    """
    if category in VALID_CATEGORIES:
        return True, ""

    # Provide helpful error message
    return False, (
        f"Invalid category: '{category}'\n"
        f"Valid categories are:\n" + "\n".join(f"  - {cat}" for cat in VALID_CATEGORIES)
    )


def determine_category(experiment: ExperimentMetadata) -> str:
    """Determine the benchmark category from experiment metadata.

    Parameters
    ----------
    experiment : ExperimentMetadata
        Experiment metadata.

    Returns
    -------
    str
        Category string like "foraging_small/classical".
    """
    env = experiment.environment
    brain = experiment.brain

    # Determine environment category
    if env.type == "static":
        env_category = "static_maze"
    elif env.predators_enabled:
        if env.grid_size <= 20:
            env_category = "predator_small"
        elif env.grid_size <= 50:
            env_category = "predator_medium"
        else:
            env_category = "predator_large"
    elif env.grid_size <= 20:
        env_category = "foraging_small"
    elif env.grid_size <= 50:
        env_category = "foraging_medium"
    else:
        env_category = "foraging_large"

    # Determine brain category using explicit mapping
    try:
        brain_type_enum = BrainType(brain.type)
        brain_category = "quantum" if brain_type_enum in QUANTUM_BRAIN_TYPES else "classical"
    except ValueError:
        # Unknown brain type - default to classical with warning
        print(f"Warning: Unknown brain type '{brain.type}', defaulting to classical category")
        brain_category = "classical"

    return f"{env_category}/{brain_category}"


def _stat_value_or_zero(values: list[float]) -> StatValue:
    """Create StatValue from values, or return all-zeros if empty.

    Parameters
    ----------
    values : list[float]
        List of values to aggregate.

    Returns
    -------
    StatValue
        Aggregated statistics, or zeros if input is empty.
    """
    if values:
        return StatValue.from_values(values)
    return StatValue(mean=0.0, std=0.0, min=0.0, max=0.0)


def aggregate_metrics(experiments: list[ExperimentMetadata]) -> AggregateMetrics:
    """Aggregate metrics across multiple experiments using StatValue.

    Parameters
    ----------
    experiments : list[ExperimentMetadata]
        List of experiment metadata to aggregate.

    Returns
    -------
    AggregateMetrics
        Aggregated metrics with mean/std/min/max across sessions.
    """
    # Collect session-level metrics
    success_rates = [exp.results.success_rate for exp in experiments]
    composite_scores = [
        exp.results.composite_benchmark_score
        for exp in experiments
        if exp.results.composite_benchmark_score is not None
    ]
    distance_efficiencies = [
        exp.results.avg_distance_efficiency
        for exp in experiments
        if exp.results.avg_distance_efficiency is not None
    ]
    learning_speeds = [
        exp.results.learning_speed for exp in experiments if exp.results.learning_speed is not None
    ]
    learning_speed_episodes = [
        float(exp.results.learning_speed_episodes)
        for exp in experiments
        if exp.results.learning_speed_episodes is not None
    ]
    stabilities = [
        exp.results.stability for exp in experiments if exp.results.stability is not None
    ]

    return AggregateMetrics(
        success_rate=_stat_value_or_zero(success_rates),
        composite_score=_stat_value_or_zero(composite_scores),
        # distance_efficiency is optional (only for dynamic envs) - None if not applicable
        distance_efficiency=StatValue.from_values(distance_efficiencies)
        if distance_efficiencies
        else None,
        learning_speed=_stat_value_or_zero(learning_speeds),
        learning_speed_episodes=_stat_value_or_zero(learning_speed_episodes),
        stability=_stat_value_or_zero(stabilities),
    )


def cmd_submit_nematodebench(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    """Handle NematodeBench multi-experiment submission command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments.
    """
    experiment_paths = [Path(p) for p in args.experiments]

    # Validate all paths exist
    for path in experiment_paths:
        if not path.exists():
            print(f"Error: Experiment folder not found: {path}", file=sys.stderr)
            sys.exit(1)
        if not path.is_dir():
            print(f"Error: Path is not a directory: {path}", file=sys.stderr)
            sys.exit(1)

    # Load all experiments
    print(f"Loading {len(experiment_paths)} experiments...")
    experiments: list[ExperimentMetadata] = []
    for path in experiment_paths:
        try:
            exp = load_experiment_from_folder(path)
            experiments.append(exp)
            print(
                f"  ✓ {exp.experiment_id}: {exp.results.success_rate:.1%} success, "
                f"{exp.results.total_runs} runs",
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  ✗ Failed to load {path}: {e}", file=sys.stderr)
            sys.exit(1)

    # Validate submission
    print("\nValidating submission...")
    is_valid, errors = validate_submission(experiments)
    if not is_valid:
        print("\n✗ Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
    print("  ✓ All validations passed")

    # Determine category
    category = args.category or determine_category(experiments[0])

    # Validate category
    is_valid_category, category_error = validate_category(category)
    if not is_valid_category:
        print(f"\nError: {category_error}", file=sys.stderr)
        sys.exit(1)

    print(f"\nCategory: {category}")

    # Get contributor info
    contributor = args.contributor
    if not contributor:
        contributor = input("\nContributor name (required): ").strip()
        if not contributor:
            print("Error: Contributor name is required", file=sys.stderr)
            sys.exit(1)

    github_username = args.github
    if not github_username and not args.no_prompt:
        github_input = input("GitHub username (optional, press Enter to skip): ").strip()
        github_username = github_input if github_input else None

    notes = args.notes
    if not notes and not args.no_prompt:
        notes_input = input("Optimization notes (optional, press Enter to skip): ").strip()
        notes = notes_input if notes_input else None

    # Copy experiments to artifacts/experiments/
    print("\nCopying experiments to artifacts/experiments/...")
    repo_root = Path(__file__).parent.parent
    artifacts_dir = repo_root / "artifacts" / "experiments"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    session_refs: list[SessionReference] = []
    for exp, src_path in zip(experiments, experiment_paths, strict=True):
        dest_path = artifacts_dir / exp.experiment_id
        if dest_path.exists():
            print(f"  ! {exp.experiment_id} already exists in artifacts, skipping copy")
        else:
            shutil.copytree(str(src_path), str(dest_path))
            print(f"  ✓ Copied {exp.experiment_id}")

        # Get master seed for session (first run's seed)
        session_seed = 0
        if exp.results.per_run_results and len(exp.results.per_run_results) > 0:
            session_seed = exp.results.per_run_results[0].seed
        else:
            print(
                f"  Warning: No per_run_results for {exp.experiment_id}, using seed=0",
                file=sys.stderr,
            )

        session_refs.append(
            SessionReference(
                experiment_id=exp.experiment_id,
                file_path=f"artifacts/experiments/{exp.experiment_id}",
                session_seed=session_seed,
                num_runs=exp.results.total_runs,
            ),
        )

    # Aggregate metrics
    print("\nAggregating metrics across sessions...")
    metrics = aggregate_metrics(experiments)
    print(f"  Success Rate: {metrics.success_rate.mean:.1%} ± {metrics.success_rate.std:.1%}")
    print(
        f"  Composite Score: {metrics.composite_score.mean:.3f} ± {metrics.composite_score.std:.3f}",
    )

    # Create submission
    submission_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    total_runs = sum(exp.results.total_runs for exp in experiments)

    # validate_submission already verified seed uniqueness, so this is True if we got here
    all_seeds_unique = True

    submission = NematodeBenchSubmission(
        submission_id=submission_id,
        brain_type=experiments[0].brain.type,
        brain_config=experiments[0].brain,
        environment=experiments[0].environment,
        category=category,
        sessions=session_refs,
        total_sessions=len(experiments),
        total_runs=total_runs,
        metrics=metrics,
        all_seeds_unique=all_seeds_unique,
        contributor=contributor,
        github_username=github_username,
        notes=notes,
    )

    # Save submission
    benchmarks_dir = repo_root / "benchmarks"
    category_dir = benchmarks_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    submission_path = category_dir / f"{submission_id}.json"

    with submission_path.open("w") as f:
        json.dump(submission.to_dict(), f, indent=2, default=str)

    print(f"\n✓ NematodeBench submission saved: {submission_path}")
    print(f"  Submission ID: {submission_id}")
    print(f"  Sessions: {len(experiments)}")
    print(f"  Total Runs: {total_runs}")
    print(f"  All Seeds Unique: {all_seeds_unique}")

    print("\nNext steps:")
    print("  1. Review the submission file")
    print(f"  2. Run: git add {submission_path} artifacts/experiments/")
    print("  3. Run: git commit -m 'Add NematodeBench submission for [description]'")
    print("  4. Create a pull request")


def cmd_leaderboard(args: argparse.Namespace) -> None:
    """Handle leaderboard command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments.
    """
    categories = [
        ("Static Maze - Quantum", "static_maze_quantum"),
        ("Static Maze - Classical", "static_maze_classical"),
        ("Foraging Small - Quantum", "foraging_small_quantum"),
        ("Foraging Small - Classical", "foraging_small_classical"),
        ("Foraging Medium - Quantum", "foraging_medium_quantum"),
        ("Foraging Medium - Classical", "foraging_medium_classical"),
        ("Foraging Large - Quantum", "foraging_large_quantum"),
        ("Foraging Large - Classical", "foraging_large_classical"),
        ("Predator Small - Quantum", "predator_small_quantum"),
        ("Predator Small - Classical", "predator_small_classical"),
        ("Predator Medium - Quantum", "predator_medium_quantum"),
        ("Predator Medium - Classical", "predator_medium_classical"),
        ("Predator Large - Quantum", "predator_large_quantum"),
        ("Predator Large - Classical", "predator_large_classical"),
    ]

    if args.category:
        # Show specific category
        submissions = list_benchmarks(args.category)
        if not submissions:
            print(f"No NematodeBench submissions found for category: {args.category}")
            return

        print(f"\n{'=' * 80}")
        print(f"NematodeBench Leaderboard: {args.category}")
        print(f"{'=' * 80}\n")

        for i, sub in enumerate(submissions[: args.limit], 1):
            contributor = f"@{sub.github_username}" if sub.github_username else sub.contributor
            m = sub.metrics
            print(f"{i}. {sub.brain_type}")
            print(f"   Score: {m.composite_score.mean:.3f} ± {m.composite_score.std:.3f}")
            print(f"   Success: {m.success_rate.mean:.1%} ± {m.success_rate.std:.1%}")
            print(f"   Sessions: {sub.total_sessions} | Runs: {sub.total_runs}")
            print(f"   Contributor: {contributor} | {sub.timestamp.strftime('%Y-%m-%d')}")
            print()
    else:
        # Show all categories
        print("\n" + "=" * 80)
        print("NematodeBench Leaderboards Summary")
        print("=" * 80 + "\n")

        total_submissions = 0
        for title, category in categories:
            submissions = list_benchmarks(category)
            total_submissions += len(submissions)
            print(f"{title}: {len(submissions)} submission(s)")
            if submissions:
                top = submissions[0]
                contributor = f"@{top.github_username}" if top.github_username else top.contributor
                print(
                    f"  Top: {top.brain_type} - "
                    f"{top.metrics.composite_score.mean:.3f} by {contributor}",
                )
            print()

        if total_submissions == 0:
            print("No NematodeBench submissions yet.")
            print("Submit your first benchmark with:")
            print("  uv run scripts/benchmark_submit.py --experiments <folder1> <folder2> ...")


def cmd_regenerate(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Handle regenerate command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments.
    """
    print("Generating leaderboards...")

    # Find repository root (assumes script is in scripts/ subdirectory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    readme_path = repo_root / "README.md"
    leaderboard_path = repo_root / "docs" / "nematodebench" / "LEADERBOARD.md"

    try:
        # Update README.md
        if readme_path.exists():
            update_readme(readme_path)
            print(f"✓ Updated {readme_path}")
        else:
            print(f"Warning: README.md not found at {readme_path}", file=sys.stderr)

        # Update/create LEADERBOARD.md
        update_leaderboard(leaderboard_path)
        print(f"✓ Updated {leaderboard_path}")

        # Generate preview of README section
        readme_section = generate_readme_section()
        print("\n" + "=" * 80)
        print("Preview of README.md 'Current Leaders' section:")
        print("=" * 80)
        print(readme_section)
        print("=" * 80)

        # Generate summary of leaderboards
        leaderboards = generate_leaderboards()
        print(f"\n✓ Generated {len(leaderboards)} category leaderboards.")

        print("\nNext steps:")
        print("  1. Review the updated files")
        print("  2. Run: git add README.md docs/nematodebench/LEADERBOARD.md")
        print("  3. Run: git commit -m 'Update benchmark leaderboards'")
        print("  4. Push changes")

    except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error regenerating documentation: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Submit benchmarks entrypoint."""
    parser = argparse.ArgumentParser(
        description="Manage Quantum Nematode benchmark submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NematodeBench submission (requires 10+ experiment sessions)
  uv run scripts/benchmark_submit.py --experiments experiments/exp1 experiments/exp2 ...

  # View leaderboards
  uv run scripts/benchmark_submit.py leaderboard

  # Regenerate documentation
  uv run scripts/benchmark_submit.py regenerate
""",
    )

    # NematodeBench submission arguments (top-level)
    parser.add_argument(
        "--experiments",
        nargs="+",
        help=f"Experiment folders to aggregate (min {MIN_SESSIONS_REQUIRED} for NematodeBench)",
    )
    parser.add_argument("--contributor", help="Contributor name")
    parser.add_argument("--github", help="GitHub username")
    parser.add_argument("--notes", help="Optimization notes")
    parser.add_argument("--category", help="Override auto-detected category")
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Don't prompt for optional fields",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Leaderboard command
    leaderboard_parser = subparsers.add_parser("leaderboard", help="View benchmark leaderboards")
    leaderboard_parser.add_argument("--category", help="Show specific category")
    leaderboard_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max results per category",
    )

    # Regenerate command
    subparsers.add_parser("regenerate", help="Regenerate leaderboard documentation")

    args = parser.parse_args()

    # Handle NematodeBench submission (top-level --experiments)
    if args.experiments:
        cmd_submit_nematodebench(args)
        return

    # Handle subcommands
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "leaderboard":
        cmd_leaderboard(args)
    elif args.command == "regenerate":
        cmd_regenerate(args)


if __name__ == "__main__":
    main()
