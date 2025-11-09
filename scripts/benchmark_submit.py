#!/usr/bin/env python
"""CLI tool for managing benchmark submissions."""

import argparse
import sys

from quantumnematode.benchmark import generate_leaderboards, generate_readme_section, save_benchmark
from quantumnematode.benchmark.submission import list_benchmarks
from quantumnematode.experiment import load_experiment


def cmd_submit(args: argparse.Namespace) -> None:
    """Handle submit command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments.
    """
    try:
        # Load experiment
        metadata = load_experiment(args.experiment_id)

        # Get contributor info interactively if not provided
        contributor = args.contributor
        if not contributor:
            contributor = input("Contributor name (required): ").strip()
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

        # Save benchmark
        filepath = save_benchmark(
            metadata=metadata,
            contributor=contributor,
            github_username=github_username,
            notes=notes,
        )

        print(f"\n✓ Benchmark saved successfully: {filepath}")
        print("\nNext steps:")
        print("  1. Review the benchmark file")
        print("  2. Run: git add", str(filepath))
        print("  3. Run: git commit -m 'Add benchmark for [description]'")
        print("  4. Create a pull request")

    except FileNotFoundError:
        print(f"Error: Experiment '{args.experiment_id}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_leaderboard(args: argparse.Namespace) -> None:
    """Handle leaderboard command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments.
    """
    if args.category:
        # Show specific category
        benchmarks = list_benchmarks(args.category)
        if not benchmarks:
            print(f"No benchmarks found for category: {args.category}")
            return

        print(f"\n{'=' * 80}")
        print(f"Leaderboard: {args.category}")
        print(f"{'=' * 80}\n")

        for i, benchmark in enumerate(benchmarks[:args.limit], 1):
            contributor = benchmark.benchmark.contributor if benchmark.benchmark else "Unknown"
            print(f"{i}. {benchmark.brain.type} - {benchmark.results.success_rate:.0%} success")
            print(f"   {contributor} | {benchmark.timestamp.strftime('%Y-%m-%d')}")
            print(f"   Avg Steps: {benchmark.results.avg_steps:.0f}")
            if benchmark.results.avg_foods_collected:
                print(f"   Avg Foods: {benchmark.results.avg_foods_collected:.1f}")
            print()
    else:
        # Show all categories
        categories = [
            ("Static Maze - Quantum", "static_maze_quantum"),
            ("Static Maze - Classical", "static_maze_classical"),
            ("Dynamic Small - Quantum", "dynamic_small_quantum"),
            ("Dynamic Small - Classical", "dynamic_small_classical"),
            ("Dynamic Medium - Quantum", "dynamic_medium_quantum"),
            ("Dynamic Medium - Classical", "dynamic_medium_classical"),
            ("Dynamic Large - Quantum", "dynamic_large_quantum"),
            ("Dynamic Large - Classical", "dynamic_large_classical"),
        ]

        print("\n" + "=" * 80)
        print("Benchmark Leaderboards Summary")
        print("=" * 80 + "\n")

        for title, category in categories:
            benchmarks = list_benchmarks(category)
            print(f"{title}: {len(benchmarks)} benchmark(s)")
            if benchmarks:
                top = benchmarks[0]
                contributor = top.benchmark.contributor if top.benchmark else "Unknown"
                print(f"  Top: {top.brain.type} - {top.results.success_rate:.0%} by {contributor}")
            print()


def cmd_regenerate(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Handle regenerate command.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments.
    """
    print("Generating leaderboards...")

    # Generate README section
    readme_section = generate_readme_section()
    print("\nREADME.md benchmark section generated.")
    print("=" * 80)
    print(readme_section)
    print("=" * 80)

    # Generate full leaderboards
    leaderboards = generate_leaderboards()
    print(f"\nGenerated {len(leaderboards)} category leaderboards.")

    print("\nTo update documentation:")
    print("  1. Copy the README section above into README.md")
    print("  2. Update BENCHMARKS.md with full leaderboard tables")
    print("  3. Commit and push changes")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage Quantum Nematode benchmark submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit an experiment as a benchmark")
    submit_parser.add_argument("experiment_id", help="Experiment ID to promote to benchmark")
    submit_parser.add_argument("--contributor", help="Contributor name")
    submit_parser.add_argument("--github", help="GitHub username")
    submit_parser.add_argument("--notes", help="Optimization notes")
    submit_parser.add_argument("--no-prompt", action="store_true", help="Don't prompt for optional fields")

    # Leaderboard command
    leaderboard_parser = subparsers.add_parser("leaderboard", help="View benchmark leaderboards")
    leaderboard_parser.add_argument("--category", help="Show specific category")
    leaderboard_parser.add_argument("--limit", type=int, default=10, help="Max results per category")

    # Regenerate command
    subparsers.add_parser("regenerate", help="Regenerate leaderboard documentation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "leaderboard":
        cmd_leaderboard(args)
    elif args.command == "regenerate":
        cmd_regenerate(args)


if __name__ == "__main__":
    main()
