# pragma: no cover

"""Leaderboard generation for benchmarks."""

from quantumnematode.benchmark.submission import list_benchmarks
from quantumnematode.experiment.metadata import ExperimentMetadata


def format_benchmark_row(benchmark: ExperimentMetadata) -> dict[str, str]:
    """Format benchmark data for table row.

    Parameters
    ----------
    benchmark : ExperimentMetadata
        Benchmark metadata.

    Returns
    -------
    dict[str, str]
        Formatted row data.
    """
    contributor = benchmark.benchmark.contributor if benchmark.benchmark else "Unknown"
    github = benchmark.benchmark.github_username if benchmark.benchmark else None
    contributor_display = f"@{github}" if github else contributor

    date = benchmark.timestamp.strftime("%Y-%m-%d")

    row = {
        "brain": benchmark.brain.type,
        "success_rate": f"{benchmark.results.success_rate:.0%}",
        "avg_steps": f"{benchmark.results.avg_steps:.0f}",
        "contributor": contributor_display,
        "date": date,
    }

    # Add foraging-specific metrics if available
    if benchmark.results.avg_foods_collected is not None:
        row["foods_per_run"] = f"{benchmark.results.avg_foods_collected:.1f}"
    if benchmark.results.avg_distance_efficiency is not None:
        row["dist_eff"] = f"{benchmark.results.avg_distance_efficiency:.2f}"

    return row


def generate_category_table(category: str, limit: int = 10) -> str:
    """Generate markdown table for a category.

    Parameters
    ----------
    category : str
        Benchmark category.
    limit : int, optional
        Maximum number of entries to include.

    Returns
    -------
    str
        Markdown table string.
    """
    benchmarks = list_benchmarks(category)[:limit]

    if not benchmarks:
        return "_No benchmarks submitted yet._\n"

    # Determine columns based on environment type
    is_foraging = "dynamic" in category

    if is_foraging:
        headers = [
            "Brain",
            "Success Rate",
            "Avg Steps",
            "Foods/Run",
            "Dist Eff",
            "Contributor",
            "Date",
        ]
        header_line = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join(["---"] * len(headers)) + "|"
    else:
        headers = ["Brain", "Success Rate", "Avg Steps", "Contributor", "Date"]
        header_line = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join(["---"] * len(headers)) + "|"

    rows = []
    for benchmark in benchmarks:
        row_data = format_benchmark_row(benchmark)
        if is_foraging:
            row = (
                f"| {row_data['brain']} | {row_data['success_rate']} | "
                f"{row_data['avg_steps']} | {row_data.get('foods_per_run', 'N/A')} | "
                f"{row_data.get('dist_eff', 'N/A')} | {row_data['contributor']} | "
                f"{row_data['date']} |"
            )
        else:
            row = (
                f"| {row_data['brain']} | {row_data['success_rate']} | "
                f"{row_data['avg_steps']} | {row_data['contributor']} | {row_data['date']} |"
            )
        rows.append(row)

    return "\n".join([header_line, separator, *rows]) + "\n"


def generate_readme_section() -> str:
    """Generate "Current Leaders" section content for README.md.

    This generates only the dynamic content that goes under the "### Current Leaders"
    heading in README.md. Replace the content between "### Current Leaders" and
    "See [BENCHMARKS.md]" with the output of this function.

    Returns
    -------
    str
        Markdown content for the Current Leaders section only.
    """
    sections = []

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

    # Check if we have any benchmarks at all
    all_benchmarks = []
    for _, category_id in categories:
        all_benchmarks.extend(list_benchmarks(category_id))

    if not all_benchmarks:
        sections.append("*No benchmarks submitted yet. Be the first to set a benchmark!*")
    else:
        # Show top performers from all categories
        for display_name, category_id in categories:
            category_benchmarks = list_benchmarks(category_id)
            if category_benchmarks:
                sections.append(f"#### {display_name}\n")
                table = generate_category_table(category_id, limit=3)
                sections.append(table)

    return "\n".join(sections)


def generate_leaderboards() -> dict[str, str]:
    """Generate all leaderboard tables.

    Returns
    -------
    dict[str, str]
        Dictionary mapping category to markdown table.
    """
    categories = [
        "static_maze_quantum",
        "static_maze_classical",
        "dynamic_small_quantum",
        "dynamic_small_classical",
        "dynamic_medium_quantum",
        "dynamic_medium_classical",
        "dynamic_large_quantum",
        "dynamic_large_classical",
    ]

    leaderboards = {}
    for category in categories:
        table = generate_category_table(category, limit=20)
        leaderboards[category] = table

    return leaderboards
