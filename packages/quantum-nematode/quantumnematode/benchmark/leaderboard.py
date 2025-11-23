# pragma: no cover

"""Leaderboard generation for benchmarks."""

import re
from pathlib import Path

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

    # Convergence indicator: ✓ if converged, ⚠ if not
    conv_indicator = "✓" if benchmark.results.converged else "⚠"

    row = {
        "brain": benchmark.brain.type,
        "score": f"{benchmark.results.composite_benchmark_score:.3f}"
        if benchmark.results.composite_benchmark_score
        else "N/A",
        "success_rate": f"{benchmark.results.post_convergence_success_rate:.0%}"
        if benchmark.results.post_convergence_success_rate is not None
        else f"{benchmark.results.success_rate:.0%}",  # Fallback for legacy benchmarks
        "avg_steps": f"{benchmark.results.post_convergence_avg_steps:.0f}"
        if benchmark.results.post_convergence_avg_steps is not None
        else f"{benchmark.results.avg_steps:.0f}",  # Fallback for legacy benchmarks
        "converge_run": f"{benchmark.results.convergence_run}"
        if benchmark.results.convergence_run
        else "N/A",
        "stability": f"{benchmark.results.post_convergence_variance:.3f}"
        if benchmark.results.post_convergence_variance is not None
        else "N/A",
        "conv_indicator": conv_indicator,
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
            "Score",
            "Success%",
            "Steps",
            "Converge@Run",
            "Stability",
            "Contributor",
            "Date",
        ]
        header_line = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join(["---"] * len(headers)) + "|"
    else:
        headers = [
            "Brain",
            "Score",
            "Success%",
            "Steps",
            "Converge@Run",
            "Stability",
            "Contributor",
            "Date",
        ]
        header_line = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join(["---"] * len(headers)) + "|"

    rows = []
    for benchmark in benchmarks:
        row_data = format_benchmark_row(benchmark)
        # Same format for both foraging and static (convergence metrics apply to both)
        row = (
            f"| {row_data['conv_indicator']} {row_data['brain']} | {row_data['score']} | "
            f"{row_data['success_rate']} | {row_data['avg_steps']} | "
            f"{row_data['converge_run']} | {row_data['stability']} | "
            f"{row_data['contributor']} | {row_data['date']} |"
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
        ("Dynamic Predator Small - Quantum", "dynamic_predator_small_quantum"),
        ("Dynamic Predator Small - Classical", "dynamic_predator_small_classical"),
        ("Dynamic Predator Medium - Quantum", "dynamic_predator_medium_quantum"),
        ("Dynamic Predator Medium - Classical", "dynamic_predator_medium_classical"),
        ("Dynamic Predator Large - Quantum", "dynamic_predator_large_quantum"),
        ("Dynamic Predator Large - Classical", "dynamic_predator_large_classical"),
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
        "dynamic_predator_small_quantum",
        "dynamic_predator_small_classical",
        "dynamic_predator_medium_quantum",
        "dynamic_predator_medium_classical",
        "dynamic_predator_large_quantum",
        "dynamic_predator_large_classical",
    ]

    leaderboards = {}
    for category in categories:
        table = generate_category_table(category, limit=20)
        leaderboards[category] = table

    return leaderboards


def generate_benchmarks_doc() -> str:
    """Generate complete BENCHMARKS.md leaderboard section.

    This generates the leaderboard tables that should replace the content
    under the "## Leaderboards" heading in BENCHMARKS.md.

    Returns
    -------
    str
        Markdown content for the Leaderboards section.
    """
    sections = []

    # Category mappings with full names
    category_groups = [
        (
            "### Static Maze",
            [
                ("#### Quantum Architectures", "static_maze_quantum"),
                ("#### Classical Architectures", "static_maze_classical"),
            ],
        ),
        (
            "### Dynamic Small (≤20x20)",
            [
                ("#### Quantum Architectures", "dynamic_small_quantum"),
                ("#### Classical Architectures", "dynamic_small_classical"),
            ],
        ),
        (
            "### Dynamic Medium (≤50x50)",
            [
                ("#### Quantum Architectures", "dynamic_medium_quantum"),
                ("#### Classical Architectures", "dynamic_medium_classical"),
            ],
        ),
        (
            "### Dynamic Large (>50x50)",
            [
                ("#### Quantum Architectures", "dynamic_large_quantum"),
                ("#### Classical Architectures", "dynamic_large_classical"),
            ],
        ),
        (
            "### Dynamic Predator Small (≤20x20)",
            [
                ("#### Quantum Architectures", "dynamic_predator_small_quantum"),
                ("#### Classical Architectures", "dynamic_predator_small_classical"),
            ],
        ),
        (
            "### Dynamic Predator Medium (≤50x50)",
            [
                ("#### Quantum Architectures", "dynamic_predator_medium_quantum"),
                ("#### Classical Architectures", "dynamic_predator_medium_classical"),
            ],
        ),
        (
            "### Dynamic Predator Large (>50x50)",
            [
                ("#### Quantum Architectures", "dynamic_predator_large_quantum"),
                ("#### Classical Architectures", "dynamic_predator_large_classical"),
            ],
        ),
    ]

    for group_title, subcategories in category_groups:
        sections.append(f"{group_title}\n")

        for subcat_title, category_id in subcategories:
            sections.append(f"{subcat_title}\n")
            table = generate_category_table(category_id, limit=20)
            sections.append(table)

    return "\n".join(sections)


def update_readme(readme_path: Path | str) -> None:
    """Update README.md with latest benchmark leaderboard.

    Parameters
    ----------
    readme_path : Path | str
        Path to README.md file.
    """
    readme_path = Path(readme_path)
    content = readme_path.read_text()

    # Generate new section
    new_section = generate_readme_section()

    # Find the section to replace (between "### Current Leaders" and "See [BENCHMARKS.md]")
    pattern = r"(### Current Leaders\n\n)(.*?)(See \[BENCHMARKS\.md\])"
    replacement = f"\\1{new_section}\\n\\3"

    updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Write back
    readme_path.write_text(updated_content)


def update_benchmarks_doc(benchmarks_path: Path | str) -> None:
    """Update BENCHMARKS.md with latest leaderboard tables.

    Parameters
    ----------
    benchmarks_path : Path | str
        Path to BENCHMARKS.md file.
    """
    benchmarks_path = Path(benchmarks_path)
    content = benchmarks_path.read_text()

    # Generate new leaderboards section
    new_section = generate_benchmarks_doc()

    # Find and replace the Leaderboards section (between heading and horizontal rule)
    # Match everything between "## Leaderboards\n\n" and "\n---\n"
    pattern = r"(## Leaderboards\n\n).*?(\n---\n)"
    replacement = f"\\1{new_section}\\2"

    updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Write back
    benchmarks_path.write_text(updated_content)
