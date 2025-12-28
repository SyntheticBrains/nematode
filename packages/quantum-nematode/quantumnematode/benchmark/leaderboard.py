# pragma: no cover

"""Leaderboard generation for NematodeBench benchmarks.

This module generates leaderboard tables from NematodeBenchSubmission JSON files.
"""

import json
import re
from pathlib import Path

from quantumnematode.experiment.metadata import StatValue
from quantumnematode.experiment.submission import NematodeBenchSubmission
from quantumnematode.logging_config import logger

# Default benchmarks storage directory
BENCHMARKS_DIR = Path.cwd() / "benchmarks"


def load_submission(filepath: Path) -> NematodeBenchSubmission | None:
    """Load a NematodeBench submission from JSON file.

    Parameters
    ----------
    filepath : Path
        Path to the submission JSON file.

    Returns
    -------
    NematodeBenchSubmission | None
        Parsed submission or None if not a valid NematodeBench format.
    """
    try:
        with filepath.open() as f:
            data = json.load(f)

        # Check if this is NematodeBench format (has submission_id and sessions)
        if "submission_id" not in data or "sessions" not in data:
            return None

        return NematodeBenchSubmission.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
        logger.warning(f"Failed to load submission {filepath.name}: {e}")
        return None


def list_benchmarks(category: str | None = None) -> list[NematodeBenchSubmission]:
    """List all NematodeBench submissions.

    Parameters
    ----------
    category : str | None, optional
        Category to filter by (None for all categories).
        Format: "foraging_small/classical" or "foraging_small_classical".

    Returns
    -------
    list[NematodeBenchSubmission]
        List of submissions, sorted by composite score (descending).
    """
    submissions: list[NematodeBenchSubmission] = []

    if not BENCHMARKS_DIR.exists():
        return submissions

    # Skip legacy folder
    for json_file in BENCHMARKS_DIR.rglob("*.json"):
        if "legacy" in str(json_file):
            continue

        submission = load_submission(json_file)
        if submission is None:
            continue

        # Apply category filter if specified
        if category:
            # Normalize category format for comparison
            normalized_category = category.replace("/", "_").replace("-", "_")
            submission_category = submission.category.replace("/", "_").replace("-", "_")
            if normalized_category != submission_category:
                continue

        submissions.append(submission)

    # Sort by mean composite score (descending)
    submissions.sort(key=lambda s: s.metrics.composite_score.mean, reverse=True)

    return submissions


def format_stat_value(stat: StatValue | dict, fmt: str = ".2f", *, is_percent: bool = False) -> str:
    """Format a StatValue as 'mean ± std'.

    Parameters
    ----------
    stat : StatValue | dict
        The statistical value to format.
    fmt : str
        Format string for numbers.
    is_percent : bool
        If True, format as percentage.

    Returns
    -------
    str
        Formatted string like "0.87 ± 0.03" or "87.0% ± 3.0%".
    """
    if stat is None:
        return "N/A"

    if isinstance(stat, dict):
        mean = stat.get("mean", 0)
        std = stat.get("std", 0)
    else:
        mean = stat.mean
        std = stat.std

    if is_percent:
        return f"{mean * 100:{fmt}}% ± {std * 100:{fmt}}%"
    return f"{mean:{fmt}} ± {std:{fmt}}"


def format_benchmark_row(submission: NematodeBenchSubmission) -> dict[str, str]:
    """Format submission data for table row.

    Parameters
    ----------
    submission : NematodeBenchSubmission
        The NematodeBench submission to format.

    Returns
    -------
    dict[str, str]
        Formatted row data.
    """
    contributor = submission.contributor
    github = submission.github_username
    contributor_display = f"@{github}" if github else contributor

    date = submission.timestamp.strftime("%Y-%m-%d")

    row = {
        "brain": submission.brain_type,
        "score": format_stat_value(submission.metrics.composite_score, ".3f"),
        "success_rate": format_stat_value(submission.metrics.success_rate, ".1f", is_percent=True),
        "learning_speed": format_stat_value(submission.metrics.learning_speed, ".2f"),
        "stability": format_stat_value(submission.metrics.stability, ".2f"),
        "distance_efficiency": "N/A",
        "sessions": str(submission.total_sessions),
        "runs": str(submission.total_runs),
        "contributor": contributor_display,
        "date": date,
    }

    # Add distance efficiency if available
    if submission.metrics.distance_efficiency:
        row["distance_efficiency"] = format_stat_value(submission.metrics.distance_efficiency, ".2f")

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
    submissions = list_benchmarks(category)[:limit]

    if not submissions:
        return "_No NematodeBench submissions yet._\n"

    # Define headers
    headers = [
        "Brain",
        "Score",
        "Success Rate",
        "Learning Speed",
        "Stability",
        "Distance Efficiency",
        "Sessions",
        "Contributor",
        "Date",
    ]

    header_line = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join(["---"] * len(headers)) + "|"

    rows = []
    for submission in submissions:
        row_data = format_benchmark_row(submission)
        row = (
            f"| {row_data['brain']} | {row_data['score']} | "
            f"{row_data['success_rate']} | {row_data['learning_speed']} | "
            f"{row_data['stability']} | {row_data['distance_efficiency']} | "
            f"{row_data['sessions']} | {row_data['contributor']} | "
            f"{row_data['date']} |"
        )
        rows.append(row)

    return "\n".join([header_line, separator, *rows]) + "\n"


def generate_readme_section() -> str:
    """Generate "Current Leaders" section content for README.md.

    Shows top 3 benchmarks per category, only for categories with submissions.

    Returns
    -------
    str
        Markdown content for the Current Leaders section.
    """
    sections = []

    categories = [
        ("Foraging Small - Classical", "foraging_small_classical"),
        ("Foraging Small - Quantum", "foraging_small_quantum"),
        ("Foraging Medium - Classical", "foraging_medium_classical"),
        ("Foraging Medium - Quantum", "foraging_medium_quantum"),
        ("Predator Small - Classical", "predator_small_classical"),
        ("Predator Small - Quantum", "predator_small_quantum"),
    ]

    # Collect all submissions to check if any exist
    all_submissions = []
    for _, category_id in categories:
        all_submissions.extend(list_benchmarks(category_id))

    if not all_submissions:
        sections.append(
            "*No NematodeBench submissions yet. Be the first to submit!*",
        )
    else:
        # Show top 3 performers from categories with submissions
        for display_name, category_id in categories:
            category_submissions = list_benchmarks(category_id)
            if category_submissions:
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
        "foraging_small_quantum",
        "foraging_small_classical",
        "foraging_medium_quantum",
        "foraging_medium_classical",
        "foraging_large_quantum",
        "foraging_large_classical",
        "predator_small_quantum",
        "predator_small_classical",
        "predator_medium_quantum",
        "predator_medium_classical",
        "predator_large_quantum",
        "predator_large_classical",
    ]

    leaderboards = {}
    for category in categories:
        table = generate_category_table(category, limit=20)
        leaderboards[category] = table

    return leaderboards


def generate_benchmarks_doc() -> str:
    """Generate complete BENCHMARKS.md leaderboard section.

    Returns
    -------
    str
        Markdown content for the Leaderboards section.
    """
    sections = []

    category_groups = [
        (
            "### Static Maze",
            [
                ("#### Quantum Architectures", "static_maze_quantum"),
                ("#### Classical Architectures", "static_maze_classical"),
            ],
        ),
        (
            "### Foraging Small (≤20x20)",
            [
                ("#### Quantum Architectures", "foraging_small_quantum"),
                ("#### Classical Architectures", "foraging_small_classical"),
            ],
        ),
        (
            "### Foraging Medium (≤50x50)",
            [
                ("#### Quantum Architectures", "foraging_medium_quantum"),
                ("#### Classical Architectures", "foraging_medium_classical"),
            ],
        ),
        (
            "### Foraging Large (>50x50)",
            [
                ("#### Quantum Architectures", "foraging_large_quantum"),
                ("#### Classical Architectures", "foraging_large_classical"),
            ],
        ),
        (
            "### Predator Small (≤20x20)",
            [
                ("#### Quantum Architectures", "predator_small_quantum"),
                ("#### Classical Architectures", "predator_small_classical"),
            ],
        ),
        (
            "### Predator Medium (≤50x50)",
            [
                ("#### Quantum Architectures", "predator_medium_quantum"),
                ("#### Classical Architectures", "predator_medium_classical"),
            ],
        ),
        (
            "### Predator Large (>50x50)",
            [
                ("#### Quantum Architectures", "predator_large_quantum"),
                ("#### Classical Architectures", "predator_large_classical"),
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
    replacement = f"\\1{new_section}\n\\3"

    updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Write back
    readme_path.write_text(updated_content)


def generate_leaderboard_md() -> str:
    """Generate complete standalone LEADERBOARD.md content.

    Returns
    -------
    str
        Full markdown content for LEADERBOARD.md.
    """
    lines = [
        "# NematodeBench Leaderboard",
        "",
        "Auto-generated leaderboard for NematodeBench benchmark submissions.",
        "",
        "> **Note**: This file is auto-generated. Do not edit manually.",
        "> Run `uv run scripts/benchmark_submit.py regenerate` to update.",
        "",
        "For submission guidelines, see [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md).",
        "",
    ]

    # Add all category tables
    lines.append(generate_benchmarks_doc())

    lines.extend(
        [
            "",
            "---",
            "",
            "*Last updated: Auto-generated from benchmark submissions.*",
            "",
        ],
    )

    return "\n".join(lines)


def update_leaderboard(leaderboard_path: Path | str) -> None:
    """Update or create LEADERBOARD.md with latest benchmark data.

    Parameters
    ----------
    leaderboard_path : Path | str
        Path to LEADERBOARD.md file.
    """
    leaderboard_path = Path(leaderboard_path)
    content = generate_leaderboard_md()
    leaderboard_path.write_text(content)
