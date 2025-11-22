# pragma: no cover

"""Benchmark submission workflow and storage."""

import json
from pathlib import Path

from quantumnematode.benchmark.categorization import (
    determine_benchmark_category,
    get_category_directory,
)
from quantumnematode.benchmark.validation import BenchmarkValidationRules, validate_benchmark
from quantumnematode.experiment.metadata import BenchmarkMetadata, ExperimentMetadata
from quantumnematode.logging_config import logger

# Default benchmarks storage directory
BENCHMARKS_DIR = Path.cwd() / "benchmarks"


def ensure_benchmark_category_dir(category: str) -> Path:
    """Ensure benchmark category directory exists.

    Parameters
    ----------
    category : str
        Benchmark category.

    Returns
    -------
    Path
        Path to category directory.
    """
    category_dir = BENCHMARKS_DIR / get_category_directory(category)
    category_dir.mkdir(parents=True, exist_ok=True)
    return category_dir


def save_benchmark(
    metadata: ExperimentMetadata,
    contributor: str,
    github_username: str | None = None,
    notes: str | None = None,
    rules: BenchmarkValidationRules | None = None,
) -> Path:
    """Save experiment as a benchmark submission.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata to save as benchmark.
    contributor : str
        Contributor name.
    github_username : str | None, optional
        GitHub username.
    notes : str | None, optional
        Notes about optimization approach.
    rules : BenchmarkValidationRules | None, optional
        Validation rules (uses defaults if None).

    Returns
    -------
    Path
        Path to saved benchmark file.

    Raises
    ------
    BenchmarkValidationError
        If validation fails.
    """
    # Determine category
    category = determine_benchmark_category(metadata)

    # Add benchmark metadata
    metadata.benchmark = BenchmarkMetadata(
        contributor=contributor,
        github_username=github_username,
        category=category,
        notes=notes,
        verified=False,  # Requires manual verification
    )

    # Validate benchmark
    validate_benchmark(metadata, rules)

    # Save to appropriate directory
    category_dir = ensure_benchmark_category_dir(category)
    filepath = category_dir / f"{metadata.experiment_id}.json"

    # Write atomically
    temp_filepath = filepath.with_suffix(".tmp")
    try:
        with temp_filepath.open("w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        temp_filepath.replace(filepath)
    except Exception as e:
        if temp_filepath.exists():
            temp_filepath.unlink()
        logger.error(f"Failed to save benchmark: {e}")
        raise
    else:
        logger.info(f"Benchmark saved to {filepath}")
        logger.info(f"Category: {category}")
        logger.info("Next steps:")
        logger.info("  1. Review the benchmark file")
        logger.info("  2. Create a pull request adding this file")
        logger.info("  3. Maintainers will verify and merge")
        return filepath


def load_benchmark(benchmark_id: str, category: str) -> ExperimentMetadata:
    """Load benchmark from file.

    Parameters
    ----------
    benchmark_id : str
        Benchmark experiment ID.
    category : str
        Benchmark category.

    Returns
    -------
    ExperimentMetadata
        Loaded benchmark metadata.

    Raises
    ------
    FileNotFoundError
        If benchmark file doesn't exist.
    """
    category_dir = BENCHMARKS_DIR / get_category_directory(category)
    filepath = category_dir / f"{benchmark_id}.json"

    if not filepath.exists():
        msg = f"Benchmark {benchmark_id} not found in category {category}"
        raise FileNotFoundError(msg)

    with filepath.open() as f:
        data = json.load(f)

    return ExperimentMetadata.from_dict(data)


def list_benchmarks(category: str | None = None) -> list[ExperimentMetadata]:
    """List all benchmarks, optionally filtered by category.

    Parameters
    ----------
    category : str | None, optional
        Category to filter by (None for all categories).

    Returns
    -------
    list[ExperimentMetadata]
        List of benchmark metadata, sorted by success rate (descending).
    """
    benchmarks = []

    if category:
        # Load from specific category
        try:
            category_dir = BENCHMARKS_DIR / get_category_directory(category)
            if category_dir.exists():
                for filepath in category_dir.glob("*.json"):
                    try:
                        with filepath.open() as f:
                            data = json.load(f)
                        metadata = ExperimentMetadata.from_dict(data)
                        benchmarks.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to load benchmark {filepath.name}: {e}")
        except ValueError:
            logger.warning(f"Invalid category format: {category}")
    # Load all benchmarks from all categories
    elif BENCHMARKS_DIR.exists():
        for json_file in BENCHMARKS_DIR.rglob("*.json"):
            if json_file.name == ".gitkeep":
                continue
            try:
                with json_file.open() as f:
                    data = json.load(f)
                metadata = ExperimentMetadata.from_dict(data)
                benchmarks.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load benchmark {json_file.name}: {e}")

    # Sort by composite score (descending), fallback to success rate if no convergence data
    def sort_key(b: ExperimentMetadata) -> tuple[float, float, float]:
        # Primary: Composite benchmark score (if available)
        composite = (
            b.results.composite_benchmark_score
            if b.results.composite_benchmark_score is not None
            else 0.0
        )
        # Secondary: Success rate (for legacy benchmarks or tiebreaker)
        success = b.results.success_rate
        # Tertiary: Foods collected (for foraging environments)
        foods = b.results.avg_foods_collected if b.results.avg_foods_collected is not None else 0.0
        return (composite, success, foods)

    benchmarks.sort(key=sort_key, reverse=True)

    return benchmarks
