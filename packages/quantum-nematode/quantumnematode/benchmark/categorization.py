"""Benchmark categorization logic."""

from quantumnematode.experiment.metadata import ExperimentMetadata


def is_quantum_brain(brain_type: str) -> bool:
    """Check if brain type is quantum-based.

    Parameters
    ----------
    brain_type : str
        Brain type identifier.

    Returns
    -------
    bool
        True if quantum brain, False if classical.
    """
    quantum_brains = {"modular", "qmodular"}
    return brain_type in quantum_brains


def get_environment_category(env_type: str, grid_size: int) -> str:
    """Get environment category string.

    Parameters
    ----------
    env_type : str
        Environment type ("static" or "dynamic").
    grid_size : int
        Grid size.

    Returns
    -------
    str
        Environment category (e.g., "static_maze", "dynamic_small").
    """
    if env_type == "static":
        return "static_maze"

    # Dynamic foraging categories based on grid size
    if grid_size <= 20:
        return "dynamic_small"
    if grid_size <= 50:
        return "dynamic_medium"
    return "dynamic_large"


def determine_benchmark_category(metadata: ExperimentMetadata) -> str:
    """Determine benchmark category from experiment metadata.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata.

    Returns
    -------
    str
        Benchmark category string (e.g., "dynamic_medium_quantum").
    """
    env_category = get_environment_category(
        metadata.environment.type,
        metadata.environment.grid_size,
    )

    brain_class = "quantum" if is_quantum_brain(metadata.brain.type) else "classical"

    return f"{env_category}_{brain_class}"


def get_category_directory(category: str) -> str:
    """Get benchmark storage directory path for a category.

    Parameters
    ----------
    category : str
        Benchmark category.

    Returns
    -------
    str
        Relative directory path (e.g., "dynamic_medium/quantum").
    """
    # Split category into environment and brain class
    parts = category.rsplit("_", 1)
    if len(parts) != 2:
        msg = f"Invalid category format: {category}"
        raise ValueError(msg)

    env_category, brain_class = parts
    return f"{env_category}/{brain_class}"
