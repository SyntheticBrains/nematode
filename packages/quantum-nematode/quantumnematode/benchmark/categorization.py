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
    quantum_brains = {"modular", "qmodular", "qvarcircuit", "qqlearning"}
    return brain_type in quantum_brains


def get_environment_category(
    grid_size: int,
    *,
    predators_enabled: bool = False,
) -> str:
    """Get environment category string.

    Parameters
    ----------
    grid_size : int
        Grid size.
    predators_enabled : bool, optional
        Whether predators are enabled, by default False.

    Returns
    -------
    str
        Environment category (e.g., "foraging_small", "predator_small").
    """
    # Size category based on grid size
    size_category = "small" if grid_size <= 20 else "medium" if grid_size <= 50 else "large"

    # Predator evasion vs foraging
    if predators_enabled:
        return f"predator_{size_category}"
    return f"foraging_{size_category}"


def determine_benchmark_category(metadata: ExperimentMetadata) -> str:
    """Determine benchmark category from experiment metadata.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata.

    Returns
    -------
    str
        Benchmark category string (e.g., "foraging_medium_quantum",
        "predator_small_quantum").
    """
    env_category = get_environment_category(
        metadata.environment.grid_size,
        predators_enabled=metadata.environment.predators_enabled,
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
        Relative directory path (e.g., "foraging_medium/quantum").
    """
    # Split category into environment and brain class
    parts = category.rsplit("_", 1)
    if len(parts) != 2:
        msg = f"Invalid category format: {category}"
        raise ValueError(msg)

    env_category, brain_class = parts
    return f"{env_category}/{brain_class}"
