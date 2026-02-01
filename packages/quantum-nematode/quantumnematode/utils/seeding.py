"""Seeding infrastructure for reproducible experiments.

This module provides utilities for generating, managing, and using random seeds
to ensure experiment reproducibility. It supports:

1. Automatic seed generation when not provided by user
2. Global seed setting for numpy and torch
3. Creating seeded random number generators
4. Per-run seed derivation for multi-run experiments

Usage:
    # Auto-generate seed if not provided
    seed = ensure_seed(user_seed)

    # Set global RNG states
    set_global_seed(seed)

    # Create a seeded numpy RNG for a specific component
    rng = get_rng(seed)
"""

import hashlib
import secrets

import numpy as np

# Maximum seed value (2^32 - 1, compatible with numpy and torch)
MAX_SEED = 2**32

# Global registry for tracking seeds used in current session
_seed_registry: dict[str, int] = {}


def generate_seed() -> int:
    """Generate a cryptographically random seed.

    Uses the secrets module to generate a high-quality random seed
    suitable for initializing random number generators.

    Returns
    -------
        A random integer in [0, 2^32)
    """
    return secrets.randbelow(MAX_SEED)


def ensure_seed(seed: int | None = None) -> int:
    """Ensure a seed is available, generating one if not provided.

    This is the primary interface for seed management. If a seed is provided,
    it is returned as-is. If None, a new random seed is generated.

    Args:
        seed: User-provided seed, or None to auto-generate

    Returns
    -------
        The provided seed or a newly generated one
    """
    if seed is not None:
        return seed
    return generate_seed()


def set_global_seed(seed: int) -> None:
    """Set global random seeds for numpy and torch.

    This ensures reproducibility across all random operations in the
    experiment by seeding both numpy and PyTorch's random number generators.

    Args:
        seed: The seed to use for global RNG state

    Note:
        PyTorch seeding is only performed if torch is available.
        This allows the module to work in environments without torch.
    """
    # Seed numpy
    np.random.seed(seed)  # noqa: NPY002

    # Seed torch if available
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Record in registry
    _seed_registry["global"] = seed


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Create a seeded numpy random number Generator.

    Creates an independent random number generator that won't affect
    the global numpy RNG state. This is preferred for component-specific
    randomness.

    Args:
        seed: Seed for the RNG, or None to use a random seed

    Returns
    -------
        A numpy Generator instance seeded with the given seed
    """
    actual_seed = ensure_seed(seed)
    return np.random.default_rng(actual_seed)


def derive_run_seed(base_seed: int, run_index: int) -> int:
    """Derive a seed for a specific run from a base seed.

    This ensures each run in a multi-run experiment has a unique but
    deterministic seed, allowing any individual run to be reproduced.

    Args:
        base_seed: The experiment's base seed
        run_index: The index of the run (0-based)

    Returns
    -------
        A derived seed for the specific run
    """
    # Use a simple but effective combination
    # Adding run_index ensures uniqueness, modulo ensures valid range
    return (base_seed + run_index) % MAX_SEED


def derive_episode_seed(base_seed: int, gen: int, candidate_idx: int, episode: int) -> int:
    """Derive a deterministic seed for a specific episode.

    Uses BLAKE2b hash of (base_seed, gen, candidate_idx, episode) to produce
    independent, reproducible seeds for each episode evaluation.

    Note: We use hashlib.blake2b instead of Python's hash() because hash()
    is salted (randomized per process since Python 3.3), which would break
    reproducibility across multiprocessing workers and separate runs.

    Args:
        base_seed: Base seed from --seed argument.
        gen: Generation number.
        candidate_idx: Index of candidate in population.
        episode: Episode number within evaluation.

    Returns
    -------
        Deterministic seed in valid range for numpy.
    """
    payload = f"{base_seed}:{gen}:{candidate_idx}:{episode}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") & 0xFFFF_FFFF


def register_seed(name: str, seed: int) -> None:
    """Register a seed in the session registry.

    Useful for tracking which seeds were used by different components
    for debugging and reproducibility verification.

    Args:
        name: Identifier for the seed (e.g., "environment", "brain")
        seed: The seed value being used
    """
    _seed_registry[name] = seed


def get_seed_registry() -> dict[str, int]:
    """Get a copy of the current seed registry.

    Returns
    -------
        Dictionary mapping component names to their seeds
    """
    return _seed_registry.copy()


def clear_seed_registry() -> None:
    """Clear the seed registry.

    Useful for starting a new experiment session.
    """
    _seed_registry.clear()
