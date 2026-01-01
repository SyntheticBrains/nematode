"""
Unified feature extraction layer for all brain architectures.

This module provides a single entry point for extracting sensory features from
BrainParams, ensuring consistent feature computation across quantum (ModularBrain)
and classical (PPOBrain) architectures.

Feature extraction modules are based on C. elegans sensory neuron types:
- ASE neurons: Combined chemotaxis (food + predator gradients)
- AWC, AWA neurons: Food-specific chemotaxis (appetitive/approach)
- ASH, ADL neurons: Nociception (aversive/escape from predators)
- AFD neurons: Thermotaxis (temperature sensing)
- URX, BAG neurons: Aerotaxis (oxygen sensing)
- ALM, PLM, AVM neurons: Mechanosensation (touch/contact)

All features are scaled to quantum-compatible ranges (typically [-π/2, π/2] or [0, π])
for direct use in quantum rotation gates, and can be concatenated for classical networks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantumnematode.brain.modules import (
    MODULE_FEATURE_EXTRACTORS,
    ModuleName,
    RotationAxis,
)

if TYPE_CHECKING:
    from quantumnematode.brain.arch import BrainParams


def extract_sensory_features(
    params: BrainParams,
    modules: list[ModuleName] | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract sensory features from BrainParams for all or specified modules.

    This is the unified entry point for feature extraction, providing consistent
    features for both quantum and classical brain architectures.

    Parameters
    ----------
    params : BrainParams
        Agent state containing sensory information from the environment.
    modules : list[ModuleName] | None
        List of modules to extract features for. If None, extracts all available modules.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping module names to feature arrays.
        Each array contains [rx, ry, rz] values for that module.

    Examples
    --------
    >>> features = extract_sensory_features(params)
    >>> features["chemotaxis"]  # array([rx, ry, rz])

    >>> features = extract_sensory_features(params, [ModuleName.CHEMOTAXIS, ModuleName.AVERSIVE])
    """
    if modules is None:
        modules = list(MODULE_FEATURE_EXTRACTORS.keys())

    result = {}
    for module in modules:
        extractor = MODULE_FEATURE_EXTRACTORS.get(module)
        if extractor is not None:
            features = extractor(params)
            # Convert to numpy array [rx, ry, rz]
            result[module.value] = np.array(
                [
                    features.get(RotationAxis.RX, 0.0),
                    features.get(RotationAxis.RY, 0.0),
                    features.get(RotationAxis.RZ, 0.0),
                ],
                dtype=np.float32,
            )

    return result


def extract_flat_features(
    params: BrainParams,
    modules: list[ModuleName] | None = None,
) -> np.ndarray:
    """
    Extract sensory features as a flat concatenated array for classical networks.

    This is useful for PPOBrain and other classical architectures that expect
    a single feature vector as input.

    Parameters
    ----------
    params : BrainParams
        Agent state containing sensory information from the environment.
    modules : list[ModuleName] | None
        List of modules to extract features for. If None, extracts all available modules.

    Returns
    -------
    np.ndarray
        Flat array of all feature values concatenated in module order.
        Shape: (num_modules * 3,) for [rx, ry, rz] per module.

    Examples
    --------
    >>> features = extract_flat_features(params, [ModuleName.CHEMOTAXIS, ModuleName.AVERSIVE])
    >>> features.shape  # (6,) for 2 modules * 3 rotations each
    """
    module_features = extract_sensory_features(params, modules)

    if not module_features:
        return np.array([], dtype=np.float32)

    # Concatenate in consistent order (sorted by module name)
    sorted_modules = sorted(module_features.keys())
    return np.concatenate([module_features[m] for m in sorted_modules])


def get_feature_dimension(modules: list[ModuleName] | None = None) -> int:
    """
    Get the dimension of the flat feature vector for given modules.

    Parameters
    ----------
    modules : list[ModuleName] | None
        List of modules. If None, uses all available modules.

    Returns
    -------
    int
        Number of features (num_modules * 3 for rx, ry, rz per module).
    """
    if modules is None:
        modules = list(MODULE_FEATURE_EXTRACTORS.keys())

    # Each module produces 3 features (rx, ry, rz)
    return len(modules) * 3
