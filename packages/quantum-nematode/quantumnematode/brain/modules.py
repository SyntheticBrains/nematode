"""
Feature extraction modules for ModularBrain.

These modules extract features from the agent's state and environment,
which are then used to inform the agent's actions. Each module corresponds to a
specific aspect of the agent's sensory input or internal state.

The features are extracted as RX, RY, and RZ values for the qubits assigned to each module.

In the future, features can take more advanced forms, such as sub-circuits, groups of qubits,
or even more complex quantum operations.

Other possible modules to add in the short term (excluding placeholders):
- Satiety/hunger
- Touch/tactile
- Memory (short-term/long-term)
- Decision-making (e.g., reinforcement learning)

"""

from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]

from quantumnematode.brain.arch import BrainParams


def proprioception_features(
    params: BrainParams,
    satiety: float = 1.0,  # noqa: ARG001
) -> dict[str, float]:
    """
    Extract proprioception features: agent's own direction only.

    In future, this could be extended to include more complex proprioceptive data
    such as joint angles, body posture/bend, local relative position, etc.

    Args:
        params: BrainParams containing agent state.
        satiety: Current satiety value (unused).

    Returns
    -------
        Dictionary with rx, ry, rz values for proprioception qubit(s).
    """
    direction_map = {"up": 0.0, "down": np.pi, "left": np.pi / 2, "right": -np.pi / 2}
    direction = direction_map.get(params.agent_direction or "up", 0.0)
    return {"rx": 0.0, "ry": 0.0, "rz": direction}


def chemotaxis_features(
    params: BrainParams,
    satiety: float = 1.0,  # noqa: ARG001
) -> dict[str, float]:
    """
    Extract chemotaxis features: gradient strength and direction.

    Args:
        params: BrainParams containing agent state.
        satiety: Current satiety value (unused).

    Returns
    -------
        Dictionary with rx, ry, rz values for chemotaxis qubit(s).
    """
    return {
        "rx": params.gradient_strength or 0.0,
        "ry": params.gradient_direction or 0.0,
        "rz": 0.0,
    }


def thermotaxis_features(
    params: BrainParams,  # noqa: ARG001
    satiety: float = 1.0,  # noqa: ARG001
) -> dict[str, float]:
    """
    Extract thermotaxis features (placeholder).

    Args:
        params: BrainParams containing agent state.
        satiety: Current satiety value (unused).

    Returns
    -------
        Dictionary with rx, ry, rz values for thermotaxis qubit(s).
    """
    return {"rx": 0.0, "ry": 0.0, "rz": 0.0}


def oxygen_features(
    params: BrainParams,  # noqa: ARG001
    satiety: float = 1.0,  # noqa: ARG001
) -> dict[str, float]:
    """
    Extract oxygen sensing features (placeholder).

    Args:
        params: BrainParams containing agent state.
        satiety: Current satiety value (unused).

    Returns
    -------
        Dictionary with rx, ry, rz values for oxygen qubit(s).
    """
    return {"rx": 0.0, "ry": 0.0, "rz": 0.0}


def vision_features(
    params: BrainParams,  # noqa: ARG001
    satiety: float = 1.0,  # noqa: ARG001
) -> dict[str, float]:
    """
    Extract vision features (placeholder).

    Args:
        params: BrainParams containing agent state.
        satiety: Current satiety value (unused).

    Returns
    -------
        Dictionary with rx, ry, rz values for vision qubit(s).
    """
    return {"rx": 0.0, "ry": 0.0, "rz": 0.0}


MODULE_FEATURE_EXTRACTORS: dict[str, Any] = {
    "proprioception": proprioception_features,
    "chemotaxis": chemotaxis_features,
    "thermotaxis": thermotaxis_features,
    "oxygen": oxygen_features,
    "vision": vision_features,
}


def extract_features_for_module(
    module: str,
    params: BrainParams,
    satiety: float = 1.0,
) -> dict[str, float]:
    """
    Extract features for a given module using the appropriate extractor.

    Args:
        module: Name of the module (e.g., 'proprioception').
        params: BrainParams containing agent state.
        satiety: Current satiety value.

    Returns
    -------
        Dictionary with rx, ry, rz values for the module's qubit(s).
    """
    extractor = MODULE_FEATURE_EXTRACTORS.get(module)
    if extractor:
        return extractor(params, satiety=satiety)
    return {"rx": 0.0, "ry": 0.0, "rz": 0.0}
