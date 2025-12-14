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

from enum import Enum
from typing import Any

import numpy as np

from quantumnematode.brain.actions import Action
from quantumnematode.brain.arch import BrainParams
from quantumnematode.env import Direction


class RotationAxis(str, Enum):
    """Rotation axes used in feature extraction."""

    RX = "rx"
    RY = "ry"
    RZ = "rz"


def proprioception_features(
    params: BrainParams,
) -> dict[RotationAxis, float]:
    """
    Extract proprioception features: agent's own direction only.

    In future, this could be extended to include more complex proprioceptive data
    such as joint angles, body posture/bend, local relative position, etc.

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for proprioception qubit(s).
    """
    direction_map = {
        Direction.UP: 0.0,
        Direction.DOWN: np.pi,
        Direction.LEFT: np.pi / 2,
        Direction.RIGHT: -np.pi / 2,
    }
    direction = direction_map.get(params.agent_direction or Direction.UP, 0.0)
    return {RotationAxis.RX: 0.0, RotationAxis.RY: 0.0, RotationAxis.RZ: direction}


def chemotaxis_features(
    params: BrainParams,
) -> dict[RotationAxis, float]:
    """
    Extract chemotaxis features: gradient strength and relative direction to goal.

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for chemotaxis qubit(s).
    """
    # Use gradient_strength directly (0-1) scaled to a moderate range
    grad_strength = params.gradient_strength or 0.0
    # Scale to [-π/2, π/2] for stability
    grad_strength_scaled = grad_strength * np.pi - np.pi / 2

    # gradient_direction is the absolute direction to the goal ([-pi, pi])
    grad_direction = params.gradient_direction or 0.0
    # agent_direction is a string ("up", "down", etc.)
    direction_map = {
        Direction.UP: np.pi / 2,
        Direction.DOWN: -np.pi / 2,
        Direction.LEFT: np.pi,
        Direction.RIGHT: 0.0,
    }
    agent_facing_angle = direction_map.get(params.agent_direction or Direction.UP, np.pi / 2)
    # Compute relative angle to goal ([-pi, pi])
    relative_angle = (grad_direction - agent_facing_angle + np.pi) % (2 * np.pi) - np.pi

    # Normalize relative angle
    relative_angle_normalized = relative_angle / np.pi  # [-1, 1]
    # Scale back to moderate quantum range
    relative_angle_scaled = relative_angle_normalized * np.pi / 2  # [-π/2, π/2]

    return {
        RotationAxis.RX: grad_strength_scaled,
        RotationAxis.RY: relative_angle_scaled,
        RotationAxis.RZ: 0.0,
    }


def thermotaxis_features(
    params: BrainParams,  # noqa: ARG001
) -> dict[RotationAxis, float]:
    """
    Extract thermotaxis features (placeholder).

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for thermotaxis qubit(s).
    """
    return {RotationAxis.RX: 0.0, RotationAxis.RY: 0.0, RotationAxis.RZ: 0.0}


def oxygen_features(
    params: BrainParams,  # noqa: ARG001
) -> dict[RotationAxis, float]:
    """
    Extract oxygen sensing features (placeholder).

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for oxygen qubit(s).
    """
    return {RotationAxis.RX: 0.0, RotationAxis.RY: 0.0, RotationAxis.RZ: 0.0}


def vision_features(
    params: BrainParams,  # noqa: ARG001
) -> dict[RotationAxis, float]:
    """
    Extract vision features (placeholder).

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for vision qubit(s).
    """
    return {RotationAxis.RX: 0.0, RotationAxis.RY: 0.0, RotationAxis.RZ: 0.0}


def appetitive_features(
    params: BrainParams,
) -> dict[RotationAxis, float]:
    """
    Extract appetitive (food-seeking) features for approach behavior.

    This module encodes signals that drive the agent toward food sources,
    inspired by C. elegans appetitive chemotaxis circuits (AWC neurons).

    Uses SEPARATED food gradient to encode pure food-seeking behavior.
    This is distinct from the chemotaxis module which uses the combined gradient.

    Feature encoding:
    - RX: Food gradient strength (how strongly food is sensed)
    - RY: Relative direction to food (where to go)

    All features scaled to [-π/2, π/2] or [0, π] for quantum gate stability.

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for appetitive qubit(s).
    """
    # Food gradient strength - use separated food gradient
    food_strength = params.food_gradient_strength or 0.0
    # Use tanh to normalize potentially unbounded gradient values to [0, 1]
    food_strength_normalized = np.tanh(food_strength)
    # Scale to [-π/2, π/2] for quantum gate stability
    food_strength_scaled = food_strength_normalized * np.pi - np.pi / 2

    # Relative angle to food source
    food_direction = params.food_gradient_direction
    if food_direction is not None and params.agent_direction is not None:
        direction_map = {
            Direction.UP: np.pi / 2,
            Direction.DOWN: -np.pi / 2,
            Direction.LEFT: np.pi,
            Direction.RIGHT: 0.0,
        }
        agent_angle = direction_map.get(params.agent_direction, np.pi / 2)
        # Compute relative angle to food ([-π, π])
        relative_angle = (food_direction - agent_angle + np.pi) % (2 * np.pi) - np.pi
        # Normalize and scale to [-π/2, π/2]
        relative_angle_normalized = relative_angle / np.pi  # [-1, 1]
        relative_angle_scaled = relative_angle_normalized * np.pi / 2
    else:
        relative_angle_scaled = 0.0

    return {
        RotationAxis.RX: food_strength_scaled,
        RotationAxis.RY: relative_angle_scaled,
        RotationAxis.RZ: 0.0,
    }


def aversive_features(
    params: BrainParams,
) -> dict[RotationAxis, float]:
    """
    Extract aversive (predator-avoidance) features for escape behavior.

    This module encodes predator gradient signals for avoidance behavior,
    inspired by C. elegans aversive response circuits (ASH neurons).

    Uses SEPARATED predator gradient to encode pure escape behavior.
    The predator_gradient_direction points AWAY from predators (escape direction).

    Feature encoding:
    - RX: Predator threat level (how strongly predator is sensed)
    - RY: Escape direction relative to agent facing (where to flee)

    All features scaled to [-π/2, π/2] or [0, π] for quantum gate stability.

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for aversive qubit(s).
    """
    # Predator gradient strength - indicates threat level
    predator_strength = params.predator_gradient_strength or 0.0
    # Use tanh to normalize potentially unbounded gradient values to [0, 1]
    predator_strength_normalized = np.tanh(predator_strength)
    # Scale to [-π/2, π/2] for quantum gate stability
    predator_strength_scaled = predator_strength_normalized * np.pi - np.pi / 2

    # Relative angle to escape direction (predator_gradient_direction points away)
    escape_direction = params.predator_gradient_direction
    if escape_direction is not None and params.agent_direction is not None:
        direction_map = {
            Direction.UP: np.pi / 2,
            Direction.DOWN: -np.pi / 2,
            Direction.LEFT: np.pi,
            Direction.RIGHT: 0.0,
        }
        agent_angle = direction_map.get(params.agent_direction, np.pi / 2)
        # Compute relative angle to escape direction ([-π, π])
        relative_angle = (escape_direction - agent_angle + np.pi) % (2 * np.pi) - np.pi
        # Normalize and scale to [-π/2, π/2]
        relative_angle_normalized = relative_angle / np.pi  # [-1, 1]
        escape_scaled = relative_angle_normalized * np.pi / 2
    else:
        escape_scaled = 0.0

    return {
        RotationAxis.RX: predator_strength_scaled,
        RotationAxis.RY: escape_scaled,
        RotationAxis.RZ: 0.0,
    }


def memory_action_features(
    params: BrainParams,
) -> dict[RotationAxis, float]:
    """
    Extract action features: encode an action taken by the agent such as most recent.

    Args:
        params: BrainParams containing agent state (should have action attribute).

    Returns
    -------
        Dictionary with rx, ry, rz values for action qubit(s).
    """
    # Map possible actions to angles (expand as needed)
    action_map = {
        Action.FORWARD: np.pi,
        Action.LEFT: np.pi / 2,
        Action.RIGHT: -np.pi / 2,
        Action.STAY: 0.0,
        None: 0.0,  # Default if no action
    }
    action_data = getattr(params, "action", None)
    action = action_data.action if action_data and hasattr(action_data, "action") else None
    angle = action_map.get(action, 0.0)
    return {RotationAxis.RX: 0.0, RotationAxis.RY: 0.0, RotationAxis.RZ: angle}


class ModuleName(str, Enum):
    """Module names used in ModularBrain."""

    PROPRIOCEPTION = "proprioception"
    CHEMOTAXIS = "chemotaxis"
    THERMOTAXIS = "thermotaxis"
    OXYGEN = "oxygen"
    VISION = "vision"
    ACTION = "action"
    APPETITIVE = "appetitive"
    AVERSIVE = "aversive"


MODULE_FEATURE_EXTRACTORS: dict[ModuleName, Any] = {
    ModuleName.PROPRIOCEPTION: proprioception_features,
    ModuleName.CHEMOTAXIS: chemotaxis_features,
    ModuleName.THERMOTAXIS: thermotaxis_features,
    ModuleName.OXYGEN: oxygen_features,
    ModuleName.VISION: vision_features,
    ModuleName.ACTION: memory_action_features,
    ModuleName.APPETITIVE: appetitive_features,
    ModuleName.AVERSIVE: aversive_features,
}

Modules = dict[ModuleName, list[int]]

DEFAULT_MODULES = {
    ModuleName.CHEMOTAXIS: [0, 1],
}


def extract_features_for_module(
    module: ModuleName,
    params: BrainParams,
) -> dict[str, float]:
    """
    Extract features for a given module using the appropriate extractor.

    Args:
        module: Name of the module (e.g., ModuleName.PROPRIOCEPTION).
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for the module's qubit(s), with string keys.
    """
    extractor = MODULE_FEATURE_EXTRACTORS.get(module)
    if extractor:
        features = extractor(params)
        return {axis.value: value for axis, value in features.items()}
    return {axis.value: 0.0 for axis in RotationAxis}


def count_total_qubits(modules: Modules) -> int:
    """
    Count the total number of unique qubits used in the given modules mapping.

    Args:
        modules: Mapping from ModuleName to list of qubit indices.

    Returns
    -------
        int: Total number of unique qubits used across all modules.
    """
    qubit_indices = set()
    for qubits in modules.values():
        qubit_indices.update(qubits)
    return len(qubit_indices)
