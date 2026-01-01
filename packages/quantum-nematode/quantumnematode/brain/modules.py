"""
Feature extraction modules for ModularBrain.

These modules extract features from the agent's state and environment,
which are then used to inform the agent's actions. Each module corresponds to a
specific aspect of the agent's sensory input or internal state.

The features are extracted as RX, RY, and RZ values for the qubits assigned to each module.
All features are scaled to quantum-compatible ranges (typically [-π/2, π/2] or [0, π])
for direct use in quantum rotation gates.

Module naming follows C. elegans neuroscience conventions with neuron references:
- proprioception: Body orientation sensing
- chemotaxis: Combined gradient sensing (ASE neurons)
- food_chemotaxis: Food-specific approach behavior (AWC, AWA neurons)
- nociception: Aversive/escape response (ASH, ADL neurons)
- thermotaxis: Temperature sensing (AFD neurons) - placeholder
- aerotaxis: Oxygen sensing (URX, BAG neurons) - placeholder
- mechanosensation: Touch/contact detection (ALM, PLM, AVM neurons)

In the future, features can take more advanced forms, such as sub-circuits, groups of qubits,
or even more complex quantum operations.

Other possible modules to add in the short term (excluding placeholders):
- Satiety/hunger
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

    Encodes the agent's current facing direction as a rotation angle.
    In future, this could be extended to include more complex proprioceptive data
    such as joint angles, body posture/bend, local relative position, etc.

    Feature encoding:
    - RZ: Agent facing direction (0=UP, π=DOWN, ±π/2=LEFT/RIGHT)

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
    Extract chemotaxis features: combined gradient strength and relative direction.

    Modeled after C. elegans ASE neurons which sense combined chemical gradients.
    Uses the combined gradient (food attraction + predator repulsion) for general
    navigation. For separated food/predator gradients, use food_chemotaxis_features
    and nociception_features respectively.

    Feature encoding:
    - RX: Combined gradient strength scaled to [-π/2, π/2]
    - RY: Relative direction to goal scaled to [-π/2, π/2]

    C. elegans neuron reference: ASE (amphid sensory neurons)

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
    Extract thermotaxis features for temperature-guided navigation (placeholder).

    Will encode temperature sensing for navigation toward preferred temperature (Tc).
    Modeled after C. elegans AFD neurons which sense temperature gradients.

    Planned feature encoding:
    - RX: Deviation from cultivation temperature (Tc)
    - RY: Temperature gradient direction relative to agent
    - RZ: Temperature gradient strength

    C. elegans neuron reference: AFD (amphid finger cell neurons)

    Note: This is a placeholder. Full implementation in add-thermotaxis-system.

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for thermotaxis qubit(s).
    """
    return {RotationAxis.RX: 0.0, RotationAxis.RY: 0.0, RotationAxis.RZ: 0.0}


def aerotaxis_features(
    params: BrainParams,  # noqa: ARG001
) -> dict[RotationAxis, float]:
    """
    Extract aerotaxis features for oxygen-guided navigation (placeholder).

    Will encode oxygen concentration sensing for navigation toward preferred O2 levels.
    Modeled after C. elegans URX and BAG neurons which sense oxygen levels.

    Planned feature encoding:
    - RX: Oxygen concentration relative to preferred level
    - RY: Oxygen gradient direction
    - RZ: Oxygen gradient strength

    C. elegans neuron reference: URX, BAG (oxygen-sensing neurons)

    Note: This is a placeholder. Implementation deferred to future work.

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for aerotaxis qubit(s).
    """
    return {RotationAxis.RX: 0.0, RotationAxis.RY: 0.0, RotationAxis.RZ: 0.0}


# Backward compatibility alias
oxygen_features = aerotaxis_features


def vision_features(
    params: BrainParams,  # noqa: ARG001
) -> dict[RotationAxis, float]:
    """
    Extract vision features (placeholder).

    Note: C. elegans has minimal light sensing capability via ASJ and AWB neurons.
    This module is primarily for future extensions to more complex agents.

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for vision qubit(s).
    """
    return {RotationAxis.RX: 0.0, RotationAxis.RY: 0.0, RotationAxis.RZ: 0.0}


def food_chemotaxis_features(
    params: BrainParams,
) -> dict[RotationAxis, float]:
    """
    Extract food chemotaxis features for appetitive/approach behavior.

    This module encodes signals that drive the agent toward food sources,
    using the SEPARATED food gradient for pure food-seeking behavior.
    This is distinct from the chemotaxis module which uses the combined gradient.

    Modeled after C. elegans AWC and AWA neurons which mediate attraction to
    volatile odorants associated with food sources.

    Feature encoding:
    - RX: Food gradient strength (how strongly food is sensed)
    - RY: Relative direction to food (where to go)

    All features scaled to [-π/2, π/2] for quantum gate stability.

    C. elegans neuron reference: AWC, AWA (amphid wing neurons)

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for food_chemotaxis qubit(s).
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


# Backward compatibility alias
appetitive_features = food_chemotaxis_features


def nociception_features(
    params: BrainParams,
) -> dict[RotationAxis, float]:
    """
    Extract nociception features for aversive/escape behavior.

    This module encodes predator gradient signals for avoidance behavior,
    using the SEPARATED predator gradient for pure escape behavior.
    The predator_gradient_direction points AWAY from predators (escape direction).

    Modeled after C. elegans ASH and ADL neurons which mediate avoidance of
    noxious stimuli including predator pheromones.

    Feature encoding:
    - RX: Predator threat level (how strongly predator is sensed)
    - RY: Escape direction relative to agent facing (where to flee)

    All features scaled to [-π/2, π/2] for quantum gate stability.

    C. elegans neuron reference: ASH, ADL (nociceptive neurons)

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for nociception qubit(s).
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


# Backward compatibility alias
aversive_features = nociception_features


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


def mechanosensation_features(
    params: BrainParams,
) -> dict[RotationAxis, float]:
    """
    Extract mechanosensation features: physical contact with boundaries and predators.

    Modeled after C. elegans touch response neurons:
    - ALM, PLM, AVM: Gentle touch neurons (boundary contact)
    - ASH, ADL: Harsh touch / nociception neurons (predator contact)

    Feature encoding:
    - RX: Boundary contact (π/2 if touching grid edge, 0 otherwise)
    - RY: Predator contact (π/2 if in physical contact with predator, 0 otherwise)
    - RZ: Combined contact urgency (max of both, used for escape response)

    All features are non-negative in range [0, π/2] for quantum gate stability.

    Note: Current implementation uses binary signals (contact vs no contact).
    Future enhancement could add direction-aware encoding to support:
    - Anterior vs posterior touch distinction (like real C. elegans ALM/PLM)
    - Direction of contact relative to agent heading for directional escape
    Binary is sufficient for now since predator direction is already
    available via aversive_features, and boundary direction can be inferred
    from proprioception + boundary_contact.

    Args:
        params: BrainParams containing agent state.

    Returns
    -------
        Dictionary with rx, ry, rz values for mechanosensation qubit(s).
    """
    # Boundary contact: gentle touch (ALM, PLM, AVM neurons)
    # Encode as π/2 when touching boundary (aversive signal)
    boundary_value = 0.0
    if params.boundary_contact is True:
        boundary_value = np.pi / 2

    # Predator contact: harsh touch / nociception (ASH, ADL neurons)
    # Encode as π/2 when in predator contact (strong aversive signal)
    predator_value = 0.0
    if params.predator_contact is True:
        predator_value = np.pi / 2

    # Combined contact urgency: max of both signals
    # This provides an overall "danger" signal for escape response
    urgency = max(boundary_value, predator_value)

    return {
        RotationAxis.RX: boundary_value,
        RotationAxis.RY: predator_value,
        RotationAxis.RZ: urgency,
    }


class ModuleName(str, Enum):
    """Module names used in ModularBrain.

    Module names follow C. elegans neuroscience conventions where possible.
    Both scientific names and legacy names are supported for compatibility:

    Scientific names (preferred):
    - FOOD_CHEMOTAXIS: Food-specific chemotaxis (AWC, AWA neurons)
    - NOCICEPTION: Aversive/escape response (ASH, ADL neurons)
    - AEROTAXIS: Oxygen sensing (URX, BAG neurons)

    Legacy names (deprecated, kept for backward compatibility):
    - APPETITIVE: Alias for food_chemotaxis
    - AVERSIVE: Alias for nociception
    - OXYGEN: Alias for aerotaxis
    """

    PROPRIOCEPTION = "proprioception"
    CHEMOTAXIS = "chemotaxis"
    THERMOTAXIS = "thermotaxis"
    VISION = "vision"
    ACTION = "action"
    MECHANOSENSATION = "mechanosensation"

    # Scientific names (preferred)
    FOOD_CHEMOTAXIS = "food_chemotaxis"
    NOCICEPTION = "nociception"
    AEROTAXIS = "aerotaxis"

    # Legacy names (deprecated, kept for backward compatibility)
    APPETITIVE = "appetitive"
    AVERSIVE = "aversive"
    OXYGEN = "oxygen"


MODULE_FEATURE_EXTRACTORS: dict[ModuleName, Any] = {
    ModuleName.PROPRIOCEPTION: proprioception_features,
    ModuleName.CHEMOTAXIS: chemotaxis_features,
    ModuleName.THERMOTAXIS: thermotaxis_features,
    ModuleName.VISION: vision_features,
    ModuleName.ACTION: memory_action_features,
    ModuleName.MECHANOSENSATION: mechanosensation_features,
    # Scientific names
    ModuleName.FOOD_CHEMOTAXIS: food_chemotaxis_features,
    ModuleName.NOCICEPTION: nociception_features,
    ModuleName.AEROTAXIS: aerotaxis_features,
    # Legacy names (map to same functions for backward compatibility)
    ModuleName.APPETITIVE: appetitive_features,  # Same as food_chemotaxis_features
    ModuleName.AVERSIVE: aversive_features,  # Same as nociception_features
    ModuleName.OXYGEN: oxygen_features,  # Same as aerotaxis_features
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
