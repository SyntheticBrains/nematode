"""
Sensory feature extraction modules for brain architectures.

This module provides a unified interface for extracting sensory features from
BrainParams, supporting both quantum (ModularBrain) and classical (PPOBrain)
brain architectures through a single SensoryModule abstraction.

Architecture:
    Each sensory module encapsulates:

    1. **Core extraction**: BrainParams -> CoreFeatures (architecture-agnostic)
       - strength: [0, 1] where 0 = no signal, 1 = max signal
       - angle: [-1, 1] where 0 = aligned with agent heading
       - binary: 0 or 1 for on/off signals

    2. **Architecture transforms**:
       - to_quantum(): Returns np.ndarray [rx, ry, rz] in [-π/2, π/2]
       - to_classical(): Returns np.ndarray [strength, angle] preserving semantics

Module naming follows C. elegans neuroscience conventions:
- chemotaxis: Combined gradient sensing (ASE neurons)
- food_chemotaxis: Food-specific approach behavior (AWC, AWA neurons)
- nociception: Aversive/escape response (ASH, ADL neurons)
- thermotaxis: Temperature sensing (AFD neurons) - placeholder
- aerotaxis: Oxygen sensing (URX, BAG neurons) - placeholder
- mechanosensation: Touch/contact detection (ALM, PLM, AVM neurons)
- proprioception: Body orientation sensing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

from quantumnematode.brain.actions import Action
from quantumnematode.env import Direction

if TYPE_CHECKING:
    from collections.abc import Callable

    from quantumnematode.brain.arch import BrainParams


# =============================================================================
# Core Feature Data Structure
# =============================================================================


@dataclass
class CoreFeatures:
    """Architecture-agnostic feature values with semantic meaning.

    All values are in normalized ranges that match their semantic meaning:
    - strength: [0, 1] where 0 = no signal, 1 = strong signal
    - angle: [-1, 1] where 0 = aligned with agent, ±1 = opposite direction
    - binary: 0 or 1 for on/off signals
    """

    strength: float = 0.0  # [0, 1] - signal intensity
    angle: float = 0.0  # [-1, 1] - relative direction
    binary: float = 0.0  # 0 or 1 - on/off signal


# =============================================================================
# Module Name Enumeration
# =============================================================================


class ModuleName(str, Enum):
    """Sensory module identifiers.

    Module names follow C. elegans neuroscience conventions where possible.
    """

    # Core modules
    PROPRIOCEPTION = "proprioception"
    CHEMOTAXIS = "chemotaxis"
    FOOD_CHEMOTAXIS = "food_chemotaxis"
    NOCICEPTION = "nociception"
    MECHANOSENSATION = "mechanosensation"

    # Placeholder modules
    THERMOTAXIS = "thermotaxis"
    AEROTAXIS = "aerotaxis"
    VISION = "vision"
    ACTION = "action"

    # Legacy aliases (deprecated)
    APPETITIVE = "appetitive"
    AVERSIVE = "aversive"
    OXYGEN = "oxygen"


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_relative_angle(
    target_direction: float | None,
    agent_direction: Direction | None,
) -> float:
    """Compute relative angle from agent heading to target direction.

    Args:
        target_direction: Absolute direction to target in radians [-π, π]
        agent_direction: Agent's facing direction

    Returns
    -------
        Relative angle normalized to [-1, 1] where:
        - 0 = target is directly ahead
        - 1 or -1 = target is directly behind
        - 0.5 = target is 90° to the right
        - -0.5 = target is 90° to the left
    """
    if target_direction is None or agent_direction is None:
        return 0.0

    direction_map = {
        Direction.UP: np.pi / 2,
        Direction.DOWN: -np.pi / 2,
        Direction.LEFT: np.pi,
        Direction.RIGHT: 0.0,
    }
    agent_angle = direction_map.get(agent_direction, np.pi / 2)

    # Compute relative angle in [-π, π]
    relative_angle = (target_direction - agent_angle + np.pi) % (2 * np.pi) - np.pi

    # Normalize to [-1, 1]
    return float(relative_angle / np.pi)


# =============================================================================
# Core Feature Extractors
# =============================================================================


def _chemotaxis_core(params: BrainParams) -> CoreFeatures:
    """Extract chemotaxis features from combined gradient."""
    strength = float(params.gradient_strength or 0.0)
    angle = _compute_relative_angle(params.gradient_direction, params.agent_direction)
    return CoreFeatures(strength=strength, angle=angle)


def _food_chemotaxis_core(params: BrainParams) -> CoreFeatures:
    """Extract food chemotaxis features from separated food gradient."""
    strength = float(params.food_gradient_strength or 0.0)
    angle = _compute_relative_angle(
        params.food_gradient_direction,
        params.agent_direction,
    )
    return CoreFeatures(strength=strength, angle=angle)


def _nociception_core(params: BrainParams) -> CoreFeatures:
    """Extract nociception features from separated predator gradient.

    Semantics (consistent with food_chemotaxis):
    - strength: [0, 1] where higher = closer predator danger
    - angle: [-1, 1] where 0 = predator is directly ahead

    The brain should learn: high strength + angle near 0 = danger ahead, turn away!
    """
    strength = float(params.predator_gradient_strength or 0.0)
    angle = _compute_relative_angle(
        params.predator_gradient_direction,
        params.agent_direction,
    )
    return CoreFeatures(strength=strength, angle=angle)


def _mechanosensation_core(params: BrainParams) -> CoreFeatures:
    """Extract mechanosensation features from touch/contact signals."""
    boundary = 1.0 if params.boundary_contact is True else 0.0
    predator = 1.0 if params.predator_contact is True else 0.0
    urgency = max(boundary, predator)
    return CoreFeatures(strength=boundary, angle=predator, binary=urgency)


def _proprioception_core(params: BrainParams) -> CoreFeatures:
    """Extract proprioception features from agent's facing direction.

    The direction is stored in the binary field, scaled to [-1, 1]:
    - UP: 0.0
    - DOWN: 1.0 (will become π with binary transform: 1 * π = π)
    - LEFT: 0.5 (will become π/2 with binary transform: 0.5 * π = π/2)
    - RIGHT: -0.5 (will become -π/2 with binary transform: -0.5 * π = -π/2)
    """
    direction_map = {
        Direction.UP: 0.0,
        Direction.DOWN: 1.0,
        Direction.LEFT: 0.5,
        Direction.RIGHT: -0.5,
    }
    direction = direction_map.get(params.agent_direction or Direction.UP, 0.0)
    return CoreFeatures(binary=direction)


def _thermotaxis_core(params: BrainParams) -> CoreFeatures:  # noqa: ARG001
    """Extract thermotaxis features (placeholder - returns zeros)."""
    return CoreFeatures()


def _aerotaxis_core(params: BrainParams) -> CoreFeatures:  # noqa: ARG001
    """Extract aerotaxis features (placeholder - returns zeros)."""
    return CoreFeatures()


def _vision_core(params: BrainParams) -> CoreFeatures:  # noqa: ARG001
    """Extract vision features (placeholder - returns zeros)."""
    return CoreFeatures()


def _action_core(params: BrainParams) -> CoreFeatures:
    """Extract action memory features from most recent action."""
    action_map = {
        Action.FORWARD: 1.0,
        Action.LEFT: 0.5,
        Action.RIGHT: -0.5,
        Action.STAY: 0.0,
        None: 0.0,
    }
    action_data = getattr(params, "action", None)
    action = action_data.action if action_data and hasattr(action_data, "action") else None
    return CoreFeatures(binary=action_map.get(action, 0.0))


# =============================================================================
# SensoryModule Class
# =============================================================================


@dataclass
class SensoryModule:
    """A sensory module with unified extraction and architecture-specific transforms.

    Each module encapsulates:
    - Core extraction: BrainParams -> CoreFeatures (semantic values)
    - Quantum transform: to_quantum() -> np.ndarray [rx, ry, rz]
    - Classical transform: to_classical() -> np.ndarray [strength, angle]

    Attributes
    ----------
        name: Module identifier from ModuleName enum
        extract: Function that extracts CoreFeatures from BrainParams
        description: Scientific description with C. elegans neuron references
        transform_type: "standard" or "binary" for quantum transform
        is_placeholder: Whether module is fully implemented
    """

    name: ModuleName
    extract: Callable[[BrainParams], CoreFeatures]
    description: str
    transform_type: Literal["standard", "binary", "proprioception", "placeholder"] = "standard"
    is_placeholder: bool = False

    def to_quantum(self, params: BrainParams) -> np.ndarray:
        """Extract and transform to quantum gate angles [rx, ry, rz].

        Returns
        -------
            np.ndarray of shape (3,) with values depending on transform_type:
            - standard: RX uses offset for full range
            - binary: all values scaled by π/2 (for on/off signals)
            - proprioception: RZ uses full π scaling for direction encoding
            - placeholder: returns zeros
        """
        core = self.extract(params)
        if self.transform_type == "placeholder":
            # Placeholder modules return zeros
            return np.zeros(3, dtype=np.float32)
        if self.transform_type == "binary":
            # Binary signals: [0, 1] -> [0, π/2]
            return np.array(
                [
                    core.strength * np.pi / 2,
                    core.angle * np.pi / 2,
                    core.binary * np.pi / 2,
                ],
                dtype=np.float32,
            )
        if self.transform_type == "proprioception":
            # Proprioception: RZ uses full π scaling for direction
            # binary in [-1, 1] -> RZ in [-π, π]
            return np.array(
                [
                    0.0,
                    0.0,
                    core.binary * np.pi,  # [-1,1] -> [-π, π]
                ],
                dtype=np.float32,
            )
        # Standard: strength offset, angle scaled, binary half-rotation
        return np.array(
            [
                core.strength * np.pi - np.pi / 2,  # [0,1] -> [-π/2, π/2]
                core.angle * np.pi / 2,  # [-1,1] -> [-π/2, π/2]
                core.binary * np.pi / 2,  # [0,1] -> [0, π/2]
            ],
            dtype=np.float32,
        )

    def to_quantum_dict(self, params: BrainParams) -> dict[str, float]:
        """Extract and transform to quantum gate angles as a dict.

        This is a convenience method for ModularBrain and QModularBrain that
        returns features in the dict format expected by circuit building code.

        Returns
        -------
            Dictionary with 'rx', 'ry', 'rz' keys and float values.
        """
        arr = self.to_quantum(params)
        return {"rx": float(arr[0]), "ry": float(arr[1]), "rz": float(arr[2])}

    def to_classical(self, params: BrainParams) -> np.ndarray:
        """Extract and transform to classical features [strength, angle].

        Returns
        -------
            np.ndarray of shape (2,) with semantic-preserving ranges:
            - strength: [0, 1] where 0 = no signal
            - angle: [-1, 1] where 0 = aligned with agent
        """
        core = self.extract(params)
        return np.array([core.strength, core.angle], dtype=np.float32)


# =============================================================================
# Sensory Module Registry
# =============================================================================

SENSORY_MODULES: dict[ModuleName, SensoryModule] = {
    # Core chemotaxis - combined gradient sensing
    ModuleName.CHEMOTAXIS: SensoryModule(
        name=ModuleName.CHEMOTAXIS,
        extract=_chemotaxis_core,
        description=(
            "Combined gradient sensing (ASE neurons). Uses the combined gradient "
            "(food attraction + predator repulsion) for general navigation."
        ),
    ),
    # Food-specific chemotaxis
    ModuleName.FOOD_CHEMOTAXIS: SensoryModule(
        name=ModuleName.FOOD_CHEMOTAXIS,
        extract=_food_chemotaxis_core,
        description=(
            "Food-specific chemotaxis (AWC, AWA neurons). Encodes signals that "
            "drive the agent toward food sources using the separated food gradient."
        ),
    ),
    # Nociception - predator avoidance
    ModuleName.NOCICEPTION: SensoryModule(
        name=ModuleName.NOCICEPTION,
        extract=_nociception_core,
        description=(
            "Aversive/escape response (ASH, ADL neurons). Encodes predator gradient "
            "signals for avoidance behavior. The predator_gradient_direction points "
            "AWAY from predators (escape direction)."
        ),
    ),
    # Mechanosensation - touch/contact
    ModuleName.MECHANOSENSATION: SensoryModule(
        name=ModuleName.MECHANOSENSATION,
        extract=_mechanosensation_core,
        description=(
            "Touch/contact detection (ALM, PLM, AVM neurons). Encodes physical "
            "contact with boundaries (gentle touch) and predators (harsh touch). "
            "Uses binary signals: strength=boundary, angle=predator, binary=urgency."
        ),
        transform_type="binary",
    ),
    # Proprioception - body orientation
    ModuleName.PROPRIOCEPTION: SensoryModule(
        name=ModuleName.PROPRIOCEPTION,
        extract=_proprioception_core,
        description=(
            "Body orientation sensing. Encodes the agent's current facing direction "
            "as a rotation angle in the binary field."
        ),
        transform_type="proprioception",
    ),
    # [PLACEHOLDER] Thermotaxis - temperature
    ModuleName.THERMOTAXIS: SensoryModule(
        name=ModuleName.THERMOTAXIS,
        extract=_thermotaxis_core,
        description=(
            "Temperature sensing (AFD neurons). Placeholder: will encode navigation "
            "toward preferred temperature (Tc)."
        ),
        transform_type="placeholder",
        is_placeholder=True,
    ),
    # [PLACEHOLDER] Aerotaxis - oxygen
    ModuleName.AEROTAXIS: SensoryModule(
        name=ModuleName.AEROTAXIS,
        extract=_aerotaxis_core,
        description=(
            "Oxygen sensing (URX, BAG neurons). Placeholder: will encode navigation "
            "toward preferred O2 levels."
        ),
        transform_type="placeholder",
        is_placeholder=True,
    ),
    # [PLACEHOLDER] Vision
    ModuleName.VISION: SensoryModule(
        name=ModuleName.VISION,
        extract=_vision_core,
        description=(
            "Light sensing (ASJ, AWB neurons). Placeholder: C. elegans has minimal "
            "light sensing. For future extensions to more complex agents."
        ),
        transform_type="placeholder",
        is_placeholder=True,
    ),
    # Action memory
    ModuleName.ACTION: SensoryModule(
        name=ModuleName.ACTION,
        extract=_action_core,
        description="Memory of most recent action. Encodes as binary (RZ-only).",
        transform_type="placeholder",
        is_placeholder=True,
    ),
}

# Legacy aliases - point to same modules
SENSORY_MODULES[ModuleName.APPETITIVE] = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS]
SENSORY_MODULES[ModuleName.AVERSIVE] = SENSORY_MODULES[ModuleName.NOCICEPTION]
SENSORY_MODULES[ModuleName.OXYGEN] = SENSORY_MODULES[ModuleName.AEROTAXIS]


# =============================================================================
# Classical Feature Extraction (for PPOBrain and other classical networks)
# =============================================================================


def extract_classical_features(
    params: BrainParams,
    modules: list[ModuleName],
) -> np.ndarray:
    """Extract features for classical neural networks with semantic-preserving ranges.

    Unlike quantum features which use rotation gate angles, classical features
    preserve semantic meaning, making them easier for neural networks to learn:

    - strength: [0, 1] where 0 = no signal, 1 = strong signal
    - angle: [-1, 1] where 0 = aligned with agent heading

    Parameters
    ----------
    params : BrainParams
        Agent state containing sensory information from the environment.
    modules : list[ModuleName]
        List of modules to extract features for.

    Returns
    -------
    np.ndarray
        Flat array of features with 2 values per module [strength, angle].
        Shape: (len(modules) * 2,)

    Examples
    --------
    >>> features = extract_classical_features(params, [ModuleName.FOOD_CHEMOTAXIS])
    >>> # Returns [food_strength, food_angle] in [0,1] and [-1,1] respectively
    """
    features = []

    # Sort modules for consistent ordering
    sorted_modules = sorted(modules, key=lambda m: m.value)

    for module in sorted_modules:
        sensory_module = SENSORY_MODULES.get(module)
        if sensory_module is not None:
            classical = sensory_module.to_classical(params)
            features.extend(classical.tolist())
        else:
            # Module not found - output zeros
            features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


def get_classical_feature_dimension(modules: list[ModuleName]) -> int:
    """Get the dimension of the classical feature vector for given modules.

    Classical features output 2 values per module: [strength, angle].

    Parameters
    ----------
    modules : list[ModuleName]
        List of modules.

    Returns
    -------
    int
        Number of features (2 per module).
    """
    return len(modules) * 2


# =============================================================================
# Module Configuration Types
# =============================================================================

Modules = dict[ModuleName, list[int]]

DEFAULT_MODULES: Modules = {
    ModuleName.CHEMOTAXIS: [0, 1],
}


def count_total_qubits(modules: Modules) -> int:
    """Count the total number of unique qubits used in the given modules mapping.

    Args:
        modules: Mapping from ModuleName to list of qubit indices.

    Returns
    -------
        Total number of unique qubits used across all modules.
    """
    qubit_indices = set()
    for qubits in modules.values():
        qubit_indices.update(qubits)
    return len(qubit_indices)
