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
    THERMOTAXIS = "thermotaxis"

    # Placeholder modules
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
    """Extract food chemotaxis features from separated food gradient.

    Food chemotaxis (AWC, AWA neurons) encodes attraction toward food sources:
    - strength: food gradient magnitude [0, 1], higher = closer/stronger food signal
    - angle: egocentric direction to food [-1, 1], 0 = food directly ahead

    This module uses SEPARATED food gradients (food_gradient_*) rather than
    combined gradients (gradient_*), allowing the brain to independently learn
    food attraction vs predator avoidance behaviors.

    Biological Reference:
        AWC neurons detect volatile attractants (benzaldehyde, isoamyl alcohol).
        AWA neurons detect additional attractants (diacetyl, pyrazine).
        Both project to interneurons that modulate turning behavior.

    Transform (standard):
        RX = tanh(strength) * π - π/2  -> [-π/2, π/2] (biased by signal strength)
        RY = angle * π/2               -> [-π/2, π/2] (turn toward food)
        RZ = 0                         -> no binary signal
    """
    strength = float(params.food_gradient_strength or 0.0)
    angle = _compute_relative_angle(
        params.food_gradient_direction,
        params.agent_direction,
    )
    return CoreFeatures(strength=strength, angle=angle)


def _nociception_core(params: BrainParams) -> CoreFeatures:
    """Extract nociception features from separated predator gradient.

    Nociception (ASH, ADL neurons) encodes aversive/danger signals:
    - strength: predator gradient magnitude [0, 1], higher = closer predator
    - angle: egocentric direction TOWARD predator [-1, 1], 0 = predator ahead

    The brain should learn: high strength + angle near 0 = danger ahead, turn away!

    Note on Gradient Direction:
        The predator gradient points TOWARD predators (danger direction), not away.
        This gives consistent semantics with food_chemotaxis where angle=0 means
        the signal source is directly ahead. The brain must learn that for
        nociception, moving AWAY from the angle direction is beneficial.

    Biological Reference:
        ASH neurons are polymodal nociceptors responding to nose touch, high
        osmolarity, and volatile repellents. ADL neurons respond to octanol
        and other repellents. Both trigger avoidance reflexes via interneurons.

    Transform (standard):
        RX = tanh(strength) * π - π/2  -> [-π/2, π/2] (danger intensity)
        RY = angle * π/2               -> [-π/2, π/2] (danger direction)
        RZ = 0                         -> no binary signal
    """
    strength = float(params.predator_gradient_strength or 0.0)
    angle = _compute_relative_angle(
        params.predator_gradient_direction,
        params.agent_direction,
    )
    return CoreFeatures(strength=strength, angle=angle)


def _mechanosensation_core(params: BrainParams) -> CoreFeatures:
    """Extract mechanosensation features from touch/contact signals.

    Mechanosensation (ALM, PLM, AVM neurons) encodes physical contact:
    - strength: boundary contact [0, 1], 1 = touching grid boundary
    - angle: predator contact [0, 1], 1 = in physical contact with predator
    - binary: urgency signal = max(boundary, predator), any contact triggers

    Unlike gradient-based modules, mechanosensation provides binary touch signals
    that indicate immediate physical contact. This is a "last resort" signal that
    fires when the agent is already in a dangerous situation.

    Biological Reference:
        ALM (Anterior Lateral Microtubule) and PLM (Posterior Lateral Microtubule)
        neurons are touch receptor neurons along the body. AVM (Anterior Ventral
        Microtubule) responds to light touch at the head. These trigger rapid
        reversal or acceleration responses.

    Transform (binary):
        RX = strength * π/2  -> [0, π/2] (boundary contact)
        RY = angle * π/2     -> [0, π/2] (predator contact)
        RZ = binary * π/2    -> [0, π/2] (any contact urgency)
    """
    boundary = 1.0 if params.boundary_contact is True else 0.0
    predator = 1.0 if params.predator_contact is True else 0.0
    urgency = max(boundary, predator)
    return CoreFeatures(strength=boundary, angle=predator, binary=urgency)


def _proprioception_core(params: BrainParams) -> CoreFeatures:
    """Extract proprioception features from agent's facing direction and movement.

    Proprioception (DVA, PVD neurons) encodes the agent's own body state:
    - strength: movement indicator (1.0 if moved this step, 0.0 if stayed)
    - angle: facing direction encoded as normalized angle [-1, 1]
    - binary: unused (0)

    Direction encoding (using angle field for semantic consistency):
    - UP: 0.5     -> 90° (facing up/north)
    - DOWN: -0.5  -> -90° (facing down/south)
    - LEFT: 1.0   -> 180° (facing left/west)
    - RIGHT: 0.0  -> 0° (facing right/east)

    Movement detection:
    - Uses params.action to detect if agent moved (any action except STAY)
    - FORWARD, LEFT, RIGHT, FORWARD_LEFT, FORWARD_RIGHT all indicate movement

    This provides the brain with awareness of its current heading and whether
    it is actively moving, enabling it to maintain consistent navigation
    strategies regardless of absolute grid orientation.

    Note on Acceleration:
        True acceleration sensing (velocity change over time) is not currently
        possible as the agent has constant speed. When variable speed or
        momentum is implemented, a separate vestibular/kinesthesia module
        could track acceleration.

    Biological Reference:
        DVA is a stretch-sensitive interneuron that modulates locomotion based
        on body posture. PVD neurons are proprioceptive neurons that sense
        body bending and curvature. Together they enable coordinated movement.

    Transform (standard):
        RX = strength * π - π/2  -> [-π/2, π/2] (movement occurred)
        RY = angle * π/2         -> [-π/2, π/2] (facing direction)
        RZ = 0                   -> no binary signal
    """
    # Facing direction encoded as angle in [-1, 1]
    # Maps to same angles as other egocentric modules
    direction_map = {
        Direction.UP: 0.5,  # 90° / π -> normalized to 0.5
        Direction.DOWN: -0.5,  # -90° / -π -> normalized to -0.5
        Direction.LEFT: 1.0,  # 180° / π -> normalized to 1.0 (or -1.0, same angle)
        Direction.RIGHT: 0.0,  # 0° -> normalized to 0.0
    }
    facing_angle = direction_map.get(params.agent_direction or Direction.UP, 0.0)

    # Movement indicator: 1.0 if agent moved this step (any action except STAY)
    movement_strength = 0.0
    if params.action is not None:
        movement_strength = 0.0 if params.action.action == Action.STAY else 1.0

    return CoreFeatures(strength=movement_strength, angle=facing_angle)


def _thermotaxis_core(params: BrainParams) -> CoreFeatures:
    """Extract thermotaxis features from temperature gradient.

    Thermotaxis (AFD neurons) encodes:
    - strength: temperature gradient magnitude (how quickly temp changes spatially)
    - angle: relative direction to warmer temperatures (egocentric)
    - binary: temperature deviation from cultivation temperature (normalized)

    Note on Biological Accuracy:
        Real C. elegans thermotaxis uses temporal comparison (sensing temperature
        change over time as the worm moves) rather than direct spatial gradient
        sensing. The spatial gradient approach used here is computationally
        equivalent for stateless brains and matches the existing chemotaxis pattern.
        When memory systems are added to the architecture (roadmap item), temporal
        sensing features should be implemented as a more biologically accurate
        alternative.
    """
    # Handle disabled thermotaxis (all fields None)
    if params.temperature is None:
        return CoreFeatures()

    # Temperature deviation from cultivation temperature
    # Normalized to [-1, 1] where:
    #   0 = at cultivation temperature (preferred)
    #   +1 = 15°C hotter than Tc (danger hot)
    #   -1 = 15°C colder than Tc (danger cold)
    # Default Tc - use explicit None check to allow Tc=0°C if ever needed
    cultivation_temp = (
        params.cultivation_temperature if params.cultivation_temperature is not None else 20.0
    )
    temp_deviation = (params.temperature - cultivation_temp) / 15.0
    temp_deviation = float(np.clip(temp_deviation, -1.0, 1.0))

    # Gradient strength (0 = no gradient, 1 = strong gradient)
    # Use tanh normalization like chemotaxis
    gradient_strength = float(params.temperature_gradient_strength or 0.0)
    normalized_strength = float(np.tanh(gradient_strength))

    # Relative angle to temperature gradient (where warmer is)
    # Uses same egocentric computation as chemotaxis
    angle = _compute_relative_angle(
        params.temperature_gradient_direction,
        params.agent_direction,
    )

    return CoreFeatures(
        strength=normalized_strength,
        angle=angle,
        binary=temp_deviation,
    )


def _aerotaxis_core(params: BrainParams) -> CoreFeatures:  # noqa: ARG001
    """Extract aerotaxis features (placeholder - returns zeros).

    Aerotaxis (URX, BAG neurons) will encode oxygen gradient navigation:
    - strength: oxygen gradient magnitude (planned)
    - angle: egocentric direction to preferred O2 level (planned)
    - binary: deviation from preferred O2 concentration (planned)

    C. elegans prefer ~10% O2 (normoxia), avoiding both hypoxia (<5%) and
    hyperoxia (>14%). Implementation will follow thermotaxis pattern with
    an oxygen field class and O2 zone classification.

    Biological Reference:
        URX neurons are oxygen sensors that detect increases in O2.
        BAG neurons detect decreases in O2 (CO2 also affects BAG).
        AQR and PQR provide additional O2 sensing at head/tail.
        Together they enable navigation to optimal oxygen levels.

    Status: PLACEHOLDER - returns zeros until oxygen field is implemented.
    """
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
        or [strength, angle, binary]

    Attributes
    ----------
        name: Module identifier from ModuleName enum
        extract: Function that extracts CoreFeatures from BrainParams
        description: Scientific description with C. elegans neuron references
        transform_type: "standard" or "binary" for quantum transform
        is_placeholder: Whether module is fully implemented
        classical_dim: Number of classical features (2 or 3). When 3, includes binary field.
    """

    name: ModuleName
    extract: Callable[[BrainParams], CoreFeatures]
    description: str
    transform_type: Literal["standard", "binary", "placeholder"] = "standard"
    is_placeholder: bool = False
    classical_dim: int = 2

    def to_quantum(self, params: BrainParams) -> np.ndarray:
        """Extract and transform to quantum gate angles [rx, ry, rz].

        Returns
        -------
            np.ndarray of shape (3,) with values depending on transform_type:
            - standard: RX uses offset for full range
            - binary: all values scaled by π/2 (for on/off signals)
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
        """Extract and transform to classical features.

        Returns either [strength, angle] or [strength, angle, binary] depending
        on the module's classical_dim setting. Most modules use 2 features, but
        some (like thermotaxis) include the binary field for richer signals.

        For thermotaxis, including the binary field (temperature deviation) is
        biologically plausible: real C. elegans AFD neurons sense absolute
        temperature, not just gradients. This enables reactive avoidance of
        dangerous temperatures ("I'm too hot right now").

        Returns
        -------
            np.ndarray of shape (classical_dim,) with semantic-preserving ranges:
            - strength: [0, 1] where 0 = no signal
            - angle: [-1, 1] where 0 = aligned with agent
            - binary (if classical_dim=3): [-1, 1] module-specific scalar
        """
        core = self.extract(params)
        if self.classical_dim == 3:  # noqa: PLR2004
            return np.array([core.strength, core.angle, core.binary], dtype=np.float32)
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
            "TOWARD predators (danger direction), so angle=0 means predator is ahead."
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
            "Body orientation sensing (DVA, PVD neurons). Encodes the agent's current "
            "facing direction in the angle field and strength if the agent is moving. "
            "Uses standard transform for semantic consistency."
        ),
        transform_type="standard",
    ),
    # Thermotaxis - temperature (AFD neurons)
    ModuleName.THERMOTAXIS: SensoryModule(
        name=ModuleName.THERMOTAXIS,
        extract=_thermotaxis_core,
        description=(
            "Temperature sensing (AFD neurons). Encodes temperature gradient for "
            "navigation toward cultivation temperature (Tc). Uses spatial gradient "
            "sensing (biologically, C. elegans uses temporal sensing). strength: "
            "gradient magnitude, angle: direction to warmer temp, binary: deviation "
            "from Tc (normalized to [-1, 1])."
        ),
        transform_type="standard",
        classical_dim=3,  # Include temp deviation for reactive avoidance
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
    - binary (for modules with classical_dim=3): [-1, 1] module-specific scalar

    Parameters
    ----------
    params : BrainParams
        Agent state containing sensory information from the environment.
    modules : list[ModuleName]
        List of modules to extract features for.

    Returns
    -------
    np.ndarray
        Flat array of features. Most modules contribute 2 values [strength, angle],
        while some (like thermotaxis) contribute 3 values [strength, angle, binary].
        Shape: (sum of classical_dim for all modules,)

    Examples
    --------
    >>> features = extract_classical_features(params, [ModuleName.FOOD_CHEMOTAXIS])
    >>> # Returns [food_strength, food_angle] in [0,1] and [-1,1] respectively
    >>> features = extract_classical_features(params, [ModuleName.THERMOTAXIS])
    >>> # Returns [grad_strength, grad_angle, temp_deviation] with temp_deviation in [-1,1]
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
            # Module not found - output zeros (default 2 features)
            features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


def get_classical_feature_dimension(modules: list[ModuleName]) -> int:
    """Get the dimension of the classical feature vector for given modules.

    Classical features output either 2 or 3 values per module depending on
    the module's classical_dim setting. Most modules output [strength, angle],
    while some (like thermotaxis) output [strength, angle, binary].

    Parameters
    ----------
    modules : list[ModuleName]
        List of modules.

    Returns
    -------
    int
        Total number of features across all modules.
    """
    total = 0
    for module in modules:
        sensory_module = SENSORY_MODULES.get(module)
        if sensory_module is not None:
            total += sensory_module.classical_dim
        else:
            total += 2  # Default for unknown modules
    return total


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
