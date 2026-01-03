"""Tests for brain feature extraction modules."""

import numpy as np
import pytest
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.modules import (
    SENSORY_MODULES,
    CoreFeatures,
    ModuleName,
    count_total_qubits,
)
from quantumnematode.env import Direction


class TestCoreFeatures:
    """Test CoreFeatures dataclass."""

    def test_default_values(self):
        """Test that CoreFeatures has correct default values."""
        core = CoreFeatures()
        assert core.strength == 0.0
        assert core.angle == 0.0
        assert core.binary == 0.0

    def test_custom_values(self):
        """Test CoreFeatures with custom values."""
        core = CoreFeatures(strength=0.5, angle=-0.25, binary=1.0)
        assert core.strength == 0.5
        assert core.angle == -0.25
        assert core.binary == 1.0


class TestSensoryModule:
    """Test SensoryModule class."""

    def test_to_quantum_standard_transform(self):
        """Test to_quantum with standard transform type."""
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.RIGHT,
        )
        features = module.to_quantum(params)

        assert features.shape == (3,)
        assert features.dtype == np.float32
        # strength 0.5 -> 0.5 * π - π/2 = 0
        assert features[0] == pytest.approx(0.0)

    def test_to_quantum_binary_transform(self):
        """Test to_quantum with binary transform type."""
        module = SENSORY_MODULES[ModuleName.MECHANOSENSATION]
        assert module.transform_type == "binary"

        params = BrainParams(boundary_contact=True, predator_contact=False)
        features = module.to_quantum(params)

        assert features.shape == (3,)
        # Binary transform: [0,1] -> [0, π/2]
        assert features[0] == pytest.approx(np.pi / 2)  # boundary
        assert features[1] == pytest.approx(0.0)  # predator
        assert features[2] == pytest.approx(np.pi / 2)  # urgency

    def test_to_classical(self):
        """Test to_classical returns semantic-preserving features."""
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        params = BrainParams(
            gradient_strength=0.8,
            gradient_direction=np.pi / 2,
            agent_direction=Direction.UP,
        )
        features = module.to_classical(params)

        assert features.shape == (2,)
        assert features.dtype == np.float32
        # strength preserved
        assert features[0] == pytest.approx(0.8)
        # angle ~0 when aligned
        assert abs(features[1]) < 0.1


class TestProprioceptionModule:
    """Test proprioception feature extraction.

    Proprioception now uses the standard transform with:
    - strength (RX): movement indicator (0.0 for now, placeholder for future)
    - angle (RY): facing direction encoded as normalized angle [-1, 1]
    - binary (RZ): unused (0)

    Direction encoding:
    - UP: angle = 0.5 (90°)
    - DOWN: angle = -0.5 (-90°)
    - LEFT: angle = 1.0 (180°)
    - RIGHT: angle = 0.0 (0°)
    """

    def test_direction_up(self):
        """Test proprioception features for UP direction."""
        module = SENSORY_MODULES[ModuleName.PROPRIOCEPTION]
        params = BrainParams(agent_direction=Direction.UP)
        features = module.to_quantum(params)

        # Standard transform: RX = strength*π - π/2, RY = angle*π/2, RZ = binary*π/2
        # UP: strength=0, angle=0.5, binary=0
        # RX: = 0*π - π/2 = -π/2
        # RY: = 0.5 * π/2 = π/4
        # RZ: = 0.0
        assert features[0] == pytest.approx(-np.pi / 2)  # RX: no movement
        assert features[1] == pytest.approx(np.pi / 4)  # RY: 90° facing up
        assert features[2] == pytest.approx(0.0)  # RZ: unused

    def test_direction_down(self):
        """Test proprioception features for DOWN direction."""
        module = SENSORY_MODULES[ModuleName.PROPRIOCEPTION]
        params = BrainParams(agent_direction=Direction.DOWN)
        features = module.to_quantum(params)

        # DOWN: strength=0, angle=-0.5, binary=0
        # RY = -0.5 * π/2 = -π/4
        assert features[0] == pytest.approx(-np.pi / 2)  # RX: no movement
        assert features[1] == pytest.approx(-np.pi / 4)  # RY: -90° facing down
        assert features[2] == pytest.approx(0.0)  # RZ: unused

    def test_direction_left(self):
        """Test proprioception features for LEFT direction."""
        module = SENSORY_MODULES[ModuleName.PROPRIOCEPTION]
        params = BrainParams(agent_direction=Direction.LEFT)
        features = module.to_quantum(params)

        # LEFT: strength=0, angle=1.0, binary=0
        # RY = 1.0 * π/2 = π/2
        assert features[0] == pytest.approx(-np.pi / 2)  # RX: no movement
        assert features[1] == pytest.approx(np.pi / 2)  # RY: 180° facing left
        assert features[2] == pytest.approx(0.0)  # RZ: unused

    def test_direction_right(self):
        """Test proprioception features for RIGHT direction."""
        module = SENSORY_MODULES[ModuleName.PROPRIOCEPTION]
        params = BrainParams(agent_direction=Direction.RIGHT)
        features = module.to_quantum(params)

        # RIGHT: strength=0, angle=0.0, binary=0
        # RY = 0.0 * π/2 = 0
        assert features[0] == pytest.approx(-np.pi / 2)  # RX: no movement
        assert features[1] == pytest.approx(0.0)  # RY: 0° facing right
        assert features[2] == pytest.approx(0.0)  # RZ: unused

    def test_movement_detection_forward(self):
        """Test that FORWARD action is detected as movement."""
        module = SENSORY_MODULES[ModuleName.PROPRIOCEPTION]
        action = ActionData(state="test", action=Action.FORWARD, probability=1.0)
        params = BrainParams(agent_direction=Direction.UP, action=action)
        features = module.to_quantum(params)

        # With movement: strength=1.0, so RX = 1.0*π - π/2 = π/2
        assert features[0] == pytest.approx(np.pi / 2)  # RX: moved
        assert features[1] == pytest.approx(np.pi / 4)  # RY: facing up
        assert features[2] == pytest.approx(0.0)  # RZ: unused

    def test_movement_detection_stay(self):
        """Test that STAY action is detected as no movement."""
        module = SENSORY_MODULES[ModuleName.PROPRIOCEPTION]
        action = ActionData(state="test", action=Action.STAY, probability=1.0)
        params = BrainParams(agent_direction=Direction.UP, action=action)
        features = module.to_quantum(params)

        # No movement: strength=0.0, so RX = 0*π - π/2 = -π/2
        assert features[0] == pytest.approx(-np.pi / 2)  # RX: stayed
        assert features[1] == pytest.approx(np.pi / 4)  # RY: facing up
        assert features[2] == pytest.approx(0.0)  # RZ: unused

    def test_movement_detection_turn(self):
        """Test that LEFT/RIGHT turns are detected as movement."""
        module = SENSORY_MODULES[ModuleName.PROPRIOCEPTION]
        action = ActionData(state="test", action=Action.LEFT, probability=1.0)
        params = BrainParams(agent_direction=Direction.LEFT, action=action)
        features = module.to_quantum(params)

        # Turning counts as movement: strength=1.0
        assert features[0] == pytest.approx(np.pi / 2)  # RX: moved (turned)
        assert features[1] == pytest.approx(np.pi / 2)  # RY: facing left
        assert features[2] == pytest.approx(0.0)  # RZ: unused


class TestChemotaxisModule:
    """Test chemotaxis feature extraction."""

    def test_zero_gradient(self):
        """Test chemotaxis features with zero gradient strength."""
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        params = BrainParams(
            gradient_strength=0.0,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = module.to_quantum(params)

        # Zero gradient should give -π/2 for RX (scaled from 0)
        assert features[0] == pytest.approx(-np.pi / 2)
        assert features[2] == pytest.approx(0.0)

    def test_max_gradient(self):
        """Test chemotaxis features with maximum gradient strength."""
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        params = BrainParams(
            gradient_strength=1.0,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = module.to_quantum(params)

        # Max gradient (1.0) should give π/2 for RX
        assert features[0] == pytest.approx(np.pi / 2)

    def test_gradient_direction_aligned(self):
        """Test when gradient direction aligns with agent direction."""
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.RIGHT,
        )
        features = module.to_quantum(params)

        # Relative angle should be 0 (aligned)
        assert features[1] == pytest.approx(0.0, abs=1e-6)

    def test_gradient_direction_opposite(self):
        """Test when gradient direction is opposite to agent direction."""
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=np.pi,
            agent_direction=Direction.RIGHT,
        )
        features = module.to_quantum(params)

        # Relative angle should be close to maximum (±π/2)
        assert abs(features[1]) == pytest.approx(np.pi / 2, rel=0.01)

    def test_none_gradient_values(self):
        """Test chemotaxis features when gradient values are None."""
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        params = BrainParams(
            gradient_strength=None,
            gradient_direction=None,
            agent_direction=Direction.UP,
        )
        features = module.to_quantum(params)

        # gradient_strength=None defaults to 0.0, scaled to -π/2
        assert features[0] == pytest.approx(-np.pi / 2)
        # gradient_direction=None means no direction info, so angle is neutral (0)
        assert features[1] == pytest.approx(0.0)


class TestThermotaxisModule:
    """Test thermotaxis (temperature sensing) module."""

    def test_thermotaxis_when_disabled(self):
        """Test thermotaxis returns default transform when temperature is None.

        When thermotaxis is disabled (temperature=None), CoreFeatures returns
        all zeros, but the standard transform still applies the offset:
        RX = 0 * π - π/2 = -π/2 (same as chemotaxis when gradient_strength=None)
        """
        module = SENSORY_MODULES[ModuleName.THERMOTAXIS]
        assert module.is_placeholder is False

        # No temperature fields set (thermotaxis disabled)
        params = BrainParams()
        features = module.to_quantum(params)

        # Standard transform: strength=0 -> RX = -π/2, angle=0 -> RY = 0, binary=0 -> RZ = 0
        assert features[0] == pytest.approx(-np.pi / 2)
        assert features[1] == pytest.approx(0.0)
        assert features[2] == pytest.approx(0.0)

    def test_thermotaxis_at_cultivation_temperature(self):
        """Test thermotaxis at cultivation temperature (comfort zone)."""
        module = SENSORY_MODULES[ModuleName.THERMOTAXIS]

        # At cultivation temperature with no gradient
        params = BrainParams(
            temperature=20.0,
            temperature_gradient_strength=0.0,
            temperature_gradient_direction=0.0,
            cultivation_temperature=20.0,
            agent_direction=Direction.UP,
        )
        features = module.to_quantum(params)

        # At cultivation temp: deviation=0, gradient_strength=0
        # Standard transform: strength=0 -> RX = -π/2, angle scales, binary=0 -> RZ = 0
        assert features[0] == pytest.approx(-np.pi / 2)  # RX: tanh(0)*π - π/2 = -π/2
        # RY encodes egocentric direction to warmer temperatures
        # Gradient points right (direction=0). Agent faces up.
        # Egocentric angle = -0.5 (warmer is to agent's right) -> RY = -π/4
        assert features[1] == pytest.approx(-np.pi / 4)
        assert features[2] == pytest.approx(0.0)  # RZ: deviation=0 -> 0

    def test_thermotaxis_hotter_than_cultivation(self):
        """Test thermotaxis when hotter than cultivation temperature."""
        module = SENSORY_MODULES[ModuleName.THERMOTAXIS]

        # 15°C hotter than cultivation (at danger threshold)
        params = BrainParams(
            temperature=35.0,
            temperature_gradient_strength=0.5,
            temperature_gradient_direction=0.0,  # Warmer to the right
            cultivation_temperature=20.0,
            agent_direction=Direction.UP,
        )
        features = module.to_quantum(params)

        # Binary should be +1 (15/15 = 1.0, clamped) -> RZ = pi/2
        assert features[2] == pytest.approx(np.pi / 2, abs=0.1)

    def test_thermotaxis_colder_than_cultivation(self):
        """Test thermotaxis when colder than cultivation temperature."""
        module = SENSORY_MODULES[ModuleName.THERMOTAXIS]

        # 15°C colder than cultivation (at danger threshold)
        params = BrainParams(
            temperature=5.0,
            temperature_gradient_strength=0.5,
            temperature_gradient_direction=0.0,
            cultivation_temperature=20.0,
            agent_direction=Direction.UP,
        )
        features = module.to_quantum(params)

        # Binary should be -1 (-15/15 = -1.0) -> RZ = -pi/2
        assert features[2] == pytest.approx(-np.pi / 2, abs=0.1)


class TestPlaceholderModules:
    """Test placeholder modules that return zero features."""

    def test_aerotaxis_returns_zeros(self):
        """Test that aerotaxis returns zeros (placeholder)."""
        module = SENSORY_MODULES[ModuleName.AEROTAXIS]
        assert module.is_placeholder is True

        params = BrainParams()
        features = module.to_quantum(params)

        # Placeholder transform returns all zeros
        np.testing.assert_array_equal(features, np.zeros(3, dtype=np.float32))

    def test_vision_returns_zeros(self):
        """Test that vision returns zeros (placeholder)."""
        module = SENSORY_MODULES[ModuleName.VISION]
        assert module.is_placeholder is True

        params = BrainParams()
        features = module.to_quantum(params)

        # Placeholder transform returns all zeros
        np.testing.assert_array_equal(features, np.zeros(3, dtype=np.float32))


class TestActionModule:
    """Test action memory feature extraction (placeholder)."""

    def test_action_forward(self):
        """Test action features for FORWARD action - placeholder returns zeros."""
        module = SENSORY_MODULES[ModuleName.ACTION]
        assert module.is_placeholder is True

        params = BrainParams()
        features = module.to_quantum(params)

        # Placeholder transform returns all zeros regardless of input
        np.testing.assert_array_equal(features, np.zeros(3, dtype=np.float32))

    def test_action_stay(self):
        """Test action features for STAY action - placeholder returns zeros."""
        module = SENSORY_MODULES[ModuleName.ACTION]
        params = BrainParams()
        features = module.to_quantum(params)

        # Placeholder transform returns all zeros
        np.testing.assert_array_equal(features, np.zeros(3, dtype=np.float32))

    def test_no_action(self):
        """Test action features when no action is present."""
        module = SENSORY_MODULES[ModuleName.ACTION]
        params = BrainParams()
        features = module.to_quantum(params)

        # Placeholder transform returns all zeros
        np.testing.assert_array_equal(features, np.zeros(3, dtype=np.float32))


class TestModuleName:
    """Test ModuleName enum."""

    def test_module_names(self):
        """Test that all module names have correct values."""
        assert ModuleName.PROPRIOCEPTION.value == "proprioception"
        assert ModuleName.CHEMOTAXIS.value == "chemotaxis"
        assert ModuleName.THERMOTAXIS.value == "thermotaxis"
        assert ModuleName.VISION.value == "vision"
        assert ModuleName.ACTION.value == "action"
        assert ModuleName.MECHANOSENSATION.value == "mechanosensation"
        assert ModuleName.FOOD_CHEMOTAXIS.value == "food_chemotaxis"
        assert ModuleName.NOCICEPTION.value == "nociception"
        assert ModuleName.AEROTAXIS.value == "aerotaxis"

    def test_legacy_aliases(self):
        """Test legacy module name aliases."""
        assert ModuleName.APPETITIVE.value == "appetitive"
        assert ModuleName.AVERSIVE.value == "aversive"
        assert ModuleName.OXYGEN.value == "oxygen"

    def test_all_modules_in_registry(self):
        """Test that all module names have corresponding entries in SENSORY_MODULES."""
        for module_name in ModuleName:
            assert module_name in SENSORY_MODULES, (
                f"Module {module_name} missing from SENSORY_MODULES"
            )


class TestToQuantumDict:
    """Test the to_quantum_dict method on SensoryModule."""

    def test_extract_chemotaxis(self):
        """Test extracting chemotaxis features as dict."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        features = module.to_quantum_dict(params)

        # Should return dict with string keys
        assert "rx" in features
        assert "ry" in features
        assert "rz" in features
        assert isinstance(features["rx"], float)

    def test_extract_proprioception(self):
        """Test extracting proprioception features as dict."""
        params = BrainParams(agent_direction=Direction.DOWN)
        module = SENSORY_MODULES[ModuleName.PROPRIOCEPTION]
        features = module.to_quantum_dict(params)

        # DOWN: strength=0, angle=-0.5, binary=0
        # Standard transform: RX = 0*π - π/2 = -π/2, RY = -0.5 * π/2 = -π/4, RZ = 0
        assert features["rx"] == pytest.approx(-np.pi / 2)
        assert features["ry"] == pytest.approx(-np.pi / 4)
        assert features["rz"] == pytest.approx(0.0)

    def test_extract_placeholder_module(self):
        """Test extracting features from placeholder module."""
        params = BrainParams()
        module = SENSORY_MODULES[ModuleName.VISION]
        features = module.to_quantum_dict(params)

        # Placeholder modules return zeros
        assert features["rx"] == pytest.approx(0.0)
        assert features["ry"] == pytest.approx(0.0)
        assert features["rz"] == pytest.approx(0.0)

    def test_to_quantum_dict_deterministic(self):
        """Test that to_quantum_dict is deterministic."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.RIGHT,
        )
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        features1 = module.to_quantum_dict(params)
        features2 = module.to_quantum_dict(params)

        assert features1 == features2


class TestCountTotalQubits:
    """Test qubit counting utility."""

    def test_single_module_single_qubit(self):
        """Test counting qubits with one module using one qubit."""
        modules = {ModuleName.CHEMOTAXIS: [0]}
        assert count_total_qubits(modules) == 1

    def test_single_module_multiple_qubits(self):
        """Test counting qubits with one module using multiple qubits."""
        modules = {ModuleName.CHEMOTAXIS: [0, 1]}
        assert count_total_qubits(modules) == 2

    def test_multiple_modules_no_overlap(self):
        """Test counting qubits with multiple modules, no qubit overlap."""
        modules = {
            ModuleName.CHEMOTAXIS: [0, 1],
            ModuleName.PROPRIOCEPTION: [2, 3],
        }
        assert count_total_qubits(modules) == 4

    def test_multiple_modules_with_overlap(self):
        """Test counting qubits with overlapping qubit assignments."""
        modules = {
            ModuleName.CHEMOTAXIS: [0, 1],
            ModuleName.PROPRIOCEPTION: [1, 2],  # Qubit 1 is shared
        }
        assert count_total_qubits(modules) == 3

    def test_empty_modules(self):
        """Test counting qubits with no modules."""
        modules = {}
        assert count_total_qubits(modules) == 0


class TestFoodChemotaxisModule:
    """Test food chemotaxis (appetitive) feature extraction."""

    def test_zero_food_gradient(self):
        """Test features with zero food gradient strength."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS]
        params = BrainParams(
            food_gradient_strength=0.0,
            food_gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = module.to_quantum(params)

        # strength=0 -> -π/2
        assert features[0] == pytest.approx(-np.pi / 2)

    def test_strong_food_gradient(self):
        """Test features with strong food gradient."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS]
        params = BrainParams(
            food_gradient_strength=1.0,
            food_gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = module.to_quantum(params)

        # strength=1 -> π/2
        assert features[0] == pytest.approx(np.pi / 2)

    def test_classical_features_semantic_range(self):
        """Test classical features preserve semantic ranges."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS]

        # No food signal
        params_no_food = BrainParams(
            food_gradient_strength=0.0,
            food_gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = module.to_classical(params_no_food)
        assert features[0] == pytest.approx(0.0)  # strength=0 means no signal

        # Strong food signal
        params_with_food = BrainParams(
            food_gradient_strength=0.9,
            food_gradient_direction=np.pi / 2,
            agent_direction=Direction.UP,
        )
        features = module.to_classical(params_with_food)
        assert features[0] == pytest.approx(0.9)  # strength preserved

    def test_legacy_alias(self):
        """Test that APPETITIVE alias points to same module."""
        assert SENSORY_MODULES[ModuleName.APPETITIVE] is SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS]


class TestNociceptionModule:
    """Test nociception (aversive/predator avoidance) feature extraction."""

    def test_zero_predator_gradient(self):
        """Test features with zero predator gradient (no threat)."""
        module = SENSORY_MODULES[ModuleName.NOCICEPTION]
        params = BrainParams(
            predator_gradient_strength=0.0,
            predator_gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = module.to_quantum(params)

        # strength=0 -> -π/2
        assert features[0] == pytest.approx(-np.pi / 2)

    def test_strong_predator_gradient(self):
        """Test features with strong predator threat."""
        module = SENSORY_MODULES[ModuleName.NOCICEPTION]
        params = BrainParams(
            predator_gradient_strength=1.0,
            predator_gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = module.to_quantum(params)

        # strength=1 -> π/2
        assert features[0] == pytest.approx(np.pi / 2)

    def test_legacy_alias(self):
        """Test that AVERSIVE alias points to same module."""
        assert SENSORY_MODULES[ModuleName.AVERSIVE] is SENSORY_MODULES[ModuleName.NOCICEPTION]


class TestMechanosensationModule:
    """Test mechanosensation (touch/contact) feature extraction."""

    def test_no_contact(self):
        """Test features when no contact detected."""
        module = SENSORY_MODULES[ModuleName.MECHANOSENSATION]
        params = BrainParams(boundary_contact=False, predator_contact=False)
        features = module.to_quantum(params)

        assert features[0] == pytest.approx(0.0)
        assert features[1] == pytest.approx(0.0)
        assert features[2] == pytest.approx(0.0)

    def test_boundary_contact_only(self):
        """Test features when only boundary contact detected."""
        module = SENSORY_MODULES[ModuleName.MECHANOSENSATION]
        params = BrainParams(boundary_contact=True, predator_contact=False)
        features = module.to_quantum(params)

        assert features[0] == pytest.approx(np.pi / 2)  # boundary
        assert features[1] == pytest.approx(0.0)  # predator
        assert features[2] == pytest.approx(np.pi / 2)  # urgency

    def test_predator_contact_only(self):
        """Test features when only predator contact detected."""
        module = SENSORY_MODULES[ModuleName.MECHANOSENSATION]
        params = BrainParams(boundary_contact=False, predator_contact=True)
        features = module.to_quantum(params)

        assert features[0] == pytest.approx(0.0)  # boundary
        assert features[1] == pytest.approx(np.pi / 2)  # predator
        assert features[2] == pytest.approx(np.pi / 2)  # urgency

    def test_both_contacts(self):
        """Test features when both contacts detected."""
        module = SENSORY_MODULES[ModuleName.MECHANOSENSATION]
        params = BrainParams(boundary_contact=True, predator_contact=True)
        features = module.to_quantum(params)

        assert features[0] == pytest.approx(np.pi / 2)
        assert features[1] == pytest.approx(np.pi / 2)
        assert features[2] == pytest.approx(np.pi / 2)

    def test_feature_values_bounded(self):
        """Test that features stay within quantum gate bounds."""
        module = SENSORY_MODULES[ModuleName.MECHANOSENSATION]
        for boundary in [True, False, None]:
            for predator in [True, False, None]:
                params = BrainParams(
                    boundary_contact=boundary,
                    predator_contact=predator,
                )
                features = module.to_quantum(params)

                # All values should be in [0, π/2] (binary transform)
                assert 0.0 <= features[0] <= np.pi / 2
                assert 0.0 <= features[1] <= np.pi / 2
                assert 0.0 <= features[2] <= np.pi / 2


class TestFeatureValueRanges:
    """Test that feature values stay within reasonable quantum rotation ranges."""

    def test_chemotaxis_features_bounded(self):
        """Test that chemotaxis features stay within [-π/2, π/2]."""
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        for strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for grad_dir in np.linspace(-np.pi, np.pi, 5):
                params = BrainParams(
                    gradient_strength=strength,
                    gradient_direction=float(grad_dir),
                    agent_direction=Direction.UP,
                )
                features = module.to_quantum(params)

                assert -np.pi / 2 <= features[0] <= np.pi / 2
                assert -np.pi / 2 <= features[1] <= np.pi / 2


class TestFeatureConsistency:
    """Test that feature extraction is consistent and deterministic."""

    def test_chemotaxis_deterministic(self):
        """Test that chemotaxis features are deterministic."""
        module = SENSORY_MODULES[ModuleName.CHEMOTAXIS]
        params = BrainParams(
            gradient_strength=0.7,
            gradient_direction=np.pi / 4,
            agent_direction=Direction.UP,
        )
        features1 = module.to_quantum(params)
        features2 = module.to_quantum(params)

        np.testing.assert_array_equal(features1, features2)

    def test_classical_features_deterministic(self):
        """Test that classical features are deterministic."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS]
        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=np.pi / 3,
            agent_direction=Direction.RIGHT,
        )
        features1 = module.to_classical(params)
        features2 = module.to_classical(params)

        np.testing.assert_array_equal(features1, features2)
