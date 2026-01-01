"""Tests for brain feature extraction modules."""

import numpy as np
import pytest
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.modules import (
    MODULE_FEATURE_EXTRACTORS,
    ModuleName,
    RotationAxis,
    appetitive_features,
    aversive_features,
    chemotaxis_features,
    count_total_qubits,
    extract_features_for_module,
    mechanosensation_features,
    memory_action_features,
    oxygen_features,
    proprioception_features,
    thermotaxis_features,
    vision_features,
)
from quantumnematode.env import Direction


class TestRotationAxis:
    """Test RotationAxis enum."""

    def test_rotation_axis_values(self):
        """Test that rotation axis enum values are correct."""
        assert RotationAxis.RX.value == "rx"
        assert RotationAxis.RY.value == "ry"
        assert RotationAxis.RZ.value == "rz"

    def test_rotation_axis_completeness(self):
        """Test that all expected rotation axes exist."""
        axes = {RotationAxis.RX, RotationAxis.RY, RotationAxis.RZ}
        assert len(axes) == 3


class TestProprioceptionFeatures:
    """Test proprioception feature extraction."""

    def test_direction_up(self):
        """Test proprioception features for UP direction."""
        params = BrainParams(agent_direction=Direction.UP)
        features = proprioception_features(params)

        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == 0.0

    def test_direction_down(self):
        """Test proprioception features for DOWN direction."""
        params = BrainParams(agent_direction=Direction.DOWN)
        features = proprioception_features(params)

        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == np.pi

    def test_direction_left(self):
        """Test proprioception features for LEFT direction."""
        params = BrainParams(agent_direction=Direction.LEFT)
        features = proprioception_features(params)

        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == pytest.approx(np.pi / 2)

    def test_direction_right(self):
        """Test proprioception features for RIGHT direction."""
        params = BrainParams(agent_direction=Direction.RIGHT)
        features = proprioception_features(params)

        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == pytest.approx(-np.pi / 2)

    def test_no_direction(self):
        """Test proprioception features when direction is None (defaults to UP)."""
        params = BrainParams(agent_direction=None)
        features = proprioception_features(params)

        assert features[RotationAxis.RZ] == 0.0

    def test_proprioception_features_deterministic(self):
        """Test that proprioception features are deterministic."""
        params = BrainParams(agent_direction=Direction.UP)
        features1 = proprioception_features(params)
        features2 = proprioception_features(params)

        assert features1 == features2


class TestChemotaxisFeatures:
    """Test chemotaxis feature extraction."""

    def test_zero_gradient(self):
        """Test chemotaxis features with zero gradient strength."""
        params = BrainParams(
            gradient_strength=0.0,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = chemotaxis_features(params)

        # Zero gradient should give -π/2 for RX (scaled from 0)
        assert features[RotationAxis.RX] == pytest.approx(-np.pi / 2)
        assert features[RotationAxis.RZ] == 0.0

    def test_max_gradient(self):
        """Test chemotaxis features with maximum gradient strength."""
        params = BrainParams(
            gradient_strength=1.0,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = chemotaxis_features(params)

        # Max gradient (1.0) should give π/2 for RX
        assert features[RotationAxis.RX] == pytest.approx(np.pi / 2)

    def test_gradient_direction_aligned(self):
        """Test when gradient direction aligns with agent direction."""
        # Agent facing right (0.0), gradient also at 0.0
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.RIGHT,
        )
        features = chemotaxis_features(params)

        # Relative angle should be 0 (aligned)
        assert features[RotationAxis.RY] == pytest.approx(0.0, abs=1e-6)

    def test_gradient_direction_opposite(self):
        """Test when gradient direction is opposite to agent direction."""
        # Agent facing right (0.0), gradient at π (opposite)
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=np.pi,
            agent_direction=Direction.RIGHT,
        )
        features = chemotaxis_features(params)

        # Relative angle should be close to maximum (±π/2)
        assert abs(features[RotationAxis.RY]) == pytest.approx(np.pi / 2, rel=0.01)

    def test_gradient_perpendicular(self):
        """Test when gradient is perpendicular to agent direction."""
        # Agent facing up (π/2), gradient at 0.0 (right)
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = chemotaxis_features(params)

        # Should have a non-zero relative angle for RY
        assert features[RotationAxis.RY] != 0.0

    def test_none_gradient_values(self):
        """Test chemotaxis features when gradient values are None."""
        params = BrainParams(
            gradient_strength=None,
            gradient_direction=None,
            agent_direction=Direction.UP,
        )
        features = chemotaxis_features(params)

        # gradient_strength=None defaults to 0.0, scaled to -π/2
        assert features[RotationAxis.RX] == pytest.approx(-np.pi / 2)
        # gradient_direction=None defaults to 0.0 with agent facing UP at angle π/2.
        # The relative angle calculation yields -π/2, which normalizes to -0.5,
        # then scales to -π/4 for the final RY rotation value.
        assert features[RotationAxis.RY] == pytest.approx(-np.pi / 4)


class TestPlaceholderModules:
    """Test placeholder modules that return zero features."""

    def test_thermotaxis_returns_zeros(self):
        """Test that thermotaxis returns all zeros (placeholder)."""
        params = BrainParams()
        features = thermotaxis_features(params)

        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == 0.0

    def test_oxygen_returns_zeros(self):
        """Test that oxygen sensing returns all zeros (placeholder)."""
        params = BrainParams()
        features = oxygen_features(params)

        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == 0.0

    def test_vision_returns_zeros(self):
        """Test that vision returns all zeros (placeholder)."""
        params = BrainParams()
        features = vision_features(params)

        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == 0.0


class TestMemoryActionFeatures:
    """Test memory action feature extraction."""

    def test_action_forward(self):
        """Test action features for FORWARD action."""
        from quantumnematode.brain.actions import Action, ActionData

        action_data = ActionData(action=Action.FORWARD, probability=0.75, state="00")
        params = BrainParams()
        params.action = action_data

        features = memory_action_features(params)

        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == np.pi

    def test_action_stay(self):
        """Test action features for STAY action."""
        from quantumnematode.brain.actions import Action, ActionData

        action_data = ActionData(action=Action.STAY, probability=0.75, state="00")
        params = BrainParams()
        params.action = action_data

        features = memory_action_features(params)

        assert features[RotationAxis.RZ] == 0.0

    def test_action_left(self):
        """Test action features for LEFT action."""
        from quantumnematode.brain.actions import Action, ActionData

        action_data = ActionData(action=Action.LEFT, probability=0.75, state="00")
        params = BrainParams()
        params.action = action_data

        features = memory_action_features(params)

        assert features[RotationAxis.RZ] == pytest.approx(np.pi / 2)

    def test_action_right(self):
        """Test action features for RIGHT action."""
        from quantumnematode.brain.actions import Action, ActionData

        action_data = ActionData(action=Action.RIGHT, probability=0.75, state="00")
        params = BrainParams()
        params.action = action_data

        features = memory_action_features(params)

        assert features[RotationAxis.RZ] == pytest.approx(-np.pi / 2)

    def test_no_action(self):
        """Test action features when no action is present."""
        params = BrainParams()
        features = memory_action_features(params)

        # Should default to 0.0
        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == 0.0


class TestModuleName:
    """Test ModuleName enum."""

    def test_module_names(self):
        """Test that all module names have correct values."""
        # Core modules
        assert ModuleName.PROPRIOCEPTION.value == "proprioception"
        assert ModuleName.CHEMOTAXIS.value == "chemotaxis"
        assert ModuleName.THERMOTAXIS.value == "thermotaxis"
        assert ModuleName.VISION.value == "vision"
        assert ModuleName.ACTION.value == "action"
        assert ModuleName.MECHANOSENSATION.value == "mechanosensation"

        # Scientific names (preferred)
        assert ModuleName.FOOD_CHEMOTAXIS.value == "food_chemotaxis"
        assert ModuleName.NOCICEPTION.value == "nociception"
        assert ModuleName.AEROTAXIS.value == "aerotaxis"

        # Legacy names (deprecated, kept for backward compatibility)
        assert ModuleName.APPETITIVE.value == "appetitive"
        assert ModuleName.AVERSIVE.value == "aversive"
        assert ModuleName.OXYGEN.value == "oxygen"

    def test_all_modules_have_extractors(self):
        """Test that all module names have corresponding feature extractors."""
        for module_name in ModuleName:
            assert module_name in MODULE_FEATURE_EXTRACTORS, (
                f"Module {module_name} missing from MODULE_FEATURE_EXTRACTORS"
            )


class TestExtractFeaturesForModule:
    """Test the extract_features_for_module utility function."""

    def test_extract_chemotaxis(self):
        """Test extracting chemotaxis features."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = extract_features_for_module(ModuleName.CHEMOTAXIS, params)

        # Should return dict with string keys
        assert "rx" in features
        assert "ry" in features
        assert "rz" in features
        assert isinstance(features["rx"], (float, np.floating))

    def test_extract_proprioception(self):
        """Test extracting proprioception features."""
        params = BrainParams(agent_direction=Direction.DOWN)
        features = extract_features_for_module(ModuleName.PROPRIOCEPTION, params)

        assert features["rz"] == np.pi
        assert features["rx"] == 0.0
        assert features["ry"] == 0.0

    def test_extract_unknown_module(self):
        """Test that unknown modules return zero features."""
        # This shouldn't happen in practice, but test defensive behavior
        params = BrainParams()
        # Create a mock module name that doesn't exist
        features = extract_features_for_module(ModuleName.VISION, params)

        # Placeholder modules should return zeros
        assert features["rx"] == 0.0
        assert features["ry"] == 0.0
        assert features["rz"] == 0.0

    def test_extract_features_works(self):
        """Test that extract_features_for_module works correctly."""
        params = BrainParams(gradient_strength=0.5)
        # Should not raise an error
        features = extract_features_for_module(
            ModuleName.CHEMOTAXIS,
            params,
        )
        assert isinstance(features, dict)
        assert "rx" in features
        assert "ry" in features
        assert "rz" in features


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
        # Should count unique qubits only
        assert count_total_qubits(modules) == 3

    def test_empty_modules(self):
        """Test counting qubits with no modules."""
        modules = {}
        assert count_total_qubits(modules) == 0

    def test_module_with_empty_qubit_list(self):
        """Test counting qubits when a module has no qubits assigned."""
        modules = {
            ModuleName.CHEMOTAXIS: [0, 1],
            ModuleName.PROPRIOCEPTION: [],
        }
        assert count_total_qubits(modules) == 2

    def test_non_sequential_qubits(self):
        """Test counting non-sequential qubit indices."""
        modules = {
            ModuleName.CHEMOTAXIS: [0, 5, 10],
            ModuleName.PROPRIOCEPTION: [2, 7],
        }
        # Should count all unique indices
        assert count_total_qubits(modules) == 5  # 0, 2, 5, 7, 10


class TestFeatureValueRanges:
    """Test that feature values stay within reasonable quantum rotation ranges."""

    def test_chemotaxis_rx_bounded(self):
        """Test that chemotaxis RX stays within [-π/2, π/2]."""
        # Test with various gradient strengths
        for strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
            params = BrainParams(
                gradient_strength=strength,
                gradient_direction=0.0,
                agent_direction=Direction.UP,
            )
            features = chemotaxis_features(params)
            assert -np.pi / 2 <= features[RotationAxis.RX] <= np.pi / 2

    def test_chemotaxis_ry_bounded(self):
        """Test that chemotaxis RY stays within [-π/2, π/2]."""
        # Test with various relative angles
        for grad_dir in np.linspace(-np.pi, np.pi, 8):
            params = BrainParams(
                gradient_strength=0.5,
                gradient_direction=float(grad_dir),
                agent_direction=Direction.UP,
            )
            features = chemotaxis_features(params)
            assert -np.pi / 2 <= features[RotationAxis.RY] <= np.pi / 2

    def test_proprioception_rz_bounded(self):
        """Test that proprioception RZ stays within [-π, π]."""
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            params = BrainParams(agent_direction=direction)
            features = proprioception_features(params)
            assert -np.pi <= features[RotationAxis.RZ] <= np.pi

    def test_action_rz_bounded(self):
        """Test that action RZ stays within [-π, π]."""
        from quantumnematode.brain.actions import Action, ActionData

        for action in [Action.FORWARD, Action.STAY, Action.LEFT, Action.RIGHT]:
            action_data = ActionData(action=action, probability=0.75, state="00")
            params = BrainParams()
            params.action = action_data
            features = memory_action_features(params)
            assert -np.pi <= features[RotationAxis.RZ] <= np.pi


class TestFeatureConsistency:
    """Test that feature extraction is consistent and deterministic."""

    def test_proprioception_deterministic(self):
        """Test that proprioception features are deterministic."""
        params = BrainParams(agent_direction=Direction.LEFT)
        features1 = proprioception_features(params)
        features2 = proprioception_features(params)

        assert features1 == features2

    def test_chemotaxis_deterministic(self):
        """Test that chemotaxis features are deterministic."""
        params = BrainParams(
            gradient_strength=0.7,
            gradient_direction=np.pi / 4,
            agent_direction=Direction.UP,
        )
        features1 = chemotaxis_features(params)
        features2 = chemotaxis_features(params)

        assert features1[RotationAxis.RX] == features2[RotationAxis.RX]
        assert features1[RotationAxis.RY] == features2[RotationAxis.RY]
        assert features1[RotationAxis.RZ] == features2[RotationAxis.RZ]

    def test_extract_features_deterministic(self):
        """Test that extract_features_for_module is deterministic."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=0.0,
            agent_direction=Direction.RIGHT,
        )
        features1 = extract_features_for_module(ModuleName.CHEMOTAXIS, params)
        features2 = extract_features_for_module(ModuleName.CHEMOTAXIS, params)

        assert features1 == features2


class TestAppetitiveFeatures:
    """Test appetitive (food-seeking) feature extraction."""

    def test_zero_food_gradient(self):
        """Test appetitive features with zero food gradient strength."""
        params = BrainParams(
            food_gradient_strength=0.0,
            food_gradient_direction=0.0,
            agent_direction=Direction.UP,
            satiety=100.0,
        )
        features = appetitive_features(params)

        # tanh(0) = 0, scaled to -π/2
        assert features[RotationAxis.RX] == pytest.approx(-np.pi / 2)

    def test_strong_food_gradient(self):
        """Test appetitive features with strong food gradient."""
        params = BrainParams(
            food_gradient_strength=2.0,  # Strong signal
            food_gradient_direction=0.0,
            agent_direction=Direction.UP,
            satiety=0.0,  # Starving
        )
        features = appetitive_features(params)

        # tanh(2.0) ≈ 0.964, scaled to near π/2
        assert features[RotationAxis.RX] > 0

    def test_food_direction_aligned(self):
        """Test when food direction aligns with agent direction."""
        params = BrainParams(
            food_gradient_strength=1.0,
            food_gradient_direction=0.0,  # Food to the right
            agent_direction=Direction.RIGHT,
            satiety=100.0,
        )
        features = appetitive_features(params)

        # Relative angle should be 0 (aligned)
        assert features[RotationAxis.RY] == pytest.approx(0.0, abs=1e-6)

    def test_food_direction_opposite(self):
        """Test when food direction is opposite to agent direction."""
        params = BrainParams(
            food_gradient_strength=1.0,
            food_gradient_direction=np.pi,  # Food behind
            agent_direction=Direction.RIGHT,
            satiety=100.0,
        )
        features = appetitive_features(params)

        # Relative angle should be close to ±π/2
        assert abs(features[RotationAxis.RY]) == pytest.approx(np.pi / 2, rel=0.01)

    def test_rz_always_zero(self):
        """Test that RZ is always 0.0 (currently unused, reserved for future)."""
        # RZ is currently unused per feature contract - always returns 0.0
        # Could be used for satiety/hunger encoding in future
        for satiety in [0.0, 100.0, 200.0]:
            params = BrainParams(
                food_gradient_strength=0.5,
                food_gradient_direction=0.0,
                agent_direction=Direction.UP,
                satiety=satiety,
            )
            features = appetitive_features(params)
            assert features[RotationAxis.RZ] == pytest.approx(0.0)

    def test_none_food_gradient(self):
        """Test appetitive features when food gradient is None."""
        params = BrainParams(
            food_gradient_strength=None,
            food_gradient_direction=None,
            agent_direction=Direction.UP,
            satiety=100.0,
        )
        features = appetitive_features(params)

        # Should default to 0 strength, scaled to -π/2
        assert features[RotationAxis.RX] == pytest.approx(-np.pi / 2)
        # Direction None should give 0.0
        assert features[RotationAxis.RY] == pytest.approx(0.0)

    def test_appetitive_deterministic(self):
        """Test that appetitive features are deterministic."""
        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=np.pi / 4,
            agent_direction=Direction.UP,
            satiety=75.0,
        )
        features1 = appetitive_features(params)
        features2 = appetitive_features(params)

        assert features1 == features2


class TestAversiveFeatures:
    """Test aversive (predator-avoidance) feature extraction."""

    def test_zero_predator_gradient(self):
        """Test aversive features with zero predator gradient (no threat)."""
        params = BrainParams(
            predator_gradient_strength=0.0,
            predator_gradient_direction=0.0,
            agent_direction=Direction.UP,
            satiety=100.0,
        )
        features = aversive_features(params)

        # tanh(0) = 0, scaled to -π/2
        assert features[RotationAxis.RX] == pytest.approx(-np.pi / 2)

    def test_strong_predator_gradient(self):
        """Test aversive features with strong predator threat."""
        params = BrainParams(
            predator_gradient_strength=2.0,  # Strong threat
            predator_gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = aversive_features(params)

        # tanh(2.0) ≈ 0.964, scaled to near π/2
        assert features[RotationAxis.RX] > 0
        # RZ is currently unused - always 0.0
        assert features[RotationAxis.RZ] == pytest.approx(0.0)

    def test_escape_direction_aligned(self):
        """Test when escape direction aligns with agent direction."""
        params = BrainParams(
            predator_gradient_strength=1.0,
            predator_gradient_direction=0.0,  # Escape to the right
            agent_direction=Direction.RIGHT,
            satiety=100.0,
        )
        features = aversive_features(params)

        # Relative angle should be 0 (already facing escape direction)
        assert features[RotationAxis.RY] == pytest.approx(0.0, abs=1e-6)

    def test_escape_direction_opposite(self):
        """Test when escape direction is behind agent."""
        params = BrainParams(
            predator_gradient_strength=1.0,
            predator_gradient_direction=np.pi,  # Escape is behind
            agent_direction=Direction.RIGHT,
            satiety=100.0,
        )
        features = aversive_features(params)

        # Relative angle should be close to ±π/2
        assert abs(features[RotationAxis.RY]) == pytest.approx(np.pi / 2, rel=0.01)

    def test_rz_always_zero(self):
        """Test that RZ is always 0.0 (currently unused, reserved for future)."""
        # RZ is currently unused per feature contract - always returns 0.0
        # Could be used for risk tolerance encoding in future
        for satiety in [0.0, 100.0, 200.0]:
            params = BrainParams(
                predator_gradient_strength=1.0,
                predator_gradient_direction=0.0,
                agent_direction=Direction.UP,
                satiety=satiety,
            )
            features = aversive_features(params)
            assert features[RotationAxis.RZ] == pytest.approx(0.0)

    def test_none_predator_gradient(self):
        """Test aversive features when predator gradient is None."""
        params = BrainParams(
            predator_gradient_strength=None,
            predator_gradient_direction=None,
            agent_direction=Direction.UP,
            satiety=100.0,
        )
        features = aversive_features(params)

        # Should default to 0 strength, scaled to -π/2
        assert features[RotationAxis.RX] == pytest.approx(-np.pi / 2)
        # Direction None should give 0.0
        assert features[RotationAxis.RY] == pytest.approx(0.0)

    def test_aversive_deterministic(self):
        """Test that aversive features are deterministic."""
        params = BrainParams(
            predator_gradient_strength=0.5,
            predator_gradient_direction=np.pi / 4,
            agent_direction=Direction.UP,
            satiety=75.0,
        )
        features1 = aversive_features(params)
        features2 = aversive_features(params)

        assert features1 == features2


class TestAppetitiveAversiveValueRanges:
    """Test that appetitive and aversive feature values stay within bounds."""

    def test_appetitive_rx_bounded(self):
        """Test that appetitive RX stays within [-π/2, π/2]."""
        for strength in [0.0, 0.5, 1.0, 2.0, 5.0]:
            params = BrainParams(
                food_gradient_strength=strength,
                food_gradient_direction=0.0,
                agent_direction=Direction.UP,
                satiety=100.0,
            )
            features = appetitive_features(params)
            assert -np.pi / 2 <= features[RotationAxis.RX] <= np.pi / 2

    def test_appetitive_ry_bounded(self):
        """Test that appetitive RY stays within [-π/2, π/2]."""
        for direction in np.linspace(-np.pi, np.pi, 8):
            params = BrainParams(
                food_gradient_strength=1.0,
                food_gradient_direction=float(direction),
                agent_direction=Direction.UP,
                satiety=100.0,
            )
            features = appetitive_features(params)
            assert -np.pi / 2 <= features[RotationAxis.RY] <= np.pi / 2

    def test_appetitive_rz_always_zero(self):
        """Test that appetitive RZ is always 0.0 (currently unused)."""
        # RZ is reserved for future use (e.g., satiety/hunger encoding)
        for satiety in [0.0, 50.0, 100.0, 150.0, 200.0]:
            params = BrainParams(
                food_gradient_strength=1.0,
                food_gradient_direction=0.0,
                agent_direction=Direction.UP,
                satiety=satiety,
            )
            features = appetitive_features(params)
            assert features[RotationAxis.RZ] == pytest.approx(0.0)

    def test_aversive_rx_bounded(self):
        """Test that aversive RX stays within [-π/2, π/2]."""
        for strength in [0.0, 0.5, 1.0, 2.0, 5.0]:
            params = BrainParams(
                predator_gradient_strength=strength,
                predator_gradient_direction=0.0,
                agent_direction=Direction.UP,
                satiety=100.0,
            )
            features = aversive_features(params)
            assert -np.pi / 2 <= features[RotationAxis.RX] <= np.pi / 2

    def test_aversive_ry_bounded(self):
        """Test that aversive RY stays within [-π/2, π/2]."""
        for direction in np.linspace(-np.pi, np.pi, 8):
            params = BrainParams(
                predator_gradient_strength=1.0,
                predator_gradient_direction=float(direction),
                agent_direction=Direction.UP,
                satiety=100.0,
            )
            features = aversive_features(params)
            assert -np.pi / 2 <= features[RotationAxis.RY] <= np.pi / 2

    def test_aversive_rz_always_zero(self):
        """Test that aversive RZ is always 0.0 (currently unused)."""
        # RZ is reserved for future use (e.g., risk tolerance encoding)
        for satiety in [0.0, 50.0, 100.0, 150.0, 200.0]:
            params = BrainParams(
                predator_gradient_strength=1.0,
                predator_gradient_direction=0.0,
                agent_direction=Direction.UP,
                satiety=satiety,
            )
            features = aversive_features(params)
            assert features[RotationAxis.RZ] == pytest.approx(0.0)


class TestMechanosensationFeatures:
    """Test mechanosensation (touch/contact) feature extraction.

    Modeled after C. elegans touch response neurons:
    - ALM, PLM, AVM: Gentle touch neurons (boundary contact)
    - ASH, ADL: Harsh touch / nociception neurons (predator contact)
    """

    def test_no_contact(self):
        """Test mechanosensation features when no contact detected."""
        params = BrainParams(
            boundary_contact=False,
            predator_contact=False,
        )
        features = mechanosensation_features(params)

        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == 0.0

    def test_boundary_contact_only(self):
        """Test mechanosensation features when only boundary contact detected."""
        params = BrainParams(
            boundary_contact=True,
            predator_contact=False,
        )
        features = mechanosensation_features(params)

        # RX encodes boundary contact (gentle touch)
        assert features[RotationAxis.RX] == pytest.approx(np.pi / 2)
        # RY encodes predator contact (harsh touch)
        assert features[RotationAxis.RY] == 0.0
        # RZ encodes combined urgency (max of both)
        assert features[RotationAxis.RZ] == pytest.approx(np.pi / 2)

    def test_predator_contact_only(self):
        """Test mechanosensation features when only predator contact detected."""
        params = BrainParams(
            boundary_contact=False,
            predator_contact=True,
        )
        features = mechanosensation_features(params)

        # RX encodes boundary contact (gentle touch)
        assert features[RotationAxis.RX] == 0.0
        # RY encodes predator contact (harsh touch)
        assert features[RotationAxis.RY] == pytest.approx(np.pi / 2)
        # RZ encodes combined urgency (max of both)
        assert features[RotationAxis.RZ] == pytest.approx(np.pi / 2)

    def test_both_contacts(self):
        """Test mechanosensation features when both contacts detected."""
        params = BrainParams(
            boundary_contact=True,
            predator_contact=True,
        )
        features = mechanosensation_features(params)

        # Both RX and RY should be active
        assert features[RotationAxis.RX] == pytest.approx(np.pi / 2)
        assert features[RotationAxis.RY] == pytest.approx(np.pi / 2)
        # RZ encodes combined urgency (max of both, which is still π/2)
        assert features[RotationAxis.RZ] == pytest.approx(np.pi / 2)

    def test_none_values_treated_as_no_contact(self):
        """Test that None values are treated as no contact."""
        params = BrainParams(
            boundary_contact=None,
            predator_contact=None,
        )
        features = mechanosensation_features(params)

        assert features[RotationAxis.RX] == 0.0
        assert features[RotationAxis.RY] == 0.0
        assert features[RotationAxis.RZ] == 0.0

    def test_mechanosensation_deterministic(self):
        """Test that mechanosensation features are deterministic."""
        params = BrainParams(
            boundary_contact=True,
            predator_contact=False,
        )
        features1 = mechanosensation_features(params)
        features2 = mechanosensation_features(params)

        assert features1 == features2

    def test_mechanosensation_in_module_name_enum(self):
        """Test that MECHANOSENSATION is in ModuleName enum."""
        assert ModuleName.MECHANOSENSATION.value == "mechanosensation"

    def test_mechanosensation_in_feature_extractors(self):
        """Test that mechanosensation has a feature extractor registered."""
        assert ModuleName.MECHANOSENSATION in MODULE_FEATURE_EXTRACTORS
        assert MODULE_FEATURE_EXTRACTORS[ModuleName.MECHANOSENSATION] == mechanosensation_features

    def test_extract_features_for_mechanosensation(self):
        """Test extract_features_for_module with mechanosensation."""
        params = BrainParams(
            boundary_contact=True,
            predator_contact=False,
        )
        features = extract_features_for_module(ModuleName.MECHANOSENSATION, params)

        # Should return dict with string keys
        assert "rx" in features
        assert "ry" in features
        assert "rz" in features
        assert features["rx"] == pytest.approx(np.pi / 2)
        assert features["ry"] == 0.0
        assert features["rz"] == pytest.approx(np.pi / 2)

    def test_feature_values_bounded(self):
        """Test that mechanosensation features stay within quantum gate bounds."""
        # Test all combinations of contact states
        for boundary in [True, False, None]:
            for predator in [True, False, None]:
                params = BrainParams(
                    boundary_contact=boundary,
                    predator_contact=predator,
                )
                features = mechanosensation_features(params)

                # All values should be in [0, π/2] (non-negative aversive signals)
                assert 0.0 <= features[RotationAxis.RX] <= np.pi / 2
                assert 0.0 <= features[RotationAxis.RY] <= np.pi / 2
                assert 0.0 <= features[RotationAxis.RZ] <= np.pi / 2
