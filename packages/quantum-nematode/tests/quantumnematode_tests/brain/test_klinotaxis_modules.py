"""Tests for klinotaxis (head-sweep) sensing modules."""

from __future__ import annotations

import numpy as np
import pytest
from quantumnematode.agent.agent import _compute_lateral_offsets
from quantumnematode.brain.arch._brain import BrainParams
from quantumnematode.brain.modules import (
    SENSORY_MODULES,
    ModuleName,
    get_classical_feature_dimension,
)
from quantumnematode.env.env import Direction


class TestComputeLateralOffsets:
    """Tests for _compute_lateral_offsets() head-sweep geometry."""

    def test_up_heading(self) -> None:
        """UP heading: left=west, right=east."""
        left, right = _compute_lateral_offsets(Direction.UP, (5, 5), 20)
        assert left == (4, 5)
        assert right == (6, 5)

    def test_right_heading(self) -> None:
        """RIGHT heading: left=north, right=south."""
        left, right = _compute_lateral_offsets(Direction.RIGHT, (5, 5), 20)
        assert left == (5, 6)
        assert right == (5, 4)

    def test_down_heading(self) -> None:
        """DOWN heading: left=east, right=west."""
        left, right = _compute_lateral_offsets(Direction.DOWN, (5, 5), 20)
        assert left == (6, 5)
        assert right == (4, 5)

    def test_left_heading(self) -> None:
        """LEFT heading: left=south, right=north."""
        left, right = _compute_lateral_offsets(Direction.LEFT, (5, 5), 20)
        assert left == (5, 4)
        assert right == (5, 6)

    def test_stay_defaults_to_up(self) -> None:
        """STAY heading defaults to UP-like offsets."""
        left, right = _compute_lateral_offsets(Direction.STAY, (5, 5), 20)
        assert left == (4, 5)
        assert right == (6, 5)

    def test_clamp_left_edge(self) -> None:
        """Position at x=0 with UP heading: left clamps to 0."""
        left, right = _compute_lateral_offsets(Direction.UP, (0, 5), 20)
        assert left == (0, 5)  # clamped
        assert right == (1, 5)

    def test_clamp_right_edge(self) -> None:
        """Position at x=19 with UP heading: right clamps to 19."""
        left, right = _compute_lateral_offsets(Direction.UP, (19, 5), 20)
        assert left == (18, 5)
        assert right == (19, 5)  # clamped

    def test_clamp_bottom_edge(self) -> None:
        """Position at y=0 with RIGHT heading: right clamps to 0."""
        left, right = _compute_lateral_offsets(Direction.RIGHT, (5, 0), 20)
        assert left == (5, 1)
        assert right == (5, 0)  # clamped


class TestKlinotaxisModuleRegistration:
    """Tests for klinotaxis module registration."""

    @pytest.mark.parametrize(
        "module_name",
        [
            ModuleName.FOOD_CHEMOTAXIS_KLINOTAXIS,
            ModuleName.NOCICEPTION_KLINOTAXIS,
            ModuleName.THERMOTAXIS_KLINOTAXIS,
            ModuleName.AEROTAXIS_KLINOTAXIS,
            ModuleName.PHEROMONE_FOOD_KLINOTAXIS,
            ModuleName.PHEROMONE_ALARM_KLINOTAXIS,
            ModuleName.PHEROMONE_AGGREGATION_KLINOTAXIS,
        ],
    )
    def test_registered(self, module_name: ModuleName) -> None:
        """All 7 klinotaxis modules are registered in SENSORY_MODULES."""
        assert module_name in SENSORY_MODULES

    @pytest.mark.parametrize(
        "module_name",
        [
            ModuleName.FOOD_CHEMOTAXIS_KLINOTAXIS,
            ModuleName.NOCICEPTION_KLINOTAXIS,
            ModuleName.THERMOTAXIS_KLINOTAXIS,
            ModuleName.AEROTAXIS_KLINOTAXIS,
            ModuleName.PHEROMONE_FOOD_KLINOTAXIS,
            ModuleName.PHEROMONE_ALARM_KLINOTAXIS,
            ModuleName.PHEROMONE_AGGREGATION_KLINOTAXIS,
        ],
    )
    def test_classical_dim_is_3(self, module_name: ModuleName) -> None:
        """All klinotaxis modules have classical_dim=3."""
        assert SENSORY_MODULES[module_name].classical_dim == 3


class TestKlinotaxisFeatureExtraction:
    """Tests for klinotaxis feature extraction."""

    def test_food_klinotaxis_features(self) -> None:
        """Food klinotaxis: strength=concentration, angle=lateral, binary=dC/dt."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS_KLINOTAXIS]
        params = BrainParams(
            food_concentration=0.5,
            food_lateral_gradient=0.01,
            food_dconcentration_dt=0.02,
        )
        features = module.to_classical(params)
        assert features.shape == (3,)
        assert features[0] == pytest.approx(0.5)
        assert features[1] == pytest.approx(np.tanh(0.01 * 50.0))
        assert features[2] == pytest.approx(np.tanh(0.02 * 50.0))

    def test_none_fields_produce_zeros(self) -> None:
        """None lateral gradient and derivative produce zero features."""
        module = SENSORY_MODULES[ModuleName.FOOD_CHEMOTAXIS_KLINOTAXIS]
        params = BrainParams()
        features = module.to_classical(params)
        assert features.shape == (3,)
        np.testing.assert_array_equal(features, np.zeros(3, dtype=np.float32))

    def test_thermotaxis_klinotaxis_features(self) -> None:
        """Thermotaxis klinotaxis: strength=|deviation/15|, normalized lateral."""
        module = SENSORY_MODULES[ModuleName.THERMOTAXIS_KLINOTAXIS]
        params = BrainParams(
            temperature=25.0,
            cultivation_temperature=20.0,
            temperature_lateral_gradient=0.03,  # Already normalized by ÷15
            temperature_ddt=0.01,
        )
        features = module.to_classical(params)
        assert features.shape == (3,)
        assert features[0] == pytest.approx(5.0 / 15.0)  # |deviation/15|

    def test_aerotaxis_klinotaxis_features(self) -> None:
        """Aerotaxis klinotaxis: strength=O2/21, normalized lateral."""
        module = SENSORY_MODULES[ModuleName.AEROTAXIS_KLINOTAXIS]
        params = BrainParams(
            oxygen_concentration=10.5,
            oxygen_lateral_gradient=0.02,  # Already normalized by ÷21
            oxygen_dconcentration_dt=0.005,
        )
        features = module.to_classical(params)
        assert features.shape == (3,)
        assert features[0] == pytest.approx(10.5 / 21.0)

    def test_feature_dimension_with_klinotaxis(self) -> None:
        """Total feature dimension sums correctly with klinotaxis modules."""
        from quantumnematode.agent.stam import compute_memory_dim

        modules = [
            ModuleName.FOOD_CHEMOTAXIS_KLINOTAXIS,
            ModuleName.NOCICEPTION_KLINOTAXIS,
            ModuleName.STAM,
        ]
        stam_dim = compute_memory_dim(2)  # food + predator
        dim = get_classical_feature_dimension(modules, stam_dim_override=stam_dim)
        # Two klinotaxis modules at 3 features each, plus STAM
        assert dim == 3 + 3 + stam_dim


class TestKlinotaxisIntegration:
    """Integration tests for klinotaxis with agent."""

    def test_klinotaxis_populates_lateral_gradient(self) -> None:
        """Klinotaxis mode populates food_lateral_gradient on BrainParams."""
        from quantumnematode.agent import QuantumNematodeAgent
        from quantumnematode.brain.arch.qvarcircuit import (
            QVarCircuitBrain,
            QVarCircuitBrainConfig,
        )
        from quantumnematode.env import DynamicForagingEnvironment, ForagingParams
        from quantumnematode.utils.config_loader import SensingConfig, SensingMode

        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            seed=42,
        )
        sensing = SensingConfig(
            chemotaxis_mode=SensingMode.KLINOTAXIS,
            stam_enabled=True,
        )
        brain_config = QVarCircuitBrainConfig(seed=42)
        brain = QVarCircuitBrain(brain_config)
        agent = QuantumNematodeAgent(
            brain=brain,
            env=env,
            sensing_config=sensing,
        )
        params = agent._create_brain_params()
        assert params.food_lateral_gradient is not None
        assert params.food_concentration is not None
        # Oracle gradients should be suppressed
        assert params.food_gradient_strength is None
        assert params.food_gradient_direction is None
