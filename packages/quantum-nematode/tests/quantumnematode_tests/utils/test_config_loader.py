"""Tests for configuration loading utilities."""

import pytest
from quantumnematode.env import ForagingParams, HealthParams, PredatorParams, ThermotaxisParams
from quantumnematode.utils.config_loader import (
    EnvironmentConfig,
    ForagingConfig,
    HealthConfig,
    PredatorConfig,
    SensingConfig,
    SensingMode,
    ThermotaxisConfig,
    apply_sensing_mode,
    validate_sensing_config,
)


class TestHealthConfig:
    """Test cases for HealthConfig."""

    def test_default_values(self):
        """Test HealthConfig default values."""
        config = HealthConfig()
        assert config.enabled is False
        assert config.max_hp == 100.0
        assert config.predator_damage == 10.0
        assert config.food_healing == 5.0

    def test_custom_values(self):
        """Test HealthConfig with custom values."""
        config = HealthConfig(
            enabled=True,
            max_hp=200.0,
            predator_damage=25.0,
            food_healing=15.0,
        )
        assert config.enabled is True
        assert config.max_hp == 200.0
        assert config.predator_damage == 25.0
        assert config.food_healing == 15.0

    def test_to_params(self):
        """Test HealthConfig.to_params() conversion."""
        config = HealthConfig(
            enabled=True,
            max_hp=150.0,
            predator_damage=20.0,
            food_healing=10.0,
        )

        params = config.to_params()

        assert isinstance(params, HealthParams)
        assert params.enabled is True
        assert params.max_hp == 150.0
        assert params.predator_damage == 20.0
        assert params.food_healing == 10.0

    def test_to_params_default_values(self):
        """Test HealthConfig.to_params() with default values."""
        config = HealthConfig()
        params = config.to_params()

        assert isinstance(params, HealthParams)
        assert params.enabled is False
        assert params.max_hp == 100.0
        assert params.predator_damage == 10.0
        assert params.food_healing == 5.0


class TestForagingConfig:
    """Test cases for ForagingConfig."""

    def test_default_values(self):
        """Test ForagingConfig default values."""
        config = ForagingConfig()
        assert config.foods_on_grid == 10
        assert config.target_foods_to_collect == 15
        assert config.min_food_distance == 5
        assert config.agent_exclusion_radius == 10
        assert config.gradient_decay_constant == 10.0
        assert config.gradient_strength == 1.0

    def test_custom_values(self):
        """Test ForagingConfig with custom values."""
        config = ForagingConfig(
            foods_on_grid=10,
            target_foods_to_collect=20,
            min_food_distance=5,
            agent_exclusion_radius=8,
            gradient_decay_constant=12.0,
            gradient_strength=2.0,
        )
        assert config.foods_on_grid == 10
        assert config.target_foods_to_collect == 20
        assert config.min_food_distance == 5
        assert config.agent_exclusion_radius == 8
        assert config.gradient_decay_constant == 12.0
        assert config.gradient_strength == 2.0

    def test_to_params(self):
        """Test ForagingConfig.to_params() conversion."""
        config = ForagingConfig(
            foods_on_grid=8,
            target_foods_to_collect=15,
            min_food_distance=4,
            agent_exclusion_radius=6,
            gradient_decay_constant=10.0,
            gradient_strength=1.5,
        )

        params = config.to_params()

        assert isinstance(params, ForagingParams)
        assert params.foods_on_grid == 8
        assert params.target_foods_to_collect == 15
        assert params.min_food_distance == 4
        assert params.agent_exclusion_radius == 6
        assert params.gradient_decay_constant == 10.0
        assert params.gradient_strength == 1.5


class TestPredatorConfig:
    """Test cases for PredatorConfig."""

    def test_default_values(self):
        """Test PredatorConfig default values."""
        config = PredatorConfig()
        assert config.enabled is False
        assert config.count == 2
        assert config.speed == 1.0
        assert config.detection_radius == 8
        assert config.kill_radius == 0
        assert config.gradient_decay_constant == 12.0
        assert config.gradient_strength == 1.0

    def test_custom_values(self):
        """Test PredatorConfig with custom values."""
        config = PredatorConfig(
            enabled=True,
            count=5,
            speed=0.5,
            detection_radius=10,
            kill_radius=2,
            gradient_decay_constant=15.0,
            gradient_strength=2.0,
        )
        assert config.enabled is True
        assert config.count == 5
        assert config.speed == 0.5
        assert config.detection_radius == 10
        assert config.kill_radius == 2
        assert config.gradient_decay_constant == 15.0
        assert config.gradient_strength == 2.0

    def test_to_params(self):
        """Test PredatorConfig.to_params() conversion."""
        config = PredatorConfig(
            enabled=True,
            count=3,
            speed=0.75,
            detection_radius=6,
            kill_radius=1,
            gradient_decay_constant=10.0,
            gradient_strength=1.5,
        )

        params = config.to_params()

        assert isinstance(params, PredatorParams)
        assert params.enabled is True
        assert params.count == 3
        assert params.speed == 0.75
        assert params.detection_radius == 6
        assert params.kill_radius == 1
        assert params.gradient_decay_constant == 10.0
        assert params.gradient_strength == 1.5


class TestEnvironmentConfig:
    """Test cases for EnvironmentConfig."""

    def test_default_values(self):
        """Test default values."""
        config = EnvironmentConfig()
        assert config.grid_size == 50
        assert config.viewport_size == (11, 11)
        assert config.foraging is None
        assert config.predators is None
        assert config.health is None

    def test_with_nested_configs(self):
        """Test EnvironmentConfig with nested configurations."""
        config = EnvironmentConfig(
            grid_size=30,
            viewport_size=(15, 15),
            foraging=ForagingConfig(foods_on_grid=20),
            predators=PredatorConfig(enabled=True, count=3),
            health=HealthConfig(enabled=True, max_hp=200.0),
        )

        assert config.grid_size == 30
        assert config.viewport_size == (15, 15)
        assert config.foraging is not None
        assert config.foraging.foods_on_grid == 20
        assert config.predators is not None
        assert config.predators.enabled is True
        assert config.predators.count == 3
        assert config.health is not None
        assert config.health.enabled is True
        assert config.health.max_hp == 200.0

    def test_get_foraging_config(self):
        """Test get_foraging_config returns configured or default."""
        # With explicit config
        config_with = EnvironmentConfig(
            foraging=ForagingConfig(foods_on_grid=25),
        )
        foraging = config_with.get_foraging_config()
        assert foraging.foods_on_grid == 25

        # Without explicit config (should return default)
        config_without = EnvironmentConfig()
        foraging_default = config_without.get_foraging_config()
        assert foraging_default.foods_on_grid == 10  # Default value

    def test_get_predator_config(self):
        """Test get_predator_config returns configured or default."""
        # With explicit config
        config_with = EnvironmentConfig(
            predators=PredatorConfig(enabled=True, count=5),
        )
        predator = config_with.get_predator_config()
        assert predator.enabled is True
        assert predator.count == 5

        # Without explicit config (should return default)
        config_without = EnvironmentConfig()
        predator_default = config_without.get_predator_config()
        assert predator_default.enabled is False  # Default value

    def test_get_health_config(self):
        """Test get_health_config returns configured or default."""
        # With explicit config
        config_with = EnvironmentConfig(
            health=HealthConfig(enabled=True, max_hp=150.0),
        )
        health = config_with.get_health_config()
        assert health.enabled is True
        assert health.max_hp == 150.0

        # Without explicit config (should return default)
        config_without = EnvironmentConfig()
        health_default = config_without.get_health_config()
        assert health_default.enabled is False  # Default value
        assert health_default.max_hp == 100.0  # Default value


class TestThermotaxisConfig:
    """Test cases for ThermotaxisConfig."""

    def test_default_values(self):
        """Test ThermotaxisConfig default values."""
        config = ThermotaxisConfig()
        assert config.enabled is False
        assert config.cultivation_temperature == 20.0
        assert config.base_temperature == 20.0
        assert config.gradient_direction == 0.0
        assert config.gradient_strength == 0.5
        assert config.hot_spots is None
        assert config.cold_spots is None
        assert config.spot_decay_constant == 5.0
        assert config.comfort_delta == 5.0
        assert config.discomfort_delta == 10.0
        assert config.danger_delta == 15.0

    def test_custom_values_with_spots(self):
        """Test ThermotaxisConfig with hot/cold spots."""
        config = ThermotaxisConfig(
            enabled=True,
            cultivation_temperature=22.0,
            base_temperature=22.0,
            gradient_strength=0.15,
            hot_spots=[[75, 50, 25.0], [25, 75, 20.0]],
            cold_spots=[[25, 25, 25.0]],
            spot_decay_constant=8.0,
        )
        assert config.enabled is True
        assert config.cultivation_temperature == 22.0
        assert config.gradient_strength == 0.15
        assert config.hot_spots == [[75, 50, 25.0], [25, 75, 20.0]]
        assert config.cold_spots == [[25, 25, 25.0]]
        assert config.spot_decay_constant == 8.0

    def test_to_params_basic(self):
        """Test ThermotaxisConfig.to_params() basic conversion."""
        config = ThermotaxisConfig(
            enabled=True,
            cultivation_temperature=22.0,
            base_temperature=22.0,
            gradient_direction=1.57,
            gradient_strength=0.3,
            comfort_delta=4.0,
            discomfort_delta=8.0,
            danger_delta=12.0,
        )

        params = config.to_params()

        assert isinstance(params, ThermotaxisParams)
        assert params.enabled is True
        assert params.cultivation_temperature == 22.0
        assert params.base_temperature == 22.0
        assert params.gradient_direction == 1.57
        assert params.gradient_strength == 0.3
        assert params.comfort_delta == 4.0
        assert params.discomfort_delta == 8.0
        assert params.danger_delta == 12.0

    def test_to_params_converts_hot_spots_to_tuples(self):
        """Test that to_params() converts hot_spots from list of lists to list of tuples."""
        config = ThermotaxisConfig(
            enabled=True,
            hot_spots=[[75, 50, 25.0], [25, 75, 20.0], [80, 80, 22.0]],
            spot_decay_constant=8.0,
        )

        params = config.to_params()

        assert params.hot_spots is not None
        assert len(params.hot_spots) == 3
        # Verify each spot is a tuple with correct values
        assert params.hot_spots[0] == (75, 50, 25.0)
        assert params.hot_spots[1] == (25, 75, 20.0)
        assert params.hot_spots[2] == (80, 80, 22.0)
        # Verify they are actual tuples (important for type safety)
        assert isinstance(params.hot_spots[0], tuple)
        assert params.spot_decay_constant == 8.0

    def test_to_params_converts_cold_spots_to_tuples(self):
        """Test that to_params() converts cold_spots from list of lists to list of tuples."""
        config = ThermotaxisConfig(
            enabled=True,
            cold_spots=[[25, 25, 25.0], [75, 25, 20.0], [50, 85, 18.0]],
            spot_decay_constant=10.0,
        )

        params = config.to_params()

        assert params.cold_spots is not None
        assert len(params.cold_spots) == 3
        # Verify each spot is a tuple with correct values
        assert params.cold_spots[0] == (25, 25, 25.0)
        assert params.cold_spots[1] == (75, 25, 20.0)
        assert params.cold_spots[2] == (50, 85, 18.0)
        # Verify they are actual tuples
        assert isinstance(params.cold_spots[0], tuple)
        assert params.spot_decay_constant == 10.0

    def test_to_params_with_none_spots(self):
        """Test that to_params() handles None hot_spots and cold_spots."""
        config = ThermotaxisConfig(
            enabled=True,
            hot_spots=None,
            cold_spots=None,
        )

        params = config.to_params()

        assert params.hot_spots is None
        assert params.cold_spots is None

    def test_to_params_converts_float_coordinates_to_int(self):
        """Test that to_params() converts float coordinates to int."""
        # YAML may parse coordinates as floats
        config = ThermotaxisConfig(
            enabled=True,
            hot_spots=[[75.0, 50.0, 25.0]],
            cold_spots=[[25.0, 25.0, 25.0]],
        )

        params = config.to_params()

        assert params.hot_spots is not None
        assert params.cold_spots is not None
        # Coordinates should be int, intensity should be float
        assert params.hot_spots[0] == (75, 50, 25.0)
        assert params.cold_spots[0] == (25, 25, 25.0)
        assert isinstance(params.hot_spots[0][0], int)
        assert isinstance(params.hot_spots[0][1], int)
        assert isinstance(params.hot_spots[0][2], float)

    def test_to_params_validates_hot_spot_format(self):
        """Test that to_params() raises ValueError for malformed hot_spots."""
        config = ThermotaxisConfig(
            enabled=True,
            hot_spots=[[75, 50]],  # Missing intensity
        )

        with pytest.raises(ValueError, match=r"Invalid hot_spot at index 0"):
            config.to_params()

    def test_to_params_validates_cold_spot_format(self):
        """Test that to_params() raises ValueError for malformed cold_spots."""
        config = ThermotaxisConfig(
            enabled=True,
            cold_spots=[[25, 25, 10.0, 5.0]],  # Extra element
        )

        with pytest.raises(ValueError, match=r"Invalid cold_spot at index 0"):
            config.to_params()

    def test_to_params_default_values(self):
        """Test ThermotaxisConfig.to_params() with default values."""
        config = ThermotaxisConfig()
        params = config.to_params()

        assert isinstance(params, ThermotaxisParams)
        assert params.enabled is False
        assert params.cultivation_temperature == 20.0
        assert params.base_temperature == 20.0
        assert params.gradient_direction == 0.0
        assert params.gradient_strength == 0.5
        assert params.hot_spots is None
        assert params.cold_spots is None
        assert params.spot_decay_constant == 5.0
        assert params.comfort_delta == 5.0
        assert params.discomfort_delta == 10.0
        assert params.danger_delta == 15.0

    def test_to_params_with_empty_spots(self):
        """Test that to_params() handles empty lists for hot_spots and cold_spots."""
        config = ThermotaxisConfig(
            enabled=True,
            hot_spots=[],
            cold_spots=[],
        )

        params = config.to_params()

        assert params.hot_spots == []
        assert params.cold_spots == []


class TestSensingConfig:
    """Test cases for SensingConfig and sensing mode logic."""

    def test_defaults_all_oracle(self) -> None:
        """Test that default sensing modes are all ORACLE with STAM disabled."""
        config = SensingConfig()
        assert config.chemotaxis_mode == SensingMode.ORACLE
        assert config.thermotaxis_mode == SensingMode.ORACLE
        assert config.nociception_mode == SensingMode.ORACLE
        assert config.stam_enabled is False
        assert config.stam_buffer_size == 30
        assert config.stam_decay_rate == 0.1

    def test_custom_modes(self) -> None:
        """Test that custom sensing modes are accepted and stored correctly."""
        config = SensingConfig(
            chemotaxis_mode=SensingMode.TEMPORAL,
            thermotaxis_mode=SensingMode.DERIVATIVE,
            nociception_mode=SensingMode.ORACLE,
        )
        assert config.chemotaxis_mode == SensingMode.TEMPORAL
        assert config.thermotaxis_mode == SensingMode.DERIVATIVE
        assert config.nociception_mode == SensingMode.ORACLE

    def test_invalid_mode_rejected(self) -> None:
        """Test that an invalid sensing mode raises ValueError."""
        with pytest.raises(ValueError, match="validation error"):
            SensingConfig.model_validate({"chemotaxis_mode": "invalid"})

    def test_buffer_size_must_be_positive(self) -> None:
        """Test that non-positive buffer size raises ValueError."""
        with pytest.raises(ValueError, match="greater than 0"):
            SensingConfig(stam_buffer_size=0)
        with pytest.raises(ValueError, match="greater than 0"):
            SensingConfig(stam_buffer_size=-1)

    def test_decay_rate_must_be_positive(self) -> None:
        """Test that non-positive decay rate raises ValueError."""
        with pytest.raises(ValueError, match="greater than 0"):
            SensingConfig(stam_decay_rate=0.0)
        with pytest.raises(ValueError, match="greater than 0"):
            SensingConfig(stam_decay_rate=-0.1)

    def test_environment_config_has_sensing(self) -> None:
        """Test that EnvironmentConfig provides default sensing configuration."""
        env_config = EnvironmentConfig()
        sensing = env_config.get_sensing_config()
        assert sensing.chemotaxis_mode == SensingMode.ORACLE
        assert sensing.stam_enabled is False

    def test_environment_config_custom_sensing(self) -> None:
        """Test that EnvironmentConfig accepts custom sensing configuration."""
        env_config = EnvironmentConfig(
            sensing=SensingConfig(chemotaxis_mode=SensingMode.TEMPORAL, stam_enabled=True),
        )
        sensing = env_config.get_sensing_config()
        assert sensing.chemotaxis_mode == SensingMode.TEMPORAL
        assert sensing.stam_enabled is True


class TestValidateSensingConfig:
    """Test validation and auto-enable logic."""

    def test_derivative_auto_enables_stam(self) -> None:
        """Test that derivative mode auto-enables STAM."""
        config = SensingConfig(chemotaxis_mode=SensingMode.DERIVATIVE)
        validated = validate_sensing_config(config)
        assert validated.stam_enabled is True

    def test_derivative_preserves_custom_stam_params(self) -> None:
        """Test that derivative mode preserves custom STAM buffer parameters."""
        config = SensingConfig(
            chemotaxis_mode=SensingMode.DERIVATIVE,
            stam_buffer_size=50,
            stam_decay_rate=0.2,
        )
        validated = validate_sensing_config(config)
        assert validated.stam_enabled is True
        assert validated.stam_buffer_size == 50
        assert validated.stam_decay_rate == 0.2

    def test_derivative_with_stam_already_enabled(self) -> None:
        """Test that derivative mode keeps STAM enabled when already set."""
        config = SensingConfig(
            chemotaxis_mode=SensingMode.DERIVATIVE,
            stam_enabled=True,
        )
        validated = validate_sensing_config(config)
        assert validated.stam_enabled is True

    def test_oracle_mode_no_stam_change(self) -> None:
        """Test that oracle mode does not auto-enable STAM."""
        config = SensingConfig()
        validated = validate_sensing_config(config)
        assert validated.stam_enabled is False

    def test_temporal_without_stam_warns(self) -> None:
        """Temporal without STAM should be accepted but is not recommended."""
        config = SensingConfig(chemotaxis_mode=SensingMode.TEMPORAL)
        validated = validate_sensing_config(config)
        # Should still be accepted (no auto-enable for temporal)
        assert validated.stam_enabled is False


class TestApplySensingMode:
    """Test sensory module name translation."""

    def test_oracle_mode_no_changes(self) -> None:
        """Test that oracle mode leaves module names unchanged."""
        sensing = SensingConfig()
        modules = ["food_chemotaxis", "mechanosensation"]
        result = apply_sensing_mode(modules, sensing)
        assert result == ["food_chemotaxis", "mechanosensation"]

    def test_temporal_replaces_food_chemotaxis(self) -> None:
        """Test that temporal mode replaces food_chemotaxis with food_chemotaxis_temporal."""
        sensing = SensingConfig(chemotaxis_mode=SensingMode.TEMPORAL)
        modules = ["food_chemotaxis", "mechanosensation"]
        result = apply_sensing_mode(modules, sensing)
        assert "food_chemotaxis_temporal" in result
        assert "food_chemotaxis" not in result
        assert "mechanosensation" in result

    def test_temporal_replaces_nociception(self) -> None:
        """Test that temporal mode replaces nociception with nociception_temporal."""
        sensing = SensingConfig(nociception_mode=SensingMode.TEMPORAL)
        modules = ["food_chemotaxis", "nociception"]
        result = apply_sensing_mode(modules, sensing)
        assert "nociception_temporal" in result
        assert "nociception" not in result

    def test_temporal_replaces_thermotaxis(self) -> None:
        """Test that derivative mode replaces thermotaxis with thermotaxis_temporal."""
        sensing = SensingConfig(thermotaxis_mode=SensingMode.DERIVATIVE)
        modules = ["food_chemotaxis", "thermotaxis"]
        result = apply_sensing_mode(modules, sensing)
        assert "thermotaxis_temporal" in result
        assert "thermotaxis" not in result

    def test_combined_chemotaxis_splits(self) -> None:
        """Test that combined chemotaxis is split into food and nociception modules."""
        sensing = SensingConfig(chemotaxis_mode=SensingMode.TEMPORAL)
        modules = ["chemotaxis"]
        result = apply_sensing_mode(modules, sensing)
        assert "food_chemotaxis_temporal" in result
        assert "nociception" in result  # Oracle nociception added
        assert "chemotaxis" not in result

    def test_combined_chemotaxis_splits_with_temporal_nociception(self) -> None:
        """Test that combined chemotaxis splits both food and nociception to temporal."""
        sensing = SensingConfig(
            chemotaxis_mode=SensingMode.TEMPORAL,
            nociception_mode=SensingMode.TEMPORAL,
        )
        modules = ["chemotaxis"]
        result = apply_sensing_mode(modules, sensing)
        assert "food_chemotaxis_temporal" in result
        assert "nociception_temporal" in result

    def test_combined_chemotaxis_no_duplicate_nociception(self) -> None:
        """Test that splitting chemotaxis does not duplicate nociception."""
        sensing = SensingConfig(chemotaxis_mode=SensingMode.TEMPORAL)
        modules = ["chemotaxis", "nociception"]
        result = apply_sensing_mode(modules, sensing)
        noci_count = sum(1 for m in result if "nociception" in m)
        assert noci_count == 1

    def test_stam_appended_when_enabled(self) -> None:
        """Test that STAM module is appended when stam_enabled is True."""
        sensing = SensingConfig(stam_enabled=True)
        modules = ["food_chemotaxis"]
        result = apply_sensing_mode(modules, sensing)
        assert "stam" in result

    def test_stam_not_duplicated(self) -> None:
        """Test that STAM is not duplicated if already in the module list."""
        sensing = SensingConfig(stam_enabled=True)
        modules = ["food_chemotaxis", "stam"]
        result = apply_sensing_mode(modules, sensing)
        assert result.count("stam") == 1

    def test_stam_not_appended_when_disabled(self) -> None:
        """Test that STAM module is not appended when stam_enabled is False."""
        sensing = SensingConfig(stam_enabled=False)
        modules = ["food_chemotaxis"]
        result = apply_sensing_mode(modules, sensing)
        assert "stam" not in result

    def test_mechanosensation_never_replaced(self) -> None:
        """Test that mechanosensation is never replaced by temporal modes."""
        sensing = SensingConfig(
            chemotaxis_mode=SensingMode.TEMPORAL,
            thermotaxis_mode=SensingMode.TEMPORAL,
            nociception_mode=SensingMode.TEMPORAL,
        )
        modules = ["food_chemotaxis", "mechanosensation", "nociception", "thermotaxis"]
        result = apply_sensing_mode(modules, sensing)
        assert "mechanosensation" in result
