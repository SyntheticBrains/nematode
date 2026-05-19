"""Tests for configuration loading utilities."""

import pytest
from quantumnematode.env import (
    AerotaxisParams,
    ForagingParams,
    HealthParams,
    PredatorParams,
    ThermotaxisParams,
)
from quantumnematode.utils.config_loader import (
    AerotaxisConfig,
    EnvironmentConfig,
    EvolutionConfig,
    ForagingConfig,
    HealthConfig,
    LawnScheduleEntry,
    PredatorConfig,
    SensingConfig,
    SensingMode,
    ThermotaxisConfig,
    TransgenerationalConfig,
    apply_sensing_mode,
    validate_sensing_config,
)


class TestHealthConfig:
    """Test cases for HealthConfig."""

    def test_default_values(self):
        """Test HealthConfig default values."""
        config = HealthConfig()
        assert config.max_hp == 100.0
        assert config.predator_damage == 10.0
        assert config.food_healing == 5.0

    def test_custom_values(self):
        """Test HealthConfig with custom values."""
        config = HealthConfig(
            max_hp=200.0,
            predator_damage=25.0,
            food_healing=15.0,
        )
        assert config.max_hp == 200.0
        assert config.predator_damage == 25.0
        assert config.food_healing == 15.0

    def test_to_params(self):
        """Test HealthConfig.to_params() conversion."""
        config = HealthConfig(
            max_hp=150.0,
            predator_damage=20.0,
            food_healing=10.0,
        )

        params = config.to_params()

        assert isinstance(params, HealthParams)
        assert params.max_hp == 150.0
        assert params.predator_damage == 20.0
        assert params.food_healing == 10.0

    def test_to_params_default_values(self):
        """Test HealthConfig.to_params() with default values."""
        config = HealthConfig()
        params = config.to_params()

        assert isinstance(params, HealthParams)
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
        assert config.gradient_decay_constant == 12.0
        assert config.gradient_strength == 1.0

    def test_custom_values(self):
        """Test PredatorConfig with custom values."""
        config = PredatorConfig(
            enabled=True,
            count=5,
            speed=0.5,
            detection_radius=10,
            gradient_decay_constant=15.0,
            gradient_strength=2.0,
        )
        assert config.enabled is True
        assert config.count == 5
        assert config.speed == 0.5
        assert config.detection_radius == 10
        assert config.gradient_decay_constant == 15.0
        assert config.gradient_strength == 2.0

    def test_to_params(self):
        """Test PredatorConfig.to_params() conversion."""
        config = PredatorConfig(
            enabled=True,
            count=3,
            speed=0.75,
            detection_radius=6,
            gradient_decay_constant=10.0,
            gradient_strength=1.5,
        )

        params = config.to_params()

        assert isinstance(params, PredatorParams)
        assert params.enabled is True
        assert params.count == 3
        assert params.speed == 0.75
        assert params.detection_radius == 6
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
            health=HealthConfig(max_hp=200.0),
        )

        assert config.grid_size == 30
        assert config.viewport_size == (15, 15)
        assert config.foraging is not None
        assert config.foraging.foods_on_grid == 20
        assert config.predators is not None
        assert config.predators.enabled is True
        assert config.predators.count == 3
        assert config.health is not None
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
            health=HealthConfig(max_hp=150.0),
        )
        health = config_with.get_health_config()
        assert health.max_hp == 150.0

        # Without explicit config (should return default)
        config_without = EnvironmentConfig()
        health_default = config_without.get_health_config()
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


class TestAerotaxisConfig:
    """Test cases for AerotaxisConfig."""

    def test_aerotaxis_config_to_params(self):
        """Test AerotaxisConfig.to_params() basic conversion."""
        config = AerotaxisConfig(
            enabled=True,
            base_oxygen=10.0,
            gradient_direction=1.57,
            gradient_strength=0.1,
        )

        params = config.to_params()

        assert isinstance(params, AerotaxisParams)
        assert params.enabled is True
        assert params.base_oxygen == 10.0
        assert params.gradient_direction == 1.57
        assert params.gradient_strength == 0.1

    def test_aerotaxis_config_defaults(self):
        """Test AerotaxisConfig default values."""
        config = AerotaxisConfig()
        assert config.enabled is False
        assert config.base_oxygen == 10.0
        assert config.gradient_direction == 0.0
        assert config.comfort_lower == 5.0
        assert config.comfort_upper == 12.0
        assert config.lethal_hypoxia_upper == 2.0
        assert config.danger_hyperoxia_upper == 17.0


class TestApplySensingModeAerotaxis:
    """Test sensing mode translation for aerotaxis."""

    def test_apply_sensing_mode_aerotaxis(self):
        """Test that aerotaxis becomes aerotaxis_temporal when mode is temporal."""
        sensing = SensingConfig(aerotaxis_mode=SensingMode.TEMPORAL)
        modules = ["food_chemotaxis", "aerotaxis"]
        result = apply_sensing_mode(modules, sensing)
        assert "aerotaxis_temporal" in result
        assert "aerotaxis" not in result
        assert "food_chemotaxis" in result


class TestValidateSensingConfigAerotaxis:
    """Test validation logic for aerotaxis sensing config."""

    def test_validate_sensing_config_aerotaxis_derivative(self):
        """Test that aerotaxis derivative mode auto-enables STAM."""
        config = SensingConfig(aerotaxis_mode=SensingMode.DERIVATIVE)
        validated = validate_sensing_config(config)
        assert validated.stam_enabled is True


class TestTransgenerationalConfig:
    """Validate ``TransgenerationalConfig`` + ``LawnScheduleEntry`` schema rules."""

    def _make_schedule(self, generations: int) -> list[LawnScheduleEntry]:
        return [
            LawnScheduleEntry(
                generation=g,
                pathogen_lawns_enabled=(g == 0),
                ppo_train_episodes=10 if g == 0 else 0,
            )
            for g in range(generations)
        ]

    def test_defaults_are_sensible(self) -> None:
        """``decay_factor`` defaults to 0.6 and ``extraction_seed`` to 424242."""
        cfg = TransgenerationalConfig(
            enabled=True,
            lawn_schedule=self._make_schedule(2),
        )
        assert cfg.enabled is True
        assert cfg.decay_factor == pytest.approx(0.6)
        assert cfg.extraction_seed == 424242
        assert len(cfg.lawn_schedule) == 2

    def test_decay_factor_out_of_range_rejected(self) -> None:
        """``decay_factor`` MUST be in ``[0.0, 1.0]``."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TransgenerationalConfig(
                enabled=True,
                decay_factor=1.5,
                lawn_schedule=self._make_schedule(1),
            )
        with pytest.raises(ValidationError):
            TransgenerationalConfig(
                enabled=True,
                decay_factor=-0.1,
                lawn_schedule=self._make_schedule(1),
            )

    def test_duplicate_generation_in_schedule_rejected(self) -> None:
        """Two schedule entries with the same generation index SHALL be rejected."""
        with pytest.raises(ValueError, match="duplicate generation"):
            TransgenerationalConfig(
                enabled=True,
                lawn_schedule=[
                    LawnScheduleEntry(
                        generation=0,
                        pathogen_lawns_enabled=True,
                        ppo_train_episodes=10,
                    ),
                    LawnScheduleEntry(
                        generation=0,
                        pathogen_lawns_enabled=False,
                        ppo_train_episodes=0,
                    ),
                ],
            )

    def test_decay_shape_default_is_geometric(self) -> None:
        """``decay_shape`` defaults to ``"geometric"`` (M6 byte-equivalent)."""
        cfg = TransgenerationalConfig(
            enabled=True,
            lawn_schedule=self._make_schedule(1),
        )
        assert cfg.decay_shape == "geometric"

    def test_decay_shape_unknown_value_rejected(self) -> None:
        """``decay_shape`` outside the Literal SHALL be rejected at YAML load."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TransgenerationalConfig(
                enabled=True,
                decay_shape="exponential",  # type: ignore[arg-type]
                lawn_schedule=self._make_schedule(1),
            )

    def test_bias_network_defaults_when_block_absent(self) -> None:
        """When ``bias_network`` is absent the loader SHALL fall back to M6 legacy path."""
        cfg = TransgenerationalConfig(
            enabled=True,
            lawn_schedule=self._make_schedule(1),
        )
        assert cfg.bias_network is None

    def test_bias_network_defaults_match_plan_v2(self) -> None:
        """Empty ``bias_network`` sub-block populates the plan v2 defaults."""
        from quantumnematode.utils.config_loader import BiasNetworkConfig

        cfg = TransgenerationalConfig(
            enabled=True,
            lawn_schedule=self._make_schedule(1),
            bias_network=BiasNetworkConfig(),
        )
        assert cfg.bias_network is not None
        assert cfg.bias_network.hidden_dim == 8
        assert cfg.bias_network.activation == "tanh"
        assert cfg.bias_network.input_features == [
            "predator_gradient_strength",
            "predator_gradient_direction_sin",
            "food_gradient_strength",
        ]

    def test_bias_network_hidden_dim_zero_means_linear(self) -> None:
        """``hidden_dim: 0`` is accepted (linear projection only)."""
        from quantumnematode.utils.config_loader import BiasNetworkConfig

        cfg = TransgenerationalConfig(
            enabled=True,
            lawn_schedule=self._make_schedule(1),
            bias_network=BiasNetworkConfig(hidden_dim=0),
        )
        assert cfg.bias_network is not None
        assert cfg.bias_network.hidden_dim == 0

    def test_bias_network_input_features_rejects_unknown_field(self) -> None:
        """Unknown ``BrainParams`` field names in ``input_features`` SHALL be rejected."""
        from pydantic import ValidationError
        from quantumnematode.utils.config_loader import BiasNetworkConfig

        with pytest.raises(ValidationError, match="invalid entries"):
            BiasNetworkConfig(input_features=["nonexistent_field"])

    def test_bias_network_input_features_accepts_sin_cos_transform(self) -> None:
        """``_sin`` and ``_cos`` suffixed names resolve to known radian fields."""
        from quantumnematode.utils.config_loader import BiasNetworkConfig

        cfg = BiasNetworkConfig(
            input_features=[
                "predator_gradient_direction_sin",
                "food_gradient_direction_cos",
                "predator_gradient_strength",  # raw field also valid
            ],
        )
        assert cfg.input_features == [
            "predator_gradient_direction_sin",
            "food_gradient_direction_cos",
            "predator_gradient_strength",
        ]

    def test_bias_network_input_features_rejects_unknown_stem(self) -> None:
        """``_sin`` / ``_cos`` suffix on an unknown stem SHALL be rejected."""
        from pydantic import ValidationError
        from quantumnematode.utils.config_loader import BiasNetworkConfig

        with pytest.raises(ValidationError, match="unknown stem"):
            BiasNetworkConfig(input_features=["nonexistent_field_sin"])

    def test_bias_network_input_features_rejects_sin_cos_on_nonradian_known_field(
        self,
    ) -> None:
        """``_sin`` / ``_cos`` on a KNOWN but NON-radian stem SHALL be rejected.

        ``predator_gradient_strength`` is a known ``BrainParams`` field but
        its value is a unit-scaled magnitude (not a radian angle), so
        applying ``math.sin`` to it is nonsense. The validator MUST catch
        this so a typo or copy-paste error in a YAML doesn't silently
        produce a substrate that reads sin(magnitude).
        """
        from pydantic import ValidationError
        from quantumnematode.utils.config_loader import BiasNetworkConfig

        with pytest.raises(ValidationError, match="not radian-valued"):
            BiasNetworkConfig(input_features=["predator_gradient_strength_sin"])
        with pytest.raises(ValidationError, match="not radian-valued"):
            BiasNetworkConfig(input_features=["food_gradient_strength_cos"])

    def test_bias_network_input_features_must_be_non_empty(self) -> None:
        """Empty ``input_features`` SHALL be rejected."""
        from pydantic import ValidationError
        from quantumnematode.utils.config_loader import BiasNetworkConfig

        with pytest.raises(ValidationError, match="at least one feature name"):
            BiasNetworkConfig(input_features=[])

    def test_probe_ring_defaults_when_block_absent(self) -> None:
        """When ``probe_ring`` is absent the loader SHALL leave it ``None``."""
        cfg = TransgenerationalConfig(
            enabled=True,
            lawn_schedule=self._make_schedule(1),
        )
        assert cfg.probe_ring is None

    def test_probe_ring_defaults_match_plan_v2(self) -> None:
        """Empty ``probe_ring`` sub-block populates the plan v2 defaults."""
        from quantumnematode.utils.config_loader import ProbeRingConfig

        cfg = TransgenerationalConfig(
            enabled=True,
            lawn_schedule=self._make_schedule(1),
            probe_ring=ProbeRingConfig(),
        )
        assert cfg.probe_ring is not None
        assert cfg.probe_ring.count == 8
        assert cfg.probe_ring.radius_offset == 1
        assert cfg.probe_ring.include_food_gradient_variants is False

    def test_probe_ring_count_must_be_at_least_1(self) -> None:
        """``probe_ring.count: 0`` SHALL be rejected (would produce zero probes)."""
        from pydantic import ValidationError
        from quantumnematode.utils.config_loader import ProbeRingConfig

        with pytest.raises(ValidationError):
            ProbeRingConfig(count=0)


class TestEvolutionConfigTransgenerationalPairing:
    """Validate the inheritance ↔ transgenerational pairing contract."""

    def _make_schedule(self, generations: int) -> list[LawnScheduleEntry]:
        return [
            LawnScheduleEntry(
                generation=g,
                pathogen_lawns_enabled=(g == 0),
                ppo_train_episodes=10 if g == 0 else 0,
            )
            for g in range(generations)
        ]

    def test_tei_enabled_requires_inheritance_transgenerational(self) -> None:
        """``transgenerational.enabled=True`` SHALL require ``inheritance=transgenerational``."""
        with pytest.raises(ValueError, match=r"transgenerational\.enabled=True requires"):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=10,
                inheritance="lamarckian",
                transgenerational=TransgenerationalConfig(
                    enabled=True,
                    lawn_schedule=self._make_schedule(2),
                ),
            )

    def test_tei_disabled_requires_inheritance_none(self) -> None:
        """``transgenerational.enabled=False`` SHALL require ``inheritance=none``."""
        with pytest.raises(ValueError, match=r"transgenerational\.enabled=False requires"):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=10,
                inheritance="lamarckian",
                transgenerational=TransgenerationalConfig(
                    enabled=False,
                    lawn_schedule=self._make_schedule(2),
                ),
            )

    def test_inheritance_transgenerational_requires_config_block(self) -> None:
        """``inheritance=transgenerational`` SHALL require a non-None block."""
        with pytest.raises(
            ValueError,
            match="transgenerational config block is missing",
        ):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=10,
                inheritance="transgenerational",
                transgenerational=None,
            )

    def test_lawn_schedule_generation_out_of_range_rejected(self) -> None:
        """A schedule entry with ``generation >= evolution.generations`` SHALL be rejected."""
        with pytest.raises(
            ValueError,
            match=r"out-of-range generations \[3\]",
        ):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=10,
                inheritance="transgenerational",
                transgenerational=TransgenerationalConfig(
                    enabled=True,
                    lawn_schedule=[
                        LawnScheduleEntry(
                            generation=0,
                            pathogen_lawns_enabled=True,
                            ppo_train_episodes=10,
                        ),
                        LawnScheduleEntry(
                            generation=3,
                            pathogen_lawns_enabled=False,
                            ppo_train_episodes=0,
                        ),
                    ],
                ),
            )

    def test_lawn_schedule_missing_generations_rejected(self) -> None:
        """A schedule that doesn't cover ``[0, generations)`` SHALL be rejected."""
        with pytest.raises(
            ValueError,
            match=r"missing generations \[1, 2\]",
        ):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=3,
                episodes_per_eval=1,
                learn_episodes_per_eval=10,
                inheritance="transgenerational",
                transgenerational=TransgenerationalConfig(
                    enabled=True,
                    lawn_schedule=[
                        LawnScheduleEntry(
                            generation=0,
                            pathogen_lawns_enabled=True,
                            ppo_train_episodes=10,
                        ),
                    ],
                ),
            )

    def test_gen0_zero_train_episodes_rejected_when_enabled(self) -> None:
        """Under enabled TEI, a gen-0 entry with ``ppo_train_episodes=0`` SHALL be rejected.

        F0 has no substrate to inherit from (F0 IS the substrate source),
        so its train phase MUST run for any subsequent extraction to have
        meaningful biases. The fitness function would otherwise reject
        this at worker entry, after pool startup.
        """
        with pytest.raises(
            ValueError,
            match=r"generation=0 has ppo_train_episodes=0",
        ):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=10,
                inheritance="transgenerational",
                transgenerational=TransgenerationalConfig(
                    enabled=True,
                    lawn_schedule=[
                        LawnScheduleEntry(
                            generation=0,
                            pathogen_lawns_enabled=True,
                            ppo_train_episodes=0,
                        ),
                        LawnScheduleEntry(
                            generation=1,
                            pathogen_lawns_enabled=False,
                            ppo_train_episodes=0,
                        ),
                    ],
                ),
            )

    def test_gen0_zero_train_episodes_allowed_when_disabled(self) -> None:
        """Under TEI-off the gen-0 train-phase check is bypassed.

        With ``enabled=False`` the control arm doesn't run substrate
        extraction; its train-phase setting is governed by the outer
        ``learn_episodes_per_eval`` field alone.
        """
        cfg = EvolutionConfig(
            algorithm="cmaes",
            population_size=4,
            generations=2,
            episodes_per_eval=1,
            learn_episodes_per_eval=10,
            inheritance="none",
            transgenerational=TransgenerationalConfig(
                enabled=False,
                lawn_schedule=[
                    LawnScheduleEntry(
                        generation=0,
                        pathogen_lawns_enabled=True,
                        ppo_train_episodes=0,
                    ),
                    LawnScheduleEntry(
                        generation=1,
                        pathogen_lawns_enabled=False,
                        ppo_train_episodes=0,
                    ),
                ],
            ),
        )
        assert cfg.transgenerational is not None
        assert cfg.transgenerational.enabled is False

    def test_well_formed_tei_config_accepted(self) -> None:
        """A schedule covering every gen in [0, generations) SHALL load cleanly."""
        cfg = EvolutionConfig(
            algorithm="cmaes",
            population_size=4,
            generations=2,
            episodes_per_eval=1,
            learn_episodes_per_eval=10,
            inheritance="transgenerational",
            transgenerational=TransgenerationalConfig(
                enabled=True,
                lawn_schedule=self._make_schedule(2),
            ),
        )
        assert cfg.transgenerational is not None
        assert cfg.transgenerational.enabled is True


class TestEvolutionConfigComposedInheritancePairing:
    """Validate the M6.13 ``weights+transgenerational`` composed-mode pairing rules.

    Cross-product matrix: every (inheritance, transgenerational.enabled)
    cell SHOULD be either explicitly accepted or explicitly rejected. The
    PR-A archived M6.9+ tests cover the four pre-M6.13 cells; this class
    covers the new composed cell plus the previously-tested cells that
    interact with the widened pairing rules.
    """

    def _composed_schedule(self, generations: int, f1_k: int = 100) -> list[LawnScheduleEntry]:
        """Build a lawn_schedule where F0 has K>0 (substrate extraction needs trained
        elite) AND every F1+ entry has ``ppo_train_episodes=f1_k`` (composed mode
        requires retraining).
        """  # noqa: D205
        return [
            LawnScheduleEntry(
                generation=g,
                pathogen_lawns_enabled=(g == 0),
                ppo_train_episodes=100 if g == 0 else f1_k,
            )
            for g in range(generations)
        ]

    def test_composed_mode_with_tei_enabled_accepted(self) -> None:
        """``inheritance=weights+transgenerational`` + ``enabled=True`` + F1+ K>0 SHALL load."""
        cfg = EvolutionConfig(
            algorithm="cmaes",
            population_size=4,
            generations=2,
            episodes_per_eval=1,
            learn_episodes_per_eval=100,
            inheritance="weights+transgenerational",
            transgenerational=TransgenerationalConfig(
                enabled=True,
                lawn_schedule=self._composed_schedule(2, f1_k=100),
            ),
        )
        assert cfg.inheritance == "weights+transgenerational"
        assert cfg.transgenerational is not None
        assert cfg.transgenerational.enabled is True

    def test_composed_mode_with_tei_disabled_rejected(self) -> None:
        """``inheritance=weights+transgenerational`` + ``enabled=False`` SHALL raise.

        Composed mode REQUIRES the substrate to be active; otherwise it
        collapses to pure Lamarckian and the user should set
        ``inheritance: lamarckian`` for that case.
        """
        with pytest.raises(
            ValueError,
            match=r"transgenerational\.enabled=False requires",
        ):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=100,
                inheritance="weights+transgenerational",
                transgenerational=TransgenerationalConfig(
                    enabled=False,
                    lawn_schedule=self._composed_schedule(2),
                ),
            )

    def test_composed_mode_without_transgenerational_block_rejected(self) -> None:
        """``inheritance=weights+transgenerational`` + ``transgenerational=None`` SHALL raise."""
        with pytest.raises(
            ValueError,
            match="transgenerational config block is missing",
        ):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=100,
                inheritance="weights+transgenerational",
                transgenerational=None,
            )

    def test_composed_mode_with_f1_k_zero_rejected(self) -> None:
        """Composed mode SHALL reject any F1+ ``ppo_train_episodes=0`` entry.

        Pure-TEI (M6.9+) uses K=0 at F1+ to test the floor; composed mode
        is the opposite — the substrate prior acts ON the training
        distribution, not in place of it. K=0 at F1+ would silently
        collapse composed mode to pure-TEI.
        """
        with pytest.raises(
            ValueError,
            match=r"F1\+ entries with ppo_train_episodes=0",
        ):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=100,
                inheritance="weights+transgenerational",
                transgenerational=TransgenerationalConfig(
                    enabled=True,
                    lawn_schedule=self._composed_schedule(2, f1_k=0),
                ),
            )

    def test_composed_mode_with_elite_count_two_rejected(self) -> None:
        """``inheritance=weights+transgenerational`` + ``elite_count=2`` SHALL raise.

        Same single-elite-broadcast contract as Lamarckian.
        """
        with pytest.raises(
            ValueError,
            match=r"inheritance_elite_count MUST be 1",
        ):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=100,
                inheritance="weights+transgenerational",
                inheritance_elite_count=2,
                transgenerational=TransgenerationalConfig(
                    enabled=True,
                    lawn_schedule=self._composed_schedule(2),
                ),
            )

    def test_pure_tei_mode_with_f1_k_zero_still_accepted(self) -> None:
        """The composed-mode F1+ K>0 rule SHALL NOT apply to pure-TEI ``transgenerational``.

        Pure-TEI is M6.9+'s K=0 floor test; the M6.13 K>0 sub-rule fires
        only when ``inheritance == "weights+transgenerational"``.
        """
        cfg = EvolutionConfig(
            algorithm="cmaes",
            population_size=4,
            generations=2,
            episodes_per_eval=1,
            learn_episodes_per_eval=100,
            inheritance="transgenerational",
            transgenerational=TransgenerationalConfig(
                enabled=True,
                lawn_schedule=[
                    LawnScheduleEntry(
                        generation=0,
                        pathogen_lawns_enabled=True,
                        ppo_train_episodes=100,
                    ),
                    LawnScheduleEntry(
                        generation=1,
                        pathogen_lawns_enabled=False,
                        ppo_train_episodes=0,
                    ),
                ],
            ),
        )
        # Pure-TEI accepts F1+ K=0 — that's the floor-test config.
        assert cfg.inheritance == "transgenerational"

    def test_lamarckian_with_substrate_enabled_rejected(self) -> None:
        """``inheritance=lamarckian`` + ``transgenerational.enabled=True`` SHALL raise.

        The M6.13 widening accepts ``weights+transgenerational`` as a
        substrate-enabled pairing — but it MUST NOT accept plain
        Lamarckian with a substrate block. Users who want both inheritance
        types must use the new composed value.
        """
        with pytest.raises(
            ValueError,
            match=r"transgenerational\.enabled=True requires",
        ):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=100,
                inheritance="lamarckian",
                transgenerational=TransgenerationalConfig(
                    enabled=True,
                    lawn_schedule=self._composed_schedule(2),
                ),
            )

    def test_baldwin_with_substrate_enabled_rejected(self) -> None:
        """``inheritance=baldwin`` + ``transgenerational.enabled=True`` SHALL raise."""
        with pytest.raises(
            ValueError,
            match=r"transgenerational\.enabled=True requires",
        ):
            EvolutionConfig(
                algorithm="cmaes",
                population_size=4,
                generations=2,
                episodes_per_eval=1,
                learn_episodes_per_eval=100,
                inheritance="baldwin",
                transgenerational=TransgenerationalConfig(
                    enabled=True,
                    lawn_schedule=self._composed_schedule(2),
                ),
            )
