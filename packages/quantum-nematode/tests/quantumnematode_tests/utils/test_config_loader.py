"""Tests for configuration loading utilities."""

from quantumnematode.env import ForagingParams, HealthParams, PredatorParams, ThermotaxisParams
from quantumnematode.utils.config_loader import (
    DynamicEnvironmentConfig,
    ForagingConfig,
    HealthConfig,
    PredatorConfig,
    ThermotaxisConfig,
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


class TestDynamicEnvironmentConfig:
    """Test cases for DynamicEnvironmentConfig."""

    def test_default_values(self):
        """Test DynamicEnvironmentConfig default values."""
        config = DynamicEnvironmentConfig()
        assert config.grid_size == 50
        assert config.viewport_size == (11, 11)
        assert config.foraging is None
        assert config.predators is None
        assert config.health is None

    def test_with_nested_configs(self):
        """Test DynamicEnvironmentConfig with nested configurations."""
        config = DynamicEnvironmentConfig(
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
        config_with = DynamicEnvironmentConfig(
            foraging=ForagingConfig(foods_on_grid=25),
        )
        foraging = config_with.get_foraging_config()
        assert foraging.foods_on_grid == 25

        # Without explicit config (should return default)
        config_without = DynamicEnvironmentConfig()
        foraging_default = config_without.get_foraging_config()
        assert foraging_default.foods_on_grid == 10  # Default value

    def test_get_predator_config(self):
        """Test get_predator_config returns configured or default."""
        # With explicit config
        config_with = DynamicEnvironmentConfig(
            predators=PredatorConfig(enabled=True, count=5),
        )
        predator = config_with.get_predator_config()
        assert predator.enabled is True
        assert predator.count == 5

        # Without explicit config (should return default)
        config_without = DynamicEnvironmentConfig()
        predator_default = config_without.get_predator_config()
        assert predator_default.enabled is False  # Default value

    def test_get_health_config(self):
        """Test get_health_config returns configured or default."""
        # With explicit config
        config_with = DynamicEnvironmentConfig(
            health=HealthConfig(enabled=True, max_hp=150.0),
        )
        health = config_with.get_health_config()
        assert health.enabled is True
        assert health.max_hp == 150.0

        # Without explicit config (should return default)
        config_without = DynamicEnvironmentConfig()
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
        import pytest

        config = ThermotaxisConfig(
            enabled=True,
            hot_spots=[[75, 50]],  # Missing intensity
        )

        with pytest.raises(ValueError, match=r"Invalid hot_spot at index 0"):
            config.to_params()

    def test_to_params_validates_cold_spot_format(self):
        """Test that to_params() raises ValueError for malformed cold_spots."""
        import pytest

        config = ThermotaxisConfig(
            enabled=True,
            cold_spots=[[25, 25, 10.0, 5.0]],  # Extra element
        )

        with pytest.raises(ValueError, match=r"Invalid cold_spot at index 0"):
            config.to_params()
