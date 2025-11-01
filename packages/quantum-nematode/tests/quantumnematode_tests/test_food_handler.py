"""Tests for the food consumption handler."""

# ruff: noqa: SLF001 - Testing private members is intentional

from unittest.mock import MagicMock

import pytest
from quantumnematode.agent import SatietyConfig
from quantumnematode.env import DynamicForagingEnvironment, MazeEnvironment
from quantumnematode.food_handler import FoodConsumptionHandler
from quantumnematode.satiety import SatietyManager


class TestFoodConsumptionHandlerInitialization:
    """Test food handler initialization."""

    def test_initialize_with_static_environment(self):
        """Test initialization with a static maze environment."""
        env = MazeEnvironment(grid_size=10)
        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)

        assert handler.env == env
        assert handler.satiety_manager == satiety_manager
        assert handler.satiety_gain_fraction == 0.2
        assert handler._initial_distance is None
        assert handler._steps_since_last_food == 0

    def test_initialize_with_dynamic_environment(self):
        """Test initialization with a dynamic foraging environment."""
        env = DynamicForagingEnvironment(grid_size=30, num_initial_foods=5)
        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)

        assert handler.env == env
        assert handler._initial_distance is not None
        assert handler._initial_distance >= 0

    def test_initialize_with_custom_satiety_gain(self):
        """Test initialization with custom satiety gain fraction."""
        env = MazeEnvironment(grid_size=10)
        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(
            env,
            satiety_manager,
            satiety_gain_fraction=0.5,
        )

        assert handler.satiety_gain_fraction == 0.5


class TestStepTracking:
    """Test step tracking for distance efficiency."""

    def test_track_step_increments_counter(self):
        """Test that track_step increments the step counter."""
        env = MazeEnvironment(grid_size=10)
        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)

        assert handler._steps_since_last_food == 0

        handler.track_step()
        assert handler._steps_since_last_food == 1

        handler.track_step()
        assert handler._steps_since_last_food == 2

    def test_track_multiple_steps(self):
        """Test tracking multiple steps."""
        env = MazeEnvironment(grid_size=10)
        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)

        for _ in range(10):
            handler.track_step()

        assert handler._steps_since_last_food == 10


class TestFoodConsumptionStaticEnvironment:
    """Test food consumption in static maze environments."""

    def test_no_food_at_position(self):
        """Test when agent is not at food position."""
        env = MagicMock(spec=MazeEnvironment)
        env.reached_goal.return_value = False
        satiety_manager = SatietyManager(SatietyConfig(initial_satiety=100.0))
        handler = FoodConsumptionHandler(env, satiety_manager)

        result = handler.check_and_consume_food()

        assert result.food_consumed is False
        assert result.satiety_restored == 0.0
        assert result.reward == 0.0
        assert result.distance_efficiency is None

    def test_consume_food_static_environment(self):
        """Test consuming food in static environment."""
        env = MagicMock(spec=MazeEnvironment)
        env.reached_goal.return_value = True
        satiety_manager = SatietyManager(SatietyConfig(initial_satiety=100.0))
        satiety_manager._current_satiety = 50.0
        handler = FoodConsumptionHandler(
            env,
            satiety_manager,
            satiety_gain_fraction=0.3,
        )

        result = handler.check_and_consume_food()

        assert result.food_consumed is True
        assert result.satiety_restored == pytest.approx(30.0)  # 100 * 0.3
        assert satiety_manager.current_satiety == pytest.approx(80.0)
        assert result.distance_efficiency is None  # No efficiency for static env


class TestFoodConsumptionDynamicEnvironment:
    """Test food consumption in dynamic foraging environments."""

    def test_consume_food_dynamic_environment(self):
        """Test consuming food in dynamic environment."""
        env = MagicMock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = True
        env.consume_food.return_value = True
        env.get_nearest_food_distance.return_value = 15

        satiety_manager = SatietyManager(SatietyConfig(initial_satiety=100.0))
        handler = FoodConsumptionHandler(env, satiety_manager)
        handler._initial_distance = 10
        handler._steps_since_last_food = 12

        result = handler.check_and_consume_food()

        assert result.food_consumed is True
        assert result.satiety_restored == pytest.approx(20.0)  # 100 * 0.2
        assert result.distance_efficiency == pytest.approx(10 / 12, rel=0.01)

    def test_consume_food_perfect_efficiency(self):
        """Test consuming food with perfect distance efficiency."""
        env = MagicMock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = True
        env.consume_food.return_value = True
        env.get_nearest_food_distance.return_value = 20

        satiety_manager = SatietyManager(SatietyConfig(initial_satiety=100.0))
        handler = FoodConsumptionHandler(env, satiety_manager)
        handler._initial_distance = 10
        handler._steps_since_last_food = 10

        result = handler.check_and_consume_food()

        assert result.distance_efficiency == pytest.approx(1.0)

    def test_consume_food_resets_step_counter(self):
        """Test that consuming food resets the step counter."""
        env = MagicMock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = True
        env.consume_food.return_value = True
        env.get_nearest_food_distance.return_value = 15

        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)
        handler._steps_since_last_food = 25

        handler.check_and_consume_food()

        assert handler._steps_since_last_food == 0

    def test_consume_food_updates_initial_distance(self):
        """Test that consuming food updates initial distance for next food."""
        env = MagicMock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = True
        env.consume_food.return_value = True
        env.get_nearest_food_distance.return_value = 25

        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)
        handler._initial_distance = 10

        handler.check_and_consume_food()

        assert handler._initial_distance == 25

    def test_no_food_consumed_in_dynamic_env(self):
        """Test when env.consume_food returns False."""
        env = MagicMock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = True
        env.consume_food.return_value = False

        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)

        result = handler.check_and_consume_food()

        assert result.food_consumed is False
        assert result.satiety_restored == 0.0


class TestDistanceEfficiencyCalculation:
    """Test distance efficiency calculation edge cases."""

    def test_efficiency_with_zero_steps(self):
        """Test efficiency when steps is zero (shouldn't happen but handle gracefully)."""
        env = MagicMock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = True
        env.consume_food.return_value = True
        env.get_nearest_food_distance.return_value = 10

        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)
        handler._initial_distance = 5
        handler._steps_since_last_food = 0

        result = handler.check_and_consume_food()

        assert result.distance_efficiency == 1.0

    def test_efficiency_with_none_initial_distance(self):
        """Test when initial distance is None."""
        env = MagicMock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = True
        env.consume_food.return_value = True
        env.get_nearest_food_distance.return_value = 10

        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)
        handler._initial_distance = None
        handler._steps_since_last_food = 10

        result = handler.check_and_consume_food()

        assert result.distance_efficiency is None

    def test_efficiency_with_zero_initial_distance(self):
        """Test when initial distance is zero."""
        env = MagicMock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = True
        env.consume_food.return_value = True
        env.get_nearest_food_distance.return_value = 10

        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)
        handler._initial_distance = 0
        handler._steps_since_last_food = 5

        result = handler.check_and_consume_food()

        assert result.distance_efficiency is None


class TestSatietyRestoration:
    """Test satiety restoration on food consumption."""

    def test_satiety_restored_to_max(self):
        """Test that satiety is clamped at max when restored."""
        env = MagicMock(spec=MazeEnvironment)
        env.reached_goal.return_value = True
        satiety_manager = SatietyManager(SatietyConfig(initial_satiety=100.0))
        satiety_manager._current_satiety = 95.0
        handler = FoodConsumptionHandler(
            env,
            satiety_manager,
            satiety_gain_fraction=0.2,
        )

        handler.check_and_consume_food()

        # Should be clamped at 100.0, not 115.0
        assert satiety_manager.current_satiety == 100.0

    def test_satiety_restored_from_low(self):
        """Test satiety restoration from very low level."""
        env = MagicMock(spec=MazeEnvironment)
        env.reached_goal.return_value = True
        satiety_manager = SatietyManager(SatietyConfig(initial_satiety=100.0))
        satiety_manager._current_satiety = 10.0
        handler = FoodConsumptionHandler(
            env,
            satiety_manager,
            satiety_gain_fraction=0.5,
        )

        result = handler.check_and_consume_food()

        assert result.satiety_restored == 50.0
        assert satiety_manager.current_satiety == 60.0


class TestReset:
    """Test handler reset functionality."""

    def test_reset_clears_step_counter(self):
        """Test that reset clears the step counter."""
        env = MazeEnvironment(grid_size=10)
        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)
        handler._steps_since_last_food = 42

        handler.reset()

        assert handler._steps_since_last_food == 0

    def test_reset_updates_initial_distance_dynamic(self):
        """Test that reset updates initial distance for dynamic environments."""
        env = DynamicForagingEnvironment(grid_size=30, num_initial_foods=5)
        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)

        handler._initial_distance = None
        handler.reset()

        # Should have updated from the environment
        assert handler._initial_distance is not None
        assert handler._initial_distance >= 0

    def test_reset_static_environment(self):
        """Test reset with static environment."""
        env = MazeEnvironment(grid_size=10)
        satiety_manager = SatietyManager(SatietyConfig())
        handler = FoodConsumptionHandler(env, satiety_manager)
        handler._steps_since_last_food = 10

        handler.reset()

        assert handler._steps_since_last_food == 0
        assert handler._initial_distance is None
