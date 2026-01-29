"""Tests for the reward calculation system."""

from unittest.mock import Mock

import pytest
from quantumnematode.agent import RewardConfig
from quantumnematode.agent.reward_calculator import RewardCalculator
from quantumnematode.env import DynamicForagingEnvironment


@pytest.fixture
def default_config():
    """Create a default reward config for testing."""
    return RewardConfig(
        reward_distance_scale=0.1,
        penalty_step=0.01,
        penalty_anti_dithering=0.05,
        penalty_stuck_position=0.02,
        stuck_position_threshold=3,
        reward_exploration=0.02,
    )


class TestRewardCalculatorInitialization:
    """Test reward calculator initialization."""

    def test_initialize_with_config(self, default_config):
        """Test that calculator initializes with config."""
        calculator = RewardCalculator(default_config)
        assert calculator.config is default_config


class TestDynamicForagingRewards:
    """Test reward calculation for dynamic foraging environments."""

    def test_foraging_distance_reward(self, default_config):
        """Test distance reward in foraging environment."""
        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [2, 2]
        env.foods = [(5, 5), (1, 1)]
        env.get_nearest_food_distance = Mock(return_value=1)
        env.visited_cells = set()
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False

        calculator = RewardCalculator(default_config)
        path = [(3, 3), (2, 2)]  # Previous nearest was Manhattan=4, now=1

        reward = calculator.calculate_reward(env, path)

        # Distance reward: 0.1 * (4 - 1) = 0.3
        # Exploration bonus: 0.02 (new cell)
        # Step penalty: -0.01
        assert reward == pytest.approx(0.31)

    def test_foraging_exploration_bonus(self, default_config):
        """Test exploration bonus for visiting new cells."""
        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [1, 2]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = set()
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False

        calculator = RewardCalculator(default_config)
        path = [(1, 2)]

        reward = calculator.calculate_reward(env, path)

        # Exploration bonus: 0.02, Step penalty: -0.01
        assert reward == pytest.approx(0.01)
        assert (1, 2) in env.visited_cells

    def test_foraging_no_exploration_bonus_for_visited_cell(self, default_config):
        """Test no exploration bonus for already visited cells."""
        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [1, 2]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(1, 2)}
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False

        calculator = RewardCalculator(default_config)
        path = [(1, 2)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01
        assert reward == pytest.approx(-0.01)


class TestAntiDitheringPenalty:
    """Test anti-dithering penalty."""

    def test_anti_dithering_penalty_applied(self, default_config):
        """Test penalty when agent oscillates back to previous position."""
        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = (1, 1)  # Use tuple to match path format
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(1, 1)}  # Mark as visited to avoid exploration bonus
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False

        calculator = RewardCalculator(default_config)
        path = [(1, 1), (2, 2), (1, 1)]  # Back to same position

        reward = calculator.calculate_reward(env, path)

        # Anti-dithering penalty: -0.05
        # Step penalty: -0.01
        assert reward == pytest.approx(-0.06)

    def test_no_anti_dithering_penalty_normal_movement(self, default_config):
        """Test no penalty for normal movement."""
        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [3, 3]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(3, 3)}  # Mark as visited to avoid exploration bonus
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False

        calculator = RewardCalculator(default_config)
        path = [(1, 1), (2, 2), (3, 3)]  # Normal progression

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01 (no distance reward since no foods)
        assert reward == pytest.approx(-0.01)


class TestPredatorProximityPenalty:
    """Test predator proximity penalty."""

    def test_predator_proximity_penalty_applied(self):
        """Test penalty when agent is within predator detection radius."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.1,
        )

        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [5, 5]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(5, 5)}  # Mark as visited to avoid exploration bonus
        env.predator = Mock()
        env.predator.enabled = True
        env.is_agent_in_danger = Mock(return_value=True)  # Within detection radius
        env.wall_collision_occurred = False

        calculator = RewardCalculator(config)
        path = [(5, 5)]

        reward = calculator.calculate_reward(env, path)

        # Proximity penalty: -0.1, Step penalty: -0.01
        assert reward == pytest.approx(-0.11)

    def test_no_predator_proximity_penalty_when_safe(self):
        """Test no penalty when agent is outside predator detection radius."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.1,
        )

        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [5, 5]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(5, 5)}  # Mark as visited to avoid exploration bonus
        env.predator = Mock()
        env.predator.enabled = True
        env.is_agent_in_danger = Mock(return_value=False)  # Outside detection radius
        env.wall_collision_occurred = False

        calculator = RewardCalculator(config)
        path = [(5, 5)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01
        assert reward == pytest.approx(-0.01)

    def test_no_predator_proximity_penalty_when_disabled(self):
        """Test no penalty when predators are disabled."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.1,
        )

        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [5, 5]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(5, 5)}  # Mark as visited to avoid exploration bonus
        env.predator = Mock()
        env.predator.enabled = False  # Predators disabled
        env.wall_collision_occurred = False

        calculator = RewardCalculator(config)
        path = [(5, 5)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01
        assert reward == pytest.approx(-0.01)

    def test_predator_proximity_penalty_zero_disabled(self):
        """Test that zero proximity penalty effectively disables the feature."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.0,  # Disabled
        )

        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [5, 5]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(5, 5)}  # Mark as visited to avoid exploration bonus
        env.predator = Mock()
        env.predator.enabled = True
        env.is_agent_in_danger = Mock(return_value=True)
        env.wall_collision_occurred = False

        calculator = RewardCalculator(config)
        path = [(5, 5)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01 (no proximity penalty)
        assert reward == pytest.approx(-0.01)


class TestStuckPositionPenalty:
    """Test stuck position penalty."""

    def test_stuck_position_penalty_applied(self, default_config):
        """Test penalty when agent is stuck."""
        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [2, 2]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(2, 2)}  # Mark as visited to avoid exploration bonus
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False

        calculator = RewardCalculator(default_config)
        path = [(2, 2), (2, 2)]

        reward = calculator.calculate_reward(env, path, stuck_position_count=5)

        # Stuck penalty: 0.02 * min(5-3, 10) = 0.02 * 2 = 0.04
        # Step penalty: 0.01
        assert reward == pytest.approx(-0.05)

    def test_no_stuck_penalty_below_threshold(self, default_config):
        """Test no penalty when below stuck threshold."""
        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [2, 2]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(2, 2)}  # Mark as visited to avoid exploration bonus
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False

        calculator = RewardCalculator(default_config)
        path = [(2, 2), (2, 2)]

        reward = calculator.calculate_reward(env, path, stuck_position_count=2)

        # Only step penalty (2 < threshold of 3)
        assert reward == pytest.approx(-0.01)

    def test_stuck_penalty_capped_at_10(self, default_config):
        """Test stuck penalty is capped at 10 extra steps."""
        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [2, 2]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(2, 2)}  # Mark as visited to avoid exploration bonus
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False

        calculator = RewardCalculator(default_config)
        path = [(2, 2), (2, 2)]

        reward = calculator.calculate_reward(env, path, stuck_position_count=20)

        # Stuck penalty: 0.02 * min(20-3, 10) = 0.02 * 10 = 0.20
        # Step penalty: 0.01
        assert reward == pytest.approx(-0.21)


class TestBoundaryCollisionPenalty:
    """Test boundary collision penalty.

    Note: Boundary penalty is collision-based, not position-based.
    Penalty applies when agent tries to move into a wall, not when at an edge cell.
    This allows agents to collect food at edges without being penalized.
    """

    def test_boundary_collision_penalty_applied(self):
        """Test penalty when agent collided with wall (tried to move into boundary)."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_boundary_collision=0.02,
        )

        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [0, 5]  # At left edge after failed move
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(0, 5)}  # Mark as visited
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = True  # Tried to move into wall

        calculator = RewardCalculator(config)
        path = [(0, 5)]

        reward = calculator.calculate_reward(env, path)

        # Boundary penalty: -0.02, Step penalty: -0.01
        assert reward == pytest.approx(-0.03)

    def test_no_boundary_collision_penalty_when_no_collision(self):
        """Test no penalty when agent didn't try to move into a wall."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_boundary_collision=0.02,
        )

        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [5, 5]  # In middle of grid
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(5, 5)}  # Mark as visited
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False  # No wall collision

        calculator = RewardCalculator(config)
        path = [(5, 5)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01
        assert reward == pytest.approx(-0.01)

    def test_no_penalty_at_edge_without_collision(self):
        """Test no penalty when agent is at edge but didn't try to move into wall.

        This is the key difference from position-based penalty: agent can be at
        edge cell (e.g., collecting food) without being penalized.
        """
        config = RewardConfig(
            penalty_step=0.01,
            penalty_boundary_collision=0.02,
        )

        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [0, 5]  # At left edge
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(0, 5)}  # Mark as visited
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False  # At edge but no collision attempt

        calculator = RewardCalculator(config)
        path = [(0, 5)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01 (no boundary penalty despite being at edge)
        assert reward == pytest.approx(-0.01)

    def test_boundary_collision_penalty_zero_disabled(self):
        """Test that zero boundary penalty effectively disables the feature."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_boundary_collision=0.0,  # Disabled
        )

        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [0, 5]  # At left edge
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(0, 5)}  # Mark as visited
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = True  # Collision occurred but penalty is 0

        calculator = RewardCalculator(config)
        path = [(0, 5)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01 (no boundary penalty)
        assert reward == pytest.approx(-0.01)
