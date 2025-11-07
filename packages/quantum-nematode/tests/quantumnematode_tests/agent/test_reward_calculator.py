"""Tests for the reward calculation system."""

from unittest.mock import Mock

import pytest
from quantumnematode.agent import RewardConfig
from quantumnematode.env import DynamicForagingEnvironment, MazeEnvironment
from quantumnematode.agent.reward_calculator import RewardCalculator


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


class TestMazeEnvironmentRewards:
    """Test reward calculation for maze environments."""

    def test_maze_distance_reward_moving_closer(self, default_config):
        """Test reward when agent moves closer to goal."""
        env = Mock(spec=MazeEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [2, 3]
        env.goal = [5, 5]
        env.reached_goal.return_value = False

        calculator = RewardCalculator(default_config)
        path = [(3, 3), (2, 3)]  # Moved from (3,3) to (2,3)

        reward = calculator.calculate_reward(env, path)

        # Prev distance: |3-5| + |3-5| = 4
        # Curr distance: |2-5| + |3-5| = 5
        # Distance reward: 0.1 * (4 - 5) = -0.1
        # Step penalty: -0.01
        assert reward == pytest.approx(-0.11)

    def test_maze_distance_reward_first_step(self, default_config):
        """Test reward on first step (no previous position)."""
        env = Mock(spec=MazeEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [1, 1]
        env.goal = [5, 5]
        env.reached_goal.return_value = False

        calculator = RewardCalculator(default_config)
        path = [(1, 1)]  # First step

        reward = calculator.calculate_reward(env, path)

        # No distance reward on first step, only step penalty
        assert reward == pytest.approx(-0.01)


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

        calculator = RewardCalculator(default_config)
        path = [(1, 2)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01
        assert reward == pytest.approx(-0.01)


class TestAntiDitheringPenalty:
    """Test anti-dithering penalty."""

    def test_anti_dithering_penalty_applied(self, default_config):
        """Test penalty when agent oscillates back to previous position."""
        env = Mock(spec=MazeEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [1, 1]
        env.goal = [5, 5]

        calculator = RewardCalculator(default_config)
        path = [(1, 1), (2, 2), [1, 1]]  # Back to same position

        reward = calculator.calculate_reward(env, path)

        # Prev distance: |2-5| + |2-5| = 6
        # Curr distance: |1-5| + |1-5| = 8
        # Distance reward: 0.1 * (6 - 8) = -0.2
        # Anti-dithering penalty: -0.05
        # Step penalty: -0.01
        assert reward == pytest.approx(-0.21)

    def test_no_anti_dithering_penalty_normal_movement(self, default_config):
        """Test no penalty for normal movement."""
        env = Mock(spec=MazeEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [3, 3]
        env.goal = [5, 5]

        calculator = RewardCalculator(default_config)
        path = [(1, 1), (2, 2), (3, 3)]  # Normal progression

        reward = calculator.calculate_reward(env, path)

        # Prev distance: |2-5| + |2-5| = 6
        # Curr distance: |3-5| + |3-5| = 4
        # Distance reward: 0.1 * (6 - 4) = 0.2
        # Step penalty: -0.01
        assert reward == pytest.approx(0.19)


class TestStuckPositionPenalty:
    """Test stuck position penalty."""

    def test_stuck_position_penalty_applied(self, default_config):
        """Test penalty when agent is stuck."""
        env = Mock(spec=MazeEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [2, 2]
        env.goal = [5, 5]

        calculator = RewardCalculator(default_config)
        path = [(2, 2), (2, 2)]

        reward = calculator.calculate_reward(env, path, stuck_position_count=5)

        # Stuck penalty: 0.02 * min(5-3, 10) = 0.02 * 2 = 0.04
        # Step penalty: 0.01
        assert reward == pytest.approx(-0.05)

    def test_no_stuck_penalty_below_threshold(self, default_config):
        """Test no penalty when below stuck threshold."""
        env = Mock(spec=MazeEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [2, 2]
        env.goal = [5, 5]

        calculator = RewardCalculator(default_config)
        path = [(2, 2), (2, 2)]

        reward = calculator.calculate_reward(env, path, stuck_position_count=2)

        # Only step penalty (2 < threshold of 3)
        assert reward == pytest.approx(-0.01)

    def test_stuck_penalty_capped_at_10(self, default_config):
        """Test stuck penalty is capped at 10 extra steps."""
        env = Mock(spec=MazeEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [2, 2]
        env.goal = [5, 5]

        calculator = RewardCalculator(default_config)
        path = [(2, 2), (2, 2)]

        reward = calculator.calculate_reward(env, path, stuck_position_count=20)

        # Stuck penalty: 0.02 * min(20-3, 10) = 0.02 * 10 = 0.20
        # Step penalty: 0.01
        assert reward == pytest.approx(-0.21)
