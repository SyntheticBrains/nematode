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


class TestPredatorEvasionReward:
    """Test distance-scaled predator evasion reward.

    The evasion reward mirrors food distance reward structure:
    - Moving AWAY from nearest predator: positive reward
    - Moving TOWARD nearest predator: negative penalty
    - Scale factor is penalty_predator_proximity (same config field as old flat penalty)
    """

    def _make_predator_env(self, agent_pos, predator_positions, *, in_danger=True):
        """Create a mock env with predators at given positions."""
        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = agent_pos
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {tuple(agent_pos)}
        env.predator = Mock()
        env.predator.enabled = True
        env.is_agent_in_danger = Mock(return_value=in_danger)
        env.wall_collision_occurred = False

        # Create predator mocks
        predators = []
        for pos in predator_positions:
            pred = Mock()
            pred.position = pos
            predators.append(pred)
        env.predators = predators

        # get_nearest_predator_distance returns Manhattan distance
        if predator_positions:
            env.get_nearest_predator_distance = Mock(
                return_value=min(
                    abs(agent_pos[0] - p[0]) + abs(agent_pos[1] - p[1]) for p in predator_positions
                ),
            )
        else:
            env.get_nearest_predator_distance = Mock(return_value=None)

        return env

    def test_evasion_reward_moving_away(self):
        """Moving away from predator while in danger zone gives positive reward."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
        )
        # Agent was at (5,5), now at (6,5). Predator at (3,5).
        # prev_dist = |5-3| + |5-5| = 2, curr_dist = |6-3| + |5-5| = 3
        # evasion_reward = 0.5 * (3 - 2) = +0.5
        env = self._make_predator_env([6, 5], [(3, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 5), (6, 5)]

        reward = calculator.calculate_reward(env, path)

        # Evasion reward: +0.5, Step penalty: -0.01
        assert reward == pytest.approx(0.49)

    def test_evasion_penalty_moving_closer(self):
        """Moving toward predator while in danger zone gives negative penalty."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
        )
        # Agent was at (6,5), now at (5,5). Predator at (3,5).
        # prev_dist = |6-3| + |5-5| = 3, curr_dist = |5-3| + |5-5| = 2
        # evasion_reward = 0.5 * (2 - 3) = -0.5
        env = self._make_predator_env([5, 5], [(3, 5)])
        calculator = RewardCalculator(config)
        path = [(6, 5), (5, 5)]

        reward = calculator.calculate_reward(env, path)

        # Evasion penalty: -0.5, Step penalty: -0.01
        assert reward == pytest.approx(-0.51)

    def test_evasion_no_change_perpendicular_far(self):
        """Moving perpendicular (same distance, beyond contact) gives zero evasion reward."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
        )
        # Agent was at (5,2), now at (5,8). Predator at (2,5).
        # prev_dist = |5-2| + |2-5| = 6, curr_dist = |5-2| + |8-5| = 6
        # evasion_reward = 0.5 * (6 - 6) = 0.0, no contact penalty (dist > 1)
        env = self._make_predator_env([5, 8], [(2, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 2), (5, 8)]

        reward = calculator.calculate_reward(env, path)

        # Zero evasion reward, Step penalty: -0.01
        assert reward == pytest.approx(-0.01)

    def test_evasion_contact_penalty_at_distance_zero(self):
        """Predator on same cell applies contact penalty even with no distance change."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
        )
        # Agent was at (5,5), now at (5,5). Predator at (5,5). dist=0 both times.
        # evasion_reward = 0.5 * (0 - 0) - 0.5 = -0.5
        env = self._make_predator_env([5, 5], [(5, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 5), (5, 5)]

        reward = calculator.calculate_reward(env, path)

        # Contact penalty: -0.5, Step penalty: -0.01
        assert reward == pytest.approx(-0.51)

    def test_evasion_contact_penalty_at_distance_one(self):
        """Predator adjacent applies contact penalty even with no distance change."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
        )
        # Agent was at (5,4), now at (5,6). Predator at (5,5).
        # prev_dist = 1, curr_dist = 1. evasion = 0.5*(1-1) - 0.5 = -0.5
        env = self._make_predator_env([5, 6], [(5, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 4), (5, 6)]

        reward = calculator.calculate_reward(env, path)

        # Contact penalty: -0.5, Step penalty: -0.01
        assert reward == pytest.approx(-0.51)

    def test_evasion_outside_detection_not_applied(self):
        """No evasion reward when agent is outside detection radius."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
        )
        env = self._make_predator_env([50, 50], [(10, 10)], in_danger=False)
        calculator = RewardCalculator(config)
        path = [(49, 50), (50, 50)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01
        assert reward == pytest.approx(-0.01)

    def test_evasion_first_step_flat_fallback(self):
        """First step in episode uses flat penalty fallback."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
        )
        env = self._make_predator_env([5, 5], [(3, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 5)]  # Only 1 position — no previous to compare

        reward = calculator.calculate_reward(env, path)

        # Flat fallback penalty: -0.5, Step penalty: -0.01
        assert reward == pytest.approx(-0.51)

    def test_no_evasion_when_predators_disabled(self):
        """No evasion reward when predators are disabled."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
        )

        env = Mock(spec=DynamicForagingEnvironment)
        env.reached_goal.return_value = False
        env.agent_pos = [5, 5]
        env.get_nearest_food_distance = Mock(return_value=None)
        env.visited_cells = {(5, 5)}
        env.predator = Mock()
        env.predator.enabled = False
        env.wall_collision_occurred = False

        calculator = RewardCalculator(config)
        path = [(4, 5), (5, 5)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01
        assert reward == pytest.approx(-0.01)

    def test_evasion_zero_scale_effectively_disabled(self):
        """Zero penalty_predator_proximity effectively disables evasion reward."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.0,
        )
        env = self._make_predator_env([5, 5], [(3, 5)])
        calculator = RewardCalculator(config)
        path = [(6, 5), (5, 5)]  # Moving toward predator

        reward = calculator.calculate_reward(env, path)

        # Zero evasion (0.0 * delta = 0.0), Step penalty: -0.01
        assert reward == pytest.approx(-0.01)

    def test_evasion_multiple_predators_uses_nearest(self):
        """With multiple predators, evasion uses nearest predator distance."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
        )
        # Agent was at (5,5), now at (6,5). Predators at (3,5) and (50,50).
        # Nearest: (3,5). prev_dist=2, curr_dist=3
        # evasion_reward = 0.5 * (3 - 2) = +0.5
        env = self._make_predator_env([6, 5], [(3, 5), (50, 50)])
        calculator = RewardCalculator(config)
        path = [(5, 5), (6, 5)]

        reward = calculator.calculate_reward(env, path)

        # Evasion reward: +0.5, Step penalty: -0.01
        assert reward == pytest.approx(0.49)


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
