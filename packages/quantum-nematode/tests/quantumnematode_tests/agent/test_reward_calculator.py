"""Tests for the reward calculation system."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from quantumnematode.agent import RewardConfig
from quantumnematode.agent.reward_calculator import RewardCalculator
from quantumnematode.env import DynamicForagingEnvironment
from quantumnematode.env.temperature import TemperatureZone

if TYPE_CHECKING:
    from collections.abc import Sequence


def _make_agent_state(
    position: Sequence[int],
    visited_cells: set[tuple[int, ...]] | None = None,
    *,
    wall_collision: bool = False,
) -> Mock:
    """Create a mock AgentState with position and visited_cells."""
    state = Mock()
    state.position = tuple(position)
    state.visited_cells = visited_cells if visited_cells is not None else set()
    state.wall_collision_occurred = wall_collision
    return state


def _make_base_env(
    agent_pos: Sequence[int],
    *,
    visited_cells: set[tuple[int, ...]] | None = None,
    wall_collision: bool = False,
    agent_id: str = "default",
) -> Mock:
    """Create a base mock env for reward calculator tests."""
    env = Mock(spec=DynamicForagingEnvironment)
    state = _make_agent_state(agent_pos, visited_cells, wall_collision=wall_collision)
    env.agents = {agent_id: state}
    env.reached_goal_for = Mock(return_value=False)
    env.get_nearest_food_distance_for = Mock(return_value=None)
    env.predator = Mock()
    env.predator.enabled = False
    env.thermotaxis = Mock()
    env.thermotaxis.enabled = False
    env.foods = []
    return env


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
        penalty_boundary_collision=0.0,
        penalty_temperature_proximity=0.0,
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
        env = _make_base_env([2, 2])
        env.foods = [(5, 5), (1, 1)]
        env.get_nearest_food_distance_for = Mock(return_value=1)

        calculator = RewardCalculator(default_config)
        path = [(3, 3), (2, 2)]  # Previous nearest was Manhattan=4, now=1

        reward = calculator.calculate_reward(env, path)

        # Distance reward: 0.1 * (4 - 1) = 0.3
        # Exploration bonus: 0.02 (new cell)
        # Step penalty: -0.01
        assert reward == pytest.approx(0.31)

    def test_foraging_exploration_bonus(self, default_config):
        """Test exploration bonus for visiting new cells."""
        env = _make_base_env([1, 2])

        calculator = RewardCalculator(default_config)
        path = [(1, 2)]

        reward = calculator.calculate_reward(env, path)

        # Exploration bonus: 0.02, Step penalty: -0.01
        assert reward == pytest.approx(0.01)
        assert (1, 2) in env.agents["default"].visited_cells

    def test_foraging_no_exploration_bonus_for_visited_cell(self, default_config):
        """Test no exploration bonus for already visited cells."""
        env = _make_base_env([1, 2], visited_cells={(1, 2)})

        calculator = RewardCalculator(default_config)
        path = [(1, 2)]

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01
        assert reward == pytest.approx(-0.01)


class TestMultiAgentReward:
    """Test reward calculation in multi-agent mode."""

    def test_per_agent_exploration_bonus(self, default_config):
        """Each agent gets independent exploration bonus for the same cell."""
        state_0 = _make_agent_state([5, 5])
        state_1 = _make_agent_state([5, 5])
        env = Mock(spec=DynamicForagingEnvironment)
        env.agents = {"agent_0": state_0, "agent_1": state_1}
        env.reached_goal_for = Mock(return_value=False)
        env.get_nearest_food_distance_for = Mock(return_value=None)
        env.predator = Mock()
        env.predator.enabled = False
        env.thermotaxis = Mock()
        env.thermotaxis.enabled = False
        env.foods = []

        calculator = RewardCalculator(default_config)

        # Agent 0 visits (5,5) — gets exploration bonus
        r0 = calculator.calculate_reward(env, [(5, 5)], agent_id="agent_0")
        assert (5, 5) in state_0.visited_cells

        # Agent 1 visits (5,5) — also gets exploration bonus (independent visited_cells)
        r1 = calculator.calculate_reward(env, [(5, 5)], agent_id="agent_1")
        assert (5, 5) in state_1.visited_cells

        # Both get exploration bonus (0.02) - step penalty (0.01) = 0.01
        assert r0 == pytest.approx(0.01)
        assert r1 == pytest.approx(0.01)

    def test_per_agent_distance_reward(self, default_config):
        """Each agent gets distance reward from its own position."""
        state_near = _make_agent_state([5, 5], visited_cells={(5, 5)})
        state_far = _make_agent_state([15, 15], visited_cells={(15, 15)})
        env = Mock(spec=DynamicForagingEnvironment)
        env.agents = {"near": state_near, "far": state_far}
        env.reached_goal_for = Mock(return_value=False)
        env.foods = [(5, 6)]
        env.predator = Mock()
        env.predator.enabled = False
        env.thermotaxis = Mock()
        env.thermotaxis.enabled = False

        # Near agent: distance 1 from food
        env.get_nearest_food_distance_for = Mock(
            side_effect=lambda aid: 1 if aid == "near" else 20,
        )

        calculator = RewardCalculator(default_config)

        # Near agent moved from (5,6) to (5,5) — closer to food at (5,6)
        r_near = calculator.calculate_reward(env, [(5, 6), (5, 5)], agent_id="near")
        # Far agent moved from (15,14) to (15,15) — farther from food at (5,6)
        r_far = calculator.calculate_reward(env, [(15, 14), (15, 15)], agent_id="far")

        # Near agent gets positive distance reward (moved closer), far gets negative
        assert r_near > r_far


class TestAntiDitheringPenalty:
    """Test anti-dithering penalty."""

    def test_anti_dithering_penalty_applied(self, default_config):
        """Test penalty when agent oscillates back to previous position."""
        env = _make_base_env((1, 1), visited_cells={(1, 1)})

        calculator = RewardCalculator(default_config)
        path = [(1, 1), (2, 2), (1, 1)]  # Back to same position

        reward = calculator.calculate_reward(env, path)

        # Anti-dithering penalty: -0.05
        # Step penalty: -0.01
        assert reward == pytest.approx(-0.06)

    def test_no_anti_dithering_penalty_normal_movement(self, default_config):
        """Test no penalty for normal movement."""
        env = _make_base_env([3, 3], visited_cells={(3, 3)})

        calculator = RewardCalculator(default_config)
        path = [(1, 1), (2, 2), (3, 3)]  # Normal progression

        reward = calculator.calculate_reward(env, path)

        # Only step penalty: -0.01 (no distance reward since no foods)
        assert reward == pytest.approx(-0.01)


class TestPredatorEvasionReward:
    """Test distance-scaled predator evasion reward."""

    def _make_predator_env(self, agent_pos, predator_positions, *, in_danger=True):
        """Create a mock env with predators at given positions."""
        env = _make_base_env(agent_pos, visited_cells={tuple(agent_pos)})
        env.predator.enabled = True
        env.is_agent_in_danger_for = Mock(return_value=in_danger)

        predators = []
        for pos in predator_positions:
            pred = Mock()
            pred.position = pos
            predators.append(pred)
        env.predators = predators

        if predator_positions:
            env.get_nearest_predator_distance_for = Mock(
                return_value=min(
                    abs(agent_pos[0] - p[0]) + abs(agent_pos[1] - p[1]) for p in predator_positions
                ),
            )
        else:
            env.get_nearest_predator_distance_for = Mock(return_value=None)

        return env

    def test_evasion_reward_moving_away(self):
        """Moving away from predator while in danger zone gives positive reward."""
        config = RewardConfig(penalty_step=0.01, penalty_predator_proximity=0.5)
        env = self._make_predator_env([6, 5], [(3, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 5), (6, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(0.49)

    def test_evasion_penalty_moving_closer(self):
        """Moving toward predator while in danger zone gives negative penalty."""
        config = RewardConfig(penalty_step=0.01, penalty_predator_proximity=0.5)
        env = self._make_predator_env([5, 5], [(3, 5)])
        calculator = RewardCalculator(config)
        path = [(6, 5), (5, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.51)

    def test_evasion_contact_penalty_at_distance_zero(self):
        """Predator on same cell applies contact penalty."""
        config = RewardConfig(penalty_step=0.01, penalty_predator_proximity=0.5)
        env = self._make_predator_env([5, 5], [(5, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 5), (5, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.51)

    def test_evasion_outside_detection_not_applied(self):
        """No evasion reward when agent is outside detection radius."""
        config = RewardConfig(penalty_step=0.01, penalty_predator_proximity=0.5)
        env = self._make_predator_env([50, 50], [(10, 10)], in_danger=False)
        calculator = RewardCalculator(config)
        path = [(49, 50), (50, 50)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.01)

    def test_evasion_first_step_flat_fallback(self):
        """First step in episode uses flat penalty fallback."""
        config = RewardConfig(penalty_step=0.01, penalty_predator_proximity=0.5)
        env = self._make_predator_env([5, 5], [(3, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.51)

    def test_no_evasion_when_predators_disabled(self):
        """No evasion reward when predators are disabled."""
        config = RewardConfig(penalty_step=0.01, penalty_predator_proximity=0.5)
        env = _make_base_env([5, 5], visited_cells={(5, 5)})

        calculator = RewardCalculator(config)
        path = [(4, 5), (5, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.01)

    def test_evasion_multiple_predators_uses_nearest(self):
        """With multiple predators, evasion uses nearest predator distance."""
        config = RewardConfig(penalty_step=0.01, penalty_predator_proximity=0.5)
        env = self._make_predator_env([6, 5], [(3, 5), (50, 50)])
        calculator = RewardCalculator(config)
        path = [(5, 5), (6, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(0.49)


class TestGradientOnlyRewardMode:
    """Validate the ``reward_mode='gradient_only'`` M6.10 audit-B remediation.

    Under ``gradient_only`` the distance-scaled evasion term and the
    first-step flat-fallback are both dropped; only the contact
    penalty at ``dist <= 1`` fires. ``default`` mode preserves the
    M3 / M4 / M5 / M6 byte-equivalent behaviour.
    """

    def _make_predator_env(self, agent_pos, predator_positions, *, in_danger=True):
        env = _make_base_env(agent_pos, visited_cells={tuple(agent_pos)})
        env.predator.enabled = True
        env.is_agent_in_danger_for = Mock(return_value=in_danger)
        predators = []
        for pos in predator_positions:
            pred = Mock()
            pred.position = pos
            predators.append(pred)
        env.predators = predators
        if predator_positions:
            env.get_nearest_predator_distance_for = Mock(
                return_value=min(
                    abs(agent_pos[0] - p[0]) + abs(agent_pos[1] - p[1]) for p in predator_positions
                ),
            )
        else:
            env.get_nearest_predator_distance_for = Mock(return_value=None)
        return env

    def test_default_mode_preserves_distance_term(self):
        """``reward_mode='default'`` (legacy) SHALL apply the distance-scaled evasion term."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
            reward_mode="default",
        )
        env = self._make_predator_env([6, 5], [(3, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 5), (6, 5)]

        # Moving away (prev_dist=2 → curr_dist=3) → evasion_reward = +0.5; step penalty -0.01.
        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(0.49)

    def test_gradient_only_drops_distance_term(self):
        """``reward_mode='gradient_only'`` SHALL drop the distance-scaled tangential term."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
            reward_mode="gradient_only",
        )
        env = self._make_predator_env([6, 5], [(3, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 5), (6, 5)]

        # Same move as above (away from predator, dist > 1) but no
        # distance-scaled reward. Only step penalty fires.
        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.01)

    def test_gradient_only_preserves_contact_penalty(self):
        """``gradient_only`` SHALL keep the contact penalty at ``dist <= 1``."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
            reward_mode="gradient_only",
        )
        env = self._make_predator_env([5, 5], [(5, 5)])
        calculator = RewardCalculator(config)
        path = [(5, 5), (5, 5)]

        # Contact (dist=0) → contact penalty -0.5; step penalty -0.01.
        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.51)

    def test_gradient_only_vs_default_diverges_on_motion_to_contact(self):
        """``gradient_only`` and ``default`` SHALL produce different rewards on motion-to-contact.

        Discriminating path: agent at (6,5) moves to (5,5) where predator
        sits — prev_dist=1, curr_dist=0. Default mode adds distance-scaled
        term `0.5*(0-1) = -0.5` (moving closer) PLUS contact penalty -0.5
        = -1.0 evasion. gradient_only adds only the contact penalty -0.5.
        With step penalty -0.01: default → -1.01, gradient_only → -0.51.
        """
        default_config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
            reward_mode="default",
        )
        gradient_config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
            reward_mode="gradient_only",
        )
        env_default = self._make_predator_env([5, 5], [(5, 5)])
        env_gradient = self._make_predator_env([5, 5], [(5, 5)])
        path = [(6, 5), (5, 5)]

        r_default = RewardCalculator(default_config).calculate_reward(env_default, path)
        r_gradient = RewardCalculator(gradient_config).calculate_reward(env_gradient, path)

        assert r_default == pytest.approx(-1.01)
        assert r_gradient == pytest.approx(-0.51)
        assert r_default != r_gradient

    def test_gradient_only_drops_flat_fallback(self):
        """``gradient_only`` SHALL drop the first-step flat-fallback penalty."""
        config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
            reward_mode="gradient_only",
        )
        env = self._make_predator_env([5, 5], [(3, 5)])
        calculator = RewardCalculator(config)
        # First step (single-position path) — no prev_dist available.
        # Default mode would apply flat fallback; gradient_only skips it.
        path = [(5, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.01)

    def test_gradient_only_skips_fallback_when_curr_pred_dist_is_none(self):
        """``gradient_only`` SHALL skip the fallback when ``curr_pred_dist`` is ``None``.

        Same exec path as the first-step case (the original condition is
        ``curr_pred_dist is not None and len(path) > 1``) but reached via
        a different env state: predator in danger zone but distance
        un-computable (env-side edge case). Default mode applies the
        flat fallback; gradient_only mode does not.
        """
        env_default = self._make_predator_env([5, 5], [(3, 5)])
        env_default.get_nearest_predator_distance_for = Mock(return_value=None)
        env_gradient = self._make_predator_env([5, 5], [(3, 5)])
        env_gradient.get_nearest_predator_distance_for = Mock(return_value=None)
        path = [(4, 5), (5, 5)]  # len > 1 so first-step guard doesn't trip

        default_config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
            reward_mode="default",
        )
        gradient_config = RewardConfig(
            penalty_step=0.01,
            penalty_predator_proximity=0.5,
            reward_mode="gradient_only",
        )
        r_default = RewardCalculator(default_config).calculate_reward(env_default, path)
        r_gradient = RewardCalculator(gradient_config).calculate_reward(env_gradient, path)

        # Default: flat-fallback -0.5 + step penalty -0.01 = -0.51.
        # gradient_only: only step penalty -0.01.
        assert r_default == pytest.approx(-0.51)
        assert r_gradient == pytest.approx(-0.01)

    def test_default_and_gradient_only_match_outside_predators(self):
        """When predators are disabled, both modes SHALL produce identical rewards."""
        env = _make_base_env([5, 5], visited_cells={(5, 5)})
        env.predator.enabled = False
        path = [(4, 5), (5, 5)]

        default_calc = RewardCalculator(
            RewardConfig(penalty_step=0.01, penalty_predator_proximity=0.5, reward_mode="default"),
        )
        gradient_calc = RewardCalculator(
            RewardConfig(
                penalty_step=0.01,
                penalty_predator_proximity=0.5,
                reward_mode="gradient_only",
            ),
        )
        assert default_calc.calculate_reward(env, path) == gradient_calc.calculate_reward(env, path)

    def test_reward_mode_default_value(self):
        """``reward_mode`` SHALL default to ``"default"`` (legacy byte-equivalence)."""
        config = RewardConfig()
        assert config.reward_mode == "default"

    def test_unknown_reward_mode_rejected_at_construction(self):
        """``RewardConfig`` SHALL reject unknown ``reward_mode`` values at construction."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RewardConfig(reward_mode="sparse")  # type: ignore[arg-type]


class TestTemperatureAvoidanceReward:
    """Test distance-scaled temperature avoidance reward."""

    def _make_thermotaxis_env(
        self,
        agent_pos,
        current_temp,
        prev_temp,
        zone,
        *,
        cultivation_temp=20.0,
    ):
        """Create a mock env with thermotaxis at given temperatures."""
        env = _make_base_env(agent_pos, visited_cells={tuple(agent_pos)})
        env.thermotaxis.enabled = True
        env.thermotaxis.cultivation_temperature = cultivation_temp
        env.get_temperature_zone_for = Mock(return_value=zone)

        def get_temp_side_effect(position=None):
            if position is None or position == tuple(agent_pos):
                return current_temp
            return prev_temp

        env.get_temperature = Mock(side_effect=get_temp_side_effect)
        return env

    def test_temp_avoidance_reward_moving_toward_tc(self):
        """Moving toward cultivation temp in discomfort zone gives positive reward."""
        config = RewardConfig(penalty_step=0.01, penalty_temperature_proximity=0.3)
        env = self._make_thermotaxis_env(
            [5, 5],
            current_temp=26.0,
            prev_temp=28.0,
            zone=TemperatureZone.DISCOMFORT_HOT,
        )
        calculator = RewardCalculator(config)
        path = [(4, 5), (5, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(0.59)

    def test_temp_avoidance_penalty_moving_away_from_tc(self):
        """Moving away from cultivation temp in discomfort zone gives negative penalty."""
        config = RewardConfig(penalty_step=0.01, penalty_temperature_proximity=0.3)
        env = self._make_thermotaxis_env(
            [5, 5],
            current_temp=28.0,
            prev_temp=26.0,
            zone=TemperatureZone.DISCOMFORT_HOT,
        )
        calculator = RewardCalculator(config)
        path = [(4, 5), (5, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.61)

    def test_temp_avoidance_not_applied_in_comfort_zone(self):
        """No avoidance reward when agent is in comfort zone."""
        config = RewardConfig(penalty_step=0.01, penalty_temperature_proximity=0.3)
        env = self._make_thermotaxis_env(
            [5, 5],
            current_temp=22.0,
            prev_temp=24.0,
            zone=TemperatureZone.COMFORT,
        )
        calculator = RewardCalculator(config)
        path = [(4, 5), (5, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.01)

    def test_temp_avoidance_not_applied_without_thermotaxis(self):
        """No avoidance reward when thermotaxis is disabled."""
        config = RewardConfig(penalty_step=0.01, penalty_temperature_proximity=0.3)
        env = _make_base_env([5, 5], visited_cells={(5, 5)})

        calculator = RewardCalculator(config)
        path = [(4, 5), (5, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.01)


class TestStuckPositionPenalty:
    """Test stuck position penalty."""

    def test_stuck_position_penalty_applied(self, default_config):
        """Test penalty when agent is stuck."""
        env = _make_base_env([2, 2], visited_cells={(2, 2)})

        calculator = RewardCalculator(default_config)
        path = [(2, 2), (2, 2)]

        reward = calculator.calculate_reward(env, path, stuck_position_count=5)
        assert reward == pytest.approx(-0.05)

    def test_no_stuck_penalty_below_threshold(self, default_config):
        """Test no penalty when below stuck threshold."""
        env = _make_base_env([2, 2], visited_cells={(2, 2)})

        calculator = RewardCalculator(default_config)
        path = [(2, 2), (2, 2)]

        reward = calculator.calculate_reward(env, path, stuck_position_count=2)
        assert reward == pytest.approx(-0.01)

    def test_stuck_penalty_capped_at_10(self, default_config):
        """Test stuck penalty is capped at 10 extra steps."""
        env = _make_base_env([2, 2], visited_cells={(2, 2)})

        calculator = RewardCalculator(default_config)
        path = [(2, 2), (2, 2)]

        reward = calculator.calculate_reward(env, path, stuck_position_count=20)
        assert reward == pytest.approx(-0.21)


class TestBoundaryCollisionPenalty:
    """Test boundary collision penalty."""

    def test_boundary_collision_penalty_applied(self):
        """Test penalty when agent collided with wall."""
        config = RewardConfig(penalty_step=0.01, penalty_boundary_collision=0.02)
        env = _make_base_env([0, 5], visited_cells={(0, 5)}, wall_collision=True)

        calculator = RewardCalculator(config)
        path = [(0, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.03)

    def test_no_boundary_collision_penalty_when_no_collision(self):
        """Test no penalty when agent didn't try to move into a wall."""
        config = RewardConfig(penalty_step=0.01, penalty_boundary_collision=0.02)
        env = _make_base_env([5, 5], visited_cells={(5, 5)})

        calculator = RewardCalculator(config)
        path = [(5, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.01)

    def test_no_penalty_at_edge_without_collision(self):
        """Test no penalty when at edge but didn't try to move into wall."""
        config = RewardConfig(penalty_step=0.01, penalty_boundary_collision=0.02)
        env = _make_base_env([0, 5], visited_cells={(0, 5)})

        calculator = RewardCalculator(config)
        path = [(0, 5)]

        reward = calculator.calculate_reward(env, path)
        assert reward == pytest.approx(-0.01)
