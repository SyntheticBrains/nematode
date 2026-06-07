"""Unit tests for the continuous-2D environment.

Covers the behaviours reachable without the runner: float-position init,
kinematic `(speed, turn)` movement, speed clamping, world-bound clamping, heading
wrap, point-worm body, capture-radius food, Euclidean distances, the coherent
discrete-action fallback, and the grid-substrate regression smoke.
"""

from __future__ import annotations

import math

import pytest
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID, AgentState


def _env(
    world: float = 20.0,
    max_step: float = 1.0,
    capture_radius: float = 1.0,
) -> Continuous2DEnvironment:
    return Continuous2DEnvironment(
        continuous=Continuous2DParams(
            world_size_mm=world,
            max_step_mm=max_step,
            capture_radius_mm=capture_radius,
        ),
    )


def _pos(env: Continuous2DEnvironment) -> tuple[float, float]:
    pc = env.agents[DEFAULT_AGENT_ID].pos_continuous
    assert pc is not None
    return pc


def _state(env: Continuous2DEnvironment) -> AgentState:
    return env.agents[DEFAULT_AGENT_ID]


class TestDiscreteFallbackCoherence:
    """Discrete action on the continuous env stays coherent.

    The float position (read by capture/reward) mirrors the moved integer cell
    (read by sensing). This is the fallback path for a discrete brain on a
    continuous env.
    """

    def test_discrete_move_syncs_pos_continuous(self) -> None:
        from quantumnematode.brain.actions import Action

        env = _env(world=20.0)
        env.move_agent(Action.FORWARD)  # discrete fallback (inherited _apply_movement)
        st = _state(env)
        assert st.pos_continuous == (float(st.position[0]), float(st.position[1]))

    def test_discrete_move_capture_reflects_new_position(self) -> None:
        from quantumnematode.brain.actions import Action

        env = _env(world=20.0, capture_radius=1.0)
        env.move_agent(Action.FORWARD)
        st = _state(env)
        env.foods = [st.position]  # food at the worm's new cell
        assert env.reached_goal_for(DEFAULT_AGENT_ID) is True  # capture follows the move


class TestInit:
    def test_float_position_centred(self) -> None:
        env = _env(world=20.0)
        st = _state(env)
        assert _pos(env) == (10.0, 10.0)
        assert st.heading_rad == 0.0
        assert st.position == (10, 10)  # rounded int view
        assert st.body == [(10, 10)]  # point-worm

    def test_parent_extent_matches_world(self) -> None:
        env = _env(world=30.0)
        assert env.grid_size == 30


class TestKinematicMovement:
    def test_forward_move_along_heading(self) -> None:
        env = _env()
        env.move_agent_continuous(speed=1.0, turn=0.0)  # heading 0 → +x
        x, y = _pos(env)
        assert x == pytest.approx(11.0)
        assert y == pytest.approx(10.0)

    def test_turn_then_move(self) -> None:
        env = _env()
        env.move_agent_continuous(speed=1.0, turn=math.pi / 2)  # heading +π/2 → +y
        x, y = _pos(env)
        assert x == pytest.approx(10.0, abs=1e-6)
        assert y == pytest.approx(11.0)
        assert _state(env).heading_rad == pytest.approx(math.pi / 2)

    def test_speed_clamped_to_max_step(self) -> None:
        env = _env(max_step=1.0)
        env.move_agent_continuous(speed=5.0, turn=0.0)  # clamped to 1.0
        x, _y = _pos(env)
        assert x == pytest.approx(11.0)

    def test_world_bound_clamp(self) -> None:
        env = _env(world=20.0, max_step=2.0)
        _state(env).pos_continuous = (19.5, 10.0)
        env.move_agent_continuous(speed=2.0, turn=0.0)  # would reach 21.5 → clamp 20
        x, _y = _pos(env)
        assert x == pytest.approx(20.0)

    def test_negative_speed_clamped_to_zero(self) -> None:
        env = _env()
        before = _pos(env)
        env.move_agent_continuous(speed=-3.0, turn=0.0)
        assert _pos(env) == pytest.approx(before)

    def test_heading_wrapped_to_pi(self) -> None:
        env = _env()
        for _ in range(5):
            env.move_agent_continuous(speed=0.0, turn=math.pi * 0.9)
            assert -math.pi <= _state(env).heading_rad <= math.pi

    def test_point_worm_body_stays_length_one(self) -> None:
        env = _env()
        for _ in range(3):
            env.move_agent_continuous(speed=1.0, turn=0.3)
        st = _state(env)
        assert len(st.body) == 1
        assert st.body[0] == st.position

    def test_int_position_view_synced_and_in_bounds(self) -> None:
        env = _env(world=20.0)
        env.move_agent_continuous(speed=1.0, turn=0.0)
        st = _state(env)
        assert st.position == env._discretise(_pos(env))
        assert 0 <= st.position[0] <= env.grid_size - 1
        assert 0 <= st.position[1] <= env.grid_size - 1


class TestCaptureRadius:
    def test_reached_goal_within_radius(self) -> None:
        env = _env(capture_radius=1.0)
        _state(env).pos_continuous = (10.0, 10.0)
        env.foods = [(11, 10)]  # distance 1.0 == radius → reached
        assert env.reached_goal_for(DEFAULT_AGENT_ID) is True

    def test_reached_goal_outside_radius(self) -> None:
        env = _env(capture_radius=1.0)
        _state(env).pos_continuous = (10.0, 10.0)
        env.foods = [(13, 10)]  # distance 3.0 > radius → not reached
        assert env.reached_goal_for(DEFAULT_AGENT_ID) is False

    def test_consume_within_radius_returns_and_removes(self) -> None:
        env = _env(capture_radius=1.0)
        _state(env).pos_continuous = (10.4, 10.0)
        env.foods = [(11, 10)]  # distance 0.6 ≤ radius
        consumed = env.consume_food_for(DEFAULT_AGENT_ID)
        assert consumed == (11, 10)
        assert (11, 10) not in env.foods  # removed (a respawn may add elsewhere)

    def test_consume_outside_radius_returns_none(self) -> None:
        env = _env(capture_radius=1.0)
        _state(env).pos_continuous = (10.0, 10.0)
        env.foods = [(13, 10)]
        assert env.consume_food_for(DEFAULT_AGENT_ID) is None
        assert (13, 10) in env.foods  # untouched

    def test_consume_picks_nearest_within_radius(self) -> None:
        env = _env(capture_radius=2.0)
        _state(env).pos_continuous = (10.0, 10.0)
        env.foods = [(12, 10), (11, 10)]  # distances 2.0 and 1.0 → nearest is (11,10)
        assert env.consume_food_for(DEFAULT_AGENT_ID) == (11, 10)

    def test_nearest_food_distance_is_euclidean(self) -> None:
        env = _env()
        _state(env).pos_continuous = (10.0, 10.0)
        env.foods = [(13, 14)]  # Euclidean 5.0 (not Manhattan 7)
        assert env.get_nearest_food_distance_for(DEFAULT_AGENT_ID) == 5
        assert env.get_nearest_food_distance() == 5

    def test_nearest_food_distance_none_when_empty(self) -> None:
        env = _env()
        env.foods = []
        assert env.get_nearest_food_distance_for(DEFAULT_AGENT_ID) is None

    def test_nearest_food_distance_is_unrounded(self) -> None:
        """Distance is the true Euclidean value, not rounded to int (Rung-2)."""
        env = _env()
        _state(env).pos_continuous = (10.4, 10.0)  # fractional worm position
        env.foods = [(12, 10)]  # distance 1.6 (round would give 2)
        dist = env.get_nearest_food_distance_for(DEFAULT_AGENT_ID)
        assert dist == pytest.approx(1.6)


class TestFloatSourcePlacement:
    """Rung-2: continuous-2D food sources are placed at real-valued coordinates."""

    def _seeded_env(self) -> Continuous2DEnvironment:
        from quantumnematode.env.env import ForagingParams

        return Continuous2DEnvironment(
            continuous=Continuous2DParams(world_size_mm=30.0),
            foraging=ForagingParams(foods_on_grid=6, min_food_distance=2),
            seed=123,
        )

    def test_sources_are_within_bounds(self) -> None:
        env = self._seeded_env()
        upper = float(env.grid_size - 1)
        assert len(env.foods) > 0
        for fx, fy in env.foods:
            assert 0.0 <= fx <= upper
            assert 0.0 <= fy <= upper

    def test_sources_are_real_valued(self) -> None:
        """At least one source has a fractional coordinate (off the integer lattice)."""
        env = self._seeded_env()
        assert any(fx != int(fx) or fy != int(fy) for fx, fy in env.foods)

    def test_candidate_generator_returns_python_floats_in_bounds(self) -> None:
        """The overridden candidate generator yields real-valued in-bounds coords."""
        env = self._seeded_env()
        upper = float(env.grid_size - 1)
        for _ in range(20):
            fx, fy = env._generate_food_candidate()
            assert isinstance(fx, float)
            assert isinstance(fy, float)
            assert 0.0 <= fx <= upper
            assert 0.0 <= fy <= upper


class TestCopyPreservesContinuousState:
    """`copy()` clones as the subclass and keeps the continuous position/heading.

    The continuous env has no runtime `copy()` caller yet (many-worlds is guarded
    off on this substrate), so these lock in the override + field-preservation
    behaviour the many-worlds-on-continuous path will depend on.
    """

    def test_copy_returns_continuous_subclass_with_same_params(self) -> None:
        env = _env(world=24.0, max_step=2.0, capture_radius=1.5)
        clone = env.copy()
        assert isinstance(clone, Continuous2DEnvironment)
        assert clone is not env
        assert clone.continuous.world_size_mm == 24.0
        assert clone.continuous.max_step_mm == 2.0
        assert clone.continuous.capture_radius_mm == 1.5
        assert clone.grid_size == env.grid_size

    def test_copy_preserves_pos_continuous_and_heading(self) -> None:
        env = _env(world=20.0, max_step=2.0)
        # Move off-centre so the float position + heading are non-default.
        env.move_agent_continuous(1.5, math.pi / 3)
        src = _state(env)
        # Snapshot the source value (pos_continuous is a tuple, so this is by
        # value) — `src` itself aliases the live env state, so comparing against
        # `src.pos_continuous` after the move would be tautological.
        pos_before = src.pos_continuous
        clone = env.copy()
        cloned = clone.agents[DEFAULT_AGENT_ID]
        assert cloned.pos_continuous == pos_before
        assert cloned.heading_rad == src.heading_rad
        # And the clone is independent — moving it leaves the source untouched.
        clone.move_agent_continuous(1.0, 0.0)
        assert _state(env).pos_continuous == pos_before


class TestGridSubstrateUnchanged:
    """The grid substrate is byte-unaffected by the continuous additions.

    A focused regression smoke; the broader grid behaviour is covered by the full
    env + agent suites (~800 tests) that pass unchanged alongside this change.
    """

    def test_factory_default_is_grid(self) -> None:
        from quantumnematode.env.env import DynamicForagingEnvironment
        from quantumnematode.utils.config_loader import (
            EnvironmentConfig,
            create_env_from_config,
        )

        env = create_env_from_config(EnvironmentConfig(grid_size=12))
        assert type(env) is DynamicForagingEnvironment  # not the continuous subclass

    def test_grid_discrete_move_is_one_cell(self) -> None:
        from quantumnematode.brain.actions import Action
        from quantumnematode.env.env import DynamicForagingEnvironment

        env = DynamicForagingEnvironment(grid_size=12, seed=7)
        start = env.agent_pos
        env.move_agent(Action.FORWARD)  # discrete cardinal move from centre
        moved = env.agent_pos
        assert abs(moved[0] - start[0]) + abs(moved[1] - start[1]) == 1
