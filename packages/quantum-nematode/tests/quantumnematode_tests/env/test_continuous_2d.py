"""Unit tests for the continuous-2D environment kinematics (T5 §3.3).

Covers the `continuous-2d-environment` spec scenarios reachable without the
runner: float-position init, kinematic `(speed, turn)` movement, speed clamping,
world-bound clamping, heading wrap, and point-worm body. Capture-radius /
placement / Euclidean (§3.4) and klinotaxis sampling (§3.5) are tested when added.
"""

from __future__ import annotations

import math

import pytest
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID


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


def _state(env: Continuous2DEnvironment):
    return env.agents[DEFAULT_AGENT_ID]


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
