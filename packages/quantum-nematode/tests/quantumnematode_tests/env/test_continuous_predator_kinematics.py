"""Unit tests for continuous-2D predator kinematics + Euclidean detection.

Covers the behaviours added when predators move continuously on the continuous-2D
substrate: continuous ``(speed, heading)`` movement (pursuit / wander / stationary),
world-bound clamping, float spawn, Euclidean detection / damage / contact-zone
geometry against the worm's float ``pos_continuous``, ``copy()`` state preservation,
and a grid-path regression smoke (the additive ``Predator`` fields stay unused on the
discrete grid).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import (
    DEFAULT_AGENT_ID,
    ContactZone,
    Predator,
    PredatorParams,
    PredatorType,
)

# World 30 mm → grid_size 30 (cells 0..29); centre cell (15, 15) so the agent's
# spawn-time integer reference and the post-init float centre coincide.
_WORLD = 30.0
_CENTRE = (15.0, 15.0)


def _env(  # noqa: PLR0913 — test factory threads predator config knobs
    *,
    count: int = 1,
    ptype: PredatorType = PredatorType.PURSUIT,
    detection: int = 20,
    damage: int = 2,
    speed: float = 1.0,
    max_step: float = 1.0,
    seed: int = 7,
) -> Continuous2DEnvironment:
    return Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=_WORLD, max_step_mm=max_step),
        predator=PredatorParams(
            enabled=True,
            count=count,
            predator_type=ptype,
            speed=speed,
            detection_radius=detection,
            damage_radius=damage,
        ),
        start_pos=(15, 15),
        seed=seed,
    )


def _set_agent(env: Continuous2DEnvironment, xy: tuple[float, float], heading: float = 0.0) -> None:
    state = env.agents[DEFAULT_AGENT_ID]
    state.pos_continuous = xy
    state.heading_rad = heading


def _pred(env: Continuous2DEnvironment) -> Predator:
    return env.predators[0]


class TestContinuousMovement:
    def test_pursuit_steers_toward_agent_and_advances_subcell(self) -> None:
        env = _env(detection=50)
        _set_agent(env, (10.0, 10.0))
        pred = _pred(env)
        pred.pos_continuous = (5.0, 5.0)
        pred.heading_rad = 0.0

        env.update_predators()

        nx, ny = pred.pos_continuous
        # Heading orients toward the agent's bearing; advance is one max_step (1 mm).
        assert pred.heading_rad == pytest.approx(math.atan2(5.0, 5.0))
        assert math.hypot(nx - 5.0, ny - 5.0) == pytest.approx(1.0)
        # Sub-cell: the float truth is not snapped to the lattice; the integer
        # position is its rounded view.
        assert abs(nx - round(nx)) > 1e-6
        assert pred.position == (round(nx), round(ny))

    def test_stationary_predator_never_moves(self) -> None:
        env = _env(ptype=PredatorType.STATIONARY, damage=5)
        pred = _pred(env)
        pred.pos_continuous = (7.0, 8.0)
        _set_agent(env, (7.5, 8.0))

        env.update_predators()

        assert pred.pos_continuous == (7.0, 8.0)

    def test_wandering_predator_advances_and_stays_in_bounds(self) -> None:
        # Agent out of detection range → the pursuit predator wanders.
        env = _env(detection=1)
        _set_agent(env, (25.0, 25.0))
        pred = _pred(env)
        pred.pos_continuous = (2.0, 2.0)
        pred.heading_rad = 0.0

        env.update_predators()

        nx, ny = pred.pos_continuous
        assert math.hypot(nx - 2.0, ny - 2.0) == pytest.approx(1.0)
        assert 0.0 <= nx <= _WORLD
        assert 0.0 <= ny <= _WORLD

    def test_world_bound_clamping_x_axis(self) -> None:
        env = _env(detection=50, damage=0)
        _set_agent(env, (40.0, 15.0))  # off-arena target → heading +x toward the wall
        pred = _pred(env)
        pred.pos_continuous = (29.8, 15.0)
        pred.heading_rad = 0.0

        env.update_predators()

        nx, ny = pred.pos_continuous
        assert nx == pytest.approx(_WORLD)  # clamped at the bound (partial move)
        assert 0.0 <= ny <= _WORLD

    def test_world_bound_clamping_y_axis(self) -> None:
        env = _env(detection=50, damage=0)
        _set_agent(env, (15.0, 40.0))  # off-arena target → heading +y toward the wall
        pred = _pred(env)
        pred.pos_continuous = (15.0, 29.8)
        pred.heading_rad = 0.0

        env.update_predators()

        nx, ny = pred.pos_continuous
        assert ny == pytest.approx(_WORLD)  # per-axis y clamp at the bound
        assert 0.0 <= nx <= _WORLD


class TestMultiAgentPursuit:
    def test_pursuit_targets_nearest_alive_agent(self) -> None:
        env = _env(detection=50)
        # Register a second agent; place predator nearer to agent "a2".
        env.add_agent("a2", position=(1, 1))
        env.agents[DEFAULT_AGENT_ID].pos_continuous = (2.0, 18.0)
        env.agents["a2"].pos_continuous = (2.0, 2.0)
        pred = _pred(env)
        pred.pos_continuous = (2.0, 5.0)  # closer to a2 (dist 3) than default (dist 13)
        pred.heading_rad = 0.0

        env.update_predators()

        # Steered toward a2 (-y, bearing about -pi/2), not the farther default agent (+y).
        assert pred.heading_rad == pytest.approx(math.atan2(2.0 - 5.0, 0.0))

    def test_dead_agent_excluded_from_targets(self) -> None:
        env = _env(detection=50)
        env.add_agent("a2", position=(1, 1))
        env.agents[DEFAULT_AGENT_ID].pos_continuous = (2.0, 18.0)
        env.agents["a2"].pos_continuous = (2.0, 2.0)
        env.agents["a2"].alive = False  # nearest, but dead → ignored
        pred = _pred(env)
        pred.pos_continuous = (2.0, 5.0)
        pred.heading_rad = 0.0

        env.update_predators()

        # Only the (living) default agent remains a target → steer +y toward it.
        assert pred.heading_rad == pytest.approx(math.atan2(18.0 - 5.0, 0.0))


class TestEuclideanDetection:
    def test_detection_is_euclidean_not_manhattan(self) -> None:
        # Offset (4, 3): Euclidean 5.0 (≤ 5 → danger), Manhattan 7 (> 5 → grid would
        # say safe). Distinguishes the two metrics.
        env = _env(detection=5, damage=0)
        _set_agent(env, (10.0, 10.0))
        pred = _pred(env)

        pred.pos_continuous = (14.0, 13.0)
        assert env.is_agent_in_danger_for(DEFAULT_AGENT_ID) is True

        pred.pos_continuous = (14.0, 14.0)  # offset (4, 4): Euclidean 5.66 > 5
        assert env.is_agent_in_danger_for(DEFAULT_AGENT_ID) is False

    def test_damage_is_euclidean(self) -> None:
        env = _env(detection=50, damage=3)
        _set_agent(env, (10.0, 10.0))
        pred = _pred(env)

        pred.pos_continuous = (12.0, 12.0)  # Euclidean 2.83 ≤ 3
        assert env.is_agent_in_damage_radius_for(DEFAULT_AGENT_ID) is True

        pred.pos_continuous = (12.5, 12.5)  # Euclidean 3.54 > 3
        assert env.is_agent_in_damage_radius_for(DEFAULT_AGENT_ID) is False


class TestContactZoneContinuousHeading:
    @pytest.mark.parametrize(
        ("predator_xy", "expected"),
        [
            ((12.0, 10.0), ContactZone.ANTERIOR),  # dead ahead (worm faces +x)
            ((8.0, 10.0), ContactZone.POSTERIOR),  # directly behind
            ((10.0, 12.0), ContactZone.LATERAL),  # abeam (+y)
        ],
    )
    def test_zone_from_heading(
        self,
        predator_xy: tuple[float, float],
        expected: ContactZone,
    ) -> None:
        env = _env(detection=50, damage=5)
        _set_agent(env, (10.0, 10.0), heading=0.0)  # facing +x
        _pred(env).pos_continuous = predator_xy

        assert env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID) == expected

    def test_zone_rotates_with_heading(self) -> None:
        # Same predator (+y of the worm); facing +y → it is now dead ahead.
        env = _env(detection=50, damage=5)
        _set_agent(env, (10.0, 10.0), heading=math.pi / 2.0)
        _pred(env).pos_continuous = (10.0, 12.0)

        assert env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID) == ContactZone.ANTERIOR

    def test_overlap_classified_anterior(self) -> None:
        # Predator exactly on the worm (rel_len == 0) → ANTERIOR by convention.
        env = _env(detection=50, damage=5)
        _set_agent(env, (10.0, 10.0), heading=0.0)
        _pred(env).pos_continuous = (10.0, 10.0)

        assert env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID) == ContactZone.ANTERIOR


class TestFloatSpawn:
    def test_predators_spawn_float_within_bounds_and_separated(self) -> None:
        env = _env(count=3, detection=4, damage=2, seed=3)
        min_sep = max(4, 2)
        upper = float(env.grid_size - 1)

        assert any(
            abs(p.pos_continuous[0] - round(p.pos_continuous[0])) > 1e-9
            for p in env.predators
            if p.pos_continuous is not None
        ), "expected at least one non-integer spawn coordinate"

        for p in env.predators:
            assert p.pos_continuous is not None
            x, y = p.pos_continuous
            assert 0.0 <= x <= upper
            assert 0.0 <= y <= upper
            assert p.position == (round(x), round(y))  # synced integer view
            assert math.hypot(x - _CENTRE[0], y - _CENTRE[1]) > min_sep


class TestCopyPreservesContinuousState:
    def test_copy_preserves_pos_continuous_and_heading(self) -> None:
        env = _env(count=2, seed=11)
        for _ in range(3):
            env.update_predators()

        clone = env.copy()

        for original, cloned in zip(env.predators, clone.predators, strict=True):
            assert cloned.pos_continuous == original.pos_continuous
            assert cloned.heading_rad == original.heading_rad


class TestGridPathUnchanged:
    def test_grid_predator_fields_stay_unused(self) -> None:
        # A grid Predator carries the additive fields at their defaults and the grid
        # movement path never touches them (byte-stability of the integer model is
        # covered end-to-end by test_predator_brain_byte_equivalence).
        pred = Predator(position=(2, 2), predator_type=PredatorType.PURSUIT, detection_radius=8)
        assert pred.pos_continuous is None
        assert pred.heading_rad == 0.0

        pred.update_position(20, np.random.default_rng(0), agent_positions=[(5, 5)], step_index=0)

        assert pred.pos_continuous is None
        assert pred.heading_rad == 0.0
