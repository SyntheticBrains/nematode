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
    dmg_fallback: float = 1.0,
    seed: int = 7,
) -> Continuous2DEnvironment:
    return Continuous2DEnvironment(
        continuous=Continuous2DParams(
            world_size_mm=_WORLD,
            max_step_mm=max_step,
            predator_damage_radius_mm=dmg_fallback,
        ),
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


class TestContinuousPredatorContactIntensity:
    """Euclidean ``predator_contact_intensity_at`` + effective-radius fallback.

    The ``predator_mechano`` channel was dead on continuous (default ``damage_radius=0``
    skipped every predator); the continuous override revives it via the effective radius
    and uses Euclidean distance.
    """

    def test_revived_with_default_damage_radius_zero(self) -> None:
        # Continuous default damage_radius=0 (would skip every predator pre-fix), fallback 1.0 mm.
        env = _env(damage=0, dmg_fallback=1.0)
        _pred(env).pos_continuous = (10.0, 10.0)
        # 0.5 mm away (Euclidean) → intensity 1 - 0.5/1.0 = 0.5, NOT 0.0 (channel revived).
        assert env.predator_contact_intensity_at((10.5, 10.0)) == pytest.approx(0.5)

    def test_euclidean_metric(self) -> None:
        env = _env(damage=0, dmg_fallback=2.0)
        _pred(env).pos_continuous = (10.0, 10.0)
        # Off-axis query: Euclidean hypot(0.6,0.8)=1.0 → 1 - 1.0/2.0 = 0.5;
        # Manhattan would be 1.4 → 0.3. Assert the Euclidean value.
        assert env.predator_contact_intensity_at((10.6, 10.8)) == pytest.approx(0.5)

    def test_zero_outside_effective_radius(self) -> None:
        env = _env(damage=0, dmg_fallback=1.0)
        _pred(env).pos_continuous = (10.0, 10.0)
        assert env.predator_contact_intensity_at((15.0, 10.0)) == 0.0  # 5 mm > 1 mm radius

    def test_explicit_positive_radius_takes_precedence(self) -> None:
        env = _env(damage=4, dmg_fallback=1.0)  # explicit 4 mm wins over the fallback
        _pred(env).pos_continuous = (10.0, 10.0)
        # 2 mm away, radius 4 → 1 - 2/4 = 0.5 (would be 0 if the 1 mm fallback were used).
        assert env.predator_contact_intensity_at((12.0, 10.0)) == pytest.approx(0.5)


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


class TestEuclideanNearestPredatorDistance:
    """The nearest-predator-distance the reward consumes is Euclidean on continuous."""

    def test_nearest_distance_is_euclidean_not_manhattan(self) -> None:
        env = _env(detection=50, damage=2)
        _set_agent(env, (10.0, 10.0))
        pred = _pred(env)
        pred.pos_continuous = (13.0, 14.0)  # offset (3, 4): Euclidean 5.0, Manhattan 7

        dist = env.get_nearest_predator_distance_for(DEFAULT_AGENT_ID)
        assert dist == pytest.approx(5.0)  # hypot(3, 4), not |3| + |4| = 7

    def test_nearest_distance_takes_the_minimum(self) -> None:
        env = _env(count=2, detection=50, damage=2)
        _set_agent(env, (10.0, 10.0))
        env.predators[0].pos_continuous = (10.0, 18.0)  # 8.0 away
        env.predators[1].pos_continuous = (13.0, 14.0)  # 5.0 away (nearest)

        assert env.get_nearest_predator_distance_for(DEFAULT_AGENT_ID) == pytest.approx(5.0)

    def test_convenience_matches_default_agent(self) -> None:
        env = _env(detection=50, damage=2)
        _set_agent(env, (10.0, 10.0))
        _pred(env).pos_continuous = (13.0, 14.0)

        assert env.get_nearest_predator_distance() == env.get_nearest_predator_distance_for(
            DEFAULT_AGENT_ID,
        )

    def test_returns_none_when_no_predators(self) -> None:
        env = _env(count=0, detection=50, damage=2)
        _set_agent(env, (10.0, 10.0))
        assert env.get_nearest_predator_distance_for(DEFAULT_AGENT_ID) is None


class TestDamageRadiusFallback:
    """Body/contact-scale fallback when damage_radius is the unreachable grid default.

    The integer grid default ``damage_radius=0`` is unreachable as a Euclidean distance,
    so the continuous substrate falls back to ``predator_damage_radius_mm``.
    """

    def test_default_zero_falls_back_to_body_scale(self) -> None:
        # damage_radius=0 (grid default) -> fallback predator_damage_radius_mm (default 1.0 mm).
        env = _env(detection=50, damage=0)
        _set_agent(env, (10.0, 10.0))
        pred = _pred(env)

        pred.pos_continuous = (10.8, 10.0)  # Euclidean 0.8 ≤ 1.0 fallback
        assert env.is_agent_in_damage_radius_for(DEFAULT_AGENT_ID) is True

        pred.pos_continuous = (11.5, 10.0)  # Euclidean 1.5 > 1.0 fallback
        assert env.is_agent_in_damage_radius_for(DEFAULT_AGENT_ID) is False

    def test_configurable_fallback_radius(self) -> None:
        # damage_radius=0 but a larger body/contact scale configured.
        env = _env(detection=50, damage=0, dmg_fallback=2.5)
        _set_agent(env, (10.0, 10.0))
        pred = _pred(env)

        pred.pos_continuous = (12.0, 10.0)  # Euclidean 2.0 ≤ 2.5 fallback
        assert env.is_agent_in_damage_radius_for(DEFAULT_AGENT_ID) is True

    def test_explicit_positive_damage_radius_takes_precedence(self) -> None:
        # An explicit positive damage_radius (3) wins over a small fallback (0.5).
        env = _env(detection=50, damage=3, dmg_fallback=0.5)
        _set_agent(env, (10.0, 10.0))
        pred = _pred(env)

        pred.pos_continuous = (12.0, 10.0)  # Euclidean 2.0: ≤ 3 (explicit), > 0.5 (fallback)
        assert env.is_agent_in_damage_radius_for(DEFAULT_AGENT_ID) is True


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


class TestDiscreteFallbackHeading:
    def test_discrete_move_syncs_heading_to_direction(self) -> None:
        from quantumnematode.brain.actions import Action
        from quantumnematode.env.env import _HEADING_OFFSET

        env = _env()
        state = env.agents[DEFAULT_AGENT_ID]
        assert state.heading_rad == 0.0  # initial facing +x

        env.move_agent(Action.LEFT)  # a turning action changes the discrete direction

        ox, oy = _HEADING_OFFSET[state.direction]
        assert (ox, oy) != (1, 0), "expected the discrete facing to have changed"
        # heading_rad must follow the new discrete facing (same _HEADING_OFFSET source).
        assert state.heading_rad == pytest.approx(math.atan2(oy, ox))

    def test_contact_zone_correct_after_discrete_move(self) -> None:
        from quantumnematode.brain.actions import Action
        from quantumnematode.env.env import _HEADING_OFFSET

        env = _env(detection=50, damage=5)
        env.move_agent(Action.LEFT)  # turn so the facing is no longer +x
        state = env.agents[DEFAULT_AGENT_ID]
        assert state.pos_continuous is not None
        ax, ay = state.pos_continuous
        ox, oy = _HEADING_OFFSET[state.direction]
        _pred(env).pos_continuous = (ax + ox, ay + oy)  # one unit dead ahead of the new facing

        # With a stale heading_rad (0.0) this predator would mis-classify as LATERAL.
        assert env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID) == ContactZone.ANTERIOR


class TestPredatorFieldUsesFloatPosition:
    def test_predator_concentration_uses_subcell_position(self) -> None:
        # Two predator float positions that round to the SAME integer cell (12, 10):
        # identical under the old integer-snap, distinct under float coordinates.
        env = _env(detection=0, damage=0)
        pred = _pred(env)
        query = (10.0, 10.0)

        pred.pos_continuous = (11.6, 10.0)  # 1.6 mm from the query point
        c_near = env.get_predator_concentration(query)
        pred.pos_continuous = (12.4, 10.0)  # 2.4 mm, but rounds to the same cell
        c_far = env.get_predator_concentration(query)

        assert c_near > c_far  # sub-cell distance is resolved (float coords, not integer-snapped)


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
