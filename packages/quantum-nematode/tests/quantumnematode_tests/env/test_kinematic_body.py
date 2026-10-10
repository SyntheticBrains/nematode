"""The neuromuscular drive map and the kinematic body.

Covers the connectome-ppo-brain requirement "The anatomical neuromuscular readout" (signs follow
the transmitter) through the drive map, and the continuous-2d-environment requirement "A kinematic
segmented body" (a forward wave moves the body forward and a backward wave backward; a
dorsal-ventral bias turns the body; the head's period is the configured period; the body stays in
the arena; the point worm is unchanged).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantumnematode.connectome.loader import load_emmons_2024_neuromuscular
from quantumnematode.connectome.muscles import (
    BODY_WALL_MUSCLES,
    muscle_position,
    muscle_segment,
)
from quantumnematode.connectome.neuromuscular import muscle_sign, neuromuscular_drive_map
from quantumnematode.env.body import (
    DRIVE_WIDTH,
    N_SEGMENTS,
    BodyParams,
    KinematicBody,
    head_line_angle,
    new_body,
    wave_amplitude,
)
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID

# ── Muscles and the drive map ────────────────────────────────────────────────────────────────


class TestMuscles:
    def test_names_parse_to_quadrant_and_position(self) -> None:
        assert muscle_position("dBWML1") == ("dBWML", 1)
        assert muscle_position("vBWML23") == ("vBWML", 23)
        with pytest.raises(ValueError, match="not a body wall muscle"):
            muscle_position("AVAL")
        with pytest.raises(ValueError, match="beyond"):
            muscle_position("vBWML24")

    def test_twelve_segments_pair_the_positions(self) -> None:
        segments = [muscle_segment(m, N_SEGMENTS) for m in BODY_WALL_MUSCLES]
        assert set(segments) == set(range(N_SEGMENTS))
        assert muscle_segment("dBWMR1", N_SEGMENTS) == muscle_segment("dBWMR2", N_SEGMENTS) == 0
        assert muscle_segment("vBWML23", N_SEGMENTS) == N_SEGMENTS - 1


class TestDriveMap:
    @pytest.fixture(scope="class")
    def drive_map(self):
        return neuromuscular_drive_map(load_emmons_2024_neuromuscular(), N_SEGMENTS)

    def test_every_junction_cell_is_a_row(self, drive_map) -> None:
        assert len(drive_map.cells) == 162
        assert drive_map.matrix.shape == (162, 4 * N_SEGMENTS)

    def test_signs_follow_the_transmitter(self, drive_map) -> None:
        for i, cell in enumerate(drive_map.cells):
            row = drive_map.matrix[i]
            sign = muscle_sign(cell)
            if sign > 0:
                assert (row >= 0).all(), cell
            elif sign < 0:
                assert (row <= 0).all(), cell
            else:
                assert (row == 0).all(), cell

    def test_the_zero_weight_cells_are_the_32_listed(self, drive_map) -> None:
        assert len(drive_map.zero_weight_cells) == 32
        assert all(muscle_sign(c) == 0 for c in drive_map.zero_weight_cells)

    def test_every_column_is_normalised(self, drive_map) -> None:
        assert np.allclose(np.abs(drive_map.matrix).sum(axis=0), 1.0)


# ── The body ─────────────────────────────────────────────────────────────────────────────────


def _drive(direction: float = 1.0, bias: float = 0.0, level: float = 0.0) -> np.ndarray:
    drive = np.zeros(DRIVE_WIDTH)
    drive[:N_SEGMENTS] = level + bias
    drive[N_SEGMENTS : 2 * N_SEGMENTS] = level - bias
    drive[-1] = direction
    return np.clip(drive, -1.0, 1.0)


def _run(
    drive: np.ndarray,
    steps: int = 12,
    params: BodyParams | None = None,
) -> tuple[np.ndarray, float]:
    body = KinematicBody(params)
    state = new_body(10.0, 10.0, 0.0)
    for _ in range(steps):
        body.step(state, drive, 20.0)
    return state.head - np.array([10.0, 10.0]), body.heading(state)


class TestBody:
    def test_a_forward_wave_settles_into_a_straight_crawl_along_its_heading(self) -> None:
        """After a start-up transient the heading holds and the head travels along it.

        A step spans 1.5 periods, so successive steps sample opposite phases of the head's swing;
        headings are compared at the same parity.
        """
        body = KinematicBody()
        state = new_body(100.0, 100.0, 0.0)
        headings, heads = [], []
        for _ in range(30):
            body.step(state, _drive(), 1000.0)
            headings.append(body.heading(state))
            heads.append(state.head.copy())
        assert abs(headings[-1] - headings[11]) < 0.05
        travel = heads[-1] - heads[11]
        along = travel @ np.array([math.cos(headings[-1]), math.sin(headings[-1])])
        assert along > 0.95 * np.linalg.norm(travel)

    def test_a_backward_wave_moves_it_the_other_way(self) -> None:
        # A sustained reversal needs the cap lifted.
        uncapped = BodyParams(max_reversal_steps=1000)
        forward, _ = _run(_drive(direction=1.0))
        backward, _ = _run(_drive(direction=-1.0), params=uncapped)
        assert forward @ backward < 0.0
        assert np.linalg.norm(backward) > 1.0

    def test_a_reversal_is_brief_and_followed_by_forward_crawling(self) -> None:
        """A held reversal request runs one step tail-to-head, then one forward, and so on."""
        body = KinematicBody()
        state = new_body(10.0, 10.0, 0.0)
        executed = []
        for _ in range(6):
            body.step(state, _drive(direction=-1.0), 20.0)
            executed.append(state.last_reversed)
        assert executed == [True, False, True, False, True, False]

    def test_the_cap_and_the_refractory_are_parameters(self) -> None:
        """Two-step reversals with a two-step refractory alternate in pairs."""
        body = KinematicBody(BodyParams(max_reversal_steps=2, reversal_refractory_steps=2))
        state = new_body(10.0, 10.0, 0.0)
        executed = []
        for _ in range(8):
            body.step(state, _drive(direction=-1.0), 20.0)
            executed.append(state.last_reversed)
        assert executed == [True, True, False, False, True, True, False, False]

    def test_a_forward_step_is_never_reversed(self) -> None:
        body = KinematicBody()
        state = new_body(10.0, 10.0, 0.0)
        body.step(state, _drive(direction=1.0), 20.0)
        assert state.last_reversed is False

    def test_a_mildly_negative_direction_still_crawls_forward(self) -> None:
        forward, _ = _run(_drive(direction=1.0))
        mild, _ = _run(_drive(direction=-0.4))
        assert np.allclose(forward, mild)

    def test_a_dorsal_bias_turns_one_way_and_a_ventral_bias_the_other(self) -> None:
        _, straight = _run(_drive(), steps=3)
        _, left = _run(_drive(bias=0.2), steps=3)
        _, right = _run(_drive(bias=-0.2), steps=3)
        assert left - straight > 0.3
        assert right - straight < -0.3

    def test_the_record_carries_the_frame_angle(self) -> None:
        """Each sub-step's frame angle places its posture: the mid-body head line is the heading."""
        body = KinematicBody()
        state = new_body(10.0, 10.0, 0.3)
        record: list = []
        body.step(state, _drive(bias=0.2), 20.0, record=record)
        _time, curvature, _head, frame_angle = record[-1]
        assert frame_angle == state.frame_angle
        assert head_line_angle(curvature, frame_angle, distance_bl=0.5) == pytest.approx(
            body.heading(state),
            abs=0.05,
        )

    def test_drive_damps_a_segment_but_never_silences_it(self) -> None:
        """Full negative drive leaves the floor, neutral half the peak, full drive the peak."""
        levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        amplitude = [wave_amplitude(_drive(level=lv))[0] for lv in levels]
        assert amplitude == pytest.approx([0.25, 0.375, 0.5, 0.75, 1.0])
        assert wave_amplitude(_drive(level=-1.0), minimum=0.0)[0] == 0.0

    def test_a_damped_crawl_moves_less_than_the_neutral_one(self) -> None:
        damped, _ = _run(_drive(level=-0.6))
        neutral, _ = _run(_drive(level=0.0))
        assert np.linalg.norm(neutral) > np.linalg.norm(damped)

    def test_the_relayed_wave_spans_plus_and_minus_one(self) -> None:
        body = KinematicBody()
        state = new_body(0.0, 0.0, 0.0)
        record: list = []
        for _ in range(8):
            body.step(state, _drive(), 1000.0, record=record)
        head_wave = np.array([entry[1][0] for entry in record[40:]])
        peak = body.params.peak_curvature * 0.5  # the neutral drive's amplitude factor
        # The switch peaks between sub-step samples, so the sampled peak sits just under it.
        assert head_wave.max() == pytest.approx(peak, rel=0.05)
        assert head_wave.min() == pytest.approx(-peak, rel=0.05)

    def test_recording_does_not_change_the_motion(self) -> None:
        body = KinematicBody()
        plain, recorded = new_body(5.0, 5.0, 0.0), new_body(5.0, 5.0, 0.0)
        record: list = []
        for _ in range(5):
            body.step(plain, _drive(), 20.0)
            body.step(recorded, _drive(), 20.0, record=record)
        assert np.array_equal(plain.head, recorded.head)
        assert len(record) == 5 * body.params.substeps

    def test_the_head_switches_at_the_configured_period(self) -> None:
        params = BodyParams(period_s=3.0)
        body = KinematicBody(params)
        state = new_body(10.0, 10.0, 0.0)
        for _ in range(12):
            body.step(state, _drive(), 1000.0)
        flips = [t for t, _bend, _target in state.events[1:]]
        intervals = np.diff(flips)
        assert np.allclose(intervals[1:], params.period_s / 2.0, rtol=1e-6)

    def test_the_motion_is_converged_at_the_default_substeps(self) -> None:
        coarse, _ = _run(_drive())
        body = KinematicBody(BodyParams(substeps=160))
        state = new_body(10.0, 10.0, 0.0)
        for _ in range(12):
            body.step(state, _drive(), 20.0)
        fine = state.head - np.array([10.0, 10.0])
        assert np.linalg.norm(coarse - fine) < 0.06 * np.linalg.norm(fine)

    def test_the_head_stays_in_the_arena(self) -> None:
        body = KinematicBody()
        state = new_body(19.8, 10.0, 0.0)
        for _ in range(20):
            body.step(state, _drive(level=1.0), 20.0)
        assert 0.0 <= state.head[0] <= 20.0
        assert 0.0 <= state.head[1] <= 20.0

    def test_a_drive_of_the_wrong_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="25 entries"):
            KinematicBody().step(new_body(1.0, 1.0, 0.0), np.zeros(2), 20.0)


class TestEnvironment:
    def test_the_point_worm_is_the_default_and_refuses_a_drive(self) -> None:
        env = Continuous2DEnvironment(continuous=Continuous2DParams(world_size_mm=20.0))
        assert env.continuous.body_model == "point"
        with pytest.raises(RuntimeError, match="kinematic"):
            env.move_agent_body(_drive())

    def test_a_kinematic_step_moves_the_head_and_sets_the_heading(self) -> None:
        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(
                world_size_mm=20.0,
                allow_reversal=True,
                body_model="kinematic",
            ),
        )
        agent = env.agents[DEFAULT_AGENT_ID]
        start = agent.pos_continuous
        assert start is not None
        for _ in range(6):
            env.move_agent_body(_drive())
        assert agent.pos_continuous != start
        assert -math.pi <= agent.heading_rad <= math.pi
        assert agent.pos_continuous is not None
        assert env.bodies[DEFAULT_AGENT_ID].head.tolist() == list(agent.pos_continuous)
