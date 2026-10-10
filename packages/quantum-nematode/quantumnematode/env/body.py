"""A kinematic segmented body: a body-level rhythm generator moved by resistive-force theory.

The body carries the rhythm and the brain sets segmental drive. A head relaxation switch sets the
phase: the head's normalised bend relaxes toward a target of +1 or -1 and the target flips when
the bend crosses a threshold, so the period follows from the relaxation time constant. A relay
passes the head's wave down the body with a fixed delay per segment, front to back; when the
brain's direction channel is negative the relay runs from the tail instead, which reverses the
wave. Each segment's curvature is the wave scaled by an amplitude and offset by a bias, both set
from the drive.

Movement follows from the change of shape. Each segment feels anisotropic drag, larger across its
length than along it, and the body's rigid translation and rotation are whatever makes the net
drag force and torque vanish: the force-free crawler of resistive-force theory. The head carries
the sensors; the heading is the direction from the body's midpoint to the head, which is steady at
the wave's scale where the head's own tangent swings with every bend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

N_SEGMENTS = 12
"""Body segments, head to tail; each pools two muscle positions per quadrant."""

DRIVE_WIDTH = 2 * N_SEGMENTS + 1
"""The drive action: dorsal drive per segment, ventral drive per segment, then direction."""

MIN_WAVE_AMPLITUDE = 0.25
"""The least share of the peak wave any segment carries, however strongly its drive damps it."""


@dataclass(frozen=True)
class BodyParams:
    """The body's geometry, its generator and its drag.

    The crawl's period (0.30 Hz) and wavelength (0.65 body lengths) are Fang-Yen et al. 2010's
    measurements on agar; the drag anisotropy (about 10) is Shen et al. 2012's and Rabets et al.
    2014's. ``peak_curvature`` is reached at full drive, so the neutral drive's half of it is the
    crawl's measured amplitude, about 9 body-lengths^-1, and full drive an omega-shaped posture.
    ``steering_gain`` has no direct measurement. It was calibrated once on the MLP positive control,
    the best of 0.5, 1, 2 and 4 by plateau success with near ties going to the smaller gain, and is
    frozen at 2 across every arm.

    The wave runs tail-to-head only below ``-reversal_threshold`` on the direction channel, so noise
    around a forward-leaning direction does not flip it. A worm's reversals are brief, one to three
    head swings, before it resumes forward crawling, so a reversal lasts at most
    ``max_reversal_steps`` steps and is followed by at least ``reversal_refractory_steps`` forward
    steps before the next. One step is 1.5 head swings, a short reversal; the long reversals that
    precede an omega turn are not reachable at one step.

    Drive damps a segment's wave down to ``min_wave_amplitude`` of the peak but never silences it:
    in forward crawling the wave propagates along the whole body through proprioceptive coupling
    (Wen et al. 2012), so no segment stops undulating while the rest crawl.
    """

    body_length_mm: float = 1.0
    step_seconds: float = 5.0
    substeps: int = 20
    period_s: float = 1.0 / 0.30
    wavelength_bl: float = 0.65
    switch_threshold: float = 0.5
    peak_curvature: float = 18.0
    steering_gain: float = 2.0
    drag_anisotropy: float = 10.0
    reversal_threshold: float = 0.5
    max_reversal_steps: int = 1
    reversal_refractory_steps: int = 1
    min_wave_amplitude: float = MIN_WAVE_AMPLITUDE

    @property
    def relax_tau(self) -> float:
        """The head switch's time constant, set so the switch completes ``period_s``."""
        theta = self.switch_threshold
        return self.period_s / (2.0 * math.log((1.0 + theta) / (1.0 - theta)))

    @property
    def relay_delay(self) -> float:
        """The wave's delay per segment, in seconds, giving ``wavelength_bl`` body lengths."""
        return self.period_s / (N_SEGMENTS * self.wavelength_bl)


@dataclass
class BodyState:
    """One body's state: its head pose, its generator and its recent head wave."""

    head: np.ndarray
    frame_angle: float
    wave: float = 0.0
    target: float = 1.0
    time: float = 0.0
    # Every switch since the oldest one the relay can still reach: (time, bend, new target). Between
    # switches the bend has a closed form, so the relay reads the past wave exactly at any delay.
    events: list[tuple[float, float, float]] = field(default_factory=lambda: [(0.0, 0.0, 1.0)])
    curvature: np.ndarray = field(default_factory=lambda: np.zeros(N_SEGMENTS))
    # Consecutive steps just run tail-to-head, and head-to-tail; a new body may reverse at once.
    reversal_run: int = 0
    forward_run: int = 1 << 30
    last_reversed: bool = False


def new_body(x: float, y: float, heading: float) -> BodyState:
    """Return a straight body with its head at ``(x, y)`` facing ``heading``."""
    return BodyState(head=np.array([x, y], dtype=float), frame_angle=heading)


def wave_amplitude(drive: np.ndarray, minimum: float = MIN_WAVE_AMPLITUDE) -> np.ndarray:
    """Each segment's share of the peak wave under ``drive``.

    The mean of a segment's dorsal and ventral drive sets it: neutral drive gives one half, full
    drive one, and full negative drive ``minimum``. In forward crawling the wave propagates along
    the whole body through proprioceptive coupling, so drive damps a segment's wave but cannot
    silence it.
    """
    dorsal, ventral = drive[..., :N_SEGMENTS], drive[..., N_SEGMENTS : 2 * N_SEGMENTS]
    level = (dorsal + ventral) / 2.0
    return 0.5 + np.where(level >= 0.0, 0.5, 0.5 - minimum) * level


def _segment_angles(curvature: np.ndarray) -> np.ndarray:
    """Each segment's angle relative to the head segment; each joint bends by ``κ_i / N``."""
    return -np.concatenate(([0.0], np.cumsum(curvature[:-1] / N_SEGMENTS)))


def _shape(curvature: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return segment midpoints and unit tangents in the head frame; the head is at the origin.

    Tangents point headward; the body extends from the head toward negative ``x`` when straight.
    """
    angles = _segment_angles(curvature)
    tangents = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    joints = np.vstack([[0.0, 0.0], -np.cumsum(tangents * ds, axis=0)])
    midpoints = (joints[:-1] + joints[1:]) / 2.0
    return midpoints, tangents


def head_line_angle(curvature: np.ndarray, frame_angle: float, distance_bl: float = 0.2) -> float:
    """Return the world-frame direction to the head from the midline ``distance_bl`` behind it.

    The point lies on the midline at that arc length from the head, in body lengths; the line runs
    from it to the head. Its turning across a head swing is how an omega turn is measured.
    """
    ds = 1.0 / N_SEGMENTS
    angles = _segment_angles(curvature)
    tangents = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    joints = np.vstack([[0.0, 0.0], -np.cumsum(tangents * ds, axis=0)])
    index = min(int(distance_bl / ds), N_SEGMENTS - 1)
    point = joints[index] - tangents[index] * (distance_bl - index * ds)
    direction = _rotate(-point[None, :], frame_angle)[0]
    return math.atan2(direction[1], direction[0])


def _rotate(vectors: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return vectors @ np.array([[c, s], [-s, c]])


def _rigid_velocity(
    rho: np.ndarray,
    tangents: np.ndarray,
    shape_velocity: np.ndarray,
    drag_t: float,
    drag_n: float,
) -> tuple[np.ndarray, float]:
    """Solve for the translation and rotation that zero the net drag force and torque.

    All vectors are in one frame. A segment at ``r`` moving at ``V + omega * perp(r) + u`` feels
    drag ``-(c_t (v.t) t + c_n (v.n) n)``; force and torque balance is linear in ``(V, omega)``.
    """
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    drag = drag_t * np.einsum("ni,nj->nij", tangents, tangents) + drag_n * np.einsum(
        "ni,nj->nij",
        normals,
        normals,
    )
    perp = np.stack([-rho[:, 1], rho[:, 0]], axis=1)  # z-hat cross r
    # Per segment, the force from (Vx, Vy, Ω) and from the shape's own motion, then summed into the
    # force rows and the torque row of one 3 x 3 system.
    cols = np.concatenate([drag, np.einsum("nij,nj->ni", drag, perp)[:, :, None]], axis=2)
    force_u = np.einsum("nij,nj->ni", drag, shape_velocity)
    a = np.vstack([cols.sum(axis=0), np.einsum("ni,nij->j", perp, cols)])
    b = -np.concatenate([force_u.sum(axis=0), [np.einsum("ni,ni->", perp, force_u)]])
    solution = np.linalg.solve(a, b)
    return solution[:2], float(solution[2])


class KinematicBody:
    """Advance bodies one environment step from a drive vector."""

    def __init__(self, params: BodyParams | None = None) -> None:
        self.params = params or BodyParams()

    def _head_wave(self, state: BodyState, dt: float) -> None:
        """Advance the head switch by ``dt`` exactly and record the head's wave.

        Between switches the bend relaxes exponentially toward its target, so the moment it crosses
        the threshold is solved in closed form and the target flips there, not at the end of the
        sub-step: the period then holds at any sub-step length.
        """
        p = self.params
        tau, theta = p.relax_tau, p.switch_threshold
        remaining = dt
        while remaining > 0.0:
            target = state.target
            gap, at_switch = state.wave - target, (theta - 1.0) * target
            crossing = tau * math.log(gap / at_switch) if gap * at_switch > 0.0 else math.inf
            if crossing >= remaining:
                state.wave = target + gap * math.exp(-remaining / tau)
                remaining = 0.0
            else:
                elapsed = max(crossing, 0.0)
                state.wave = theta * target
                state.target = -target
                state.events.append((state.time + (dt - remaining) + elapsed, state.wave, -target))
                remaining -= elapsed
        state.time += dt
        horizon = state.time - N_SEGMENTS * p.relay_delay - dt
        while len(state.events) > 1 and state.events[1][0] <= horizon:
            state.events.pop(0)

    def _relayed(self, state: BodyState, *, forward: bool) -> np.ndarray:
        """Return each segment's wave value, relayed from the head (forward) or the tail.

        Each segment reads the head's bend as it was a fixed delay per segment ago, evaluated in
        closed form from the last switch before that moment; before the body first moved it is 0.
        """
        times = state.time - np.arange(N_SEGMENTS) * self.params.relay_delay
        events = np.asarray(state.events)
        index = np.searchsorted(events[:, 0], times, side="right") - 1
        start, bend, target = events[np.maximum(index, 0)].T
        values = target + (bend - target) * np.exp(-(times - start) / self.params.relax_tau)
        # The switch flips at +-threshold, so the bend spans +-threshold; dividing gives a wave in
        # [-1, 1], the scale the amplitude mapping assumes.
        values = np.where(times < 0.0, 0.0, values) / self.params.switch_threshold
        return values if forward else values[::-1]

    def curvature(self, state: BodyState, drive: np.ndarray, *, forward: bool) -> np.ndarray:
        """Return each segment's curvature, in units of one over the body length."""
        p = self.params
        dorsal, ventral = drive[:N_SEGMENTS], drive[N_SEGMENTS : 2 * N_SEGMENTS]
        bias = (dorsal - ventral) / 2.0
        amplitude = wave_amplitude(drive, p.min_wave_amplitude)
        return p.peak_curvature * amplitude * self._relayed(state, forward=forward) + (
            p.steering_gain * bias
        )

    def step(
        self,
        state: BodyState,
        drive: np.ndarray,
        world_size_mm: float,
        record: list[tuple[float, np.ndarray, np.ndarray, float]] | None = None,
    ) -> None:
        """Advance ``state`` by one environment step under ``drive``, clamped to the arena.

        ``record``, when given, receives each sub-step's ``(time, curvature, head, frame_angle)``,
        copies that leave the motion untouched.
        """
        p = self.params
        if drive.shape != (DRIVE_WIDTH,):
            msg = f"drive must have {DRIVE_WIDTH} entries, got {drive.shape}"
            raise ValueError(msg)
        drive = np.clip(drive, -1.0, 1.0)
        forward = not self._reverses(state, requested=bool(drive[-1] < -p.reversal_threshold))
        ds = p.body_length_mm / N_SEGMENTS
        dt = p.step_seconds / p.substeps
        rho1, _ = _shape(state.curvature, ds)
        for _ in range(p.substeps):
            rho0 = rho1
            self._head_wave(state, dt)
            after = self.curvature(state, drive, forward=forward)
            rho1, _ = _shape(after, ds)
            shape_velocity = (rho1 - rho0) / dt
            # The drag is taken at the sub-step's mid-shape, a second-order step in the pose.
            rho_mid, tangents_mid = _shape((state.curvature + after) / 2.0, ds)
            velocity, omega = _rigid_velocity(
                rho_mid,
                tangents_mid,
                shape_velocity,
                1.0,
                p.drag_anisotropy,
            )
            state.head = state.head + _rotate(velocity[None, :], state.frame_angle)[0] * dt
            state.frame_angle = state.frame_angle + omega * dt
            state.head = np.clip(state.head, 0.0, world_size_mm)
            state.curvature = after
            if record is not None:
                record.append((state.time, after.copy(), state.head.copy(), state.frame_angle))

    def _reverses(self, state: BodyState, *, requested: bool) -> bool:
        """Decide whether this step runs tail-to-head, and advance the reversal bookkeeping."""
        p = self.params
        if state.reversal_run > 0:
            reverse = requested and state.reversal_run < p.max_reversal_steps
        else:
            reverse = requested and state.forward_run >= p.reversal_refractory_steps
        state.reversal_run = state.reversal_run + 1 if reverse else 0
        state.forward_run = 0 if reverse else state.forward_run + 1
        state.last_reversed = reverse
        return reverse

    def heading(self, state: BodyState) -> float:
        """Return the direction from the body's midpoint to the head, in the world frame."""
        rho, _ = _shape(state.curvature, self.params.body_length_mm / N_SEGMENTS)
        mid = rho[N_SEGMENTS // 2]
        direction = _rotate(-mid[None, :], state.frame_angle)[0]
        return math.atan2(direction[1], direction[0])
