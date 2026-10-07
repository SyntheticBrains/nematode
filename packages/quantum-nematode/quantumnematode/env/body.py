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


@dataclass(frozen=True)
class BodyParams:
    """The body's geometry, its generator and its drag.

    ``peak_curvature``, ``steering_gain`` and ``drag_anisotropy`` are placeholders until they are
    checked against their sources and calibrated once on the MLP positive control, then frozen
    across every arm.
    """

    body_length_mm: float = 1.0
    step_seconds: float = 5.0
    substeps: int = 20
    period_s: float = 3.1
    wavelength_bl: float = 0.65
    switch_threshold: float = 0.5
    peak_curvature: float = 6.0
    steering_gain: float = 1.0
    drag_anisotropy: float = 20.0

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


def new_body(x: float, y: float, heading: float) -> BodyState:
    """Return a straight body with its head at ``(x, y)`` facing ``heading``."""
    return BodyState(head=np.array([x, y], dtype=float), frame_angle=heading)


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
        values = np.where(times < 0.0, 0.0, values)
        return values if forward else values[::-1]

    def curvature(self, state: BodyState, drive: np.ndarray, *, forward: bool) -> np.ndarray:
        """Return each segment's curvature, in units of one over the body length."""
        p = self.params
        dorsal, ventral = drive[:N_SEGMENTS], drive[N_SEGMENTS : 2 * N_SEGMENTS]
        amplitude = (1.0 + (dorsal + ventral) / 2.0) / 2.0
        bias = (dorsal - ventral) / 2.0
        return p.peak_curvature * amplitude * self._relayed(state, forward=forward) + (
            p.steering_gain * bias
        )

    def step(self, state: BodyState, drive: np.ndarray, world_size_mm: float) -> None:
        """Advance ``state`` by one environment step under ``drive``, clamped to the arena."""
        p = self.params
        if drive.shape != (DRIVE_WIDTH,):
            msg = f"drive must have {DRIVE_WIDTH} entries, got {drive.shape}"
            raise ValueError(msg)
        drive = np.clip(drive, -1.0, 1.0)
        forward = drive[-1] >= 0.0
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

    def heading(self, state: BodyState) -> float:
        """Return the direction from the body's midpoint to the head, in the world frame."""
        rho, _ = _shape(state.curvature, self.params.body_length_mm / N_SEGMENTS)
        mid = rho[N_SEGMENTS // 2]
        direction = _rotate(-mid[None, :], state.frame_angle)[0]
        return math.atan2(direction[1], direction[0])
