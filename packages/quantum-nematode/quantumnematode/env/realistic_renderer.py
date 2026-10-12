"""A realistic, bright-field rendering of a recorded kinematic-body run, at any playback speed.

The drawing follows a worm on an agar plate under a bright-field microscope: pale agar, a darker
translucent body, lawns as a film that thins where it is grazed and thickens at its rim. It is drawn
from a replay recording (:mod:`quantumnematode.report.replay`), interpolated between the body's
sub-steps, so a crawl is smooth at any speed. A label always names the model and the speed, so the
picture claims no more than the 12-segment kinematic body it shows.

The shading, width profile, agar, lawn-rim and pacing formulas are adapted from Wormlight's WebGPU
renderer (https://github.com/chrisjz/wormlight, licensed under the Apache License 2.0), with
brightness inverted for bright-field and per-pixel shading replaced by pre-computed surfaces and
offset polygons. The body's width profile is the prolate ellipse of Boyle, Berri & Cohen (2012).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from quantumnematode.env.body import N_SEGMENTS, BodyParams, midline, new_body

if TYPE_CHECKING:
    import pygame

    from quantumnematode.report.replay import Replay

# Playback speeds in worm-seconds per wall-second, and the largest wall-clock gap one frame may
# advance by, so a paused or dragged window does not jump.
SPEEDS: tuple[float, ...] = (1.0, 10.0, 60.0)
MAX_WALL_STEP_S = 0.1
# Catmull-Rom points between joints, and the tips' openness in the width profile (in segments).
SPLINE_SUBDIVISIONS = 8
TIP_OPENNESS = 0.2
# A worm is about 15 times longer than it is wide.
RADIUS_FRACTION = 1.0 / 30.0
# The pharyngeal bulbs' positions along the body, and the body length on screen above which they
# are drawn.
BULB_POSITIONS: tuple[float, ...] = (0.058, 0.11)
BULB_MIN_LENGTH_PX = 60.0
# The share of agar pixels that carry a fine speck.
SPECK_FRACTION = 0.0015
# The width of a lawn's thicker rim (mm).
LAWN_RIM_MM = 0.2
# Bright-field palette.
AGAR_COLOR = (226, 222, 208)
MENISCUS_COLOR = (150, 146, 134)
LAWN_COLOR = (176, 184, 168)
BODY_EDGE_COLOR = (104, 98, 82)
BODY_CORE_COLOR = (196, 190, 168)
BODY_HIGHLIGHT_COLOR = (222, 218, 200)
BULB_COLOR = (132, 124, 102)
LABEL_COLOR = (40, 40, 40)
PANEL_COLOR = (250, 248, 240, 190)


# ── Geometry and time ────────────────────────────────────────────────────────────────────────
def catmull_rom(points: np.ndarray, subdivisions: int = SPLINE_SUBDIVISIONS) -> np.ndarray:
    """Return a uniform Catmull-Rom curve through ``points`` (N, 2), passing through each one.

    The ends are extended by mirrored control points, so the curve starts and ends on the first and
    last points.
    """
    p = np.asarray(points, dtype=float)
    padded = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = np.arange(subdivisions, dtype=float)[:, None] / subdivisions
    t2, t3 = t * t, t * t * t
    curve = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        curve.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            ),
        )
    curve.append(p[-1][None, :])
    return np.vstack(curve)


def width_profile(n_points: int, segments: int = N_SEGMENTS) -> np.ndarray:
    """Return the body's half-width along ``n_points`` evenly spaced points, as a fraction of R.

    A prolate ellipse, widest at the middle and symmetric, its tips kept open by ``TIP_OPENNESS``
    of a segment: ``sqrt(1 - x**2)`` with ``x = (s - M/2) / (M/2 + openness)``.
    """
    s = np.linspace(0.0, segments, n_points)
    x = (s - segments / 2) / (segments / 2 + TIP_OPENNESS)
    return np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))


@dataclass(frozen=True)
class Pose:
    """The body at one moment of a replay: its curvatures, head, frame angle and step."""

    curvature: np.ndarray
    head: np.ndarray
    frame_angle: float
    step: int


def interpolate(replay: Replay, time: float) -> Pose:
    """Return the body at worm-time ``time``, linear between the two sub-steps around it.

    The frame angle is interpolated along its shorter arc. Times outside the recording clamp to its
    first or last sub-step.
    """
    times = replay.substep_time
    if time <= times[0]:
        index, frac = 0, 0.0
    elif time >= times[-1]:
        index, frac = len(times) - 2, 1.0
    else:
        index = int(np.searchsorted(times, time, side="right") - 1)
        span = times[index + 1] - times[index]
        frac = float((time - times[index]) / span) if span > 0 else 0.0
    index = max(0, min(index, len(times) - 2)) if len(times) > 1 else 0
    nxt = min(index + 1, len(times) - 1)
    a0, a1 = float(replay.substep_frame[index]), float(replay.substep_frame[nxt])
    turn = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    return Pose(
        curvature=(1 - frac) * replay.substep_curvature[index]
        + frac * replay.substep_curvature[nxt],
        head=(1 - frac) * replay.substep_head[index] + frac * replay.substep_head[nxt],
        frame_angle=a0 + frac * turn,
        step=int(replay.step_of_substep[nxt if frac >= 1.0 else index]),
    )


def advance(clock: float, wall_seconds: float, speed: float, duration: float) -> float:
    """Advance a worm-time clock by ``wall_seconds`` at ``speed``, the gap capped, up to the end."""
    return min(duration, clock + min(wall_seconds, MAX_WALL_STEP_S) * speed)


def pose_midline(pose: Pose, body_length_mm: float) -> np.ndarray:
    """Return a pose's 13 world-frame joints (mm), head first, as the body module places them."""
    state = new_body(float(pose.head[0]), float(pose.head[1]), 0.0)
    state.curvature = np.asarray(pose.curvature, dtype=float)
    state.frame_angle = float(pose.frame_angle)
    return midline(state, BodyParams(body_length_mm=body_length_mm))


def label_text(speed: float, time: float) -> str:
    """Return the always-on label: the model, the speed and the worm-time."""
    return f"kinematic body · {N_SEGMENTS} segments · {speed:g}× · t = {time:6.1f} s"  # noqa: RUF001


# ── Drawing ──────────────────────────────────────────────────────────────────────────────────
def _value_noise(shape: tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
    """Return smooth value noise in [0, 1] over ``shape`` with feature size ``scale`` pixels."""
    gh, gw = int(shape[0] / scale) + 2, int(shape[1] / scale) + 2
    grid = rng.random((gh, gw))
    ys = np.arange(shape[0]) / scale
    xs = np.arange(shape[1]) / scale
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a = grid[np.ix_(y0, x0)]
    b = grid[np.ix_(y0, x0 + 1)]
    c = grid[np.ix_(y0 + 1, x0)]
    d = grid[np.ix_(y0 + 1, x0 + 1)]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def agar_rgb(world_size_mm: float, px_per_mm: float, seed: int = 7) -> np.ndarray:
    """Return the agar as an (H, W, 3) uint8 image: mottle, specks, and a meniscus at the walls."""
    size = max(1, round(world_size_mm * px_per_mm))
    rng = np.random.default_rng(seed)
    mottle = (
        0.5 * _value_noise((size, size), 1.1 * px_per_mm, rng)
        + 0.3 * _value_noise((size, size), 0.3 * px_per_mm, rng)
        + 0.2 * _value_noise((size, size), max(2.0, 0.09 * px_per_mm), rng)
    )
    shade = (mottle - 0.5) * 14.0
    specks = rng.random((size, size)) < SPECK_FRACTION
    coords = (np.arange(size) + 0.5) / px_per_mm
    wall = np.minimum.outer(
        np.minimum(coords, world_size_mm - coords),
        np.minimum(coords, world_size_mm - coords),
    )
    meniscus = np.exp(-wall / 0.6)[..., None]
    base = np.array(AGAR_COLOR, dtype=float) + shade[..., None]
    base[specks] -= 40.0
    image = base * (1 - 0.55 * meniscus) + np.array(MENISCUS_COLOR, dtype=float) * 0.55 * meniscus
    return np.clip(image, 0, 255).astype(np.uint8)


def _blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur, applied twice, of a 2-D array."""
    if radius < 1:
        return image
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    out = image
    for _ in range(2):
        out = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 0, out)
        out = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, out)
    return out


def lawn_alpha(replay: Replay, step: int, px_per_mm: float) -> np.ndarray | None:
    """Return the lawn film's opacity (H, W) in [0, 1] at ``step``; None without lawns.

    Each cell's remaining density sets the film's thickness, smoothed across cells, and a band
    inside each lawn's edge is thicker, as a real lawn's rim is.
    """
    if replay.lawn_cells is None or replay.step_lawn_density is None or replay.lawn_centres is None:
        return None
    size = max(1, round(replay.world_size_mm * px_per_mm))
    density = replay.step_lawn_density[min(step, len(replay.step_lawn_density) - 1)]
    raster = np.zeros((size, size))
    coverage = np.zeros((size, size))
    half = (replay.lawn_cell_mm or 1.0) / 2
    for (cx, cy), value in zip(replay.lawn_cells, density, strict=True):
        x0, x1 = round((cx - half) * px_per_mm), round((cx + half) * px_per_mm)
        y0 = round((replay.world_size_mm - cy - half) * px_per_mm)
        y1 = round((replay.world_size_mm - cy + half) * px_per_mm)
        raster[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = float(value)
        coverage[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = 1.0
    blur = max(1, round(0.25 * px_per_mm))
    # Normalised by the blurred coverage, so each cell's density reaches the lawn's circular edge
    # rather than fading where the square cells stop short of it.
    film = _blur(raster, blur) / np.maximum(_blur(coverage, blur), 1e-3)
    ys, xs = np.mgrid[0:size, 0:size]
    wx, wy = (xs + 0.5) / px_per_mm, replay.world_size_mm - (ys + 0.5) / px_per_mm
    radius = replay.lawn_radius_mm or 0.0
    inside = np.zeros((size, size), dtype=bool)
    rim = np.zeros((size, size))
    for cx, cy in replay.lawn_centres:
        distance = np.hypot(wx - cx, wy - cy)
        inside |= distance <= radius
        band = np.exp(-np.clip(radius - distance, 0, None) / LAWN_RIM_MM)
        rim = np.maximum(rim, band * (distance <= radius))
    return np.clip((0.5 * film + 0.5 * rim) * inside, 0.0, 1.0)


class RealisticRenderer:
    """Draw replay frames into a pygame surface: plate, lawns, body, label and status."""

    def __init__(
        self,
        replay: Replay,
        size_px: int = 800,
        *,
        texture_px_per_mm: float = 40.0,
    ) -> None:
        import pygame

        self._pg = pygame
        self.replay = replay
        self.size_px = size_px
        self._texture_ppm = texture_px_per_mm
        self._agar = agar_rgb(replay.world_size_mm, texture_px_per_mm)
        self._plate: Any = None
        self._plate_step = -1
        self.follow = True
        # The width of the followed view, in body lengths.
        self.view_body_lengths = 4.0
        pygame.font.init()
        self._font = pygame.font.SysFont("helvetica", 16)

    def _plate_surface(self, step: int) -> pygame.Surface:
        """Return the agar with the lawns at ``step``, rebuilt only when the step changes."""
        if self._plate is not None and step == self._plate_step:
            return self._plate
        image = self._agar.astype(float)
        alpha = lawn_alpha(self.replay, step, self._texture_ppm)
        if alpha is not None:
            image = (
                image * (1 - alpha[..., None])
                + np.array(LAWN_COLOR, dtype=float) * alpha[..., None]
            )
        array = np.clip(image, 0, 255).astype(np.uint8)
        self._plate = self._pg.surfarray.make_surface(array.transpose(1, 0, 2))
        self._plate_step = step
        return self._plate

    def view(self, head: np.ndarray) -> tuple[float, float, float]:
        """Return the view as (left mm, top mm, mm across), following the head when ``follow``."""
        world = self.replay.world_size_mm
        if not self.follow:
            return 0.0, world, world
        span = min(world, self.view_body_lengths * self.replay.body_length_mm)
        left = float(np.clip(head[0] - span / 2, 0.0, world - span))
        top = float(np.clip(head[1] + span / 2, span, world))
        return left, top, span

    def draw(self, surface: pygame.Surface, time: float, speed: float) -> None:
        """Draw the frame at worm-time ``time`` onto ``surface``."""
        pose = interpolate(self.replay, time)
        left, top, span = self.view(pose.head)
        plate = self._plate_surface(pose.step)
        tex = self._texture_ppm
        rect = self._pg.Rect(
            round(left * tex),
            round((self.replay.world_size_mm - top) * tex),
            max(1, round(span * tex)),
            max(1, round(span * tex)),
        )
        rect = rect.clip(plate.get_rect())
        view = self._pg.transform.smoothscale(plate.subsurface(rect), (self.size_px, self.size_px))
        surface.blit(view, (0, 0))
        ppm = self.size_px / span
        joints = pose_midline(pose, self.replay.body_length_mm)
        pixels = np.column_stack([(joints[:, 0] - left) * ppm, (top - joints[:, 1]) * ppm])
        self._draw_body(surface, pixels, self.replay.body_length_mm * RADIUS_FRACTION * ppm)
        self._draw_text(surface, pose, time, speed)

    def _draw_body(self, surface: pygame.Surface, joints_px: np.ndarray, radius_px: float) -> None:
        """Draw the body: a halo, a dark edge, a lighter core and a highlight, then the bulbs."""
        curve = catmull_rom(joints_px)
        widths = width_profile(len(curve)) * max(radius_px, 1.0)
        tangent = np.gradient(curve, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        overlay = self._pg.Surface(surface.get_size(), self._pg.SRCALPHA)
        for scale, colour in (
            (1.8, (*BODY_EDGE_COLOR, 40)),
            (1.0, (*BODY_EDGE_COLOR, 235)),
            (0.7, (*BODY_CORE_COLOR, 235)),
            (0.25, (*BODY_HIGHLIGHT_COLOR, 160)),
        ):
            offset = normal * (widths * scale)[:, None]
            outline = np.vstack([curve + offset, (curve - offset)[::-1]])
            self._pg.draw.polygon(overlay, colour, [tuple(p) for p in outline])
        length_px = float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))
        if length_px >= BULB_MIN_LENGTH_PX:
            cumulative = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))],
            )
            for position in BULB_POSITIONS:
                index = int(np.searchsorted(cumulative, position * length_px))
                centre = tuple(int(v) for v in curve[min(index, len(curve) - 1)])
                bulb = max(2, round(0.55 * widths[min(index, len(widths) - 1)]))
                self._pg.draw.circle(overlay, (*BULB_COLOR, 200), centre, bulb)
        surface.blit(overlay, (0, 0))

    def _draw_text(self, surface: pygame.Surface, pose: Pose, time: float, speed: float) -> None:
        """Draw the label and the step's status on a pale panel at the top-left."""
        replay = self.replay
        step = min(pose.step, len(replay.step_intake) - 1) if len(replay.step_intake) else 0
        lines = [label_text(speed, time)]
        if len(replay.step_intake):
            lawn = "on a lawn" if bool(replay.step_on_lawn[step]) else "off lawns"
            intake, satiety = replay.step_intake[step], replay.step_satiety[step]
            lines.append(f"intake {intake:.4f} ({lawn})  satiety {satiety:.0f}")
            serotonin, pdf = replay.step_modulators[step]
            if not math.isnan(float(serotonin)):
                lines.append(f"serotonin {serotonin:.3f}  PDF {pdf:.3f}")
        if len(replay.step_reversed) and bool(replay.step_reversed[step]):
            lines.append("reversing")
        rendered = [self._font.render(text, True, LABEL_COLOR) for text in lines]  # noqa: FBT003
        width = max(r.get_width() for r in rendered) + 16
        height = sum(r.get_height() + 2 for r in rendered) + 12
        panel = self._pg.Surface((width, height), self._pg.SRCALPHA)
        panel.fill(PANEL_COLOR)
        y = 6
        for r in rendered:
            panel.blit(r, (8, y))
            y += r.get_height() + 2
        surface.blit(panel, (8, 8))
