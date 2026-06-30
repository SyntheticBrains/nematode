"""Pygame-based graphical renderer for the Pixel theme.

Renders the simulation in a Pygame window with layered surfaces:
1. Background (soil)
2. Temperature zone overlays
3. Oxygen zone overlays
4. Toxic zone overlays
5. Entities (food, predators, nematode) - drawn with transparency so zones show through
6. UI status bar
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from quantumnematode.env.sprites import (
    AGENT_COLOR_PALETTE,
    CELL_SIZE,
    SOIL_COLOR,
    create_dead_agent_overlay,
    create_sprites,
    create_tinted_head_sprites,
    create_zone_overlay,
    draw_body_segment,
    tint_body_colors,
    zone_overlay_color,
)
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.env.env import DynamicForagingEnvironment, Viewport


@dataclass(frozen=True)
class AgentRenderState:
    """Lightweight snapshot of per-agent state for rendering.

    Built by the orchestrator before each render call. Keeps the renderer
    decoupled from QuantumNematodeAgent internals.
    """

    agent_id: str
    position: tuple[int, int]
    body: tuple[tuple[int, int], ...]
    direction: str  # Direction enum .value ("up", "down", "left", "right")
    alive: bool
    hp: float
    max_hp: float
    foods_collected: int
    satiety: float
    max_satiety: float
    color_index: int  # index into AGENT_COLOR_PALETTE (cycles via % 8)


@dataclass(frozen=True)
class ContinuousRenderState:
    """Lightweight snapshot of continuous-substrate state for rendering.

    Built by the agent before each continuous render call (mirrors the
    ``AgentRenderState`` decoupling pattern). Keeps ``Continuous2DRenderer``
    decoupled from the agent's internal sensor objects: the renderer reads pose,
    klinotaxis sample points, and adaptive-sensor state from this snapshot and the
    environment only — never from agent private attributes.
    """

    pos: tuple[float, float]
    heading_rad: float
    # Klinotaxis head-sweep sample points (left/right of heading at `sweep`). The
    # continuous agent always populates these (the sweep geometry is meaningful
    # regardless of the active sensing mode); the optional type lets a caller pass
    # None to suppress the sample markers.
    left_sample: tuple[float, float] | None
    right_sample: tuple[float, float] | None
    sweep: float
    # Adaptive chemosensory sensor state (None when the sensor is disabled).
    adaptive_background: float | None
    adaptive_readout: float | None
    adaptive_mode: str | None
    # Status-bar fields (mirror the grid renderer's status bar).
    step: int
    max_steps: int
    foods_collected: int
    target_foods: int
    health: float
    max_health: float
    satiety: float
    max_satiety: float
    in_danger: bool
    temperature: float | None
    zone_name: str | None
    oxygen: float | None
    oxygen_zone_name: str | None


# Minimum pheromone concentration to render an overlay cell
PHEROMONE_RENDER_THRESHOLD = 0.01

# Status bar configuration
STATUS_BAR_HEIGHT = 120
STATUS_FONT_SIZE = 14
STATUS_BG_COLOR = (30, 25, 22)
STATUS_TEXT_COLOR = (200, 200, 200)
STATUS_DANGER_COLOR = (220, 60, 40)
STATUS_SAFE_COLOR = (80, 200, 80)
STATUS_SESSION_COLOR = (170, 170, 220)
STATUS_PADDING = 8

# Window title
WINDOW_TITLE = "Quantum Nematode - Pixel Theme"
WINDOW_TITLE_CONTINUOUS = "Quantum Nematode - Continuous-2D"


def build_run_status_lines(  # noqa: PLR0913
    *,
    step: int,
    max_steps: int,
    foods_collected: int,
    target_foods: int,
    health: float,
    max_health: float,
    satiety: float,
    max_satiety: float,
    in_danger: bool,
    temperature: float | None,
    zone_name: str | None,
    oxygen: float | None,
    oxygen_zone_name: str | None,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Build the run-level status-bar lines shared by the grid and continuous renderers.

    Returns the ``(text, colour)`` lines for the run section (step, food, HP,
    satiety, danger state, temperature, oxygen) so both renderers format these
    fields identically. Session-level text and renderer-specific extras (e.g. the
    continuous adaptive-sensor readout) are appended by the caller.
    """
    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("Run:", STATUS_TEXT_COLOR),
        (f"Step: {step}/{max_steps}", STATUS_TEXT_COLOR),
        (f"Food: {foods_collected}/{target_foods}", STATUS_TEXT_COLOR),
        (f"HP: {health:.0f}/{max_health:.0f}", STATUS_TEXT_COLOR),
        (f"Satiety: {satiety:.0f}/{max_satiety:.0f}", STATUS_TEXT_COLOR),
    ]
    danger_text = "IN DANGER" if in_danger else "SAFE"
    danger_color = STATUS_DANGER_COLOR if in_danger else STATUS_SAFE_COLOR
    lines.append((f"Status: {danger_text}", danger_color))
    if temperature is not None and zone_name:
        lines.append((f"Temp: {temperature:.1f}C ({zone_name})", STATUS_TEXT_COLOR))
    if oxygen is not None and oxygen_zone_name:
        lines.append((f"O2: {oxygen:.1f}% ({oxygen_zone_name})", STATUS_TEXT_COLOR))
    return lines


def wrap_status_line(
    font: Any,  # noqa: ANN401
    text: str,
    color: tuple[int, int, int],
    max_width: int,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Word-wrap a status line to ``max_width`` pixels using ``font``.

    Falls back to character-level breaking for single tokens wider than
    ``max_width``. Mirrors the grid renderer's ``_wrap_text`` so a narrow window
    overflows status text onto extra rows instead of clipping it on the right.
    """
    antialias = True
    words = text.split()
    result: list[tuple[str, tuple[int, int, int]]] = []
    current_line = ""
    for word in words:
        candidate = f"{current_line} {word}".strip()
        if font.render(candidate, antialias, color).get_width() <= max_width:
            current_line = candidate
            continue
        if current_line:
            result.append((current_line, color))
        if font.render(word, antialias, color).get_width() > max_width:
            chunk = ""
            for ch in word:
                if font.render(chunk + ch, antialias, color).get_width() > max_width:
                    if chunk:
                        result.append((chunk, color))
                    chunk = ch
                else:
                    chunk += ch
            current_line = chunk
        else:
            current_line = word
    if current_line:
        result.append((current_line, color))
    return result or [("", color)]


class PygameRenderer:
    """Renders the nematode simulation using Pygame with layered surfaces.

    Parameters
    ----------
    viewport_size : tuple[int, int]
        Grid dimensions of the viewport (width, height in cells).
    cell_size : int
        Pixel size of each grid cell (default 32).
    """

    def __init__(
        self,
        viewport_size: tuple[int, int] = (11, 11),
        cell_size: int = CELL_SIZE,
    ) -> None:
        import pygame

        self._pg = pygame
        self._cell_size = cell_size
        self._viewport_size = viewport_size

        # Window dimensions
        self._width = viewport_size[0] * cell_size
        self._height = viewport_size[1] * cell_size + STATUS_BAR_HEIGHT

        # Initialize Pygame
        pygame.init()
        self._screen = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption(WINDOW_TITLE)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", STATUS_FONT_SIZE)

        # Create sprite assets
        self._sprites = create_sprites(pygame)

        # Pre-create zone overlays
        self._zone_overlays: dict[str, Any] = {}
        for name in (
            "lethal_cold",
            "danger_cold",
            "discomfort_cold",
            "comfort",
            "discomfort_hot",
            "danger_hot",
            "lethal_hot",
            "toxic",
            # Oxygen zones
            "lethal_hypoxia",
            "danger_hypoxia",
            "comfort_oxygen",
            "danger_hyperoxia",
            "lethal_hyperoxia",
        ):
            self._zone_overlays[name] = create_zone_overlay(pygame, name)

        self._closed = False
        self._last_status_line_count = 0

        # Multi-agent state (lazily populated)
        self._tinted_sprites: dict[int, dict[str, Any]] = {}
        self._tinted_body_colors: dict[
            int,
            tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        ] = {}
        self._dead_overlay = create_dead_agent_overlay(pygame)
        self._pheromone_overlay_enabled = False
        logger.info(
            f"PygameRenderer initialized: {self._width}x{self._height} "
            f"({viewport_size[0]}x{viewport_size[1]} cells @ {cell_size}px)",
        )

    @property
    def closed(self) -> bool:
        """Whether the window has been closed."""
        return self._closed

    def _resize_if_needed(self, line_count: int) -> None:
        """Resize the window if the status bar needs more space."""
        if line_count <= self._last_status_line_count:
            return
        self._last_status_line_count = line_count
        line_height = STATUS_FONT_SIZE + 2
        needed_height = line_count * line_height + 8
        if needed_height > STATUS_BAR_HEIGHT:
            self._height = self._viewport_size[1] * self._cell_size + needed_height
            self._screen = self._pg.display.set_mode((self._width, self._height))

    def pump_events(self) -> bool:
        """Process Pygame events. Returns False if window was closed."""
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                self.close()
                return False
        return True

    def render_frame(  # noqa: PLR0913
        self,
        env: DynamicForagingEnvironment,
        *,
        step: int = 0,
        max_steps: int = 0,
        foods_collected: int = 0,
        target_foods: int = 0,
        health: float = 0.0,
        max_health: float = 0.0,
        satiety: float = 0.0,
        max_satiety: float = 0.0,
        in_danger: bool = False,
        temperature: float | None = None,
        zone_name: str | None = None,
        oxygen: float | None = None,
        oxygen_zone_name: str | None = None,
        session_text: str | None = None,
    ) -> None:
        """Render one complete frame."""
        if self._closed:
            return

        if not self.pump_events():
            return

        # Estimate total status lines and resize window if needed
        session_lines = 0
        if session_text:
            session_lines = (
                sum(
                    1
                    for line in session_text.strip().splitlines()
                    if line.strip() and not line.strip().startswith("--")
                )
                + 1
            )  # +1 for separator
        # 8 = max run-level lines (title, step, food, hp, satiety, status, temp, o2)
        self._resize_if_needed(session_lines + 8)

        viewport = env.get_viewport_bounds()

        # Clear screen
        self._screen.fill(STATUS_BG_COLOR)

        self._render_background(viewport)
        self._render_temperature_zones(env, viewport)
        self._render_oxygen_zones(env, viewport)
        self._render_toxic_zones(env, viewport)
        self._render_entities(env, viewport)
        self._render_status_bar(
            step=step,
            max_steps=max_steps,
            foods_collected=foods_collected,
            target_foods=target_foods,
            health=health,
            max_health=max_health,
            satiety=satiety,
            max_satiety=max_satiety,
            in_danger=in_danger,
            temperature=temperature,
            zone_name=zone_name,
            oxygen=oxygen,
            oxygen_zone_name=oxygen_zone_name,
            session_text=session_text,
        )

        self._pg.display.flip()
        self._clock.tick(30)  # Cap at 30 FPS

    def _cell_to_pixel(self, grid_x: int, grid_y: int, viewport: Viewport) -> tuple[int, int]:
        """Convert grid coordinates to pixel position on screen.

        The grid's y-axis points up (y=0 is bottom) but Pygame's y-axis points
        down (y=0 is top), so we invert.
        """
        min_x, min_y, _, max_y = viewport
        rel_x = grid_x - min_x
        height = max_y - min_y
        rel_y = height - 1 - (grid_y - min_y)
        return rel_x * self._cell_size, rel_y * self._cell_size

    def _render_background(self, viewport: Viewport) -> None:
        """Fill grid area with soil tiles."""
        min_x, min_y, max_x, max_y = viewport
        soil = self._sprites["empty"]
        for gy in range(min_y, max_y):
            for gx in range(min_x, max_x):
                px, py = self._cell_to_pixel(gx, gy, viewport)
                self._screen.blit(soil, (px, py))

    def _render_temperature_zones(
        self,
        env: DynamicForagingEnvironment,
        viewport: Viewport,
    ) -> None:
        """Render temperature zone overlays on all cells."""
        if not env.thermotaxis.enabled or env.temperature_field is None:
            return

        from quantumnematode.env.temperature import TemperatureZone, TemperatureZoneThresholds

        thresholds = TemperatureZoneThresholds(
            comfort_delta=env.thermotaxis.comfort_delta,
            discomfort_delta=env.thermotaxis.discomfort_delta,
            danger_delta=env.thermotaxis.danger_delta,
        )

        zone_to_overlay_name = {
            TemperatureZone.LETHAL_COLD: "lethal_cold",
            TemperatureZone.DANGER_COLD: "danger_cold",
            TemperatureZone.DISCOMFORT_COLD: "discomfort_cold",
            TemperatureZone.COMFORT: "comfort",
            TemperatureZone.DISCOMFORT_HOT: "discomfort_hot",
            TemperatureZone.DANGER_HOT: "danger_hot",
            TemperatureZone.LETHAL_HOT: "lethal_hot",
        }

        min_x, min_y, max_x, max_y = viewport
        for gy in range(min_y, max_y):
            for gx in range(min_x, max_x):
                temp = env.temperature_field.get_temperature((gx, gy))
                zone = env.temperature_field.get_zone(
                    temp,
                    env.thermotaxis.cultivation_temperature,
                    thresholds,
                )
                overlay_name = zone_to_overlay_name.get(zone)
                if overlay_name and overlay_name != "comfort":
                    overlay = self._zone_overlays[overlay_name]
                    px, py = self._cell_to_pixel(gx, gy, viewport)
                    self._screen.blit(overlay, (px, py))

    def _render_oxygen_zones(
        self,
        env: DynamicForagingEnvironment,
        viewport: Viewport,
    ) -> None:
        """Render oxygen zone overlays on all cells."""
        if not env.aerotaxis.enabled:
            return

        from quantumnematode.env.oxygen import OxygenZone

        zone_to_overlay_name = {
            OxygenZone.LETHAL_HYPOXIA: "lethal_hypoxia",
            OxygenZone.DANGER_HYPOXIA: "danger_hypoxia",
            OxygenZone.COMFORT: "comfort_oxygen",
            OxygenZone.DANGER_HYPEROXIA: "danger_hyperoxia",
            OxygenZone.LETHAL_HYPEROXIA: "lethal_hyperoxia",
        }

        min_x, min_y, max_x, max_y = viewport
        for gy in range(min_y, max_y):
            for gx in range(min_x, max_x):
                zone = env.get_oxygen_zone((gx, gy))
                if zone is None:
                    continue
                overlay_name = zone_to_overlay_name.get(zone)
                if overlay_name and overlay_name != "comfort_oxygen":
                    overlay = self._zone_overlays[overlay_name]
                    px, py = self._cell_to_pixel(gx, gy, viewport)
                    self._screen.blit(overlay, (px, py))

    def _render_toxic_zones(
        self,
        env: DynamicForagingEnvironment,
        viewport: Viewport,
    ) -> None:
        """Render toxic zone overlays around stationary predators."""
        if not env.predator.enabled:
            return

        from quantumnematode.env.env import PredatorType

        toxic_overlay = self._zone_overlays["toxic"]
        min_x, min_y, max_x, max_y = viewport

        for pred in env.predators:
            if pred.predator_type != PredatorType.STATIONARY or pred.damage_radius <= 0:
                continue
            px_pos, py_pos = pred.position
            for gy in range(
                max(min_y, py_pos - pred.damage_radius),
                min(max_y, py_pos + pred.damage_radius + 1),
            ):
                for gx in range(
                    max(min_x, px_pos - pred.damage_radius),
                    min(max_x, px_pos + pred.damage_radius + 1),
                ):
                    distance = abs(gx - px_pos) + abs(gy - py_pos)
                    if distance <= pred.damage_radius:
                        sx, sy = self._cell_to_pixel(gx, gy, viewport)
                        self._screen.blit(toxic_overlay, (sx, sy))

    def _render_entities(
        self,
        env: DynamicForagingEnvironment,
        viewport: Viewport,
    ) -> None:
        """Render food, predators, nematode body, and head.

        Entity sprites use SRCALPHA so zone overlays show through.
        """
        min_x, min_y, max_x, max_y = viewport

        def _in_view(x: int, y: int) -> bool:
            return min_x <= x < max_x and min_y <= y < max_y

        # Food
        food_sprite = self._sprites["food"]
        for food_pos in env.foods:
            # Snap real-valued (continuous-2D) sources to a cell, then view-test and
            # place that same cell so edge sources don't disappear.
            fx, fy = round(food_pos[0]), round(food_pos[1])
            if _in_view(fx, fy):
                px, py = self._cell_to_pixel(fx, fy, viewport)
                self._screen.blit(food_sprite, (px, py))

        # Predators
        self._render_predator_sprites(env, viewport, _in_view)

        # Body segments with connectors
        self._render_nematode_body(env, viewport, _in_view)

        # Head (drawn last so it's on top)
        agent_x, agent_y = env.agent_pos[0], env.agent_pos[1]
        if _in_view(agent_x, agent_y):
            direction = env.current_direction.value
            head_key = f"head_{direction}"
            if head_key not in self._sprites:
                head_key = "head_up"
            head_sprite = self._sprites[head_key]
            px, py = self._cell_to_pixel(agent_x, agent_y, viewport)
            self._screen.blit(head_sprite, (px, py))

    def _render_nematode_body(
        self,
        env: DynamicForagingEnvironment,
        viewport: Viewport,
        in_view_fn: Any,  # noqa: ANN401
    ) -> None:
        """Render nematode body segments with connectors between neighbors."""
        body = env.body
        if not body:
            return

        # Build a set of all nematode positions (body + head) for neighbor lookup
        head_pos = (env.agent_pos[0], env.agent_pos[1])
        body_positions = {(seg[0], seg[1]) for seg in body}
        all_positions = body_positions | {head_pos}

        for i, seg in enumerate(body):
            sx, sy = seg[0], seg[1]
            if not in_view_fn(sx, sy):
                continue

            px, py = self._cell_to_pixel(sx, sy, viewport)
            is_tail = i == len(body) - 1

            # Note: grid y-up, pygame y-down. "up" in grid = "up" visually
            # but pixel y decreases. _cell_to_pixel handles inversion, so
            # connect_up means neighbor at (sx, sy+1) in grid coords.
            draw_body_segment(
                self._pg,
                self._screen,
                px,
                py,
                connect_up=(sx, sy + 1) in all_positions,
                connect_down=(sx, sy - 1) in all_positions,
                connect_left=(sx - 1, sy) in all_positions,
                connect_right=(sx + 1, sy) in all_positions,
                is_tail=is_tail,
            )

    def _render_predator_sprites(
        self,
        env: DynamicForagingEnvironment,
        viewport: Viewport,
        in_view_fn: Any,  # noqa: ANN401
    ) -> None:
        """Render predator sprites."""
        if not env.predator.enabled:
            return

        from quantumnematode.env.env import PredatorType

        for pred in env.predators:
            if not in_view_fn(pred.position[0], pred.position[1]):
                continue
            if pred.predator_type == PredatorType.STATIONARY:
                sprite = self._sprites["predator_stationary"]
            elif pred.predator_type == PredatorType.PURSUIT:
                sprite = self._sprites["predator_pursuit"]
            else:
                sprite = self._sprites["predator_random"]
            px, py = self._cell_to_pixel(pred.position[0], pred.position[1], viewport)
            self._screen.blit(sprite, (px, py))

    def _render_status_bar(  # noqa: PLR0913
        self,
        *,
        step: int,
        max_steps: int,
        foods_collected: int,
        target_foods: int,
        health: float,
        max_health: float,
        satiety: float,
        max_satiety: float,
        in_danger: bool,
        temperature: float | None,
        zone_name: str | None,
        oxygen: float | None = None,
        oxygen_zone_name: str | None = None,
        session_text: str | None = None,
    ) -> None:
        """Render status information below the grid."""
        bar_y = self._viewport_size[1] * self._cell_size
        self._pg.draw.rect(
            self._screen,
            STATUS_BG_COLOR,
            (0, bar_y, self._width, STATUS_BAR_HEIGHT),
        )

        lines: list[tuple[str, tuple[int, int, int]]] = []

        # Session-level info from render_text
        if session_text:
            for raw_line in session_text.strip().splitlines():
                stripped = raw_line.strip()
                if stripped and not stripped.startswith("--"):
                    lines.append((stripped, STATUS_SESSION_COLOR))
            # Separator
            lines.append(("", STATUS_TEXT_COLOR))

        # Run-level info (shared formatting with the continuous renderer)
        lines.extend(
            build_run_status_lines(
                step=step,
                max_steps=max_steps,
                foods_collected=foods_collected,
                target_foods=target_foods,
                health=health,
                max_health=max_health,
                satiety=satiety,
                max_satiety=max_satiety,
                in_danger=in_danger,
                temperature=temperature,
                zone_name=zone_name,
                oxygen=oxygen,
                oxygen_zone_name=oxygen_zone_name,
            ),
        )

        line_height = STATUS_FONT_SIZE + 2
        antialias = True
        for i, (text, color) in enumerate(lines):
            if text:
                text_surf = self._font.render(text, antialias, color)
                self._screen.blit(text_surf, (STATUS_PADDING, bar_y + 4 + i * line_height))

    # ── Multi-agent rendering ─────────────────────────────────────────

    def render_multi_agent_frame(  # noqa: PLR0913
        self,
        env: DynamicForagingEnvironment,
        agents: list[AgentRenderState],
        followed_agent_id: str,
        *,
        step: int = 0,
        max_steps: int = 0,
        current_step: int = 0,
        session_text: str | None = None,
    ) -> str:
        """Render one complete multi-agent frame.

        Parameters
        ----------
        env : DynamicForagingEnvironment
            Shared environment (for foods, predators, zones, pheromones).
        agents : list[AgentRenderState]
            Per-agent rendering snapshots.
        followed_agent_id : str
            Agent the viewport is currently following.
        step : int
            Current step within the episode.
        max_steps : int
            Maximum steps for the episode.
        current_step : int
            Simulation step for pheromone concentration queries.
        session_text : str or None
            Additional session-level text for the status bar.

        Returns
        -------
        str
            The (possibly updated) followed_agent_id after keyboard input.
        """
        if self._closed:
            return followed_agent_id

        new_followed, pheromone_toggle = self._pump_multi_agent_events(
            agents,
            followed_agent_id,
        )
        followed_agent_id = new_followed

        if self._closed:
            return followed_agent_id

        if pheromone_toggle:
            self._pheromone_overlay_enabled = not self._pheromone_overlay_enabled

        # Find the followed agent for status bar
        followed = next((a for a in agents if a.agent_id == followed_agent_id), None)
        if followed is None and agents:
            followed = agents[0]
            followed_agent_id = followed.agent_id

        # Pre-compute status bar lines and resize BEFORE drawing
        status_lines = self._build_multi_agent_status_lines(
            agents=agents,
            followed=followed,
            step=step,
            max_steps=max_steps,
            session_text=session_text,
        )
        self._resize_if_needed(len(status_lines))

        viewport = env.get_viewport_bounds_for(followed_agent_id)

        # Clear screen
        self._screen.fill(STATUS_BG_COLOR)

        # Reuse existing layer renderers
        self._render_background(viewport)
        self._render_temperature_zones(env, viewport)
        self._render_oxygen_zones(env, viewport)
        self._render_toxic_zones(env, viewport)

        # Pheromone overlay (between zones and entities)
        if self._pheromone_overlay_enabled:
            self._render_pheromone_overlay(env, viewport, current_step)

        # Multi-agent entities
        self._render_multi_agent_entities(env, agents, followed_agent_id, viewport)

        # Blit pre-computed status bar
        self._blit_status_lines(status_lines)

        self._pg.display.flip()
        self._clock.tick(30)

        return followed_agent_id

    def _pump_multi_agent_events(  # noqa: C901
        self,
        agents: list[AgentRenderState],
        followed_agent_id: str,
    ) -> tuple[str, bool]:
        """Process Pygame events for multi-agent mode.

        Returns
        -------
        tuple[str, bool]
            (new_followed_agent_id, pheromone_toggle_requested)
        """
        pheromone_toggle = False
        agent_ids = [a.agent_id for a in agents]

        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                self.close()
                return followed_agent_id, False

            if event.type != self._pg.KEYDOWN:
                continue

            # Arrow keys: cycle through agents
            if event.key == self._pg.K_RIGHT and agent_ids:
                try:
                    idx = agent_ids.index(followed_agent_id)
                except ValueError:
                    idx = -1
                followed_agent_id = agent_ids[(idx + 1) % len(agent_ids)]

            elif event.key == self._pg.K_LEFT and agent_ids:
                try:
                    idx = agent_ids.index(followed_agent_id)
                except ValueError:
                    idx = 1
                followed_agent_id = agent_ids[(idx - 1) % len(agent_ids)]

            # Number keys 1-9: jump to agent
            elif self._pg.K_1 <= event.key <= self._pg.K_9:
                target_idx = event.key - self._pg.K_1
                if target_idx < len(agent_ids):
                    followed_agent_id = agent_ids[target_idx]

            # P key: toggle pheromone overlay
            elif event.key == self._pg.K_p:
                pheromone_toggle = True

        return followed_agent_id, pheromone_toggle

    def _render_pheromone_overlay(
        self,
        env: DynamicForagingEnvironment,
        viewport: Viewport,
        current_step: int,
    ) -> None:
        """Render pheromone concentration as colored semi-transparent overlays."""
        min_x, min_y, max_x, max_y = viewport

        # Map pheromone fields to overlay colors (R, G, B)
        fields: list[tuple[Any, tuple[int, int, int]]] = []
        if env.pheromone_field_food is not None:
            fields.append((env.pheromone_field_food, (60, 200, 60)))  # green
        if env.pheromone_field_alarm is not None:
            fields.append((env.pheromone_field_alarm, (220, 60, 60)))  # red
        if env.pheromone_field_aggregation is not None:
            fields.append((env.pheromone_field_aggregation, (60, 120, 220)))  # blue

        if not fields:
            return

        for gy in range(min_y, max_y):
            for gx in range(min_x, max_x):
                pos = (gx, gy)
                for field, color in fields:
                    conc = field.get_concentration(pos, current_step)
                    if conc < PHEROMONE_RENDER_THRESHOLD:
                        continue
                    # Alpha proportional to concentration (max 160 to stay readable)
                    alpha = int(min(conc, 1.0) * 160)
                    overlay = self._pg.Surface(
                        (self._cell_size, self._cell_size),
                        self._pg.SRCALPHA,
                    )
                    overlay.fill((*color, alpha))
                    px, py = self._cell_to_pixel(gx, gy, viewport)
                    self._screen.blit(overlay, (px, py))

    def _render_multi_agent_entities(
        self,
        env: DynamicForagingEnvironment,
        agents: list[AgentRenderState],
        followed_agent_id: str,
        viewport: Viewport,
    ) -> None:
        """Render food, predators, and all agent bodies/heads."""
        min_x, min_y, max_x, max_y = viewport

        def _in_view(x: int, y: int) -> bool:
            return min_x <= x < max_x and min_y <= y < max_y

        # Food
        food_sprite = self._sprites["food"]
        for food_pos in env.foods:
            # Snap real-valued (continuous-2D) sources to a cell, then view-test and
            # place that same cell so edge sources don't disappear.
            fx, fy = round(food_pos[0]), round(food_pos[1])
            if _in_view(fx, fy):
                px, py = self._cell_to_pixel(fx, fy, viewport)
                self._screen.blit(food_sprite, (px, py))

        # Predators
        self._render_predator_sprites(env, viewport, _in_view)

        # All agents (body + head)
        for agent in agents:
            self._render_single_agent(agent, followed_agent_id, viewport, _in_view)

    def _render_single_agent(  # noqa: C901
        self,
        agent: AgentRenderState,
        followed_agent_id: str,
        viewport: Viewport,
        in_view_fn: Any,  # noqa: ANN401
    ) -> None:
        """Render one agent's body and head with appropriate coloring."""
        color_idx = agent.color_index % len(AGENT_COLOR_PALETTE)
        tint = AGENT_COLOR_PALETTE[color_idx]

        # Get or create tinted sprites for this color
        if color_idx not in self._tinted_sprites:
            self._tinted_sprites[color_idx] = create_tinted_head_sprites(self._pg, tint)
        head_sprites = self._tinted_sprites[color_idx]

        # Get tinted body colors
        if color_idx not in self._tinted_body_colors:
            self._tinted_body_colors[color_idx] = tint_body_colors(tint)
        bc, oc, hc = self._tinted_body_colors[color_idx]

        # Build position set for body connector lookup
        head_pos = agent.position
        body_positions = {(seg[0], seg[1]) for seg in agent.body}
        all_positions = body_positions | {head_pos}

        # Render body segments
        for i, seg in enumerate(agent.body):
            sx, sy = seg[0], seg[1]
            if not in_view_fn(sx, sy):
                continue
            px, py = self._cell_to_pixel(sx, sy, viewport)
            is_tail = i == len(agent.body) - 1
            draw_body_segment(
                self._pg,
                self._screen,
                px,
                py,
                connect_up=(sx, sy + 1) in all_positions,
                connect_down=(sx, sy - 1) in all_positions,
                connect_left=(sx - 1, sy) in all_positions,
                connect_right=(sx + 1, sy) in all_positions,
                is_tail=is_tail,
                body_color=bc,
                outline_color=oc,
                highlight_color=hc,
            )

        # Render head
        ax, ay = head_pos
        if in_view_fn(ax, ay):
            head_key = f"head_{agent.direction}"
            if head_key not in head_sprites:
                head_key = "head_up"
            px, py = self._cell_to_pixel(ax, ay, viewport)
            self._screen.blit(head_sprites[head_key], (px, py))

            # Followed agent highlight ring
            if agent.agent_id == followed_agent_id:
                c = self._cell_size // 2
                self._pg.draw.circle(
                    self._screen,
                    (255, 255, 255, 120),
                    (px + c, py + c),
                    self._cell_size // 2 + 2,
                    2,
                )

        # Dead agent overlay
        if not agent.alive:
            # Overlay on head position
            if in_view_fn(ax, ay):
                px, py = self._cell_to_pixel(ax, ay, viewport)
                self._screen.blit(self._dead_overlay, (px, py))
            # Overlay on body segments
            for seg in agent.body:
                if in_view_fn(seg[0], seg[1]):
                    px, py = self._cell_to_pixel(seg[0], seg[1], viewport)
                    self._screen.blit(self._dead_overlay, (px, py))

    def _build_multi_agent_status_lines(  # noqa: C901, PLR0912
        self,
        *,
        agents: list[AgentRenderState],
        followed: AgentRenderState | None,
        step: int,
        max_steps: int,
        session_text: str | None = None,
    ) -> list[tuple[str, tuple[int, int, int]]]:
        """Build and wrap status bar lines. Returns wrapped lines ready to blit."""
        lines: list[tuple[str, tuple[int, int, int]]] = []

        # Session-level info
        if session_text:
            for raw_line in session_text.strip().splitlines():
                stripped = raw_line.strip()
                if stripped and not stripped.startswith("--"):
                    lines.append((stripped, STATUS_SESSION_COLOR))
            lines.append(("", STATUS_TEXT_COLOR))

        # Agent switcher indicator
        if followed is not None:
            agent_ids = [a.agent_id for a in agents]
            try:
                idx = agent_ids.index(followed.agent_id) + 1
            except ValueError:
                idx = 0
            total = len(agents)
            lines.append(
                (
                    f"Following: {followed.agent_id} [{idx}/{total}]",
                    STATUS_SESSION_COLOR,
                ),
            )
            lines.append(
                (
                    "</> to switch, 1-9 to jump",
                    STATUS_SESSION_COLOR,
                ),
            )

        # Followed agent metrics
        if followed is not None:
            lines.append((f"Step: {step}/{max_steps}", STATUS_TEXT_COLOR))
            lines.append((f"Food: {followed.foods_collected}", STATUS_TEXT_COLOR))
            lines.append(
                (f"HP: {followed.hp:.0f}/{followed.max_hp:.0f}", STATUS_TEXT_COLOR),
            )
            lines.append(
                (
                    f"Satiety: {followed.satiety:.0f}/{followed.max_satiety:.0f}",
                    STATUS_TEXT_COLOR,
                ),
            )

        # All-agent summary
        parts = []
        for a in agents:
            status = "alive" if a.alive else "DEAD"
            parts.append(f"{a.agent_id}: {a.foods_collected}F {status}")
        if parts:
            lines.append((" | ".join(parts), STATUS_TEXT_COLOR))

        # Pheromone overlay indicator
        if self._pheromone_overlay_enabled:
            lines.append(("[P] Pheromone overlay ON", STATUS_SAFE_COLOR))

        # Wrap long lines to fit window width
        wrapped: list[tuple[str, tuple[int, int, int]]] = []
        max_text_width = self._width - STATUS_PADDING * 2
        antialias = True
        for text, color in lines:
            if not text:
                wrapped.append(("", color))
                continue
            text_surf = self._font.render(text, antialias, color)
            if text_surf.get_width() <= max_text_width:
                wrapped.append((text, color))
            else:
                wrapped.extend(self._wrap_text(text, color, max_text_width))

        return wrapped

    def _blit_status_lines(
        self,
        wrapped: list[tuple[str, tuple[int, int, int]]],
    ) -> None:
        """Blit pre-computed status lines onto the screen."""
        bar_y = self._viewport_size[1] * self._cell_size
        self._pg.draw.rect(
            self._screen,
            STATUS_BG_COLOR,
            (0, bar_y, self._width, self._height - bar_y),
        )
        line_height = STATUS_FONT_SIZE + 2
        antialias = True
        for i, (text, color) in enumerate(wrapped):
            if text:
                text_surf = self._font.render(text, antialias, color)
                self._screen.blit(
                    text_surf,
                    (STATUS_PADDING, bar_y + 4 + i * line_height),
                )

    def _wrap_text(
        self,
        text: str,
        color: tuple[int, int, int],
        max_width: int,
    ) -> list[tuple[str, tuple[int, int, int]]]:
        """Word-wrap text to fit within max_width pixels.

        Falls back to character-level breaking for single tokens wider
        than max_width.
        """
        words = text.split()
        result: list[tuple[str, tuple[int, int, int]]] = []
        current_line = ""
        antialias = True
        for word in words:
            candidate = f"{current_line} {word}".strip()
            surf = self._font.render(candidate, antialias, color)
            if surf.get_width() <= max_width:
                current_line = candidate
            else:
                if current_line:
                    result.append((current_line, color))
                # Check if the word itself exceeds max_width
                word_surf = self._font.render(word, antialias, color)
                if word_surf.get_width() > max_width:
                    # Character-level breaking
                    chunk = ""
                    for ch in word:
                        test = chunk + ch
                        if self._font.render(test, antialias, color).get_width() > max_width:
                            if chunk:
                                result.append((chunk, color))
                            chunk = ch
                        else:
                            chunk = test
                    current_line = chunk
                else:
                    current_line = word
        if current_line:
            result.append((current_line, color))
        return result or [("", color)]

    def close(self) -> None:
        """Clean up Pygame resources."""
        if not self._closed:
            self._closed = True
            self._pg.quit()
            logger.info("PygameRenderer closed.")


# ── Continuous-2D substrate renderer ──────────────────────────────────────────

# Default zoom: ~12 px/mm → a 50 mm plate renders at 600x600 + status bar.
DEFAULT_PIXELS_PER_MM = 12.0
# Heatmap lattice resolution (bounds the per-(re)sample field-call cost).
HEATMAP_LATTICE_N = 64
# Coarse quiver lattice (kept small — gradient is O(sources) per sample).
QUIVER_LATTICE_N = 12
# Minimum food-gradient strength below which a quiver arrow is not drawn.
QUIVER_MIN_STRENGTH = 1e-3
# Uniform alpha for the field heatmap blit (keeps entities/zones readable).
HEATMAP_ALPHA = 130
# Zoom multiplier for the optional agent-following camera (vs the base full-arena
# zoom). The followed view shows ``world / FOLLOW_ZOOM_FACTOR`` mm around the worm.
FOLLOW_ZOOM_FACTOR = 2.5
# Heatmap colormap (perceptually uniform).
HEATMAP_COLORMAP = "viridis"
# Selectable heatmap fields, in cycle order (food is the default first entry).
HEATMAP_FIELDS = ("food", "predator", "temperature", "oxygen", "pheromone")

# Worm / sensor overlay colours.
WORM_MARKER_COLOR = (220, 195, 160)
WORM_OUTLINE_COLOR = (120, 95, 60)
WORM_HEAD_COLOR = (235, 210, 170)  # bright head end (distinct from the body)
WORM_BODY_COLOR = (188, 162, 128)  # tapering body, darker so the head stands out
WORM_HEADING_COLOR = (
    255,
    110,
    70,
)  # heading indicator: a contrasting hue, not blended into the body
KLINOTAXIS_SAMPLE_COLOR = (120, 200, 255)

# Path-following undulating body geometry.
_WORM_BODY_SEGMENTS = 12  # recent positions kept; the body is a stylised trail over this path
_WORM_RESET_JUMP_MM = 3.0  # a pos jump larger than any legal step (<= max_step_mm) ⇒ episode reset
_WORM_UNDULATION_WAVELENGTH_SEG = 8.0  # body segments per crawl wave
_WORM_UNDULATION_PHASE_STEP = 0.55  # phase advance per frame (rad) → the wave travels down the body
_WORM_UNDULATION_AMPLITUDE_FRAC = 0.7  # lateral wiggle as a fraction of the worm radius
PREDATOR_DETECTION_RING_COLOR = (200, 160, 220)
PREDATOR_DAMAGE_RING_COLOR = (220, 70, 70)
QUIVER_ARROW_COLOR = (120, 230, 140)
STATUS_OVERLAY_HINT_COLOR = (150, 150, 170)


class Continuous2DRenderer:
    """Renders the continuous-2D substrate with sub-cell fidelity.

    A sibling of :class:`PygameRenderer` (not a subclass — the grid renderer's
    methods are hardwired to the integer scrolling viewport and ``_cell_to_pixel``
    y-inversion). The default view is the **whole arena** (no scroll): the worm is a
    point on a plate. A keyboard toggle (``C``) switches to an **agent-following**
    zoomed view that scrolls to keep the worm centred (clamped to the plate edges) —
    useful on large worlds where the worm is a small dot in the full-arena view.
    Real-valued world coordinates (millimetres) map to pixels via
    :meth:`_world_to_pixel`, which is camera-aware (full-arena or following).

    The renderer reads only the per-step :class:`ContinuousRenderState` snapshot and
    the environment (for fields, foods, predators) — never agent internals.

    Layers (back → front): background, temperature / oxygen / toxic zones,
    concentration heatmap, gradient quiver, klinotaxis + predator sensor zones,
    entities (food / predators / worm), status bar.

    Parameters
    ----------
    world_size_mm : float
        Side length of the square arena, in millimetres.
    pixels_per_mm : float
        Zoom: screen pixels per millimetre (default ``DEFAULT_PIXELS_PER_MM``).
    heatmap_n : int
        Heatmap lattice resolution per side (default ``HEATMAP_LATTICE_N``).
    """

    def __init__(
        self,
        world_size_mm: float,
        *,
        pixels_per_mm: float = DEFAULT_PIXELS_PER_MM,
        heatmap_n: int = HEATMAP_LATTICE_N,
    ) -> None:
        import pygame

        self._pg = pygame
        self._world_size_mm = float(world_size_mm)
        self._pixels_per_mm = float(pixels_per_mm)
        self._grid_size = round(world_size_mm)
        self._heatmap_n = max(2, heatmap_n)

        self._arena_px = max(1, round(self._world_size_mm * self._pixels_per_mm))
        self._width = self._arena_px
        self._height = self._arena_px + STATUS_BAR_HEIGHT

        pygame.init()
        self._screen = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption(WINDOW_TITLE_CONTINUOUS)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", STATUS_FONT_SIZE)

        # Raw entity sprites; scaled copies are cached per active pixels-per-mm
        # (the camera changes the effective zoom when following).
        sprites = create_sprites(pygame)
        self._raw_sprites: dict[str, Any] = {
            key: sprites[key]
            for key in ("food", "predator_random", "predator_stationary", "predator_pursuit")
        }
        self._sprite_cache: dict[int, dict[str, Any]] = {}

        # Camera state: full-arena by default; `C` toggles agent-following. The
        # active pixels-per-mm and the camera origin (world coords mapping to the
        # top-left of the arena viewport) are recomputed each frame in
        # `_update_camera`.
        self._follow_enabled = False
        self._active_ppm = self._pixels_per_mm
        self._cam_x0 = 0.0
        self._cam_y_top = self._world_size_mm

        # Overlay toggle state (heatmap on by default; quiver off for perf).
        self._heatmap_enabled = True
        self._quiver_enabled = False
        self._heatmap_field_idx = 0
        self._heatmap_cache: tuple[Any, Any] | None = None  # (cache_key, surface)

        # Worm body: a trailing history of recent positions (drawn as an undulating,
        # tapered body) + a crawl-wave phase. Reset on an episode-boundary position jump.
        self._body_history: deque[tuple[float, float]] = deque(maxlen=_WORM_BODY_SEGMENTS)
        self._last_worm_pos: tuple[float, float] | None = None
        self._undulation_phase: float = 0.0

        self._closed = False
        self._last_status_line_count = 0
        logger.info(
            f"Continuous2DRenderer initialized: {self._width}x{self._height} "
            f"({self._world_size_mm:.0f}mm @ {self._pixels_per_mm:.1f}px/mm)",
        )

    @property
    def closed(self) -> bool:
        """Whether the window has been closed."""
        return self._closed

    def _world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Map a real-valued world point (mm, y-up) to a screen pixel (y-down).

        Camera-aware: ``px = (x - cam_x0) * active_ppm``;
        ``py = (cam_y_top - y) * active_ppm`` (world y-up → screen y-down). In the
        default full-arena view ``cam_x0 = 0``, ``cam_y_top = world_size_mm`` and
        ``active_ppm = pixels_per_mm`` (the whole arena is in view); when following,
        the camera origin and zoom track the worm (see :meth:`_update_camera`).
        """
        px = (x - self._cam_x0) * self._active_ppm
        py = (self._cam_y_top - y) * self._active_ppm
        return round(px), round(py)

    def _update_camera(self, state: ContinuousRenderState) -> None:
        """Recompute the active zoom + camera origin for this frame.

        Full-arena (default, or when the followed view would exceed the plate):
        the whole arena maps to the viewport. Following: a
        ``world / FOLLOW_ZOOM_FACTOR`` mm window centred on the worm, clamped so it
        never shows past the plate edges.
        """
        base = self._pixels_per_mm
        if not self._follow_enabled:
            self._active_ppm, self._cam_x0, self._cam_y_top = base, 0.0, self._world_size_mm
            return
        active_ppm = base * FOLLOW_ZOOM_FACTOR
        view_mm = self._arena_px / active_ppm
        if view_mm >= self._world_size_mm:  # zoom would exceed the plate → full arena
            self._active_ppm, self._cam_x0, self._cam_y_top = base, 0.0, self._world_size_mm
            return
        half = view_mm / 2.0
        cam_cx = min(self._world_size_mm - half, max(half, state.pos[0]))
        cam_cy = min(self._world_size_mm - half, max(half, state.pos[1]))
        self._active_ppm = active_ppm
        self._cam_x0 = cam_cx - half
        self._cam_y_top = cam_cy + half

    @property
    def _camera_following(self) -> bool:
        """Whether the active camera is the zoomed agent-following view."""
        return self._active_ppm != self._pixels_per_mm

    def _sprites_for(self, ppm: float) -> tuple[dict[str, Any], int]:
        """Return entity sprites scaled for ``ppm`` (cached) and the marker size."""
        entity_px = max(10, round(2.0 * ppm))
        cached = self._sprite_cache.get(entity_px)
        if cached is None:
            cached = {
                key: self._pg.transform.smoothscale(raw, (entity_px, entity_px))
                for key, raw in self._raw_sprites.items()
            }
            self._sprite_cache[entity_px] = cached
        return cached, entity_px

    def _cell_rect(self, gx: int, gy: int) -> tuple[int, int, int, int]:
        """Pixel rect (x, y, w, h) covering the 1 mm cell centred at ``(gx, gy)``."""
        px0, py0 = self._world_to_pixel(gx - 0.5, gy + 0.5)  # top-left (y inverted)
        px1, py1 = self._world_to_pixel(gx + 0.5, gy - 0.5)  # bottom-right
        return (px0, py0, max(1, px1 - px0), max(1, py1 - py0))

    def pump_events(self) -> bool:
        """Process Pygame events. Returns False if the window was closed.

        Keyboard toggles (single-agent renderers have none today, so this is new):
        ``H`` heatmap on/off, ``F`` cycle heatmap field, ``G`` gradient quiver
        on/off, ``C`` full-arena/agent-following camera. Window-close follows the
        grid renderer's ``pump_events`` shape.
        """
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                self.close()
                return False
            if event.type != self._pg.KEYDOWN:
                continue
            if event.key == self._pg.K_h:
                self._heatmap_enabled = not self._heatmap_enabled
            elif event.key == self._pg.K_g:
                self._quiver_enabled = not self._quiver_enabled
            elif event.key == self._pg.K_c:
                self._follow_enabled = not self._follow_enabled
            elif event.key == self._pg.K_f:
                self._heatmap_field_idx = (self._heatmap_field_idx + 1) % len(HEATMAP_FIELDS)
                self._heatmap_cache = None  # field changed → invalidate cache
        return True

    def render_frame(
        self,
        env: DynamicForagingEnvironment,
        state: ContinuousRenderState,
        *,
        session_text: str | None = None,
    ) -> None:
        """Render one complete continuous-substrate frame from a state snapshot."""
        if self._closed:
            return
        if not self.pump_events():
            return

        self._update_camera(state)
        self._update_body_history(state.pos)
        status_lines = self._build_status_lines(state, session_text)
        self._resize_if_needed(len(status_lines))

        self._screen.fill(STATUS_BG_COLOR)
        self._screen.fill(SOIL_COLOR, (0, 0, self._arena_px, self._arena_px))

        self._render_temperature_zones(env)
        self._render_oxygen_zones(env)
        self._render_toxic_zones(env)
        if self._heatmap_enabled:
            self._render_heatmap(env, state)
        if self._quiver_enabled:
            self._render_quiver(env)
        self._render_sensor_zones(env, state)
        self._render_entities(env, state)
        self._blit_status_lines(status_lines)

        self._pg.display.flip()
        self._clock.tick(30)

    def _update_body_history(self, pos: tuple[float, float]) -> None:
        """Accumulate the worm's recent positions for the trailing body.

        Clears the history on an episode-boundary position jump (a discontinuity larger
        than any single legal step, e.g. the reset teleport to the arena centre), so the
        body never streaks across a reset. Advances the crawl-wave phase each frame.
        """
        import math

        last = self._last_worm_pos
        if (
            last is not None
            and math.hypot(pos[0] - last[0], pos[1] - last[1]) > _WORM_RESET_JUMP_MM
        ):
            self._body_history.clear()
        self._body_history.append((float(pos[0]), float(pos[1])))
        self._last_worm_pos = (float(pos[0]), float(pos[1]))
        self._undulation_phase += _WORM_UNDULATION_PHASE_STEP

    # ── Parity layers (ported to continuous coordinates) ──────────────────────

    def _render_temperature_zones(self, env: DynamicForagingEnvironment) -> None:
        """Fill temperature comfort/danger zones over the arena (categorical colours)."""
        if not env.thermotaxis.enabled or env.temperature_field is None:
            return

        from quantumnematode.env.temperature import TemperatureZone, TemperatureZoneThresholds

        thresholds = TemperatureZoneThresholds(
            comfort_delta=env.thermotaxis.comfort_delta,
            discomfort_delta=env.thermotaxis.discomfort_delta,
            danger_delta=env.thermotaxis.danger_delta,
        )
        zone_to_name = {
            TemperatureZone.LETHAL_COLD: "lethal_cold",
            TemperatureZone.DANGER_COLD: "danger_cold",
            TemperatureZone.DISCOMFORT_COLD: "discomfort_cold",
            TemperatureZone.DISCOMFORT_HOT: "discomfort_hot",
            TemperatureZone.DANGER_HOT: "danger_hot",
            TemperatureZone.LETHAL_HOT: "lethal_hot",
        }
        overlay = self._pg.Surface((self._arena_px, self._arena_px), self._pg.SRCALPHA)
        for gy in range(self._grid_size):
            for gx in range(self._grid_size):
                temp = env.temperature_field.get_temperature((gx, gy))
                zone = env.temperature_field.get_zone(
                    temp,
                    env.thermotaxis.cultivation_temperature,
                    thresholds,
                )
                name = zone_to_name.get(zone)
                if name is not None:
                    overlay.fill(zone_overlay_color(name), self._cell_rect(gx, gy))
        self._screen.blit(overlay, (0, 0))

    def _render_oxygen_zones(self, env: DynamicForagingEnvironment) -> None:
        """Fill oxygen comfort/danger zones over the arena (categorical colours)."""
        if not env.aerotaxis.enabled:
            return

        from quantumnematode.env.oxygen import OxygenZone

        zone_to_name = {
            OxygenZone.LETHAL_HYPOXIA: "lethal_hypoxia",
            OxygenZone.DANGER_HYPOXIA: "danger_hypoxia",
            OxygenZone.DANGER_HYPEROXIA: "danger_hyperoxia",
            OxygenZone.LETHAL_HYPEROXIA: "lethal_hyperoxia",
        }
        overlay = self._pg.Surface((self._arena_px, self._arena_px), self._pg.SRCALPHA)
        for gy in range(self._grid_size):
            for gx in range(self._grid_size):
                zone = env.get_oxygen_zone((gx, gy))
                name = zone_to_name.get(zone) if zone is not None else None
                if name is not None:
                    overlay.fill(zone_overlay_color(name), self._cell_rect(gx, gy))
        self._screen.blit(overlay, (0, 0))

    def _render_toxic_zones(self, env: DynamicForagingEnvironment) -> None:
        """Fill stationary-predator toxic (damage-radius) discs over the arena."""
        if not env.predator.enabled:
            return

        from quantumnematode.env.env import PredatorType

        overlay = self._pg.Surface((self._arena_px, self._arena_px), self._pg.SRCALPHA)
        drew = False
        for pred in env.predators:
            if pred.predator_type != PredatorType.STATIONARY or pred.damage_radius <= 0:
                continue
            px, py = pred.pos_continuous or (float(pred.position[0]), float(pred.position[1]))
            cx, cy = self._world_to_pixel(px, py)
            radius_px = max(1, round(pred.damage_radius * self._active_ppm))
            self._pg.draw.circle(overlay, zone_overlay_color("toxic"), (cx, cy), radius_px)
            drew = True
        if drew:
            self._screen.blit(overlay, (0, 0))

    # ── Fidelity overlays ─────────────────────────────────────────────────────

    def _heatmap_getter(
        self,
        env: DynamicForagingEnvironment,
        field_name: str,
        step: int,
    ) -> Any:  # noqa: ANN401
        """Return a ``(x, y) -> float`` sampler for the selected heatmap field.

        ``step`` is the current episode step — required for the time-varying
        pheromone field (it applies temporal decay keyed on the query step).
        """
        if field_name == "predator":
            return env.get_predator_concentration
        if field_name == "temperature":
            return lambda pos: env.get_temperature(pos) or 0.0
        if field_name == "oxygen":
            return lambda pos: env.get_oxygen_concentration(pos) or 0.0
        if field_name == "pheromone" and env.pheromone_field_food is not None:
            field = env.pheromone_field_food
            return lambda pos: field.get_concentration(pos, step)
        # Default / fallback: food concentration.
        return env.get_food_concentration

    def _heatmap_cache_key(
        self,
        env: DynamicForagingEnvironment,
        field_name: str,
        step: int,
    ) -> tuple:
        """Cache key keyed on the field and the sources that define it.

        The food/predator/thermal/oxygen fields are static within an episode except
        when their sources change, so the (expensive) lattice sampling is recomputed
        only when this key changes. The pheromone field is time-varying (deposits +
        temporal decay), so its key includes ``step`` to recompute every frame.
        """
        if field_name == "food":
            if env.foraging.source_depletion_enabled:
                # Depleting field: the source amplitude changes with positions fixed, so include
                # the (rounded) per-source amount in the key — else the cached heatmap stays frozen
                # as a patch flattens in place.
                sig: Any = tuple(
                    sorted(
                        (round(fx, 3), round(fy, 3), round(a, 3))
                        for (fx, fy), a in zip(env.foods, env.food_amounts, strict=True)
                    ),
                )
            else:
                sig = tuple(sorted((round(fx, 3), round(fy, 3)) for fx, fy in env.foods))
        elif field_name == "predator":
            sig = tuple(sorted((p.position[0], p.position[1]) for p in env.predators))
        elif field_name == "pheromone":
            sig = ("step", step)  # time-varying → rebuild each step
        else:
            sig = "static"
        return (field_name, sig, self._heatmap_n)

    def _render_heatmap(
        self,
        env: DynamicForagingEnvironment,
        state: ContinuousRenderState,
    ) -> None:
        """Sample the selected field over a lattice → colormap → alpha-blit.

        The sampled-and-colormapped surface is cached and only recomputed when the
        field's sources change (see :meth:`_heatmap_cache_key`).
        """
        field_name = HEATMAP_FIELDS[self._heatmap_field_idx]
        key = self._heatmap_cache_key(env, field_name, state.step)
        if self._heatmap_cache is None or self._heatmap_cache[0] != key:
            surface = self._build_heatmap_surface(env, field_name, state.step)
            self._heatmap_cache = (key, surface)
        base_surf = self._heatmap_cache[1]  # full-arena RGB surface (no baked alpha)
        if not self._camera_following:
            base_surf.set_alpha(HEATMAP_ALPHA)
            self._screen.blit(base_surf, (0, 0))
            return
        # Following: crop the cached full-arena surface to the visible world window
        # and scale it up to fill the viewport (reuses the cache — no re-sample).
        scale = self._arena_px / self._world_size_mm  # cached px per mm
        view_mm = self._arena_px / self._active_ppm
        top_mm = self._world_size_mm - self._cam_y_top
        sx0 = max(0, min(self._arena_px - 1, round(self._cam_x0 * scale)))
        sy0 = max(0, min(self._arena_px - 1, round(top_mm * scale)))
        side = max(1, min(self._arena_px - max(sx0, sy0), round(view_mm * scale)))
        sub = base_surf.subsurface((sx0, sy0, side, side))
        scaled = self._pg.transform.smoothscale(sub, (self._arena_px, self._arena_px))
        scaled.set_alpha(HEATMAP_ALPHA)
        self._screen.blit(scaled, (0, 0))

    def _build_heatmap_surface(
        self,
        env: DynamicForagingEnvironment,
        field_name: str,
        step: int,
    ) -> Any:  # noqa: ANN401
        """Build the full-arena RGB heatmap surface for one field at ``step``."""
        import matplotlib as mpl
        import numpy as np

        getter = self._heatmap_getter(env, field_name, step)
        n = self._heatmap_n
        world = self._world_size_mm
        values = np.zeros((n, n), dtype=float)  # indexed [i_x][j_y] (j=0 → top)
        for i in range(n):
            x = world * i / (n - 1)
            for j in range(n):
                y = world * (1.0 - j / (n - 1))
                values[i, j] = float(getter((x, y)))

        vmin = float(values.min())
        vmax = float(values.max())
        norm = (values - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(values)
        cmap = mpl.colormaps[HEATMAP_COLORMAP]
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)  # (n, n, 3), [i_x][j_y]

        surf = self._pg.surfarray.make_surface(rgb)
        # Full-arena RGB surface (no baked alpha — the alpha is applied at blit time,
        # so the cache can be cropped/scaled for the following camera).
        return self._pg.transform.smoothscale(surf, (self._arena_px, self._arena_px))

    def _render_quiver(self, env: DynamicForagingEnvironment) -> None:
        """Draw coarse up-gradient food arrows scaled by gradient strength."""
        import math

        n = QUIVER_LATTICE_N
        world = self._world_size_mm
        spacing_px = self._arena_px / n
        max_len = spacing_px * 0.9
        for i in range(n):
            x = world * (i + 0.5) / n
            for j in range(n):
                y = world * (j + 0.5) / n
                grad = env.get_separated_gradients((x, y), disable_log=True)  # type: ignore[arg-type]
                strength = float(grad.get("food_gradient_strength", 0.0))
                if strength <= QUIVER_MIN_STRENGTH:
                    continue
                direction = float(grad.get("food_gradient_direction", 0.0))
                cx, cy = self._world_to_pixel(x, y)
                length = max_len * min(1.0, strength)
                # World y-up → screen y-down: negate the y-component.
                ex = cx + math.cos(direction) * length
                ey = cy - math.sin(direction) * length
                self._pg.draw.line(self._screen, QUIVER_ARROW_COLOR, (cx, cy), (ex, ey), 1)
                self._draw_arrowhead(cx, cy, ex, ey, length)

    def _draw_arrowhead(
        self,
        cx: float,
        cy: float,
        ex: float,
        ey: float,
        length: float,
    ) -> None:
        """Draw a small two-line arrowhead at ``(ex, ey)`` pointing away from origin."""
        import math

        head = max(2.0, length * 0.3)
        angle = math.atan2(ey - cy, ex - cx)
        for sign in (1, -1):
            a = angle + sign * (math.pi * 0.8)
            self._pg.draw.line(
                self._screen,
                QUIVER_ARROW_COLOR,
                (ex, ey),
                (ex + math.cos(a) * head, ey + math.sin(a) * head),
                1,
            )

    def _render_sensor_zones(
        self,
        env: DynamicForagingEnvironment,
        state: ContinuousRenderState,
    ) -> None:
        """Draw klinotaxis sample points and predator detection/damage rings."""
        # Klinotaxis head-sweep sample points (left/right of heading).
        for sample in (state.left_sample, state.right_sample):
            if sample is None:
                continue
            sx, sy = self._world_to_pixel(sample[0], sample[1])
            self._pg.draw.circle(self._screen, KLINOTAXIS_SAMPLE_COLOR, (sx, sy), 3)
            self._pg.draw.circle(self._screen, (30, 30, 40), (sx, sy), 3, 1)

        # Predator detection / damage range rings.
        if env.predator.enabled:
            for pred in env.predators:
                px, py = pred.pos_continuous or (
                    float(pred.position[0]),
                    float(pred.position[1]),
                )
                cx, cy = self._world_to_pixel(px, py)
                if pred.detection_radius > 0:
                    self._pg.draw.circle(
                        self._screen,
                        PREDATOR_DETECTION_RING_COLOR,
                        (cx, cy),
                        max(1, round(pred.detection_radius * self._active_ppm)),
                        1,
                    )
                if pred.damage_radius > 0:
                    self._pg.draw.circle(
                        self._screen,
                        PREDATOR_DAMAGE_RING_COLOR,
                        (cx, cy),
                        max(1, round(pred.damage_radius * self._active_ppm)),
                        1,
                    )

    def _render_entities(
        self,
        env: DynamicForagingEnvironment,
        state: ContinuousRenderState,
    ) -> None:
        """Draw food / predator sprites and the worm (marker + heading line)."""
        import math

        sprites, entity_px = self._sprites_for(self._active_ppm)
        half = entity_px // 2
        food_sprite = sprites["food"]
        for food in env.foods:
            cx, cy = self._world_to_pixel(float(food[0]), float(food[1]))
            self._screen.blit(food_sprite, (cx - half, cy - half))

        if env.predator.enabled:
            from quantumnematode.env.env import PredatorType

            for pred in env.predators:
                if pred.predator_type == PredatorType.STATIONARY:
                    sprite = sprites["predator_stationary"]
                elif pred.predator_type == PredatorType.PURSUIT:
                    sprite = sprites["predator_pursuit"]
                else:
                    sprite = sprites["predator_random"]
                px, py = pred.pos_continuous or (
                    float(pred.position[0]),
                    float(pred.position[1]),
                )
                cx, cy = self._world_to_pixel(px, py)
                self._screen.blit(sprite, (cx - half, cy - half))

        # Worm: a path-following, tapered, undulating body trailing the head, then a
        # distinct head marker + a contrasting heading indicator. Sized to the active
        # (camera) zoom; width scales with body_length_mm. The body is a stylised visual
        # overlay (the worm is a point kinematically) — it does not affect the simulation.
        body_len_mm = getattr(getattr(env, "continuous", None), "body_length_mm", 1.0)
        worm_radius = max(4, round(0.6 * body_len_mm * self._active_ppm))
        wx, wy = self._world_to_pixel(state.pos[0], state.pos[1])

        # Body backbone = recent positions (tail→head) in pixel space, with a travelling
        # sinusoidal lateral undulation and a head→tail taper. Drawn as a CONNECTED tube:
        # thick links between consecutive points fill the gaps, with rounded circles at the
        # joints (so it reads as one continuous worm, not disjoint balls).
        backbone = [self._world_to_pixel(hx_w, hy_w) for hx_w, hy_w in self._body_history]
        n = len(backbone)
        if n >= 2:  # noqa: PLR2004 — need at least two points to define a backbone direction
            wave_k = 2.0 * math.pi / _WORM_UNDULATION_WAVELENGTH_SEG
            amp = _WORM_UNDULATION_AMPLITUDE_FRAC * worm_radius
            # Undulated pixel point + local radius (thin tail → full head) per segment.
            seg_pts: list[tuple[int, int, int]] = []
            for i, (bx, by) in enumerate(backbone):
                nxt, prv = backbone[min(i + 1, n - 1)], backbone[max(i - 1, 0)]
                dx, dy = nxt[0] - prv[0], nxt[1] - prv[1]
                seg = math.hypot(dx, dy) or 1.0
                perp_x, perp_y = -dy / seg, dx / seg  # unit normal to the local backbone
                offset = amp * math.sin(self._undulation_phase + i * wave_k)
                seg_r = max(2, round(worm_radius * (0.3 + 0.7 * (i / (n - 1)))))
                seg_pts.append((round(bx + perp_x * offset), round(by + perp_y * offset), seg_r))
            # Connecting links first (thickness ≈ local diameter), then rounded joints on top.
            for (x0, y0, r0), (x1, y1, r1) in pairwise(seg_pts):
                self._pg.draw.line(
                    self._screen,
                    WORM_BODY_COLOR,
                    (x0, y0),
                    (x1, y1),
                    max(2, r0 + r1),
                )
            for x, y, r in seg_pts:
                self._pg.draw.circle(self._screen, WORM_BODY_COLOR, (x, y), r)

        # Head marker (at the true, un-undulated position) — distinct, brighter end.
        self._pg.draw.circle(self._screen, WORM_HEAD_COLOR, (wx, wy), worm_radius)
        self._pg.draw.circle(self._screen, WORM_OUTLINE_COLOR, (wx, wy), worm_radius, 1)
        # Heading indicator: short, contrasting hue (not blended into the body).
        heading_len = worm_radius + max(6, round(1.2 * self._active_ppm))
        hx = wx + math.cos(state.heading_rad) * heading_len
        hy = wy - math.sin(state.heading_rad) * heading_len  # world y-up → screen y-down
        self._pg.draw.line(self._screen, WORM_HEADING_COLOR, (wx, wy), (hx, hy), 2)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_lines(
        self,
        state: ContinuousRenderState,
        session_text: str | None,
    ) -> list[tuple[str, tuple[int, int, int]]]:
        """Build status-bar lines: session text, shared run lines, adaptive readout, hints."""
        lines: list[tuple[str, tuple[int, int, int]]] = []
        if session_text:
            for raw_line in session_text.strip().splitlines():
                stripped = raw_line.strip()
                if stripped and not stripped.startswith("--"):
                    lines.append((stripped, STATUS_SESSION_COLOR))
            lines.append(("", STATUS_TEXT_COLOR))

        lines.extend(
            build_run_status_lines(
                step=state.step,
                max_steps=state.max_steps,
                foods_collected=state.foods_collected,
                target_foods=state.target_foods,
                health=state.health,
                max_health=state.max_health,
                satiety=state.satiety,
                max_satiety=state.max_satiety,
                in_danger=state.in_danger,
                temperature=state.temperature,
                zone_name=state.zone_name,
                oxygen=state.oxygen,
                oxygen_zone_name=state.oxygen_zone_name,
            ),
        )

        # Adaptive-sensor readout (omitted when the sensor is disabled).
        if state.adaptive_mode is not None:
            bg = state.adaptive_background if state.adaptive_background is not None else 0.0
            readout = state.adaptive_readout if state.adaptive_readout is not None else 0.0
            lines.append(
                (
                    f"Adaptive[{state.adaptive_mode}]: B={bg:.3f} r={readout:.3f}",
                    STATUS_TEXT_COLOR,
                ),
            )

        field_name = HEATMAP_FIELDS[self._heatmap_field_idx]
        heatmap_state = f"{field_name} ON" if self._heatmap_enabled else "OFF"
        quiver_state = "ON" if self._quiver_enabled else "OFF"
        # Use the *effective* camera (not the toggle) so the label is correct when
        # `_update_camera` falls back to full-arena (e.g. zoom would exceed the plate).
        camera_state = "follow" if self._camera_following else "arena"
        lines.append(
            (
                f"[H]eatmap: {heatmap_state}  [F]ield  [G]radient: {quiver_state}  "
                f"[C]amera: {camera_state}",
                STATUS_OVERLAY_HINT_COLOR,
            ),
        )

        # Wrap to the (square, often narrow) window width so long lines — the
        # adaptive readout and the key hint — overflow onto extra rows instead of
        # clipping on the right; _resize_if_needed then grows the window height.
        max_text_width = self._width - STATUS_PADDING * 2
        wrapped: list[tuple[str, tuple[int, int, int]]] = []
        for text, color in lines:
            if not text:
                wrapped.append(("", color))
                continue
            wrapped.extend(wrap_status_line(self._font, text, color, max_text_width))
        return wrapped

    def _resize_if_needed(self, line_count: int) -> None:
        """Grow the window if the status bar needs more than ``STATUS_BAR_HEIGHT``."""
        if line_count <= self._last_status_line_count:
            return
        self._last_status_line_count = line_count
        line_height = STATUS_FONT_SIZE + 2
        needed_height = line_count * line_height + 8
        if needed_height > STATUS_BAR_HEIGHT:
            self._height = self._arena_px + needed_height
            self._screen = self._pg.display.set_mode((self._width, self._height))

    def _blit_status_lines(
        self,
        lines: list[tuple[str, tuple[int, int, int]]],
    ) -> None:
        """Blit the status bar below the arena."""
        bar_y = self._arena_px
        self._pg.draw.rect(
            self._screen,
            STATUS_BG_COLOR,
            (0, bar_y, self._width, self._height - bar_y),
        )
        line_height = STATUS_FONT_SIZE + 2
        antialias = True
        for i, (text, color) in enumerate(lines):
            if text:
                text_surf = self._font.render(text, antialias, color)
                self._screen.blit(text_surf, (STATUS_PADDING, bar_y + 4 + i * line_height))

    def close(self) -> None:
        """Clean up Pygame resources."""
        if not self._closed:
            self._closed = True
            self._pg.quit()
            logger.info("Continuous2DRenderer closed.")
