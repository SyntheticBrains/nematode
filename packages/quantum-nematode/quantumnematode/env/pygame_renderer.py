"""Pygame-based graphical renderer for the Pixel theme.

Renders the simulation in a Pygame window with layered surfaces:
1. Background (soil)
2. Temperature zone overlays
3. Toxic zone overlays
4. Entities (food, predators, nematode) - drawn with transparency so zones show through
5. UI status bar
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantumnematode.env.sprites import (
    CELL_SIZE,
    create_sprites,
    create_zone_overlay,
    draw_body_segment,
)
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.env.env import DynamicForagingEnvironment, Viewport

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
        ):
            self._zone_overlays[name] = create_zone_overlay(pygame, name)

        self._closed = False
        self._last_status_line_count = 0
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
        # 7 = max run-level lines (title, step, food, hp, satiety, status, temp)
        self._resize_if_needed(session_lines + 7)

        viewport = env.get_viewport_bounds()

        # Clear screen
        self._screen.fill(STATUS_BG_COLOR)

        self._render_background(viewport)
        self._render_temperature_zones(env, viewport)
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
            if _in_view(food_pos[0], food_pos[1]):
                px, py = self._cell_to_pixel(food_pos[0], food_pos[1], viewport)
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

        # Run-level info
        lines.append(("Run:", STATUS_TEXT_COLOR))
        lines.append((f"Step: {step}/{max_steps}", STATUS_TEXT_COLOR))
        lines.append((f"Food: {foods_collected}/{target_foods}", STATUS_TEXT_COLOR))
        lines.append((f"HP: {health:.0f}/{max_health:.0f}", STATUS_TEXT_COLOR))
        lines.append((f"Satiety: {satiety:.0f}/{max_satiety:.0f}", STATUS_TEXT_COLOR))

        danger_text = "IN DANGER" if in_danger else "SAFE"
        danger_color = STATUS_DANGER_COLOR if in_danger else STATUS_SAFE_COLOR
        lines.append((f"Status: {danger_text}", danger_color))

        if temperature is not None and zone_name:
            lines.append((f"Temp: {temperature:.1f}C ({zone_name})", STATUS_TEXT_COLOR))

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
            logger.info("PygameRenderer closed.")
