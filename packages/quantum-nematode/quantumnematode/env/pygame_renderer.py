"""Pygame-based graphical renderer for the Pixel theme.

Renders the simulation in a Pygame window with layered surfaces:
1. Background (soil)
2. Temperature zone overlays
3. Toxic zone overlays
4. Entities (food, predators, nematode)
5. UI status bar
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantumnematode.env.sprites import CELL_SIZE, create_sprites, create_zone_overlay
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.env.env import DynamicForagingEnvironment, Viewport

# Status bar height in pixels
STATUS_BAR_HEIGHT = 80
STATUS_FONT_SIZE = 16
STATUS_BG_COLOR = (30, 25, 22)
STATUS_TEXT_COLOR = (200, 200, 200)
STATUS_DANGER_COLOR = (220, 60, 40)
STATUS_SAFE_COLOR = (80, 200, 80)

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
        logger.info(
            f"PygameRenderer initialized: {self._width}x{self._height} "
            f"({viewport_size[0]}x{viewport_size[1]} cells @ {cell_size}px)",
        )

    @property
    def closed(self) -> bool:
        """Whether the window has been closed."""
        return self._closed

    def pump_events(self) -> bool:
        """Process Pygame events. Returns False if window was closed."""
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                self._closed = True
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
    ) -> None:
        """Render one complete frame."""
        if self._closed:
            return

        if not self.pump_events():
            return

        viewport = env.get_viewport_bounds()

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
        """Render temperature zone overlays."""
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
        """Render food, predators, nematode body, and head."""
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

        # Body segments
        body_sprite = self._sprites["body"]
        for seg in env.body:
            if _in_view(seg[0], seg[1]):
                px, py = self._cell_to_pixel(seg[0], seg[1], viewport)
                self._screen.blit(body_sprite, (px, py))

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
    ) -> None:
        """Render status information below the grid."""
        bar_y = self._viewport_size[1] * self._cell_size
        self._pg.draw.rect(
            self._screen,
            STATUS_BG_COLOR,
            (0, bar_y, self._width, STATUS_BAR_HEIGHT),
        )

        lines = [
            f"Step: {step}/{max_steps}   Food: {foods_collected}/{target_foods}   "
            f"HP: {health:.0f}/{max_health:.0f}   Satiety: {satiety:.0f}/{max_satiety:.0f}",
        ]

        status_parts = []
        status_parts.append("IN DANGER" if in_danger else "SAFE")

        if temperature is not None and zone_name:
            status_parts.append(f"Temp: {temperature:.1f}C ({zone_name})")

        lines.append("  ".join(status_parts))

        antialias = True
        for i, line in enumerate(lines):
            color = STATUS_DANGER_COLOR if (i == 1 and in_danger) else STATUS_TEXT_COLOR
            text_surf = self._font.render(line, antialias, color)
            self._screen.blit(text_surf, (8, bar_y + 6 + i * (STATUS_FONT_SIZE + 4)))

    def close(self) -> None:
        """Clean up Pygame resources."""
        if not self._closed:
            self._closed = True
            self._pg.quit()
            logger.info("PygameRenderer closed.")
