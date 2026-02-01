"""Procedural sprite generation for the Pixel theme.

Sprites are 32x32 pixel surfaces drawn using pygame.draw primitives,
designed to be biologically accurate to C. elegans and its natural environment.
"""

from __future__ import annotations

import math
import random as _rng
from typing import Any

# --------------------------------------------------------------------------- #
# Color palette
# --------------------------------------------------------------------------- #

# Nematode (C. elegans) -- translucent cream/tan
NEMATODE_HEAD_COLOR = (220, 195, 160)
NEMATODE_HEAD_OUTLINE = (180, 155, 120)
NEMATODE_PHARYNX_COLOR = (200, 170, 130)
NEMATODE_BODY_COLOR = (210, 185, 150)
NEMATODE_BODY_OUTLINE = (170, 145, 110)
NEMATODE_EYE_COLOR = (60, 60, 60)

# Food -- bacterial colony (E. coli / OP50 lawn)
FOOD_COLOR = (120, 180, 80)
FOOD_HIGHLIGHT = (160, 210, 120)
FOOD_DARK = (80, 140, 50)

# Random predator -- nematode-trapping fungal hyphae (Arthrobotrys oligospora)
PREDATOR_RANDOM_COLOR = (160, 120, 180)
PREDATOR_RANDOM_HIGHLIGHT = (190, 160, 210)

# Stationary predator -- fungal constricting ring trap (Drechslerella)
PREDATOR_STATIONARY_COLOR = (130, 80, 150)
PREDATOR_STATIONARY_RING = (160, 100, 180)

# Pursuit predator -- predatory mite
PREDATOR_PURSUIT_COLOR = (200, 100, 60)
PREDATOR_PURSUIT_HIGHLIGHT = (230, 140, 80)
PREDATOR_PURSUIT_DARK = (160, 70, 40)

# Environment
SOIL_COLOR = (45, 35, 30)
SOIL_VARIATION = (55, 45, 38)

# Temperature zone overlay colors (with alpha)
ZONE_LETHAL_COLD = (30, 60, 200, 90)
ZONE_DANGER_COLD = (50, 130, 210, 70)
ZONE_DISCOMFORT_COLD = (100, 180, 220, 50)
ZONE_COMFORT = (0, 0, 0, 0)  # transparent
ZONE_DISCOMFORT_HOT = (220, 200, 80, 50)
ZONE_DANGER_HOT = (220, 140, 40, 70)
ZONE_LETHAL_HOT = (220, 50, 30, 90)

# Toxic zone overlay
ZONE_TOXIC = (140, 60, 180, 80)

CELL_SIZE = 32


def create_sprites(pg: Any) -> dict[str, Any]:  # noqa: ANN401
    """Create all sprites and return them in a dict.

    Parameters
    ----------
    pg : module
        The pygame module (passed to avoid top-level import).

    Returns
    -------
    dict[str, Any]
        Mapping of sprite name to pygame.Surface.
    """
    sprites: dict[str, Any] = {}

    sprites["empty"] = _make_soil(pg)
    sprites["food"] = _make_food(pg)
    sprites["predator_random"] = _make_predator_random(pg)
    sprites["predator_stationary"] = _make_predator_stationary(pg)
    sprites["predator_pursuit"] = _make_predator_pursuit(pg)
    sprites["body"] = _make_body(pg)

    # Directional head sprites
    for direction in ("up", "down", "left", "right"):
        sprites[f"head_{direction}"] = _make_head(pg, direction)

    return sprites


# --------------------------------------------------------------------------- #
# Sprite factory helpers
# --------------------------------------------------------------------------- #


def _make_soil(pg: Any) -> Any:  # noqa: ANN401
    """Dark earth/soil background tile with subtle texture."""
    surf = pg.Surface((CELL_SIZE, CELL_SIZE))
    surf.fill(SOIL_COLOR)
    # Subtle speckle texture
    _rng.seed(42)  # deterministic texture
    for _ in range(12):
        x = _rng.randint(0, CELL_SIZE - 2)  # noqa: S311
        y = _rng.randint(0, CELL_SIZE - 2)  # noqa: S311
        pg.draw.rect(surf, SOIL_VARIATION, (x, y, 2, 2))
    return surf


def _make_food(pg: Any) -> Any:  # noqa: ANN401
    """Bacterial colony -- green clustered dots on soil."""
    surf = _make_soil(pg).copy()
    c = CELL_SIZE // 2
    pg.draw.circle(surf, FOOD_COLOR, (c, c), 10)
    pg.draw.circle(surf, FOOD_HIGHLIGHT, (c - 3, c - 3), 5)
    pg.draw.circle(surf, FOOD_COLOR, (c + 7, c + 5), 4)
    pg.draw.circle(surf, FOOD_DARK, (c - 6, c + 6), 3)
    pg.draw.circle(surf, FOOD_COLOR, (c + 4, c - 7), 3)
    pg.draw.circle(surf, FOOD_HIGHLIGHT, (c - 5, c - 8), 2)
    return surf


def _make_head(pg: Any, direction: str) -> Any:  # noqa: ANN401
    """Nematode head with pharynx bulb, facing *direction*."""
    surf = _make_soil(pg).copy()
    c = CELL_SIZE // 2

    if direction == "up":
        pg.draw.ellipse(surf, NEMATODE_HEAD_COLOR, (c - 7, c - 10, 14, 20))
        pg.draw.ellipse(surf, NEMATODE_HEAD_OUTLINE, (c - 7, c - 10, 14, 20), 1)
        pg.draw.ellipse(surf, NEMATODE_PHARYNX_COLOR, (c - 5, c - 2, 10, 10))
        pg.draw.circle(surf, NEMATODE_EYE_COLOR, (c - 3, c - 6), 2)
        pg.draw.circle(surf, NEMATODE_EYE_COLOR, (c + 3, c - 6), 2)
    elif direction == "down":
        pg.draw.ellipse(surf, NEMATODE_HEAD_COLOR, (c - 7, c - 10, 14, 20))
        pg.draw.ellipse(surf, NEMATODE_HEAD_OUTLINE, (c - 7, c - 10, 14, 20), 1)
        pg.draw.ellipse(surf, NEMATODE_PHARYNX_COLOR, (c - 5, c - 8, 10, 10))
        pg.draw.circle(surf, NEMATODE_EYE_COLOR, (c - 3, c + 6), 2)
        pg.draw.circle(surf, NEMATODE_EYE_COLOR, (c + 3, c + 6), 2)
    elif direction == "left":
        pg.draw.ellipse(surf, NEMATODE_HEAD_COLOR, (c - 10, c - 7, 20, 14))
        pg.draw.ellipse(surf, NEMATODE_HEAD_OUTLINE, (c - 10, c - 7, 20, 14), 1)
        pg.draw.ellipse(surf, NEMATODE_PHARYNX_COLOR, (c - 2, c - 5, 10, 10))
        pg.draw.circle(surf, NEMATODE_EYE_COLOR, (c - 6, c - 3), 2)
        pg.draw.circle(surf, NEMATODE_EYE_COLOR, (c - 6, c + 3), 2)
    else:  # right
        pg.draw.ellipse(surf, NEMATODE_HEAD_COLOR, (c - 10, c - 7, 20, 14))
        pg.draw.ellipse(surf, NEMATODE_HEAD_OUTLINE, (c - 10, c - 7, 20, 14), 1)
        pg.draw.ellipse(surf, NEMATODE_PHARYNX_COLOR, (c - 8, c - 5, 10, 10))
        pg.draw.circle(surf, NEMATODE_EYE_COLOR, (c + 6, c - 3), 2)
        pg.draw.circle(surf, NEMATODE_EYE_COLOR, (c + 6, c + 3), 2)

    return surf


def _make_body(pg: Any) -> Any:  # noqa: ANN401
    """Nematode body segment -- rounded cream/tan segment."""
    surf = _make_soil(pg).copy()
    c = CELL_SIZE // 2
    pg.draw.circle(surf, NEMATODE_BODY_COLOR, (c, c), 8)
    pg.draw.circle(surf, NEMATODE_BODY_OUTLINE, (c, c), 8, 1)
    pg.draw.circle(surf, NEMATODE_HEAD_COLOR, (c - 2, c - 2), 4)
    return surf


def _make_predator_random(pg: Any) -> Any:  # noqa: ANN401
    """Nematode-trapping fungal hyphae -- branching purple tendrils."""
    surf = _make_soil(pg).copy()
    c = CELL_SIZE // 2
    pg.draw.line(surf, PREDATOR_RANDOM_COLOR, (c, c + 10), (c, c - 10), 2)
    pg.draw.line(surf, PREDATOR_RANDOM_COLOR, (c, c - 4), (c + 8, c - 10), 2)
    pg.draw.line(surf, PREDATOR_RANDOM_COLOR, (c, c - 4), (c - 8, c - 10), 2)
    pg.draw.line(surf, PREDATOR_RANDOM_COLOR, (c, c + 4), (c + 7, c + 10), 2)
    pg.draw.line(surf, PREDATOR_RANDOM_COLOR, (c, c + 4), (c - 7, c + 10), 2)
    knob_positions = [
        (c, c - 10),
        (c + 8, c - 10),
        (c - 8, c - 10),
        (c + 7, c + 10),
        (c - 7, c + 10),
    ]
    for pos in knob_positions:
        pg.draw.circle(surf, PREDATOR_RANDOM_HIGHLIGHT, pos, 3)
    return surf


def _make_predator_stationary(pg: Any) -> Any:  # noqa: ANN401
    """Fungal constricting ring trap -- circular ring structure."""
    surf = _make_soil(pg).copy()
    c = CELL_SIZE // 2
    pg.draw.circle(surf, PREDATOR_STATIONARY_COLOR, (c, c), 12, 3)
    pg.draw.circle(surf, PREDATOR_STATIONARY_RING, (c, c), 7, 2)
    pg.draw.circle(surf, (80, 40, 100), (c, c), 3)
    for angle_offset in range(0, 360, 60):
        rad = math.radians(angle_offset)
        ex = int(c + 14 * math.cos(rad))
        ey = int(c + 14 * math.sin(rad))
        pg.draw.line(
            surf,
            PREDATOR_STATIONARY_COLOR,
            (c + int(12 * math.cos(rad)), c + int(12 * math.sin(rad))),
            (ex, ey),
            1,
        )
    return surf


def _make_predator_pursuit(pg: Any) -> Any:  # noqa: ANN401
    """Predatory mite -- small arachnid shape in orange-red."""
    surf = _make_soil(pg).copy()
    c = CELL_SIZE // 2
    pg.draw.ellipse(surf, PREDATOR_PURSUIT_COLOR, (c - 6, c - 8, 12, 16))
    pg.draw.circle(surf, PREDATOR_PURSUIT_HIGHLIGHT, (c, c - 10), 5)
    for dy, spread in [(-4, 10), (-1, 12), (2, 11), (5, 9)]:
        pg.draw.line(surf, PREDATOR_PURSUIT_DARK, (c - 2, c + dy), (c - spread, c + dy - 3), 1)
        pg.draw.line(surf, PREDATOR_PURSUIT_DARK, (c + 2, c + dy), (c + spread, c + dy - 3), 1)
    pg.draw.circle(surf, (40, 20, 20), (c - 2, c - 12), 1)
    pg.draw.circle(surf, (40, 20, 20), (c + 2, c - 12), 1)
    return surf


def create_zone_overlay(
    pg: Any,  # noqa: ANN401
    zone_name: str,
) -> Any:  # noqa: ANN401
    """Create a semi-transparent zone overlay surface.

    Parameters
    ----------
    pg : module
        The pygame module.
    zone_name : str
        One of: lethal_cold, danger_cold, discomfort_cold, comfort,
        discomfort_hot, danger_hot, lethal_hot, toxic.

    Returns
    -------
    Any
        A CELL_SIZE x CELL_SIZE SRCALPHA surface with the zone tint.
    """
    color_map = {
        "lethal_cold": ZONE_LETHAL_COLD,
        "danger_cold": ZONE_DANGER_COLD,
        "discomfort_cold": ZONE_DISCOMFORT_COLD,
        "comfort": ZONE_COMFORT,
        "discomfort_hot": ZONE_DISCOMFORT_HOT,
        "danger_hot": ZONE_DANGER_HOT,
        "lethal_hot": ZONE_LETHAL_HOT,
        "toxic": ZONE_TOXIC,
    }
    rgba = color_map.get(zone_name, ZONE_COMFORT)
    surf = pg.Surface((CELL_SIZE, CELL_SIZE), pg.SRCALPHA)
    surf.fill(rgba)
    return surf
