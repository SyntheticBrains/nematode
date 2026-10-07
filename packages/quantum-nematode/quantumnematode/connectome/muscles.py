"""The 95 body wall muscles of the hermaphrodite, named as the adjacency matrices name them.

Body wall muscle runs in four longitudinal quadrants: dorsal and ventral, each on the left and the
right. A cell is named by its quadrant and its position along the body counted from the head, so
``dBWML1`` is the most anterior dorsal-left cell. Three quadrants hold 24 cells and the ventral-left
quadrant holds 23, which is what the Cook 2019 matrices list.
"""

from types import MappingProxyType

BODY_WALL_MUSCLE_QUADRANTS = MappingProxyType(
    {"dBWML": 24, "dBWMR": 24, "vBWML": 23, "vBWMR": 24},
)
"""Cells per quadrant: dorsal-left, dorsal-right, ventral-left, ventral-right."""

BODY_WALL_MUSCLES: tuple[str, ...] = tuple(
    f"{quadrant}{position}"
    for quadrant, count in BODY_WALL_MUSCLE_QUADRANTS.items()
    for position in range(1, count + 1)
)
"""Every body wall muscle, quadrant by quadrant, each from head to tail."""

EXPECTED_BODY_WALL_MUSCLE_COUNT = 95
"""Expected size of BODY_WALL_MUSCLES. Asserted at import time."""

if len(BODY_WALL_MUSCLES) != EXPECTED_BODY_WALL_MUSCLE_COUNT:
    msg = (
        f"BODY_WALL_MUSCLES has {len(BODY_WALL_MUSCLES)} entries; "
        f"expected exactly {EXPECTED_BODY_WALL_MUSCLE_COUNT}."
    )
    raise AssertionError(msg)

MAX_QUADRANT_POSITIONS = 24
"""The most cells any quadrant holds; positions are spread over segments on this scale."""


def muscle_position(name: str) -> tuple[str, int]:
    """Return a body wall muscle's quadrant and its position, 1 being the most anterior."""
    quadrant, digits = name[:5], name[5:]
    if quadrant not in BODY_WALL_MUSCLE_QUADRANTS or not digits.isdigit():
        msg = f"{name!r} is not a body wall muscle name"
        raise ValueError(msg)
    position = int(digits)
    if not 1 <= position <= BODY_WALL_MUSCLE_QUADRANTS[quadrant]:
        msg = f"{name!r} names position {position}, beyond its quadrant's cells"
        raise ValueError(msg)
    return quadrant, position


def muscle_segment(name: str, n_segments: int) -> int:
    """Return the 0-based body segment, head to tail, that a body wall muscle belongs to.

    Positions are spread evenly over the segments on the 24-cell scale, so at 12 segments
    positions 1-2 make segment 0, 3-4 segment 1, and so on; the ventral-left quadrant's 23rd cell
    sits alone in the last segment.
    """
    if not 1 <= n_segments <= MAX_QUADRANT_POSITIONS:
        msg = f"n_segments must be in [1, {MAX_QUADRANT_POSITIONS}], got {n_segments}"
        raise ValueError(msg)
    _quadrant, position = muscle_position(name)
    return (position - 1) * n_segments // MAX_QUADRANT_POSITIONS
