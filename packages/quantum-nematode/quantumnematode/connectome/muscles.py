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
