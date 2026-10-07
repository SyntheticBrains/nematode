"""The signed map from cells to body wall muscle drive, pooled by quadrant and segment.

Every cell with a neuromuscular junction drives the muscles it synapses onto, weighted by the
junction's serial-section count and signed by what body wall muscle responds to: its receptors
are excited by acetylcholine and inhibited by GABA. Cells releasing anything else (glutamate,
dopamine, or no known transmitter) have no established effect on body wall muscle and contribute
nothing. Each quadrant-segment column is divided by the summed magnitude of its signed entries, so
a column's drive from rates in ``[-1, 1]`` is a weighted mean in ``[-1, 1]``, comparable across
segments however densely each is innervated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantumnematode.connectome.muscles import (
    BODY_WALL_MUSCLE_QUADRANTS,
    muscle_position,
    muscle_segment,
)
from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantumnematode.connectome.model import NeuromuscularJunction

QUADRANTS: tuple[str, ...] = tuple(BODY_WALL_MUSCLE_QUADRANTS)
"""Dorsal-left, dorsal-right, ventral-left, ventral-right."""

MUSCLE_SIGN: dict[str, int] = {"ACh": 1, "GABA": -1}
"""How body wall muscle responds to a released transmitter; any other transmitter is 0."""


@dataclass(frozen=True)
class DriveMap:
    """A fixed cell-to-muscle-drive map.

    Attributes
    ----------
    cells : tuple[str, ...]
        Every cell with a neuromuscular junction, sorted.
    columns : tuple[tuple[str, int], ...]
        ``(quadrant, segment)`` for each column, quadrant-major, segments head to tail.
    matrix : np.ndarray
        ``(len(cells), len(columns))``: signed, column-normalised weights.
    zero_weight_cells : tuple[str, ...]
        Cells with a junction whose transmitter has no established body wall effect.
    """

    cells: tuple[str, ...]
    columns: tuple[tuple[str, int], ...]
    matrix: np.ndarray
    zero_weight_cells: tuple[str, ...]


def muscle_sign(cell: str) -> int:
    """Return how body wall muscle responds to a cell's primary transmitter: +1, -1 or 0."""
    _cell_class, transmitter = NEURON_CLASSIFICATION[cell]
    return MUSCLE_SIGN.get(transmitter or "", 0)


def neuromuscular_drive_map(
    junctions: Sequence[NeuromuscularJunction],
    n_segments: int,
) -> DriveMap:
    """Build the signed, column-normalised map from cells to quadrant-segment drive."""
    cells = tuple(sorted({j.pre for j in junctions}))
    columns = tuple((q, s) for q in QUADRANTS for s in range(n_segments))
    row = {cell: i for i, cell in enumerate(cells)}
    col = {key: i for i, key in enumerate(columns)}
    matrix = np.zeros((len(cells), len(columns)))
    for junction in junctions:
        quadrant, _position = muscle_position(junction.muscle)
        key = (quadrant, muscle_segment(junction.muscle, n_segments))
        matrix[row[junction.pre], col[key]] += muscle_sign(junction.pre) * junction.weight
    magnitude = np.abs(matrix).sum(axis=0)
    matrix = np.divide(matrix, magnitude, out=np.zeros_like(matrix), where=magnitude > 0)
    zero = tuple(cell for cell in cells if muscle_sign(cell) == 0)
    return DriveMap(cells=cells, columns=columns, matrix=matrix, zero_weight_cells=zero)
