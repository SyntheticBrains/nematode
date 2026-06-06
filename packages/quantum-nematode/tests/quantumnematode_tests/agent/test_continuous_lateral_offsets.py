"""Unit tests for continuous-heading klinotaxis lateral sampling (T5 §3.5).

`_continuous_lateral_offsets` samples the integer cells perpendicular to a
continuous heading. It must match the grid `_compute_lateral_offsets` at the four
cardinal headings (so grid behaviour is unchanged where they coincide) and rotate
smoothly for non-cardinal headings.
"""

from __future__ import annotations

import math

from quantumnematode.agent.agent import (
    _compute_lateral_offsets,
    _continuous_lateral_offsets,
)
from quantumnematode.env.env import Direction

_POS = (10, 10)
_GRID = 20


class TestMatchesGridAtCardinals:
    def test_up(self) -> None:
        assert _continuous_lateral_offsets(_POS, math.pi / 2, 1, _GRID) == _compute_lateral_offsets(
            Direction.UP,
            _POS,
            _GRID,
        )

    def test_right(self) -> None:
        assert _continuous_lateral_offsets(_POS, 0.0, 1, _GRID) == _compute_lateral_offsets(
            Direction.RIGHT,
            _POS,
            _GRID,
        )

    def test_down(self) -> None:
        assert _continuous_lateral_offsets(
            _POS,
            -math.pi / 2,
            1,
            _GRID,
        ) == _compute_lateral_offsets(
            Direction.DOWN,
            _POS,
            _GRID,
        )

    def test_left(self) -> None:
        assert _continuous_lateral_offsets(_POS, math.pi, 1, _GRID) == _compute_lateral_offsets(
            Direction.LEFT,
            _POS,
            _GRID,
        )


class TestRotatesAndClamps:
    def test_diagonal_heading(self) -> None:
        # Heading up-right (45°): left is up-left, right is down-right.
        left, right = _continuous_lateral_offsets(_POS, math.pi / 4, 1, _GRID)
        assert left == (9, 11)
        assert right == (11, 9)

    def test_sweep_scales_offset(self) -> None:
        left, right = _continuous_lateral_offsets(_POS, math.pi / 2, 3, _GRID)  # UP, sweep 3
        assert left == (7, 10)
        assert right == (13, 10)

    def test_clamped_to_bounds(self) -> None:
        # At the left edge, an upward heading's left sample clamps to x=0.
        left, _right = _continuous_lateral_offsets((0, 10), math.pi / 2, 3, _GRID)
        assert left[0] == 0
