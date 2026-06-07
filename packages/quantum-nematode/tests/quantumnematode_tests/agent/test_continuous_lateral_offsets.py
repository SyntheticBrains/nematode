"""Unit tests for continuous-heading klinotaxis lateral sampling.

`_continuous_lateral_offsets` samples real-valued points perpendicular to a
continuous heading (no integer-cell snapping — Rung-2 continuous-field sampling).
It must match the grid `_compute_lateral_offsets` at the four cardinal headings
(where the offsets are integer-valued, so grid behaviour is unchanged where they
coincide) and rotate smoothly for non-cardinal headings.
"""

from __future__ import annotations

import math

import pytest
from quantumnematode.agent.agent import (
    _compute_lateral_offsets,
    _continuous_lateral_offsets,
)
from quantumnematode.env.env import Direction

_POS: tuple[float, float] = (10.0, 10.0)
_IPOS: tuple[int, int] = (10, 10)  # integer position for the grid offsets comparator
_GRID: int = 20


class TestMatchesGridAtCardinals:
    """Continuous sampling reduces to the grid offsets at the four cardinal headings."""

    def test_up(self) -> None:
        """Heading +y (π/2) matches `Direction.UP`'s lateral offsets."""
        assert _continuous_lateral_offsets(_POS, math.pi / 2, 1, _GRID) == _compute_lateral_offsets(
            Direction.UP,
            _IPOS,
            _GRID,
        )

    def test_right(self) -> None:
        """Heading +x (0) matches `Direction.RIGHT`'s lateral offsets."""
        assert _continuous_lateral_offsets(_POS, 0.0, 1, _GRID) == _compute_lateral_offsets(
            Direction.RIGHT,
            _IPOS,
            _GRID,
        )

    def test_down(self) -> None:
        """Heading -y (-π/2) matches `Direction.DOWN`'s lateral offsets."""
        assert _continuous_lateral_offsets(
            _POS,
            -math.pi / 2,
            1,
            _GRID,
        ) == _compute_lateral_offsets(
            Direction.DOWN,
            _IPOS,
            _GRID,
        )

    def test_left(self) -> None:
        """Heading -x (π) matches `Direction.LEFT`'s lateral offsets."""
        assert _continuous_lateral_offsets(_POS, math.pi, 1, _GRID) == _compute_lateral_offsets(
            Direction.LEFT,
            _IPOS,
            _GRID,
        )


class TestRotatesAndClamps:
    """Off-cardinal headings rotate the sample cells and stay clamped to bounds."""

    def test_diagonal_heading(self) -> None:
        """A 45° heading puts the left sample up-left and the right sample down-right.

        Real-valued (un-snapped) sample points: ±sweep·(cos45°, sin45°) from centre.
        """
        left, right = _continuous_lateral_offsets(_POS, math.pi / 4, 1, _GRID)
        half_diag = math.sqrt(0.5)  # sin/cos of 45°
        assert left == pytest.approx((10.0 - half_diag, 10.0 + half_diag))
        assert right == pytest.approx((10.0 + half_diag, 10.0 - half_diag))

    def test_sweep_scales_offset(self) -> None:
        """A larger sweep amplitude widens the perpendicular sample spacing."""
        left, right = _continuous_lateral_offsets(_POS, math.pi / 2, 3, _GRID)  # UP, sweep 3
        assert left == (7, 10)
        assert right == (13, 10)

    def test_clamped_to_bounds(self) -> None:
        """A sample that would fall outside the grid clamps to the world edge."""
        # At the left edge, an upward heading's left sample clamps to x=0.
        left, _right = _continuous_lateral_offsets((0, 10), math.pi / 2, 3, _GRID)
        assert left[0] == 0
