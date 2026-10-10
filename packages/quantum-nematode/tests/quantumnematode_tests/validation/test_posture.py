"""The vendored eigenworm basis and real postures, and the posture instruments built on them.

Covers the realworm-behavioural-validation requirement "Posture instruments for the segmented body"
(the basis captures a real posture).
"""

from __future__ import annotations

import numpy as np
import pytest
from quantumnematode.validation.posture import (
    N_ANGLES,
    body_tangent_angles,
    eigenworm_variance,
    load_eigenworms,
    load_real_postures,
    mode_amplitude,
)


def test_the_basis_is_orthonormal_with_modes_in_columns() -> None:
    """The vendored basis loads at its pinned digest, one eigenworm per column."""
    basis = load_eigenworms()
    assert basis.shape == (N_ANGLES, N_ANGLES)
    assert np.allclose(basis.T @ basis, np.eye(N_ANGLES), atol=1e-3)


def test_the_basis_captures_the_real_postures() -> None:
    """Four eigenworms capture at least 96% of the 6,655 real postures, as the reference reports."""
    postures = load_real_postures()
    assert postures.shape == (6655, N_ANGLES)
    assert eigenworm_variance(postures, load_eigenworms()) >= 0.96


def test_a_straight_body_has_no_shape() -> None:
    """Zero curvature gives a flat, mean-removed posture of zero peak curvature."""
    angles = body_tangent_angles(np.zeros(12))
    assert angles.shape == (N_ANGLES,)
    assert np.allclose(angles, 0.0)
    assert mode_amplitude(angles[None, :], load_eigenworms())[0] == pytest.approx(0.0)


def test_a_constant_bend_has_the_curvature_as_its_slope() -> None:
    """A body bent uniformly at kappa*L = 6 has slope -6 between its segment midpoints."""
    angles = body_tangent_angles(np.full(12, 6.0))
    slope = np.gradient(angles, 1.0 / (N_ANGLES - 1))
    assert np.median(slope[10:-10]) == pytest.approx(-6.0, rel=0.05)
    assert abs(angles.mean()) < 1e-9


def test_a_larger_wave_has_a_larger_amplitude() -> None:
    """Doubling a travelling wave's curvature doubles its eigenworm amplitude."""
    basis = load_eigenworms()
    wave = 6.0 * np.sin(2.0 * np.pi * np.arange(12) / (12 * 0.65))
    small = mode_amplitude(body_tangent_angles(wave)[None, :], basis)[0]
    large = mode_amplitude(body_tangent_angles(2.0 * wave)[None, :], basis)[0]
    assert large == pytest.approx(2.0 * small)
