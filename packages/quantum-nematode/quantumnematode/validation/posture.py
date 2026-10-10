"""Posture instruments against real *C. elegans* postures: the eigenworm basis and amplitude.

A posture here is the worm's tangent angle at 100 points from head to tail, with its mean removed,
the form the eigenworm basis is defined on (Stephens et al. 2008). Each body segment holds one
angle, placed at its midpoint and interpolated linearly to the 100 points, so the slope between
midpoints is the segment's curvature, as on a smooth worm.

* **Eigenworm variance**: the share of the postures' pooled sum of squares that the first ``k``
  modes capture. Real postures give about 96% at four modes.
* **Undulation amplitude**: each posture's radius in the plane of the first two eigenworms, whose
  angle there is the undulation's phase (Stephens et al. 2008). Unlike a peak of the angle's
  derivative, it does not amplify the tracking noise in real postures.

The basis and the real postures are vendored under ``data/posture`` with their provenance; their
digests are pinned here and checked on load.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
POSTURE_DIR = _PROJECT_ROOT / "data" / "posture"
EIGENWORMS_PATH = POSTURE_DIR / "EigenWorms.csv"
SHAPES_PATH = POSTURE_DIR / "shapes.csv"
EIGENWORMS_SHA256 = "bc806b9073972f6c6f6b0c48ecf8b9278bff2433ee2b39906423d58b156d930f"
SHAPES_SHA256 = "410abb65af193b86d273bdbba670406c088032b9774e5808b02bb138950fb4b1"
N_ANGLES = 100


def _load(path: Path, digest: str) -> np.ndarray:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        msg = f"{path} does not match its pinned SHA-256; it may be an LFS pointer or modified"
        raise ValueError(msg)
    return np.loadtxt(path, delimiter=",")


def load_eigenworms() -> np.ndarray:
    """Return the basis as ``(100 angles, 100 modes)``: each column is one eigenworm."""
    return _load(EIGENWORMS_PATH, EIGENWORMS_SHA256)


def load_real_postures() -> np.ndarray:
    """Return the 6,655 real postures as ``(n, 100)`` mean-removed tangent angles."""
    return _load(SHAPES_PATH, SHAPES_SHA256)


def body_tangent_angles(curvature: np.ndarray) -> np.ndarray:
    """Turn segment curvatures ``(..., n_segments)`` in kappa*L into ``(..., 100)`` tangent angles.

    Each segment bends its successor by ``kappa / n``. The segment angles sit at their midpoints and
    are interpolated linearly to the 100 points head to tail, held flat beyond the end midpoints;
    the mean is removed.
    """
    n = curvature.shape[-1]
    segment_angles = -np.concatenate(
        [np.zeros((*curvature.shape[:-1], 1)), np.cumsum(curvature[..., :-1] / n, axis=-1)],
        axis=-1,
    )
    return segment_angles @ _interpolation(n).T - _mean_of(segment_angles @ _interpolation(n).T)


def _interpolation(n: int) -> np.ndarray:
    """Return the ``(100, n)`` weights that interpolate ``n`` midpoint values to 100 points."""
    midpoints = (np.arange(n) + 0.5) / n
    weights = np.zeros((N_ANGLES, n))
    for j, s in enumerate(np.linspace(0.0, 1.0, N_ANGLES)):
        hi = int(np.clip(np.searchsorted(midpoints, s), 1, n - 1))
        lo = hi - 1
        frac = float(np.clip((s - midpoints[lo]) / (midpoints[hi] - midpoints[lo]), 0.0, 1.0))
        weights[j, lo], weights[j, hi] = 1.0 - frac, frac
    return weights


def _mean_of(angles: np.ndarray) -> np.ndarray:
    return angles.mean(axis=-1, keepdims=True)


def eigenworm_variance(postures: np.ndarray, basis: np.ndarray, modes: int = 4) -> float:
    """Return the share of the postures' pooled sum of squares the first ``modes`` capture."""
    projections = postures @ basis[:, :modes]
    return float((projections**2).sum() / (postures**2).sum())


def mode_amplitude(postures: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return each posture's radius in the plane of the first two eigenworms."""
    projections = postures @ basis[:, :2]
    return np.hypot(projections[..., 0], projections[..., 1])
