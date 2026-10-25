"""The region check's helpers: the gain fraction and the band cohorts' observation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import npe_region_check as rc  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_the_gain_is_one_at_the_floor_and_zero_at_the_prior() -> None:
    """A posterior as close as the truth's own re-simulation gains everything; the prior nothing."""
    assert rc.gain_fraction(7.0, 1.0, 1.0) == pytest.approx(1.0)
    assert rc.gain_fraction(7.0, 7.0, 1.0) == pytest.approx(0.0)
    assert rc.gain_fraction(7.0, 4.0, 1.0) == pytest.approx(0.5)


def test_the_band_observation_has_the_summary_shape() -> None:
    """Eleven band centres and the band batches' median presence bound, all finite."""
    import npe_recovery as nr  # pyright: ignore[reportMissingImports]

    x = rc.band_observation()
    assert x.shape == (len(nr.statistic_names()),)
    assert np.all(np.isfinite(x))
    assert x[-1] > 1.0
