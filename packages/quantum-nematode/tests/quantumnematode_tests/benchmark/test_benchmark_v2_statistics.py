"""B v2's new statistics on toy tracks: the kinematics and chemotaxis families.

Covers the benchmark-grading spec's scenarios "A worm that roams too fast" and "A worm that ignores
the gradient".
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantumnematode.benchmark import chemotaxis as ch
from quantumnematode.benchmark import grading as g


def _linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a field rising 1 unit per mm along y."""
    del x
    return np.asarray(y, dtype=float)


def test_a_worm_heading_up_the_gradient_drifts_at_its_speed() -> None:
    """Straight up a linear field at 0.1 mm/s, the drift is 0.1 mm/s."""
    t = np.arange(300.0)
    x, y = np.zeros_like(t) + 5.0, 1.0 + 0.1 * t
    stats = ch.chemotaxis_statistics([(x, y)], _linear)
    assert stats["drift_mm_s"] == pytest.approx(0.1, rel=1e-6)


def test_a_worm_that_ignores_the_gradient_does_not_drift() -> None:
    """Running straight across the gradient gives no up-gradient drift."""
    t = np.arange(300.0)
    x, y = 1.0 + 0.1 * t, np.full_like(t, 5.0)
    stats = ch.chemotaxis_statistics([(x, y)], _linear)
    assert stats["drift_mm_s"] == pytest.approx(0.0, abs=1e-9)


def test_steering_toward_the_gradient_gives_a_positive_weathervane_slope() -> None:
    """A worm turning toward higher concentration at a rate set by the bearing has slope > 0."""
    heading, x, y = 0.0, [0.0], [0.0]
    for _ in range(400):
        bearing = math.pi / 2 - heading  # the gradient points along +y
        heading += math.radians(5.0) * math.sin(bearing)
        x.append(x[-1] + 0.1 * math.cos(heading))
        y.append(y[-1] + 0.1 * math.sin(heading))
    stats = ch.chemotaxis_statistics([(np.array(x), np.array(y))], _linear)
    assert stats["weathervane_deg_s"] > 0


def test_fragments_split_at_missing_samples() -> None:
    """Runs of present samples become fragments."""
    x = np.array([0.0, 1.0, np.nan, 2.0, 3.0, 4.0])
    frags = ch.split_fragments(x, x)
    assert [len(f[0]) for f in frags] == [2, 3]


def test_roaming_survival_with_and_without_censoring() -> None:
    """Kaplan-Meier: all bouts shorter than 5 min gives 0; censoring keeps survivors at risk."""
    completed = np.zeros(g.SURVIVAL_WINDOWS)
    censored = np.zeros(g.SURVIVAL_WINDOWS)
    completed[9] = 4  # four bouts end at 100 s
    assert g.roaming_survival(completed, censored, longer=0.0) == pytest.approx(0.0)
    assert g.roaming_survival(completed, censored, longer=4.0) == pytest.approx(0.5)
    censored[4] = 2  # two bouts censored at 50 s leave the risk set before 100 s
    assert g.roaming_survival(completed, censored, longer=4.0) == pytest.approx(0.5)


def test_histogram_quantile_recovers_a_known_speed() -> None:
    """A population of windows all at 0.04 mm/s has median and p90 near 0.04."""
    counts = g._speed_histogram(np.full(1000, 0.04))
    assert g._histogram_quantile(counts, 0.5) == pytest.approx(0.04, rel=0.15)
    assert g._histogram_quantile(counts, 0.9) == pytest.approx(0.04, rel=0.15)


def test_reversals_count_only_on_the_lawn() -> None:
    """A track entirely off the lawn has no on-lawn seconds or onsets."""
    t = np.arange(200.0)
    x, y = 0.1 * t, np.zeros_like(t)
    onsets, seconds = g.on_lawn_reversals(x, y, np.zeros(200, dtype=bool))
    assert (onsets, seconds) == (0, 0)
    _, seconds = g.on_lawn_reversals(x, y, np.ones(200, dtype=bool))
    assert seconds == 199
