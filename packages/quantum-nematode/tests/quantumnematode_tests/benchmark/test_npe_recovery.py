"""NPE recovery's tooling: the prior, the summary, the calibration gate and the informative gate.

Covers the fit-recovery spec's scenarios "A worm that never leaves" (sentinel and flag), "An
overconfident posterior fails" and "A posterior equal to the prior fails" (on toy problems with a
known posterior).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import npe_recovery as nr  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_the_prior_maps_the_unit_box_onto_its_ranges() -> None:
    """Every parameter's prior has a range, and the box's corners land on its ends."""
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    assert set(nr.PRIOR) == set(MECHANISTIC_PARAMETERS)
    low = nr.parameters_from_unit(np.zeros(20))
    high = nr.parameters_from_unit(np.ones(20))
    for name, (_, lo, hi) in nr.PRIOR.items():
        assert low[name] == pytest.approx(lo)
        assert high[name] == pytest.approx(hi)
    middle = nr.parameters_from_unit(np.full(20, 0.5))
    assert middle["r_dwell"] == pytest.approx(math.sqrt(1e-3 * 0.5))


def test_the_prior_draws_are_registered() -> None:
    """The training draws and the truths come from one seeded stream, in that order."""
    draws = nr.prior_draws()
    assert draws.shape == (nr.TRAINING + nr.TRUTHS, 20)
    assert np.array_equal(draws, np.random.default_rng(5401).random(draws.shape))


def test_an_undefined_statistic_is_a_sentinel_with_a_flag() -> None:
    """A worm that never leaves has no pre-exit enrichment: -5 in its place, and a flag of 1."""
    raw = np.array([[0.2, np.nan, 3.0], [0.4, 1.0, 5.0]])
    mean, sd = nr.standardiser(raw)
    assert mean[1] == pytest.approx(1.0)
    x = nr.summarise(raw, mean, sd)
    assert x.shape == (2, 6)
    assert x[0, 1] == nr.SENTINEL
    assert x[0, 4] == 1.0
    assert x[1, 4] == 0.0
    assert np.all(np.isfinite(x))


def _toy(posterior_sd_factor: float, *, from_prior: bool = False) -> float:
    """Pooled coverage on a Gaussian toy: theta ~ N(0,1) and x = theta + N(0,1), posterior known."""
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(400):
        theta = rng.normal(size=3)
        x = theta + rng.normal(size=3)
        mean, sd = x / 2.0, math.sqrt(0.5) * posterior_sd_factor
        samples = (
            rng.normal(size=(2000, 3)) if from_prior else mean + sd * rng.normal(size=(2000, 3))
        )
        covered.append(nr.coverage(samples, theta))
    return float(np.mean(covered))


def test_an_exact_posterior_is_calibrated() -> None:
    """The exact posterior's 90% intervals cover the truth about 90% of the time."""
    assert nr.COVERAGE_GATE[0] <= _toy(1.0) <= nr.COVERAGE_GATE[1]


def test_an_overconfident_posterior_fails() -> None:
    """A posterior a third as wide as it should be falls below the gate."""
    assert _toy(1.0 / 3.0) < nr.COVERAGE_GATE[0]


def test_a_posterior_equal_to_the_prior_fails() -> None:
    """It is calibrated by construction, so only the informative gate can catch it."""
    assert nr.COVERAGE_GATE[0] <= _toy(1.0, from_prior=True) <= nr.COVERAGE_GATE[1]
    scales = dict.fromkeys(nr.statistic_names()[:-1], 0.05)
    truth = {"statistics": dict.fromkeys(scales, 0.2)}
    near = {"statistics": dict.fromkeys(scales, 0.21)}
    far = {"statistics": dict.fromkeys(scales, 0.6)}
    assert nr.predictive_distance(truth, near, scales) <= nr.INFORMATIVE_GATE
    assert nr.predictive_distance(truth, far, scales) > nr.INFORMATIVE_GATE


def test_undefined_in_both_agrees_and_in_one_is_capped() -> None:
    """A statistic undefined in both simulations counts 0; undefined in one counts the cap."""
    from quantumnematode.benchmark import grading as g

    scales = dict.fromkeys(nr.statistic_names()[:-1], 0.05)
    base = dict.fromkeys(scales, 0.2)
    key = next(iter(scales))
    both = nr.predictive_distance(
        {"statistics": {**base, key: None}},
        {"statistics": {**base, key: None}},
        scales,
    )
    one = nr.predictive_distance({"statistics": {**base, key: None}}, {"statistics": base}, scales)
    assert both == 0.0
    weight = g.weights(list(g.FAMILIES))[("small_lawn", key.split("/")[1])]
    assert one == pytest.approx(math.sqrt(weight * g.Z_CAP**2))


def test_contraction_is_zero_at_the_prior_and_one_at_a_point() -> None:
    """On the unit box the prior's SD is 1/sqrt(12)."""
    rng = np.random.default_rng(1)
    assert nr.contraction(rng.random((200_000, 2))) == pytest.approx([0.0, 0.0], abs=0.01)
    assert nr.contraction(np.full((10, 2), 0.3)) == pytest.approx([1.0, 1.0])
