"""F.0b-i re-read on benchmark v2: the objective, the guarded plateau, the truths and the seeds.

Covers the fit-recovery spec's scenarios "The objective is the one C.1b fits", "A truth on v2 is
stable and near the wild type" and "A false plateau at wide sigma".
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import cma_recovery as cr  # noqa: E402  # pyright: ignore[reportMissingImports]
import cma_recovery_v2 as v2  # noqa: E402  # pyright: ignore[reportMissingImports]

FAMILIES = {"a": ("x", "y"), "b": ("z",)}
SCALES = {"a/x": 1.0, "a/y": 1.0, "b/z": 2.0}


def test_the_distance_weighs_families_equally_and_ignores_reported_statistics() -> None:
    """Covers "The objective is the one C.1b fits": families equal, ungraded statistics ignored."""
    truth = {"statistics": {"a/x": 0.0, "a/y": 0.0, "b/z": 0.0, "a/reported": 0.0}}
    draw = {"statistics": {"a/x": 1.0, "a/y": 1.0, "b/z": 4.0, "a/reported": 99.0}}
    # Family a: z = 1, 1 (weights 1/4 each); family b: z = 2 (weight 1/2).
    expected = ((0.25 * 1 + 0.25 * 1 + 0.5 * 4) / 1.0) ** 0.5
    assert v2.predictive_distance_v2(truth, draw, SCALES, FAMILIES) == pytest.approx(expected)


def test_undefined_statistics_count_zero_together_and_the_cap_alone() -> None:
    """A statistic undefined in both counts 0; undefined in one counts the cap."""
    both = v2.predictive_distance_v2(
        {"statistics": {"a/x": None, "a/y": 0.0, "b/z": 0.0}},
        {"statistics": {"a/x": None, "a/y": 0.0, "b/z": 0.0}},
        SCALES,
        FAMILIES,
    )
    assert both == 0.0
    one = v2.predictive_distance_v2(
        {"statistics": {"a/x": None, "a/y": 0.0, "b/z": 0.0}},
        {"statistics": {"a/x": 1.0, "a/y": 0.0, "b/z": 0.0}},
        SCALES,
        FAMILIES,
    )
    assert one == pytest.approx((0.25 * 100) ** 0.5)


def test_a_false_plateau_at_wide_sigma_does_not_stop_a_fit() -> None:
    """Covers "A false plateau at wide sigma": the guarded rule waits for sigma <= sigma0."""
    flat = [5.0] * 60
    assert v2.guarded_stop(flat, cr.SIGMA0) is True
    assert v2.guarded_stop(flat, 0.3) is False
    sigmas = [0.3] * 50 + [0.15] * 10
    assert v2.guarded_plateau_generation(flat, sigmas) == 51


def test_a_candidate_must_be_defined_and_present_on_v2() -> None:
    """Eligibility needs every statistic defined, local-search presence above 1.5, chemotaxis."""
    good = {"statistics": {"a/x": 1.0}, "presence": {"local_search": 2.0, "chemotaxis": True}}
    assert v2.eligible(good)
    assert not v2.eligible({**good, "statistics": {"a/x": None}})
    assert not v2.eligible({**good, "presence": {"local_search": 1.4, "chemotaxis": True}})
    assert not v2.eligible({**good, "presence": {"local_search": 2.0, "chemotaxis": False}})


def test_a_truth_whose_statistics_flip_is_skipped() -> None:
    """Covers "A truth on v2 is stable and near the wild type": flips and thin gaps skip it."""
    keys = ["a/x", "a/y"]
    observation = {"statistics": {"a/x": 1.0, "a/y": 1.0}}
    stable = [{"statistics": {"a/x": 1.1, "a/y": 0.9}}] * 3
    flipped = [*stable, {"statistics": {"a/x": None, "a/y": 0.9}}]
    assert v2.keep_candidate_v2(observation, stable, 0.5, 2.0, keys) == (True, "")
    keep, reason = v2.keep_candidate_v2(observation, flipped, 0.5, 2.0, keys)
    assert not keep
    assert "a/x" in reason
    keep, reason = v2.keep_candidate_v2(observation, stable, 0.5, 1.2, keys)
    assert not keep
    assert "prior - floor" in reason


def test_the_registered_seed_ranges_are_disjoint() -> None:
    """Fits, the pilot, the candidates and the discarded probe and smoke never share an animal."""
    animals = 200
    fit_max = v2.FIT_BASE + animals * (cr.FIT_BLOCK * 7 + cr.ANSWER_OFFSET + 9) + animals
    pilot_max = v2.PILOT_BASE + animals * (max(cr.PILOT_ANSWER_OFFSETS) + 9) + animals
    candidate_span = animals * (v2.CENTRE_K + 9) + animals
    ranges = [
        (v2.FIT_BASE, fit_max),
        (v2.PILOT_BASE, pilot_max),
        (v2.CANDIDATE_BASE, v2.CANDIDATE_BASE + candidate_span),
        (v2.PROBE_BASE, v2.PROBE_BASE + candidate_span),
        (v2.SMOKE_BASE, v2.SMOKE_BASE + animals * 32),
    ]
    ranges.sort()
    for (_, end), (start, _) in itertools.pairwise(ranges):
        assert end <= start
    assert v2.REPEAT_K + cr.REPEATS * 43 <= v2.REFERENCE_K
    assert v2.REFERENCE_K + v2.REFERENCES <= v2.CENTRE_K
