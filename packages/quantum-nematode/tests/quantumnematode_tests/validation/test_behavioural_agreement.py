"""Tests for the behavioural bias-statistic agreement grading (§4.1)."""

from quantumnematode.validation.behavioural_agreement import (
    Verdict,
    bootstrap_ci,
    grade_statistic,
)
from quantumnematode.validation.datasets import BiasCurveReference

_RANGED = BiasCurveReference(
    strategy="klinokinesis",
    statistic="down_up_turn_ratio",
    null_value=1.0,
    sign=1,
    magnitude_range=(1.5, 3.0),
    citation="Pierce-Shimomura et al. (1999)",
    notes="test",
)
_SIGN_ONLY = BiasCurveReference(
    strategy="klinotaxis",
    statistic="weathervane_slope",
    null_value=0.0,
    sign=1,
    magnitude_range=None,
    citation="Iino & Yoshida (2009)",
    notes="test",
)


def test_bootstrap_ci_orders_and_brackets_mean():
    """The CI brackets the mean and lo <= hi; a singleton collapses to the point."""
    mean, lo, hi = bootstrap_ci([1.8, 2.0, 2.2, 2.1, 1.9, 2.3, 2.0, 2.1])
    assert lo <= mean <= hi
    assert abs(mean - 2.05) < 0.2
    assert bootstrap_ci([3.0]) == (3.0, 3.0, 3.0)
    assert bootstrap_ci([]) != bootstrap_ci([])  # NaNs are not equal to themselves


def test_ranged_reproduced_when_significant_and_in_range():
    """Ratios tightly around 2.0 (in [1.5, 3.0], CI above 1.0) -> REPRODUCED."""
    res = grade_statistic([1.9, 2.0, 2.1, 2.0, 1.95, 2.05, 2.0, 2.1], _RANGED)
    assert res.verdict is Verdict.REPRODUCED
    assert res.ci_lo > _RANGED.null_value


def test_ranged_partial_when_significant_but_below_range():
    """A significant but weak bias (ratio ~1.2, below [1.5, 3.0]) -> PARTIAL."""
    res = grade_statistic([1.18, 1.2, 1.22, 1.2, 1.19, 1.21, 1.2, 1.2], _RANGED)
    assert res.verdict is Verdict.PARTIAL
    assert res.ci_lo > _RANGED.null_value  # direction is significant


def test_ranged_partial_when_leaning_but_ci_includes_null():
    """A noisy correct-lean whose CI still spans 1.0 -> PARTIAL (direction not significant)."""
    res = grade_statistic([0.5, 0.6, 0.5, 0.5, 2.8, 2.9, 0.5, 0.6], _RANGED)
    assert res.mean > _RANGED.null_value
    assert res.ci_lo <= _RANGED.null_value <= res.ci_hi
    assert res.verdict is Verdict.PARTIAL


def test_absent_when_mean_wrong_side():
    """Ratios below the null (turning elevated up-gradient) -> ABSENT."""
    res = grade_statistic([0.7, 0.8, 0.75, 0.6, 0.9, 0.7, 0.8, 0.72], _RANGED)
    assert res.verdict is Verdict.ABSENT


def test_sign_only_reproduced_on_significant_positive_slope():
    """A significantly positive slope with no magnitude range -> REPRODUCED on sign alone."""
    res = grade_statistic([0.3, 0.35, 0.28, 0.32, 0.31, 0.29, 0.33, 0.3], _SIGN_ONLY)
    assert res.verdict is Verdict.REPRODUCED
    assert res.magnitude_range is None


def test_empty_is_absent_and_serialisable():
    """No values -> ABSENT, and the result serialises to a JSON-friendly dict."""
    res = grade_statistic([], _SIGN_ONLY)
    assert res.verdict is Verdict.ABSENT
    assert res.n == 0
    d = res.to_dict()
    assert d["verdict"] == "ABSENT"
    assert d["magnitude_range"] is None
