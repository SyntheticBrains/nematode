"""Tests for the behavioural bias-curve reference signatures (§3)."""

from quantumnematode.validation.datasets import (
    BiasCurveReference,
    load_bias_signatures,
)


def test_reference_loads_both_strategies():
    """The JSON reference loads with both documented klinotaxis strategies."""
    refs = load_bias_signatures()
    assert set(refs) == {"klinokinesis", "klinotaxis"}
    assert all(isinstance(r, BiasCurveReference) for r in refs.values())


def test_klinokinesis_reference_has_sign_range_and_citation():
    """Klinokinesis: a dimensionless ratio with null 1.0, positive sign, and a magnitude range."""
    ref = load_bias_signatures()["klinokinesis"]
    assert ref.statistic == "down_up_turn_ratio"
    assert ref.null_value == 1.0
    assert ref.sign == 1
    assert ref.magnitude_range is not None
    lo, hi = ref.magnitude_range
    assert lo < hi
    assert lo > ref.null_value  # a present bias exceeds the no-bias null
    assert "Pierce-Shimomura" in ref.citation


def test_klinotaxis_reference_is_sign_only_with_citation():
    """Klinotaxis: a sign-only slope reference (null 0.0, positive sign, no magnitude range)."""
    ref = load_bias_signatures()["klinotaxis"]
    assert ref.statistic == "weathervane_slope"
    assert ref.null_value == 0.0
    assert ref.sign == 1
    assert ref.magnitude_range is None  # sign-only: the no-bias null is representable
    assert "Iino" in ref.citation


def test_fallback_matches_json(tmp_path):
    """A missing file falls back to the hardcoded defaults (same strategies + statistics)."""
    from_disk = load_bias_signatures()
    fallback = load_bias_signatures(tmp_path / "does_not_exist.json")
    assert set(fallback) == set(from_disk)
    for key, ref in fallback.items():
        assert ref.statistic == from_disk[key].statistic
        assert ref.null_value == from_disk[key].null_value
        assert ref.sign == from_disk[key].sign
        assert ref.magnitude_range == from_disk[key].magnitude_range
