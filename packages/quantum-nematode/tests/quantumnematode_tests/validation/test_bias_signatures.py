"""Tests for the behavioural bias-curve reference signatures."""

import pytest
from quantumnematode.validation.datasets import (
    BiasCurveReference,
    _default_bias_signatures,
    load_bias_signatures,
)


def test_reference_loads_all_strategies():
    """The JSON reference loads the thresholded + threshold-free companion signatures."""
    refs = load_bias_signatures()
    assert set(refs) == {
        "klinokinesis",
        "klinotaxis",
        "klinokinesis_magnitude",
        "klinotaxis_all",
    }
    assert all(isinstance(r, BiasCurveReference) for r in refs.values())


def test_threshold_free_companions_are_sign_only():
    """A companion shares its partner's direction/null but is sign-only (no magnitude range)."""
    refs = load_bias_signatures()
    mag, thr = refs["klinokinesis_magnitude"], refs["klinokinesis"]
    assert mag.statistic == "down_up_magnitude_ratio"
    assert mag.null_value == thr.null_value
    assert mag.sign == thr.sign
    assert mag.magnitude_range is None  # sign-only cross-check
    allw, slope = refs["klinotaxis_all"], refs["klinotaxis"]
    assert allw.statistic == "weathervane_slope_all"
    assert allw.null_value == slope.null_value
    assert allw.sign == slope.sign
    assert allw.magnitude_range is None


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


def test_hardcoded_fallback_matches_json():
    """The hardcoded default signatures mirror the packaged JSON (same strategies + statistics)."""
    from_disk = load_bias_signatures()
    fallback = _default_bias_signatures()
    assert set(fallback) == set(from_disk)
    for key, ref in fallback.items():
        assert ref.statistic == from_disk[key].statistic
        assert ref.null_value == from_disk[key].null_value
        assert ref.sign == from_disk[key].sign
        assert ref.magnitude_range == from_disk[key].magnitude_range


def test_explicit_missing_path_raises(tmp_path):
    """A caller-supplied path that does not exist is an error, not a silent fallback to defaults."""
    with pytest.raises(FileNotFoundError):
        load_bias_signatures(tmp_path / "does_not_exist.json")


def test_thermotaxis_modality_loads_sign_only_setpoint_references():
    """The thermotaxis reference set has four statistics, all sign-only (homeostatic setpoint)."""
    refs = load_bias_signatures(modality="thermotaxis")
    assert set(refs) == {
        "klinokinesis",
        "klinokinesis_magnitude",
        "klinotaxis",
        "klinotaxis_all",
    }
    assert all(r.magnitude_range is None for r in refs.values())  # sign-only
    assert refs["klinokinesis"].null_value == 1.0
    assert refs["klinotaxis"].null_value == 0.0
    assert "Ryu" in refs["klinokinesis"].citation
    # Distinct from the food references (different citations for the same statistic keys).
    assert refs["klinokinesis"].citation != load_bias_signatures()["klinokinesis"].citation


def test_unknown_modality_raises():
    """An unknown modality raises rather than silently grading against the food references."""
    with pytest.raises(ValueError, match="Unknown modality"):
        load_bias_signatures(modality="temperature")
