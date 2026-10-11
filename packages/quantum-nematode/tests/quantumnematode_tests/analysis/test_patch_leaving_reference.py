"""The patch-leaving reference: band and last-window enrichment, pooled over exits.

Covers the realworm-behavioural-validation scenarios "Enrichment compares with the run's own
baseline" and "Real and simulated leaving are read alike" (one set of functions for both).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from quantumnematode.validation import roaming_dwelling as rd

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))

import patch_leaving_reference as ref  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_enrichment_compares_with_the_runs_own_baseline() -> None:
    """A band all roaming, against a quarter of on-lawn windows roaming, enriches by 0.75."""
    r, d, off = rd.ROAMING, rd.DWELLING, rd.OFF_FOOD
    # 60 on-lawn windows: 15 band windows roaming, the rest dwelling, so a quarter roam overall.
    states = np.array([d] * 42 + [r] * 15 + [d] * 3 + [off])
    out = ref.leaving([states], [[60]])
    assert out["overall_on_lawn_roaming"] == pytest.approx(0.25)
    assert out["band_30_180s_roaming"] == 1.0
    assert out["band_enrichment"] == pytest.approx(0.75)
    assert out["last_window_roaming"] == 0.0


def test_the_vendored_reference_is_positive_and_matched() -> None:
    """The committed reference leaves out the last 30 s, and its band enrichment is positive."""
    import json

    data = json.loads((_REPO / "data/roaming_dwelling/scheer2023_leaving.json").read_text())
    assert data["band_seconds"] == [30.0, 180.0]
    assert data["classifier"]["band_enrichment"] > 0
    assert data["authors_labels"]["band_enrichment"] > 0
