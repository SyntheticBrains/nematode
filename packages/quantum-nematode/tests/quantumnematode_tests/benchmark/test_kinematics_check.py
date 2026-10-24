"""The on-food kinematics check: its bout rule, boundary strata, fixed spans and what it reads.

Covers the benchmark-references spec's scenarios "The check reads no graded statistic" and "A
disagreement is brought back, not applied".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from quantumnematode.benchmark import references as refs

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))

import extract_benchmark_references as ex  # noqa: E402  # pyright: ignore[reportMissingImports]
import primitive_kinematics_check as kc  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_bouts_merge_one_frame_gaps_and_drop_single_frames() -> None:
    """A one-frame gap joins two runs; a lone frame is dropped; unfiltered keeps every run."""
    flag = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert kc.bouts(flag, filtered=True) == [(1, 6), (11, 14)]
    assert kc.bouts(flag, filtered=False) == [(1, 3), (4, 6), (8, 9), (11, 14)]


def test_boundary_distance_is_signed_inside_a_circle() -> None:
    """On a 2 mm lawn, the centre is 2 mm inside and a point 3 mm out is 1 mm outside."""
    radii = np.full(64, 2.0)
    d = refs.boundary_distance(np.array([0.0, 3.0, 0.0]), np.array([0.0, 0.0, 1.9]), radii)
    assert np.allclose(d, [2.0, -1.0, 0.1])
    assert [kc.stratum(v) for v in d] == ["inside", "outside", "edge"]


def test_the_reversal_span_is_two_sem_around_the_sources() -> None:
    """The acted-on span runs 2.3 - 2(1.6/sqrt 249) to 3.4 + 2(1.8/sqrt 135) seconds."""
    low, high = kc.REFERENCE["reversal_duration_s"]["span"]
    assert low == pytest.approx(2.0972, abs=1e-4)
    assert high == pytest.approx(3.7098, abs=1e-4)
    acted = [q for q, r in kc.REFERENCE.items() if r["acted_on"]]
    assert acted == ["reversal_duration_s"]


def _store(n: int = 6, frames: int = 90) -> refs.ScheerTracks:
    rng = np.random.default_rng(3)
    seconds = frames // 3
    x = np.cumsum(rng.normal(0, 0.05, (n, seconds)), axis=1)
    y = np.cumsum(rng.normal(0, 0.05, (n, seconds)), axis=1)
    reversal = np.zeros((n, frames), dtype=bool)
    reversal[:, 10:19] = True  # 3-s reversals
    flags = [reversal, np.zeros_like(reversal), rng.random((n, frames)) < 0.2]
    flags += [np.zeros_like(reversal), np.ones_like(reversal), np.zeros_like(reversal)]
    return refs.ScheerTracks(
        x=x,
        y=y,
        frame_flags=ex.pack_flags(flags),
        flag_names=tuple(name for name, _ in ex.FRAME_FLAGS),
        in_lawn_run=None,  # type: ignore[arg-type]
        out_lawn_run=None,  # type: ignore[arg-type]
        labels=None,  # type: ignore[arg-type]
        boundary_radius=np.full((n, 64), 5.0),
        genotype=np.zeros(n, dtype=np.int16),
        genotypes=("PD1074_od2_Data",),
        date=np.array([20210101 + i % 3 for i in range(n)]),
        video=np.zeros(n, dtype=np.int32),
        videos=("v",),
        origin=np.zeros(n, dtype=np.int8),
        fig1_row=np.full(n, -1),
        side=np.array([refs.HELD_OUT] * (n - 1) + [refs.BANDS]),
    )


def test_the_check_reads_no_graded_statistic() -> None:
    """With labels and run masks absent, the check still runs, on held-out animals only."""
    result = kc.measure(_store(), filtered=True)
    assert result["animals"] == 5
    inside = result["reversal_duration_s"]["inside"]
    assert inside["value"] == pytest.approx(3.0)
    assert inside["n"] == 5


def test_a_disagreement_is_reported_and_edits_nothing() -> None:
    """A reversal duration outside the span is flagged; the primitives file is not touched."""
    path = _REPO / "data" / "locomotion" / "primitives.json"
    before = path.read_bytes()
    filtered = kc.measure(_store(), filtered=True)
    filtered["reversal_duration_s"]["inside"]["interval_95"] = [5.0, 6.0]
    verdict = kc.verdicts(filtered)["reversal_duration_s"]
    assert verdict["disagreement"]
    assert path.read_bytes() == before
