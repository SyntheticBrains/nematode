"""A.3's registered hop predictor: its statistic, its verdict map and its committed sources."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import hop_predictor as hp  # noqa: E402  # pyright: ignore[reportMissingImports]


class TestStatistic:
    def test_spearman_matches_a_known_case(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert hp.spearman(x, x[::-1].copy()) == pytest.approx(-1.0)
        assert hp.spearman(x, np.array([1.0, 3.0, 2.0, 5.0, 4.0])) == pytest.approx(0.8)

    def test_ties_share_their_rank(self) -> None:
        assert list(hp._ranks(np.array([3.0, 1.0, 3.0]))) == [1.5, 0.0, 1.5]

    def test_a_strong_negative_relation_is_detected(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.integers(4, 15, 48).astype(float)
        y = -0.01 * x + rng.normal(0, 0.02, 48)
        got = hp.registered_test(x, y)
        assert got["rho"] < -0.3
        assert got["verdict"] == "predicts"


class TestVerdictMap:
    @pytest.mark.parametrize(
        ("rho", "p", "ci", "verdict"),
        [
            (-0.45, 0.01, (-0.6, -0.3), "predicts"),
            (0.45, 0.01, (0.3, 0.6), "opposite"),
            (-0.2, 0.04, (-0.35, -0.05), "below_minimum"),
            (0.02, 0.8, (-0.15, 0.2), "no_prediction"),
            (-0.2, 0.2, (-0.45, 0.05), "unresolved"),
        ],
    )
    def test_each_verdict(self, rho, p, ci, verdict) -> None:
        assert hp.classify(rho, p, ci) == verdict


class TestSources:
    def test_the_committed_panels_give_48_disjoint_seeds(self) -> None:
        a6, a2 = hp.gaps_a6(), hp.gaps_a2_centre()
        assert sorted(a6) == list(range(305, 337))
        assert sorted(a2) == list(range(161, 177))
        assert not set(a6) & set(a2)
