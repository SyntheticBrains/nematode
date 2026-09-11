"""The horizon campaign's reading, and what it refuses to score.

Two failures these pin, both of which would look exactly like a clean result. A campaign missing
runs would shrink a horizon's pairing silently and still assign `does_not_transfer`, which is a
verdict about evidence that was never collected. And a record asserting why a contrast is
unavailable, rather than deriving it, can contradict the numbers printed beside it.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_horizon_multistep as hm  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]

_COMPETENT = ms.COMPETENT_THRESHOLD + 5.0
_DEAD = 1.0


def _scanned(*, drop: tuple[str, str, int] | None = None) -> dict:
    """Build a complete scan, optionally missing one cell."""
    out: dict = {h: {"learning": {}, "frozen": {}, "logs": {}} for h in hm.HORIZONS}
    for horizon in hm.HORIZONS:
        for arm in ("learning", "frozen"):
            for seed in hm.SEEDS:
                if drop == (horizon, arm, seed):
                    continue
                out[horizon][arm][seed] = object()
    return out


class TestAnIncompleteCampaignIsNotScored:
    def test_a_complete_campaign_passes(self) -> None:
        hm.require_complete(_scanned())

    def test_one_missing_run_refuses_a_verdict(self) -> None:
        # Without this the horizon would simply pair fewer seeds and still return a verdict.
        with pytest.raises(ValueError, match="incomplete"):
            hm.require_complete(_scanned(drop=("td099", "frozen", 5)))

    def test_the_refusal_names_the_missing_cell(self) -> None:
        with pytest.raises(ValueError, match=r"td0999/learning seeds \[3\]"):
            hm.require_complete(_scanned(drop=("td0999", "learning", 3)))

    def test_every_registered_cell_is_required(self) -> None:
        # Three horizons, two arms, eight seeds: the campaign as registered.
        assert len(hm.HORIZONS) * 2 * len(hm.SEEDS) == 48


class TestTheCompetenceNoteIsDerived:
    def test_it_names_the_arm_that_lacks_a_competent_seed(self) -> None:
        # The case this campaign hit: the frozen arm has one, the learning arm none.
        out = hm._full_clear(
            dict.fromkeys(hm.SEEDS, _DEAD),
            {1: _COMPETENT, **dict.fromkeys(hm.SEEDS[1:], _DEAD)},
        )
        assert out["level_contrast_available"] is False
        assert out["competent_seeds"] == {"learning": [], "frozen": [1]}
        assert "the learning arm has no seed" in out["note"]
        assert "[1]" in out["note"]

    def test_it_does_not_claim_no_competent_seed_when_there_is_one(self) -> None:
        # The defect: a hardcoded note asserting none while reporting one beside it.
        out = hm._full_clear(
            dict.fromkeys(hm.SEEDS, _DEAD),
            {1: _COMPETENT, **dict.fromkeys(hm.SEEDS[1:], _DEAD)},
        )
        assert "neither arm" not in out["note"]

    def test_neither_arm_competent_says_so(self) -> None:
        out = hm._full_clear(dict.fromkeys(hm.SEEDS, _DEAD), dict.fromkeys(hm.SEEDS, _DEAD))
        assert out["level_contrast_available"] is False
        assert "neither arm" in out["note"]

    def test_both_arms_competent_makes_the_contrast_available(self) -> None:
        competent = dict.fromkeys(hm.SEEDS, _COMPETENT)
        out = hm._full_clear(competent, dict(competent))
        assert out["level_contrast_available"] is True
        assert "available" in out["note"]

    def test_the_threshold_is_the_committed_one(self) -> None:
        out = hm._full_clear(dict.fromkeys(hm.SEEDS, _DEAD), dict.fromkeys(hm.SEEDS, _DEAD))
        assert out["threshold"] == ms.COMPETENT_THRESHOLD


class TestTheRegisteredRule:
    def _cell(self, *, effect: float, q: float) -> dict:
        return {"graded": {"effect": effect, "p_improve": 0.0, "q_improve": q, "defined": True}}

    def test_a_significant_shift_below_the_minimum_does_not_count(self) -> None:
        # The reason the minimum exists: the rank test fires on sign consistency, so a tiny
        # shift can be significant on a platform whose arms sit near zero.
        assert hm.MIN_EFFECT_FOODS == 0.5
        cells = {h: self._cell(effect=0.2, q=0.01) for h in hm.HORIZONS}
        for cell in cells.values():
            cell["beats_control"] = cell["graded"]["q_improve"] <= ms.SIG_Q and (
                cell["graded"]["effect"] >= hm.MIN_EFFECT_FOODS
            )
        assert not any(c["beats_control"] for c in cells.values())

    def test_the_pinned_horizon_beating_its_control_is_void_not_a_result(self) -> None:
        # It contradicts the pilot that motivated the change, so something else moved.
        cells = {h: {"beats_control": h == "td09"} for h in hm.HORIZONS}
        assert hm._verdict(cells)["verdict"] == "void"

    def test_no_horizon_beating_its_control_does_not_transfer(self) -> None:
        cells = {h: {"beats_control": False} for h in hm.HORIZONS}
        assert hm._verdict(cells)["verdict"] == "does_not_transfer"

    def test_a_raised_horizon_alone_transfers(self) -> None:
        cells = {h: {"beats_control": h == "td099"} for h in hm.HORIZONS}
        assert hm._verdict(cells)["verdict"] == "transfers"
