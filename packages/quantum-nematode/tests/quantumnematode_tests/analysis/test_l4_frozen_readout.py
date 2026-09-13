"""R.1d's reading, and the ordering it will not over-interpret.

Four failures these pin:

* the anatomical label absorbing a substituted arm, or the reverse: the readout suffix is optional,
  so a greedy pattern would file `..._motor_ppo` as the baseline;
* an arm reading as the plausible learner working, when every substituted arm takes its readout
  from a gradient-trained run and so cannot satisfy the deliverable whatever it returns;
* beating a damaged floor read as learning the cell, which is what would wrongly unblock R.1b;
* the scale and the direction conflated: PPO's readout is 5.5x the anatomical norm *and*
  near-orthogonal, so an ordering has to say which of the two it supports.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_frozen_readout as fr  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]


@dataclass
class _Record:
    """The fields of a scanned run this harness reads."""

    success: float
    foods: float


def _arm(
    learning: float,
    frozen: float,
    *,
    clear: float = 0.0,
    spread: float = 0.1,
) -> dict[str, Any]:
    return {
        "learning": {s: _Record(clear, learning + spread * i) for i, s in enumerate(fr.SEEDS)},
        "frozen": {s: _Record(0.0, frozen) for s in fr.SEEDS},
        "logs": {},
    }


def _scanned(**overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base = {name: _arm(3.8, 3.2) for name in fr.ARMS}
    base.update(overrides)
    return base


class TestTheLabelSeparatesTheBaselineFromTheSubstitutedArms:
    """The readout suffix is optional, so the baseline and the substituted arms share a stem."""

    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            ("", ("anatomical", "learning")),
            ("_frozen", ("anatomical", "frozen")),
            ("_ppo", ("ppo", "learning")),
            ("_ppo_frozen", ("ppo", "frozen")),
            ("_rotated", ("rotated", "learning")),
            ("_anatomical_scaled", ("anatomical_scaled", "learning")),
            ("_anatomical_scaled_frozen", ("anatomical_scaled", "frozen")),
        ],
    )
    def test_each_label_reads_as_its_own_arm(self, suffix: str, expected: tuple[str, str]) -> None:
        match = fr._LABEL.match(f"{fr._STEM}{suffix}-seed1.log")
        assert match is not None
        got = (
            match.group("arm") or "anatomical",
            "frozen" if match.group("frozen") else "learning",
        )
        assert got == expected

    def test_a_different_mask_is_not_this_comparison(self) -> None:
        # R.1c's directory carries every mask; only its `motor` arm is the baseline here.
        stem = fr._STEM.replace("_motor", "_hop1")
        assert fr._LABEL.match(f"{stem}-seed1.log") is None


class TestItCannotSatisfyTheDeliverable:
    def test_the_flag_is_false_whatever_the_result(self) -> None:
        competent = ms.COMPETENT_THRESHOLD + 5.0
        for cells in (_scanned(), _scanned(ppo=_arm(15.0, 3.2, clear=competent))):
            out = fr.analyse(cells)
            assert out["satisfies_plausibility_deliverable"] is False
            assert "not the plausible learner working" in out["deliverable_note"]

    def test_a_positive_result_names_the_follow_up_and_it_is_not_ppo(self) -> None:
        competent = ms.COMPETENT_THRESHOLD + 5.0
        out = fr.analyse(_scanned(ppo=_arm(15.0, 3.2, clear=competent)))
        assert out["verdict"] == "readout_is_the_handicap"
        assert "better-grounded readout, not from PPO" in out["why"]


class TestBeatingAFloorIsNotLearningTheCell:
    def test_an_arm_can_beat_its_floor_without_competence(self) -> None:
        out = fr.analyse(_scanned(ppo=_arm(8.0, 3.2, clear=0.05)))
        assert out["readouts"]["ppo"]["verdict"] == "beats_floor"
        assert not out["readouts"]["ppo"]["reaches_competence"]
        assert out["verdict"] == "readout_helps_but_not_enough"
        assert "R.1b stays blocked" in out["why"]

    def test_competence_takes_precedence_over_the_floor_verdict(self) -> None:
        competent = ms.COMPETENT_THRESHOLD + 1.0
        out = fr.analyse(_scanned(rotated=_arm(15.0, 3.2, clear=competent)))
        assert out["verdict"] == "readout_is_the_handicap"
        assert out["reaching_competence"] == ["rotated"]

    def test_no_substituted_arm_beating_its_floor_eliminates_the_readout(self) -> None:
        out = fr.analyse(_scanned())
        assert out["verdict"] == "readout_not_the_handicap"
        assert out["substituted_beating_floor"] == []
        assert "R.2 becomes the live path" in out["why"]

    def test_the_baseline_beating_its_own_floor_is_not_a_substitution_result(self) -> None:
        # `anatomical` is the comparator; it clearing its own floor says nothing about the readout.
        out = fr.analyse(_scanned(anatomical=_arm(8.0, 3.2)))
        assert out["substituted_beating_floor"] == []
        assert out["verdict"] == "readout_not_the_handicap"


class TestTheOrderingSeparatesScaleFromDirection:
    def test_scale_alone_is_reported_as_scale(self) -> None:
        # anatomical_scaled and ppo both rise, and they rise together: the norm did it, not the
        # direction, so the follow-up is the readout-scale/action-noise interaction.
        out = fr.analyse(
            _scanned(anatomical_scaled=_arm(9.0, 3.2), rotated=_arm(9.0, 3.2), ppo=_arm(9.0, 3.2)),
        )
        assert "SCALE is what mattered" in out["ordering"]["reading"]

    def test_ppos_direction_winning_is_reported_as_direction(self) -> None:
        out = fr.analyse(
            _scanned(anatomical_scaled=_arm(4.0, 3.2), rotated=_arm(4.0, 3.2), ppo=_arm(12.0, 3.2)),
        )
        assert "DIRECTION matters" in out["ordering"]["reading"]

    def test_any_direction_helping_is_not_credited_to_ppo(self) -> None:
        out = fr.analyse(
            _scanned(
                anatomical_scaled=_arm(4.0, 3.2),
                rotated=_arm(12.0, 3.2),
                ppo=_arm(12.2, 3.2),
            ),
        )
        assert "actively bad" in out["ordering"]["reading"]

    def test_four_alike_is_not_the_handicap(self) -> None:
        out = fr.analyse(_scanned())
        assert "not the handicap" in out["ordering"]["reading"]

    def test_an_unregistered_ordering_is_reported_as_mixed(self) -> None:
        # PPO's own direction actively underperforming every other arm is informative and matches no
        # registered pattern, so it is reported as it is rather than resolved toward the nearest
        # one.
        out = fr.analyse(_scanned(ppo=_arm(1.0, 3.2)))
        assert out["ordering"]["reading"].startswith("mixed")

    def test_the_gains_are_reported_numerically(self) -> None:
        out = fr.analyse(_scanned(anatomical_scaled=_arm(9.0, 3.2)))
        order = out["ordering"]
        assert order["scale_gain"] == pytest.approx(5.2, abs=0.2)
        assert set(order["levels"]) == set(fr.ARMS)


class TestTheMinimaUseTheMatchedReference:
    def test_the_matched_reference_is_the_harvests(self) -> None:
        # 18.945 at these eight seeds and this action scale, against 058's 19.31 at 32 seeds and std
        # 1.0.
        assert pytest.approx(18.945) == fr.PPO_MATCHED_FOODS
        assert pytest.approx(19.31) == fr.PPO_REFERENCE_058

    def test_both_are_reported_and_the_binding_one_is_the_larger(self) -> None:
        result = fr.minima(effect=1.4, frozen_mean=3.2)
        assert result["matched_minimum"] == pytest.approx(1.5745, abs=1e-3)
        assert result["minimum_against_058"] == pytest.approx(1.611, abs=1e-3)
        assert result["binding_minimum"] == pytest.approx(1.5745, abs=1e-3)
        assert not result["passes"]

    def test_the_absolute_minimum_binds_when_the_gap_is_small(self) -> None:
        result = fr.minima(effect=0.6, frozen_mean=18.0)
        assert result["binding_minimum"] == pytest.approx(fr.MIN_FOODS)
        assert not result["passes"]


class TestAnIncompleteCampaignIsNotScored:
    def _complete(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "learning": dict.fromkeys(fr.SEEDS, object()),
                "frozen": dict.fromkeys(fr.SEEDS, object()),
                "logs": {},
            }
            for name in fr.ARMS
        }

    def test_a_complete_campaign_passes(self) -> None:
        fr.require_complete(self._complete())

    def test_a_missing_run_refuses_a_verdict(self) -> None:
        scanned = self._complete()
        del scanned["ppo"]["frozen"][4]
        with pytest.raises(ValueError, match=r"ppo/frozen seeds \[4\]"):
            fr.require_complete(scanned)

    def test_the_new_runs_are_forty_eight(self) -> None:
        # Three substituted readouts; the anatomical pair is R.1c's, reused under the equivalence
        # test.
        assert len(fr.SUBSTITUTED) * 2 * len(fr.SEEDS) == 48
