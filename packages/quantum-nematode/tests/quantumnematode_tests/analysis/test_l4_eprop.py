"""R.2's reading, and the four things it must not do.

* read a campaign the one-step control does not license. That control's ``eprop_symmetric`` arm is
  the exact REINFORCE gradient of its plastic layer, so a failure there means the implementation is
  wrong and no connectome number means anything -- which makes stage 1 a stop clause the harness has
  to consult rather than a task someone ticks;
* call beating a damaged floor "learning the cell". Only competence makes a time-to-competence
  contrast defined, which is what R.1b needs and what 7b's gate asks for;
* attribute a difference between ``symmetric`` and ``random`` to the signal's direction, when the
  two also differ in which units the signal can reach. The matched contrasts are what separate them;
* let a missing run shrink an arm's pairing silently. ``does_not_learn`` stops the programme, so it
  must not be assignable on partial evidence.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_eprop as ep  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]


@dataclass
class _Record:
    """The fields of a scanned run this harness reads."""

    success: float
    foods: float


def _arm(foods: float, *, clear: float = 0.0, spread: float = 0.1) -> dict[int, _Record]:
    return {s: _Record(clear, foods + spread * i) for i, s in enumerate(ep.SEEDS)}


def _scanned(frozen: float = 3.2, **overrides: dict[int, _Record]) -> dict[str, Any]:
    runs: dict[str, Any] = {name: _arm(3.8) for name in ep.ARMS}
    runs["frozen"] = {s: _Record(0.0, frozen) for s in ep.SEEDS}
    runs.update(overrides)
    # No logs: the drift and hop columns report unavailable rather than building a topology.
    return {"runs": runs, "logs": {}}


def _stage_one(tmp_path: Path, reading: str = "valid", **extra: Any) -> Path:
    record = {
        "outcome": "fail",
        "arms": {"eprop": {"reading": reading, "broadcast_passes": False, **extra}},
    }
    path = tmp_path / "stage1.json"
    path.write_text(json.dumps(record))
    return path


class TestStageOneGatesTheCampaign:
    def test_a_void_control_voids_the_reading(self, tmp_path: Path) -> None:
        competent = ms.COMPETENT_THRESHOLD + 5.0
        out = ep.analyse(
            _scanned(random=_arm(15.0, clear=competent)),
            stage_one=_stage_one(tmp_path, "void", void_reason="the gradient arm did not learn"),
        )
        assert out["verdict"] == "void"
        assert "does not license" in out["why"]

    def test_a_missing_control_record_is_not_treated_as_a_pass(self) -> None:
        # Absent a record the campaign is unreadable, not readable-by-default.
        out = ep.analyse(_scanned(random=_arm(15.0)))
        assert out["verdict"] == "void"
        assert out["stage_one"]["reading"] == "unknown"

    def test_a_valid_control_lets_the_campaign_read(self, tmp_path: Path) -> None:
        out = ep.analyse(_scanned(), stage_one=_stage_one(tmp_path))
        assert out["verdict"] != "void"

    def test_the_broadcast_arms_position_is_carried(self, tmp_path: Path) -> None:
        # Reported whatever the reading: it bounds what this campaign could show.
        out = ep.analyse(
            _scanned(),
            stage_one=_stage_one(tmp_path, broadcast_passes=True),
        )
        assert out["stage_one"]["broadcast_passes"] is True


class TestBeatingAFloorIsNotLearningTheCell:
    def test_an_arm_can_beat_its_floor_without_competence(self, tmp_path: Path) -> None:
        out = ep.analyse(_scanned(random=_arm(8.0, clear=0.05)), stage_one=_stage_one(tmp_path))
        assert out["arms"]["random"]["verdict"] == "beats_floor"
        assert not out["arms"]["random"]["reaches_competence"]
        assert out["verdict"] == "learns_below_competence"
        assert "R.1b stays blocked" in out["why"]

    def test_competence_plus_the_floor_is_learning_the_cell(self, tmp_path: Path) -> None:
        competent = ms.COMPETENT_THRESHOLD + 1.0
        out = ep.analyse(
            _scanned(symmetric=_arm(15.0, clear=competent)),
            stage_one=_stage_one(tmp_path),
        )
        assert out["verdict"] == "learns_the_cell"
        assert out["learned_the_cell"] == ["symmetric"]

    def test_competence_without_the_floor_does_not_count(self, tmp_path: Path) -> None:
        # An arm clearing the cell while its floor clears it just as well has not been shown to have
        # learned anything: the floor is what says the update did the work.
        competent = ms.COMPETENT_THRESHOLD + 5.0
        out = ep.analyse(
            _scanned(frozen=3.8, random=_arm(3.8, clear=competent)),
            stage_one=_stage_one(tmp_path),
        )
        assert out["reaching_competence"] == ["random"]
        assert out["learned_the_cell"] == []
        assert out["verdict"] != "learns_the_cell"

    def test_no_arm_beating_its_floor_stops_the_programme(self, tmp_path: Path) -> None:
        out = ep.analyse(_scanned(), stage_one=_stage_one(tmp_path))
        assert out["verdict"] == "does_not_learn"
        assert out["beating_floor"] == []
        assert "two independent eligibilities" in out["why"]
        assert "7b proceeds under PPO" in out["why"]


class TestTheMatchedContrastsSeparateSourceFromReach:
    def test_the_direction_contrast_is_matched_on_reach(self, tmp_path: Path) -> None:
        # symmetric and random_motor both reach the 39 pooled units, so their difference is the
        # signal's direction and nothing else.
        out = ep.analyse(
            _scanned(symmetric=_arm(9.0), random_motor=_arm(5.0)),
            stage_one=_stage_one(tmp_path),
        )
        contrast = out["matched_contrasts"]["direction_at_matched_reach"]
        assert contrast["arms"] == ["symmetric", "random_motor"]
        assert contrast["difference_foods"] == pytest.approx(4.0, abs=1e-6)

    def test_the_reach_contrast_is_matched_on_source(self, tmp_path: Path) -> None:
        out = ep.analyse(
            _scanned(random=_arm(9.0), random_motor=_arm(6.0)),
            stage_one=_stage_one(tmp_path),
        )
        contrast = out["matched_contrasts"]["reach_at_matched_source"]
        assert contrast["arms"] == ["random", "random_motor"]
        assert contrast["difference_foods"] == pytest.approx(3.0, abs=1e-6)

    def test_both_are_reported_in_every_branch(self, tmp_path: Path) -> None:
        for scanned in (_scanned(), _scanned(random=_arm(9.0, clear=99.0))):
            out = ep.analyse(scanned, stage_one=_stage_one(tmp_path))
            assert set(out["matched_contrasts"]) == set(ep.MATCHED_CONTRASTS)

    def test_the_arms_reach_is_recorded_with_each_row(self, tmp_path: Path) -> None:
        out = ep.analyse(_scanned(), stage_one=_stage_one(tmp_path))
        assert out["arms"]["symmetric"]["reach"] == "the 39-unit pool"
        assert out["arms"]["random"]["reach"] == "all 302"


class TestTheForbiddenCellIsStated:
    def test_it_travels_with_every_result(self, tmp_path: Path) -> None:
        # True directions reaching all 302 units is a cell the mechanism forbids, not one that was
        # skipped, and the record has to say which.
        out = ep.analyse(_scanned(), stage_one=_stage_one(tmp_path))
        assert out["forbidden_cell"]["reach"] == "all 302"
        assert out["forbidden_cell"]["source"] == "readout transpose"
        assert "identically zero" in out["forbidden_cell"]["why"]

    def test_it_is_not_among_the_arms(self) -> None:
        pairs = {(meta["source"], meta["reach"]) for meta in ep.ARMS.values()}
        assert (ep.FORBIDDEN_CELL["source"], ep.FORBIDDEN_CELL["reach"]) not in pairs


class TestTheMinimaUseTheMatchedReference:
    def test_the_gap_is_taken_against_the_harvested_ppo_level(self) -> None:
        out = ep.minima(effect=1.0, frozen_mean=3.0)
        assert out["matched_gap"] == pytest.approx(ep.PPO_MATCHED_FOODS - 3.0)
        assert out["matched_minimum"] == pytest.approx(0.1 * (ep.PPO_MATCHED_FOODS - 3.0))

    def test_the_more_demanding_minimum_binds(self) -> None:
        # A high floor shrinks the reachable gap, at which point the absolute minimum is the binding
        # one; a low floor makes the relative one bind.
        assert ep.minima(1.0, frozen_mean=18.0)["binding_minimum"] == pytest.approx(ep.MIN_FOODS)
        assert ep.minima(1.0, frozen_mean=3.0)["binding_minimum"] > ep.MIN_FOODS

    def test_an_effect_below_the_binding_minimum_fails(self) -> None:
        out = ep.minima(effect=0.5, frozen_mean=3.0)
        assert not out["passes"]
        assert "below" in out["why"]


class TestAnIncompleteCampaignIsNotScored:
    def _logs(self) -> dict[str, Any]:
        return {"runs": {name: _arm(3.8) for name in (*ep.ARMS, "frozen")}, "logs": {}}

    def test_a_complete_campaign_passes(self) -> None:
        ep.require_complete(self._logs())

    def test_a_missing_run_refuses_a_verdict(self) -> None:
        scanned = self._logs()
        del scanned["runs"]["random"][ep.SEEDS[-1]]
        with pytest.raises(ValueError, match="campaign is incomplete"):
            ep.require_complete(scanned)

    def test_a_missing_frozen_run_refuses_a_verdict(self) -> None:
        scanned = self._logs()
        del scanned["runs"]["frozen"][ep.SEEDS[0]]
        with pytest.raises(ValueError, match="campaign is incomplete"):
            ep.require_complete(scanned)

    def test_the_registered_seed_count_is_sixteen(self) -> None:
        # Logbook 059 registered 16 for this gate, not R.1c's and R.1d's 8.
        assert tuple(range(1, 17)) == ep.SEEDS


class TestTheLabelReadsOneArmEach:
    @pytest.mark.parametrize(
        "suffix",
        ["symmetric", "random_motor", "random", "scalar", "frozen"],
    )
    def test_each_arm_label_matches_itself(self, suffix: str) -> None:
        match = ep._LABEL.match(f"{ep._STEM}_{suffix}-seed3.log")
        assert match is not None
        assert match.group("arm") == suffix
        assert match.group("seed") == "3"

    def test_the_perturbation_arms_are_not_this_comparison(self) -> None:
        # R.1c's directory carries the node-perturbation arms; none of them is an e-prop arm.
        stem = ep._STEM.replace("_eprop", "_nodepert")
        assert ep._LABEL.match(f"{stem}_motor-seed1.log") is None


class TestTheReferencesAreTheCommittedOnes:
    """Every comparator in this harness is a number another campaign measured and committed.

    Transcribed by hand, so a typo would move a verdict boundary quietly. These read the committed
    records rather than trusting the constants -- the cross-check R.2's registration asks for, in
    place of re-scoring R.1c's logs through a harness whose labels do not match them.
    """

    @staticmethod
    def _record(relative: str) -> dict[str, Any]:
        path = _root / "docs" / "experiments" / "logbooks" / "supporting" / relative
        return json.loads(path.read_text())

    def test_the_nodepert_pair_is_r1cs_motor_arm(self) -> None:
        record = self._record("061-l4-reduced-perturbation/reduced_perturbation.json")
        motor = record.get("result", record)["sets"]["motor"]
        assert motor["learning_mean_foods"] == pytest.approx(ep.NODEPERT_LEARNING_FOODS, abs=5e-4)
        assert motor["frozen_mean_foods"] == pytest.approx(ep.NODEPERT_FROZEN_FOODS, abs=5e-4)
        assert motor["graded"]["effect"] == pytest.approx(ep.NODEPERT_SHIFT_FOODS, abs=5e-4)

    def test_the_ppo_reference_is_r1ds_matched_harvest(self) -> None:
        record = self._record("062-l4-frozen-readout/frozen_readout.json")
        assert record["protocol"]["ppo_matched_foods"] == pytest.approx(ep.PPO_MATCHED_FOODS)

    def test_the_competence_threshold_is_the_committed_one(self) -> None:
        assert ep.COMPETENCE_THRESHOLD == ms.COMPETENT_THRESHOLD

    def test_the_drift_reference_brackets_both_prior_campaigns(self) -> None:
        # R.1c measured 1.37-1.38x across every perturbation dimension, R.1d 1.38-1.42x across four
        # readouts. The range this harness reports against has to contain both.
        low, high = ep.DRIFT_REFERENCE
        assert low <= 1.37
        assert high >= 1.42
