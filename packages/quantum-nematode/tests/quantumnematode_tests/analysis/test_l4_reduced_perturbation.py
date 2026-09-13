"""R.1c's reading, and the two questions it refuses to merge.

Three failures these pin, each of which would misreport the campaign:

* beating a damaged frozen floor read as learning the cell, which is what would wrongly unblock
  R.1b -- block V's contrast is on time to competence, and that is undefined for an arm which never
  becomes competent;
* the relative minimum taken against block V's frozen figure, which runs at a different action noise
  and implies a reachable gap this campaign does not have;
* a campaign missing runs scored anyway, assigning `not_reducible` on partial evidence.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_reduced_perturbation as rp  # noqa: E402  # pyright: ignore[reportMissingImports]


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
    """One declared set's pair. ``spread`` breaks the ties a rank test needs broken."""
    return {
        "learning": {s: _Record(clear, learning + spread * i) for i, s in enumerate(rp.SEEDS)},
        "frozen": {s: _Record(0.0, frozen) for s in rp.SEEDS},
        "logs": {},
    }


def _scanned(**overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base = {name: _arm(2.0, 1.8) for name in rp.ARMS}
    base.update(overrides)
    return base


class TestBeatingAFloorIsNotLearningTheCell:
    """The distinction R.1b turns on, and the one this change's own registration conflated."""

    def test_an_arm_can_beat_its_floor_without_reaching_competence(self) -> None:
        # The measured pilot shape: a real, significant shift at ~0% full clear.
        out = rp.analyse(_scanned(motor=_arm(4.4, 1.7, clear=0.03)))
        cell = out["sets"]["motor"]
        assert cell["verdict"] == "beats_floor"
        assert not cell["reaches_competence"]
        assert out["verdict"] == "dimension_reducible"
        assert out["r1b_unblocked"] is False
        assert "stays blocked" in out["r1b_note"]

    def test_reaching_competence_is_what_unblocks_r1b(self) -> None:
        competent = ms.COMPETENT_THRESHOLD + 5.0
        out = rp.analyse(_scanned(motor=_arm(14.0, 1.7, clear=competent)))
        assert out["sets"]["motor"]["reaches_competence"]
        assert out["r1b_unblocked"] is True
        assert "unblocked at motor" in out["r1b_note"]

    def test_the_two_lists_are_reported_separately(self) -> None:
        competent = ms.COMPETENT_THRESHOLD + 5.0
        out = rp.analyse(
            _scanned(motor=_arm(4.4, 1.7, clear=0.03), hop1=_arm(14.0, 1.7, clear=competent)),
        )
        assert "motor" in out["sets_beating_floor"]
        assert out["sets_reaching_competence"] == ["hop1"]


class TestTheMinimaUseTheMatchedReference:
    def test_the_binding_minimum_is_the_more_demanding_of_the_two(self) -> None:
        # Against this arm's own frozen 1.73 the gap is 17.58 and the minimum 1.76; against block
        # V's 3.82 it is 15.49 and 1.55. The demanding one binds.
        result = rp.minima(effect=1.65, frozen_mean=1.73)
        assert result["registered_minimum"] == pytest.approx(1.549, abs=1e-3)
        assert result["matched_minimum"] == pytest.approx(1.758, abs=1e-3)
        assert result["binding_minimum"] == pytest.approx(1.758, abs=1e-3)
        assert not result["passes"], "1.65 clears the registered bar but not the matched one"

    def test_both_references_are_reported(self) -> None:
        result = rp.minima(effect=3.0, frozen_mean=1.73)
        assert result["passes"]
        assert set(result) >= {"matched_minimum", "registered_minimum", "binding_minimum"}

    def test_the_absolute_minimum_still_binds_on_a_tiny_gap(self) -> None:
        # A frozen arm near the PPO level leaves almost no reachable gap, so the relative
        # minimum becomes trivial and the absolute one is what stops a meaningless shift counting.
        result = rp.minima(effect=0.5, frozen_mean=19.0)
        assert not result["passes"]
        assert "1.0 foods minimum" in result["why"]

    def test_a_shift_below_only_the_relative_minimum_names_it(self) -> None:
        result = rp.minima(effect=1.2, frozen_mean=1.73)
        assert not result["passes"]
        assert "of-gap minimum" in result["why"]


class TestTheRegisteredVerdicts:
    def test_no_set_beating_its_floor_is_not_reducible(self) -> None:
        out = rp.analyse(_scanned())
        assert out["verdict"] == "not_reducible"
        assert out["sets_beating_floor"] == []

    def test_the_causal_mask_alone_is_its_own_verdict(self) -> None:
        # The strongest available result: nothing given up to remove the disconnected draws.
        out = rp.analyse(_scanned(causal=_arm(6.0, 1.8)))
        assert out["verdict"] == "causal_mask_sufficient"

    def test_a_reduced_set_winning_beside_causal_reads_as_reducible(self) -> None:
        out = rp.analyse(_scanned(causal=_arm(6.0, 1.8), motor=_arm(6.0, 1.8)))
        assert out["verdict"] == "dimension_reducible"

    def test_full_winning_alone_is_not_a_reduction(self) -> None:
        # `full` is today's behaviour; beating its own floor says nothing about restricting credit.
        out = rp.analyse(_scanned(full=_arm(6.0, 1.8)))
        assert out["verdict"] == "not_reducible"
        assert out["sets_beating_floor"] == ["full"]

    def test_motor_last_winning_where_motor_does_not_is_reported(self) -> None:
        # The partial reading the registration asked to be kept rather than resolved: draws per
        # decision separate from units, since both sets carry 39 units.
        out = rp.analyse(_scanned(motor_last=_arm(6.0, 1.8)))
        assert out["verdict"] == "dimension_reducible"
        assert out["sets_beating_floor"] == ["motor_last"]
        assert rp.ARMS["motor_last"]["units"] == rp.ARMS["motor"]["units"]
        assert rp.ARMS["motor_last"]["draws"] < rp.ARMS["motor"]["draws"]


class TestTheDimensionTravelsWithTheResult:
    def test_every_set_carries_its_numbers(self) -> None:
        out = rp.analyse(_scanned())
        for name, expected in rp.ARMS.items():
            assert out["sets"][name]["dimension"] == expected

    def test_the_registered_dimensions_match_the_substrate(self) -> None:
        # The same figures the topology's own test asserts against the built masks.
        assert rp.ARMS["full"]["draws"] == 1208
        assert rp.ARMS["full"]["causal_draws"] == 672
        assert rp.ARMS["causal"]["draws"] == 672
        assert rp.ARMS["motor"]["adaptable_synapses"] == 323

    def test_the_trend_is_descriptive(self) -> None:
        cells = {name: _arm(2.0 + i, 1.8) for i, name in enumerate(rp.ARMS)}
        out = rp.analyse(cells)
        assert out["trend"]["descriptive_only"] is True

    def test_a_constant_contrast_is_reported_as_constant(self) -> None:
        # Spearman on a flat series returns not-a-number, which in a record reads as a failed test.
        cells = {name: _arm(2.0, 1.8, spread=0.0) for name in rp.ARMS}
        out = rp.analyse(cells)
        assert not out["trend"]["defined"]


class TestAnIncompleteCampaignIsNotScored:
    def _complete(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "learning": dict.fromkeys(rp.SEEDS, object()),
                "frozen": dict.fromkeys(rp.SEEDS, object()),
                "logs": {},
            }
            for name in rp.ARMS
        }

    def test_a_complete_campaign_passes(self) -> None:
        rp.require_complete(self._complete())

    def test_a_missing_run_refuses_a_verdict(self) -> None:
        scanned = self._complete()
        del scanned["motor"]["frozen"][3]
        with pytest.raises(ValueError, match=r"motor/frozen seeds \[3\]"):
            rp.require_complete(scanned)

    def test_the_registered_campaign_is_eighty_runs(self) -> None:
        assert len(rp.ARMS) * 2 * len(rp.SEEDS) == 80


class TestScanKeepsTheSetsApart:
    _BODY = "\n".join(f"Run: {i} Status: SUCCESS Eaten: 20/20" for i in range(1, 41)) + "\n"

    def _dir(self, tmp_path: Path, names: list[str]) -> Path:
        logs = tmp_path / "logs"
        logs.mkdir()
        for name in names:
            (logs / name).write_text(self._BODY)
        return tmp_path

    def test_motor_and_motor_last_are_not_confused(self) -> None:
        # `motor` is a prefix of `motor_last`; a greedy label would file one as the other.
        assert rp._LABEL.match(f"{rp._STEM}_motor_last-seed1.log").group("arm") == "motor_last"
        assert rp._LABEL.match(f"{rp._STEM}_motor-seed1.log").group("arm") == "motor"

    def test_a_frozen_label_is_read_as_the_control(self, tmp_path: Path) -> None:
        out = rp.scan(self._dir(tmp_path, [f"{rp._STEM}_motor_last_frozen-seed1.log"]))
        assert 1 in out["motor_last"]["frozen"]
        assert not out["motor_last"]["learning"]

    def test_an_unregistered_set_is_skipped(self, tmp_path: Path) -> None:
        out = rp.scan(self._dir(tmp_path, [f"{rp._STEM}_hop2-seed1.log"]))
        assert all(not cell["learning"] for cell in out.values())

    def test_a_duplicate_run_is_refused(self, tmp_path: Path) -> None:
        root = self._dir(
            tmp_path,
            [f"{rp._STEM}_motor-seed1.log", f"{rp._STEM}_motor-seed01.log"],
        )
        with pytest.raises(ValueError, match="duplicates a read run"):
            rp.scan(root)

    def test_the_fixture_body_parses(self, tmp_path: Path) -> None:
        out = rp.scan(self._dir(tmp_path, [f"{rp._STEM}_motor-seed1.log"]))
        assert out["motor"]["learning"][1].foods == pytest.approx(20.0)
        assert math.isfinite(out["motor"]["learning"][1].success)
