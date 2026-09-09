"""The structured-instruction harness: registry, family, verdict map, annotations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_structured_instruction as si  # noqa: E402  # pyright: ignore[reportMissingImports]

_SEEDS = tuple(range(1, 17))


def _arm(base: float) -> dict[int, float]:
    """Build a per-seed arm around ``base`` with enough spread to be a real sample."""
    return {s: base + 2.0 * ((s % 5) - 2) for s in _SEEDS}


def _values(**over: float) -> dict[str, dict[int, float]]:
    base = {"wt_pathway": 10.0, "rn_pathway": 8.0, "wt_global": 10.0, "rn_global": 8.0}
    base.update(over)
    return {arm: _arm(value) for arm, value in base.items()}


class TestTheRegistry:
    def test_four_arms_two_routings_two_wirings(self) -> None:
        assert si.ARM_KEYS == ("wt_pathway", "rn_pathway", "wt_global", "rn_global")

    def test_every_arm_config_exists(self) -> None:
        configs = _root / "configs" / "scenarios" / "foraging_predator_thermal"
        for stem in si.ARMS:
            assert (configs / f"{stem}.yml").is_file(), stem

    def test_the_protocol_is_the_panel_s(self) -> None:
        assert si.SEEDS == _SEEDS
        assert si.BUDGET == 3000
        assert si.EXTENSION == 1.5


class TestTheVerdictMap:
    def _verdict(self, **over: float) -> str:
        return si.verdict(si.family_tests(_values(**over)))

    def test_no_change_gives_no_routing_effect(self) -> None:
        assert self._verdict() == "no_routing_effect"

    def test_routing_can_fail_on_both_wirings(self) -> None:
        assert self._verdict(wt_pathway=4.0, rn_pathway=3.0) == "no_routing_effect"

    def test_helping_the_wild_type_only(self) -> None:
        assert self._verdict(wt_pathway=40.0) == "routing_helps_wild_type_only"

    def test_helping_the_rewired_only(self) -> None:
        assert self._verdict(rn_pathway=40.0) == "routing_helps_rewired_only"

    def test_helping_both_is_its_own_outcome(self) -> None:
        assert self._verdict(wt_pathway=40.0, rn_pathway=40.0) == "routing_helps_both"

    def test_missing_seeds_take_precedence(self) -> None:
        values = _values(wt_pathway=40.0)
        del values["wt_pathway"][2]
        assert si.verdict(si.family_tests(values)) == "insufficient_seeds"


class TestTheWiringContrastOnlyAnnotates:
    def test_a_confirmed_contrast_does_not_decide(self) -> None:
        values = _values(rn_pathway=1.0, rn_global=1.0)
        tests = si.family_tests(values)
        assert si.verdict(tests) == "no_routing_effect"
        assert tests["S3"]["confirms"] or tests["S4"]["confirms"]


class TestTheCommittedPanelIsOnlyAConsistencyCheck:
    def test_it_is_labelled_as_such(self) -> None:
        out = si.panel1_check(_values())
        for arm in ("wt_global", "rn_global"):
            assert "never a comparator" in out[arm]["note"]

    def test_it_reads_the_committed_table(self) -> None:
        committed = si.read_panel1()
        assert committed["wt_global"], "panel 1's wild-type plastic arm should be readable"
        assert set(committed["wt_global"]) <= set(range(1, 65))

    def test_no_test_in_the_family_uses_it(self) -> None:
        # Every contrast is between two arms this test ran.
        for test in si.family_tests(_values()).values():
            assert "committed" not in test["contrast"]


class TestOutput:
    def _out(self) -> dict:
        empty: dict = {arm: {} for arm in si.ARM_KEYS}
        return si.analyse(empty, empty)  # type: ignore[arg-type]

    def test_an_empty_panel_is_insufficient(self) -> None:
        assert self._out()["verdict"] == "insufficient_seeds"

    def test_the_pathway_model_is_recorded_as_a_proxy(self) -> None:
        model = self._out()["pathway_model"]
        assert "lower bound" in model
        assert "refutes this proxy" in model

    def test_the_helps_both_reading_is_recorded_in_advance(self) -> None:
        reading = self._out()["annotations"]["routing_helps_both_reading"]
        assert "not about this" in reading

    def test_it_is_json_serialisable(self) -> None:
        json.dumps(self._out())


class TestGrouping:
    def _record(
        self,
        success: float,
        episodes: int = si.BUDGET,
        *,
        converged: bool | None = True,
    ) -> object:
        from l4_panel import SeedRecord  # pyright: ignore[reportMissingImports]

        return SeedRecord(
            success=success,
            foods=0.0,
            episodes=episodes,
            converged=converged,
            onset=None,
            evasion_rate=None,
            temp_comfort=None,
            curve=[],
        )

    def test_a_run_off_the_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="registered extension"):
            si.group_panel([("wt_pathway", 1, self._record(10.0, episodes=1234), Path("a.log"))])

    def test_a_seed_outside_the_test_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside the test's seeds"):
            si.group_panel([("wt_pathway", 99, self._record(10.0), Path("a.log"))])

    def test_a_duplicate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            si.group_panel(
                [
                    ("wt_pathway", 1, self._record(10.0), Path("a.log")),
                    ("wt_pathway", 1, self._record(11.0), Path("b.log")),
                ],
            )

    def test_the_extension_wins(self) -> None:
        panel, _ = si.group_panel(
            [
                ("wt_global", 1, self._record(10.0), Path("a.log")),
                ("wt_global", 1, self._record(20.0, episodes=4500), Path("b.log")),
            ],
        )
        assert panel["wt_global"][1].episodes == 4500

    def test_a_non_converged_run_is_listed(self) -> None:
        panel, _ = si.group_panel(
            [("wt_global", 1, self._record(10.0, converged=False), Path("a.log"))],
        )
        assert si.extensions_needed(panel) == [
            {"arm": "wt_global", "seed": 1, "extend_to": 4500},
        ]
