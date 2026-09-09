"""The decorrelation harness: the registry, the family, the verdict map, the annotations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_decorrelation as dc  # noqa: E402  # pyright: ignore[reportMissingImports]

_SEEDS = tuple(range(1, 17))


def _arm(base: float) -> dict[int, float]:
    """Build a per-seed arm around ``base`` with enough spread to be a real sample."""
    return {s: base + 2.0 * ((s % 5) - 2) for s in _SEEDS}


def _grounded(wt: float = 14.0, rn: float = 9.0) -> dict[str, dict[int, float]]:
    return {"wt_hebbian_atlas": _arm(wt), "rn_hebbian_atlas": _arm(rn)}


def _values(**over: float) -> dict[str, dict[int, float]]:
    base = {"wt_antihebb": 14.0, "rn_antihebb": 9.0, "wt_oja": 14.0, "rn_oja": 9.0}
    base.update(over)
    return {arm: _arm(value) for arm, value in base.items()}


class TestTheRegistry:
    def test_four_arms_two_variants_two_wirings(self) -> None:
        assert dc.ARM_KEYS == ("wt_antihebb", "rn_antihebb", "wt_oja", "rn_oja")

    def test_every_arm_config_exists(self) -> None:
        configs = _root / "configs" / "scenarios" / "foraging_predator_thermal"
        for stem in dc.ARMS:
            assert (configs / f"{stem}.yml").is_file(), stem

    def test_the_protocol_matches_the_sign_grounding_test(self) -> None:
        assert dc.SEEDS == _SEEDS
        assert dc.BUDGET == 1000
        assert dc.EXTENSION == 1.5

    def test_the_random_sign_targets_are_the_committed_ones(self) -> None:
        assert dc.RANDOM_SIGN_MEAN == {"wt": 31.5, "rn": 17.4}


class TestTheComparator:
    def test_it_reads_the_committed_table(self) -> None:
        grounded = dc.read_grounded()
        assert set(grounded) == {"wt_hebbian_atlas", "rn_hebbian_atlas"}
        for seeds in grounded.values():
            assert set(seeds) == set(_SEEDS)

    def test_the_committed_means_are_the_published_ones(self) -> None:
        grounded = dc.read_grounded()
        wt = sum(grounded["wt_hebbian_atlas"].values()) / len(_SEEDS)
        rn = sum(grounded["rn_hebbian_atlas"].values()) / len(_SEEDS)
        assert wt == pytest.approx(14.0, abs=0.1)
        assert rn == pytest.approx(9.1, abs=0.1)

    def test_a_table_missing_seeds_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "per-seed.csv"
        path.write_text("arm,seed,success\nwt_hebbian_atlas,1,10.0\nrn_hebbian_atlas,1,5.0\n")
        with pytest.raises(ValueError, match="lacks"):
            dc.read_grounded(path)


class TestTheVerdictMap:
    def _verdict(self, **over: float) -> str:
        tests = dc.family_tests(_values(**over), _grounded())
        return dc.verdict(tests)

    def test_no_change_gives_no_recovery(self) -> None:
        assert self._verdict() == "no_recovery"

    def test_the_prediction_can_fail(self) -> None:
        # Both variants worse than the grounded arms: the registered outcome in which the
        # sign-grounding test's prediction is wrong.
        assert self._verdict(wt_antihebb=6.0, wt_oja=6.0) == "no_recovery"

    def test_only_the_sign_keyed_variant_recovering(self) -> None:
        assert self._verdict(wt_antihebb=40.0) == "recovery_specific"

    def test_only_the_general_term_recovering(self) -> None:
        assert self._verdict(wt_oja=40.0) == "recovery_general"

    def test_both_recovering(self) -> None:
        assert self._verdict(wt_antihebb=40.0, wt_oja=40.0) == "recovery_both"

    def test_missing_seeds_take_precedence(self) -> None:
        values = _values(wt_antihebb=40.0, wt_oja=40.0)
        del values["wt_antihebb"][3]
        assert dc.verdict(dc.family_tests(values, _grounded())) == "insufficient_seeds"


class TestTheWiringContrastOnlyAnnotates:
    def test_a_confirmed_contrast_does_not_change_the_verdict(self) -> None:
        # Wild-type far above rewired under both variants, but neither recovers.
        values = _values(rn_antihebb=1.0, rn_oja=1.0)
        tests = dc.family_tests(values, _grounded())
        assert dc.verdict(tests) == "no_recovery"
        assert tests["D3"]["confirms"] or tests["D4"]["confirms"]


class TestFullRecovery:
    def test_an_arm_at_the_random_sign_level_reaches_it(self) -> None:
        out = dc.full_recovery(_values(wt_antihebb=35.0))
        assert out["wt_antihebb"]["reaches"]
        assert out["wt_antihebb"]["target"] == 31.5

    def test_an_arm_that_only_helps_does_not(self) -> None:
        out = dc.full_recovery(_values(wt_antihebb=20.0))
        assert not out["wt_antihebb"]["reaches"]

    def test_the_rewired_target_is_its_own(self) -> None:
        out = dc.full_recovery(_values(rn_oja=20.0))
        assert out["rn_oja"]["target"] == 17.4
        assert out["rn_oja"]["reaches"]

    def test_a_missing_arm_reaches_nothing(self) -> None:
        out = dc.full_recovery({})
        assert not out["wt_antihebb"]["reaches"]


class TestOutput:
    def _out(self) -> dict:
        panel = {arm: {} for arm in dc.ARM_KEYS}
        return dc.analyse(panel, {arm: {} for arm in dc.ARM_KEYS})  # type: ignore[arg-type]

    def test_an_empty_panel_is_insufficient_not_scored(self) -> None:
        assert self._out()["verdict"] == "insufficient_seeds"

    def test_the_comparator_source_is_recorded(self) -> None:
        assert "never re-run" in self._out()["comparator"]["source"].replace(
            "no arm re-run",
            "never re-run",
        )

    def test_it_is_json_serialisable(self) -> None:
        json.dumps(self._out())


class TestGrouping:
    def _record(
        self,
        success: float,
        episodes: int = dc.BUDGET,
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
            dc.group_panel([("wt_oja", 1, self._record(10.0, episodes=1234), Path("a.log"))])

    def test_a_seed_outside_the_test_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside the test's seeds"):
            dc.group_panel([("wt_oja", 99, self._record(10.0), Path("a.log"))])

    def test_a_duplicate_at_the_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            dc.group_panel(
                [
                    ("wt_oja", 1, self._record(10.0), Path("a.log")),
                    ("wt_oja", 1, self._record(11.0), Path("b.log")),
                ],
            )

    def test_the_extension_wins_over_the_shorter_run(self) -> None:
        panel, logs = dc.group_panel(
            [
                ("wt_oja", 1, self._record(10.0), Path("a.log")),
                ("wt_oja", 1, self._record(20.0, episodes=1500), Path("b.log")),
            ],
        )
        assert panel["wt_oja"][1].episodes == 1500
        assert logs["wt_oja"][1] == Path("b.log")

    def test_a_non_converged_run_is_listed_for_extension(self) -> None:
        panel, _ = dc.group_panel(
            [("wt_oja", 1, self._record(10.0, converged=False), Path("a.log"))],
        )
        assert dc.extensions_needed(panel) == [{"arm": "wt_oja", "seed": 1, "extend_to": 1500}]
