"""The clone assay harness: the pass rule, the comparator, and what it refuses to impute."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_consolidation_screen as cs  # noqa: E402  # pyright: ignore[reportMissingImports]


def _values(**over: float) -> dict[int, float]:
    """Per-seed values equal to the committed frozen clone, then shifted as asked."""
    values = dict(cs.FROZEN_CLONE)
    for seed, value in over.items():
        values[int(seed.removeprefix("s"))] = value
    return values


class TestTheComparatorIsTheCommittedTable:
    def test_the_frozen_clone_values_are_the_published_ones(self) -> None:
        assert cs.FROZEN_CLONE == {
            1: 39.3,
            2: 44.0,
            3: 40.0,
            4: 21.3,
            5: 47.1,
            6: 33.3,
            7: 61.3,
            8: 23.3,
        }
        assert pytest.approx(38.7, abs=0.05) == cs.FROZEN_MEAN

    def test_the_pass_rule_is_the_registered_one(self) -> None:
        assert (cs.HOLD_MEAN, cs.HOLD_SEED, cs.HOLD_SEEDS) == (5.0, 10.0, 6)
        assert tuple(range(1, 9)) == cs.SEEDS
        assert cs.BUDGET == 2000


class TestTheAssayIsUnchangedForANewArm:
    def test_the_eligibility_variant_is_screened_by_the_same_rule(self) -> None:
        # The variant's result must be comparable with the three mechanisms that failed, which
        # it is only if the comparator, budget, metric and pass rule are untouched.
        assert "node_perturbation" in cs.ARM_KEYS
        assert cs.FROZEN_CLONE[1] == 39.3  # the committed comparator, unchanged
        assert (cs.HOLD_MEAN, cs.HOLD_SEED, cs.HOLD_SEEDS) == (5.0, 10.0, 6)
        assert cs.BUDGET == 2000

    def test_its_arm_config_carries_the_control_pinned_sigma(self) -> None:
        import yaml

        configs = _root / "configs" / "scenarios" / "foraging_predator_thermal"
        stem = next(k for k, v in cs.ARMS.items() if v == "node_perturbation")
        config = yaml.safe_load((configs / f"{stem}.yml").read_text())["brain"]["config"]
        assert config["plasticity_eligibility"] == "node_perturbation"
        # The value the positive control pinned; re-tuning it here would make the gate a search.
        assert config["plasticity_node_noise"] == 0.2

    def test_its_arm_is_a_single_key_block_delta_from_the_clone_arm(self) -> None:
        import yaml

        configs = _root / "configs" / "scenarios" / "foraging_predator_thermal"
        parent = configs / "connectomeppo_small_continuous2d_combined_klinotaxis_plastic_clone.yml"
        stem = next(k for k, v in cs.ARMS.items() if v == "node_perturbation")

        def flat(data: object, prefix: str = "") -> dict:
            if isinstance(data, dict):
                out: dict = {}
                for key, value in data.items():
                    out.update(flat(value, f"{prefix}{key}."))
                return out
            return {prefix.rstrip("."): data}

        base = flat(yaml.safe_load(parent.read_text()))
        variant = flat(yaml.safe_load((configs / f"{stem}.yml").read_text()))
        assert set(variant) - set(base) == {
            "brain.config.plasticity_eligibility",
            "brain.config.plasticity_node_noise",
        }
        assert {k for k in set(base) & set(variant) if base[k] != variant[k]} == set()


class TestTheHoldRule:
    def test_an_identical_arm_holds(self) -> None:
        result = cs.assess(_values())
        assert result["holds"]
        assert result["pass"]
        assert not result["improves"]

    def test_a_mean_five_points_down_still_holds(self) -> None:
        # Exactly on the boundary: every seed down five, so the mean is down five.
        result = cs.assess({s: v - 5.0 for s, v in cs.FROZEN_CLONE.items()})
        assert result["holds"]

    def test_a_mean_further_down_fails(self) -> None:
        result = cs.assess({s: v - 5.6 for s, v in cs.FROZEN_CLONE.items()})
        assert not result["holds"]
        assert not result["pass"]

    def test_three_collapsed_seeds_fail_even_at_a_passing_mean(self) -> None:
        # Three seeds far below their own, the rest lifted so the mean survives:
        # the per-seed clause is what stops a mean from hiding a destroyed arm.
        values = dict(cs.FROZEN_CLONE)
        for seed in (1, 2, 3):
            values[seed] = 0.0
        for seed in (5, 6, 7, 8):
            values[seed] += 40.0
        result = cs.assess(values)
        assert result["mean"] > cs.FROZEN_MEAN
        assert result["seeds_within_hold"] == 5
        assert not result["holds"]
        assert not result["improves"]
        assert not result["pass"]

    def test_a_seed_exactly_ten_down_is_within_the_hold(self) -> None:
        values = dict(cs.FROZEN_CLONE)
        values[1] -= 10.0
        assert cs.assess(values)["seeds_within_hold"] == 8


class TestTheImproveRule:
    def test_every_seed_up_improves(self) -> None:
        result = cs.assess({s: v + 3.0 for s, v in cs.FROZEN_CLONE.items()})
        assert result["improves"]
        assert result["pass"]

    def test_a_higher_mean_on_two_seeds_does_not_improve(self) -> None:
        values = {s: v - 2.0 for s, v in cs.FROZEN_CLONE.items()}
        values[7] += 60.0
        values[5] += 20.0
        result = cs.assess(values)
        assert result["mean"] > cs.FROZEN_MEAN
        assert result["seeds_at_or_above"] == 2
        assert not result["improves"]


class TestMissingRunsAreNeverImputed:
    def test_an_incomplete_arm_cannot_pass(self) -> None:
        values = _values()
        del values[4]
        del values[8]
        result = cs.assess(values)
        assert result["missing_seeds"] == [4, 8]
        assert result["n"] == 6
        assert not result["pass"]
        assert not result["holds"]

    def test_no_arm_at_all_is_reported_not_scored(self) -> None:
        result = cs.assess({})
        assert result["n"] == 0
        assert result["missing_seeds"] == list(cs.SEEDS)
        assert not result["pass"]


class TestGrouping:
    def _record(self, success: float, episodes: int = cs.BUDGET) -> object:
        from l4_panel import SeedRecord  # pyright: ignore[reportMissingImports]

        return SeedRecord(
            success=success,
            foods=0.0,
            episodes=episodes,
            converged=None,
            onset=None,
            evasion_rate=None,
            temp_comfort=None,
            curve=[],
        )

    def test_a_run_off_the_budget_is_refused(self) -> None:
        scanned = [("anchor", 1, self._record(30.0, episodes=3000), Path("a.log"))]
        with pytest.raises(ValueError, match="registers no extension"):
            cs.group(scanned)  # type: ignore[arg-type]

    def test_a_seed_outside_the_assay_is_refused(self) -> None:
        scanned = [("anchor", 9, self._record(30.0), Path("a.log"))]
        with pytest.raises(ValueError, match="outside the assay's seeds"):
            cs.group(scanned)  # type: ignore[arg-type]

    def test_a_duplicate_is_refused(self) -> None:
        scanned = [
            ("anchor", 1, self._record(30.0), Path("a.log")),
            ("anchor", 1, self._record(31.0), Path("b.log")),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            cs.group(scanned)  # type: ignore[arg-type]

    def test_arms_and_seeds_are_grouped(self) -> None:
        scanned = [
            ("anchor", 1, self._record(30.0), Path("a.log")),
            ("rigidity", 1, self._record(31.0), Path("b.log")),
        ]
        panel, logs = cs.group(scanned)  # type: ignore[arg-type]
        assert set(panel) == {"anchor", "rigidity"}
        assert logs["anchor"][1] == Path("a.log")


class TestTheRegistry:
    def test_the_registry_covers_every_screened_arm(self) -> None:
        assert cs.ARM_KEYS == (
            "anchor",
            "rigidity",
            "oracle",
            "node_perturbation",
            # Its frozen control: perturbation applied, no weight written, so a failing
            # plastic arm can be attributed to the rule rather than to the exploration.
            "perturbation_frozen",
        )

    def test_every_arm_config_exists(self) -> None:
        configs = _root / "configs" / "scenarios" / "foraging_predator_thermal"
        for stem in cs.ARMS:
            assert (configs / f"{stem}.yml").is_file(), stem


class TestOutput:
    def _out(self) -> dict:
        panel = {arm: {} for arm in cs.ARM_KEYS}
        return cs.analyse(panel, {arm: {} for arm in cs.ARM_KEYS})  # type: ignore[arg-type]

    def test_the_record_says_it_is_a_screen(self) -> None:
        out = self._out()
        assert "licenses running the registered panel and nothing more" in out["screen_not_test"]
        assert "verdict" in out["screen_not_test"]

    def test_the_comparator_and_rule_are_recorded(self) -> None:
        out = self._out()
        assert out["comparator"]["mean"] == pytest.approx(cs.FROZEN_MEAN)
        assert out["rule"]["hold_seeds_of_eight"] == cs.HOLD_SEEDS

    def test_the_record_is_strict_json(self) -> None:
        # Bare NaN is not JSON; an arm with no endpoint weights has no cosine to report,
        # and the record must carry that as null so a strict reader accepts it.
        out = self._out()
        assert math.isnan(out["arms"]["anchor"]["cosine_mean"])
        text = json.dumps(cs._jsonable(out), allow_nan=False)
        json.loads(text, parse_constant=lambda c: pytest.fail(f"bare {c} in the record"))

    def test_unavailable_measurements_become_null(self) -> None:
        assert cs._jsonable(float("nan")) is None
        assert cs._jsonable(float("inf")) is None
        assert cs._jsonable(float("-inf")) is None

    def test_it_replaces_them_wherever_they_are_nested(self) -> None:
        out = cs._jsonable({"a": [{"b": float("nan")}, 1.0], "c": {"d": [float("inf")]}})
        assert out == {"a": [{"b": None}, 1.0], "c": {"d": [None]}}

    def test_it_leaves_finite_values_and_non_numbers_alone(self) -> None:
        out = cs._jsonable({"f": 0.5, "i": 3, "s": "x", "n": None, "b": True})
        assert out == {"f": 0.5, "i": 3, "s": "x", "n": None, "b": True}

    def test_the_csv_carries_the_cosine_and_multiplier(self, tmp_path: Path) -> None:
        out = self._out()
        out["arms"]["anchor"] = cs.assess(_values())
        out["arms"]["anchor"]["cosine_to_clone"] = {str(s): 0.9 for s in cs.SEEDS}
        out["arms"]["anchor"]["rate_multiplier"] = {str(s): 0.5 for s in cs.SEEDS}
        path = tmp_path / "per-seed.csv"
        cs.write_per_seed_csv(out, path)
        rows = list(csv.DictReader(path.open(newline="")))
        assert len(rows) == len(cs.SEEDS)
        assert rows[0]["cosine"] == "0.9"
        assert rows[0]["rate_mult"] == "0.5"
        assert float(rows[0]["frozen_clone"]) == pytest.approx(cs.FROZEN_CLONE[1])
