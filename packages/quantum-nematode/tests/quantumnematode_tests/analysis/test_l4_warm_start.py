"""Warm-start panel harness: registry, seed range, the family, the verdict map, annotations."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_warm_start as ws  # noqa: E402  # pyright: ignore[reportMissingImports]

_SEEDS = tuple(range(1, 9))


def _arm(base: float) -> dict[int, float]:
    return {s: base + 0.5 * ((s % 3) - 1) for s in _SEEDS}


def _values(**over: float) -> dict[str, dict[int, float]]:
    base = {
        "wt_clone_frozen": 60.0,
        "rn_clone_frozen": 40.0,
        "wt_clone_hebbian": 62.0,
        "rn_clone_hebbian": 42.0,
        "wt_clone_plastic": 75.0,
        "rn_clone_plastic": 50.0,
        "wt_fullclone_frozen": 70.0,
        "rn_fullclone_frozen": 55.0,
        "wt_fullclone_ppo": 80.0,
        "rn_fullclone_ppo": 60.0,
        "wt_ppo": 45.0,
        "rn_ppo": 40.0,
    }
    base.update(over)
    return {arm: _arm(v) for arm, v in base.items()}


_FLOOR = _arm(8.0)


def _record(
    success: float,
    episodes: int = 2000,
    *,
    converged: bool | None = True,
) -> ws.SeedRecord:
    return ws.SeedRecord(
        success=success,
        foods=5.0,
        episodes=episodes,
        converged=converged,
        onset=100 if converged else None,
        evasion_rate=None,
        temp_comfort=None,
        curve=[success],
    )


def _panel(values: dict[str, dict[int, float]]) -> dict[str, dict[int, ws.SeedRecord]]:
    return {arm: {s: _record(v) for s, v in seeds.items()} for arm, seeds in values.items()}


class TestRegistry:
    def test_twelve_arms_and_their_configs_exist(self) -> None:
        assert len(ws.ARMS) == 12
        assert set(ws.BUDGETS) == set(ws.ARM_KEYS)
        for stem in ws.ARMS:
            assert (
                ws.REPO / "configs" / "scenarios" / "foraging_predator_thermal" / f"{stem}.yml"
            ).is_file(), stem

    def test_budgets(self) -> None:
        assert ws.BUDGETS["wt_clone_frozen"] == 600
        assert ws.BUDGETS["wt_clone_plastic"] == 2000
        assert ws.BUDGETS["wt_ppo"] == 3000

    def test_seed_outside_1_8_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the panel seeds 1-8"):
            ws.group_panel([("wt_ppo", 9, _record(1.0))])

    def test_longer_run_wins_a_duplicate_either_order(self) -> None:
        short, long_ = _record(1.0, episodes=2000, converged=False), _record(5.0, episodes=3000)
        for order in ([short, long_], [long_, short]):
            panel = ws.group_panel([("wt_clone_plastic", 3, r) for r in order])
            assert panel["wt_clone_plastic"][3].episodes == 3000
        same = [
            ("wt_ppo", 1, _record(1.0, episodes=3000)),
            ("wt_ppo", 1, _record(2.0, episodes=3000)),
        ]
        with pytest.raises(ValueError, match="not an extension"):
            ws.group_panel(same)

    def test_random_floor_comes_from_panel2(self) -> None:
        floor = ws.read_random_floor()
        assert set(floor) == set(_SEEDS)
        assert floor[1] == pytest.approx(36.0, abs=0.5)  # panel 2's wt_frozen seed 1


class TestFamilyAndVerdict:
    def test_six_tests_directions_and_pass(self) -> None:
        tests = ws.family_tests(_values(), _FLOOR)
        assert set(tests) == set(ws.FAMILY)
        assert tests["W1"]["mean_delta"] == pytest.approx(52.0)
        assert tests["W2"]["mean_delta"] == pytest.approx(20.0)
        assert tests["W3"]["mean_delta"] == pytest.approx(25.0)
        assert tests["W4"]["mean_delta"] == pytest.approx(15.0)
        assert tests["W5"]["mean_delta"] == pytest.approx(13.0)
        assert tests["W6"]["mean_delta"] == pytest.approx(35.0)
        assert all(t["pass"] for t in tests.values())
        assert all(t["complete"] for t in tests.values())
        assert ws.verdict(tests) == "specific_wiring"

    def test_clone_fail_first(self) -> None:
        tests = ws.family_tests(_values(wt_clone_frozen=8.0), _FLOOR)
        assert ws.verdict(tests) == "clone_fail"

    def test_sanity_floor_fail_when_the_rule_does_not_improve(self) -> None:
        tests = ws.family_tests(_values(wt_clone_plastic=60.0), _FLOOR)  # equals frozen clone
        assert ws.verdict(tests) == "sanity_floor_fail"
        tests = ws.family_tests(_values(wt_clone_hebbian=75.0), _FLOOR)  # equals plastic
        assert ws.verdict(tests) == "sanity_floor_fail"

    def test_rewired_beats_wild_type(self) -> None:
        tests = ws.family_tests(_values(wt_clone_plastic=75.0, rn_clone_plastic=95.0), _FLOOR)
        assert ws.verdict(tests) == "rewired_beats_wild_type"

    def test_degree_statistics(self) -> None:
        tests = ws.family_tests(_values(rn_clone_plastic=75.0), _FLOOR)
        assert ws.verdict(tests) == "degree_statistics"

    def test_inconclusive(self) -> None:
        tests = ws.family_tests(_values(), _FLOOR)
        tests["W3"].update({"pass": False, "ci_lo": 1.0, "ci_hi": 5.0})
        assert ws.verdict(tests) == "inconclusive"

    def test_insufficient_seeds(self) -> None:
        values = _values()
        values["wt_clone_plastic"] = {1: 90.0}
        assert ws.verdict(ws.family_tests(values, _FLOOR)) == "insufficient_seeds"

    def test_annotations_never_change_the_verdict(self) -> None:
        values = _values(rn_clone_frozen=90.0, wt_ppo=99.0)  # W2 and W6 reversed
        tests = ws.family_tests(values, _FLOOR)
        assert ws.verdict(tests) == "specific_wiring"
        notes = ws.annotate(tests)
        assert notes == {
            "wild_type_holds_better": False,
            "warm_start_helps_ppo": False,
            "rule_destroys_clone": False,
        }

    def test_rule_destroys_clone_is_named(self) -> None:
        tests = ws.family_tests(
            _values(wt_clone_plastic=20.0),
            _FLOOR,
        )  # far below the frozen clone at 60
        assert ws.verdict(tests) == "sanity_floor_fail"
        assert ws.annotate(tests)["rule_destroys_clone"] is True


class TestDescriptiveAndOutput:
    def test_descriptive_excludes_the_family_and_names_the_pairs(self) -> None:
        rows = ws.descriptive_pairs(_values())
        keys = {(r["a"], r["b"]) for r in rows}
        for pair in ws.FAMILY_PAIRS.values():
            assert pair not in keys
        assert ("wt_fullclone_ppo", "rn_fullclone_ppo") in keys
        assert sum(r["named"] for r in rows) == len(ws.DESCRIPTIVE_PAIRS)

    def test_extensions_use_each_arm_budget(self) -> None:
        panel = _panel(_values())
        panel["wt_ppo"][2] = _record(5.0, episodes=3000, converged=False)
        panel["wt_clone_frozen"][2] = _record(5.0, episodes=600, converged=False)
        panel["wt_clone_plastic"][4] = _record(
            5.0,
            episodes=3000,
            converged=False,
        )  # already extended
        rows = ws.extensions_needed(panel)
        assert {(r["arm"], r["seed"], r["extend_to"]) for r in rows} == {
            ("wt_clone_frozen", 2, 900),
            ("wt_ppo", 2, 4500),
        }

    def test_analyse_ceiling_and_clone_fits(self) -> None:
        clones = [
            {
                "parameter_set": "plastic",
                "wiring": "wt",
                "seed": 1,
                "held_out_loss": 0.01,
                "weak": False,
            },
            {
                "parameter_set": "plastic",
                "wiring": "wt",
                "seed": 2,
                "held_out_loss": 0.5,
                "weak": True,
            },
            {"parameter_set": "full", "wiring": "rn", "seed": 3, "failed": 1, "weak": True},
        ]
        out = ws.analyse(_panel(_values()), _FLOOR, {}, ceiling=90.0, clones=clones)
        mean = out["per_arm"]["wt_clone_plastic"]["mean"]
        assert out["against_ceiling"]["wt_clone_plastic"]["fraction_of_ceiling"] == pytest.approx(
            mean / 90.0,
        )
        assert out["clone_fits"]["plastic_wt"]["weak_seeds"] == [2]
        assert out["clone_fits"]["full_rn"]["failed_seeds"] == [3]
        assert out["verdict"]["verdict"] == "specific_wiring"

    def test_main_end_to_end(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        for stem, arm in ws.ARMS.items():
            good = arm.startswith("wt_clone") or arm in ("wt_fullclone_ppo", "wt_fullclone_frozen")
            for seed in _SEEDS:
                budget = ws.BUDGETS[arm]
                statuses = ["SUCCESS" if good or seed % 4 == 0 else "FAILED"] * budget
                if arm == "wt_clone_plastic":
                    statuses = ["SUCCESS"] * budget
                (logs / f"{stem}-seed{seed}.log").write_text(
                    "\n".join(
                        f"Run: {i}   Status: {s:<7} Reason: x Steps: 10    "
                        f"Eaten: {10 if s == 'SUCCESS' else 1}/10  "
                        for i, s in enumerate(statuses, 1)
                    ),
                )
        out = tmp_path / "panel.json"
        code = ws.main(
            ["--campaign-dir", str(tmp_path), "--out", str(out), "--csv", str(tmp_path / "s.csv")],
        )
        assert code == 0
        data = json.loads(out.read_text())
        assert set(data["family"]) == set(ws.FAMILY)
        assert data["verdict"]["verdict"] in {
            "specific_wiring",
            "sanity_floor_fail",
            "degree_statistics",
            "inconclusive",
        }
        with (tmp_path / "s.csv").open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 12 * 8


class TestReviewGuards:
    def test_descriptive_pairs_are_unique_across_orderings(self) -> None:
        rows = ws.descriptive_pairs(_values())
        unordered = [frozenset((r["a"], r["b"])) for r in rows]
        assert len(unordered) == len(set(unordered))
        # every named pair present exactly once, in the named orientation
        for a, b in ws.DESCRIPTIVE_PAIRS:
            assert sum(1 for r in rows if {r["a"], r["b"]} == {a, b}) == 1
            assert any(r["a"] == a and r["b"] == b for r in rows)

    def test_unregistered_run_length_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="neither the budget nor its registered extension"):
            ws.group_panel([("wt_ppo", 1, _record(1.0, episodes=6000))])
        panel = ws.group_panel(
            [
                ("wt_ppo", 1, _record(1.0, episodes=3000)),
                ("wt_ppo", 1, _record(2.0, episodes=4500)),
            ],
        )
        assert panel["wt_ppo"][1].episodes == 4500
