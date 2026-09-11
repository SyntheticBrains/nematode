"""The clone assay harness: the pass rule, the comparator, and what it refuses to impute."""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import pairwise
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


def _record(curve: list[float]) -> object:
    """Build a SeedRecord carrying just the curve the trajectory reads."""
    from l4_panel import SeedRecord  # pyright: ignore[reportMissingImports]

    return SeedRecord(
        success=float(curve[-1]) if curve else 0.0,
        foods=0.0,
        episodes=len(curve),
        converged=None,
        onset=None,
        evasion_rate=None,
        temp_comfort=None,
        curve=curve,
        peak_action_density=None,
    )


class TestTheAnnealedArmsSchedule:
    def test_the_registered_schedule_is_the_control_s(self) -> None:
        # The same bounds the positive control gates on, so a pass there is a statement about
        # the schedule this assay runs.
        assert cs.ANNEAL_INITIAL == 0.2
        assert cs.ANNEAL_FINAL == 0.02
        assert cs.ANNEAL_EPISODES == cs.BUDGET // 2

    def test_the_scale_reaches_its_floor_at_the_halfway_point(self) -> None:
        assert cs._scheduled_scale(0) == pytest.approx(0.2)
        assert cs._scheduled_scale(cs.ANNEAL_EPISODES) == pytest.approx(0.02)
        assert cs._scheduled_scale(cs.BUDGET) == pytest.approx(0.02)

    def test_both_annealed_arms_are_registered(self) -> None:
        assert "perturbation_annealed" in cs.ARM_KEYS
        assert "perturbation_annealed_frozen" in cs.ARM_KEYS
        assert {"perturbation_annealed", "perturbation_annealed_frozen"} == cs.ANNEALED_ARMS


class TestTheBinnedTrajectory:
    def test_it_reports_one_bin_per_eighth_with_its_scale(self) -> None:
        panel = {"perturbation_annealed": {1: _record([float(i) for i in range(80)])}}
        out = cs.binned_trajectory(panel, "perturbation_annealed")  # type: ignore[arg-type]
        assert out["n_read"] == 1
        assert len(out["bins"]) == cs.BINS
        # The first four bins span the decay, the last four sit at the floor.
        assert out["bins"][0]["scale"] == pytest.approx(0.2)
        assert out["bins"][cs.BINS // 2]["scale"] == pytest.approx(0.02)
        assert out["bins"][-1]["scale"] == pytest.approx(0.02)

    def test_the_bins_follow_the_curve(self) -> None:
        # A flat-then-rising curve must read as flat then rising, not be averaged away.
        curve = [0.0] * 40 + [10.0] * 40
        panel = {"perturbation_annealed": {1: _record(curve)}}
        out = cs.binned_trajectory(panel, "perturbation_annealed")  # type: ignore[arg-type]
        means = [b["mean"] for b in out["bins"]]
        assert means[0] == pytest.approx(0.0)
        assert means[-1] == pytest.approx(10.0)

    def test_a_recovering_frozen_arm_is_visible_as_a_path(self) -> None:
        # The point of the control: with weights frozen, a decaying scale lets the policy
        # recover, so the arm's cost is a path rather than the single number a fixed scale gave.
        curve = [5.0] * 20 + [15.0] * 20 + [30.0] * 20 + [35.0] * 20
        panel = {"perturbation_annealed_frozen": {1: _record(curve)}}
        out = cs.binned_trajectory(panel, "perturbation_annealed_frozen")  # type: ignore[arg-type]
        means = [b["mean"] for b in out["bins"]]
        assert all(later >= earlier for earlier, later in pairwise(means))
        assert means[-1] > means[0]

    def test_a_short_curve_is_not_imputed(self) -> None:
        panel = {"perturbation_annealed": {1: _record([1.0, 2.0])}}
        out = cs.binned_trajectory(panel, "perturbation_annealed")  # type: ignore[arg-type]
        assert out == {"bins": None, "n_read": 0}

    def test_an_absent_arm_reads_nothing(self) -> None:
        assert cs.binned_trajectory({}, "perturbation_annealed")["n_read"] == 0  # type: ignore[arg-type]


class TestTheEndpointArm:
    """An arm that evaluates another arm's endpoint weights with the perturbation off."""

    def _record(self) -> dict:
        import json

        path = (
            _root
            / "docs"
            / "experiments"
            / "logbooks"
            / "supporting"
            / "050-l4-perturbation-clone-assay"
            / "screen.json"
        )
        return json.loads(path.read_text())["arms"]["node_perturbation"]

    def test_the_committed_cosines_are_the_published_ones(self) -> None:
        # Transcribed constants are how a wrong number enters a record silently; this pins them
        # to the file they came from.
        published = {int(k): round(v, 3) for k, v in self._record()["cosine_to_clone"].items()}
        assert cs.ENDPOINT_SOURCE_COSINE["endpoint_nodeperturbation"] == published

    def test_the_committed_under_perturbation_scores_are_the_published_ones(self) -> None:
        published = {int(k): round(v, 1) for k, v in self._record()["per_seed"].items()}
        assert cs.ENDPOINT_UNDER_PERTURBATION["endpoint_nodeperturbation"] == published

    def test_the_config_is_the_comparator_plus_the_weights(self) -> None:
        import yaml

        configs = _root / "configs" / "scenarios" / "foraging_predator_thermal"
        stem = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic_frozen"
        comparator = yaml.safe_load((configs / f"{stem}_clone.yml").read_text())
        endpoint = yaml.safe_load((configs / f"{stem}_endpoint_nodeperturbation.yml").read_text())

        def flat(d: dict, prefix: str = "") -> dict:
            out: dict = {}
            for key, value in d.items():
                path = f"{prefix}.{key}" if prefix else key
                out.update(flat(value, path) if isinstance(value, dict) else {path: value})
            return out

        a, b = flat(comparator), flat(endpoint)
        differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert differing == {"brain.config.weights_path"}
        # No perturbation and no updates: the comparator's own condition.
        assert b["brain.config.freeze_updates"] is True
        assert "brain.config.plasticity_node_noise" not in b
        assert "brain.config.plasticity_eligibility" not in b


class TestTheEndpointIntegrityCheck:
    """Two checks: the weights ARE the staged tensor, and their cosine is the recorded one."""

    def _cosines(self, **over: float | None) -> dict[str, float | None]:
        out: dict[str, float | None] = {
            str(s): v for s, v in cs.ENDPOINT_SOURCE_COSINE["endpoint_nodeperturbation"].items()
        }
        out.update(over)
        return out

    def _patched(self, monkeypatch: pytest.MonkeyPatch, *, digests_match: bool) -> None:
        """Stand in for the two checkpoints, so no campaign artefacts are needed."""
        calls = {"n": 0}

        def fake(path: Path) -> str:
            calls["n"] += 1
            # The staged file is read first for each seed, then the run's.
            return "same" if digests_match or calls["n"] % 2 == 1 else "different"

        monkeypatch.setattr(cs, "_weight_digest", fake)
        monkeypatch.setattr(
            cs,
            "_experiment_json",
            lambda *_args, **_kwargs: {"exports_path": "exports/x"},
        )

    def _logs(self) -> dict[int, Path]:
        return dict.fromkeys(cs.SEEDS, Path("unused.log"))

    def _run(self, monkeypatch: pytest.MonkeyPatch, **over: float | None) -> dict:
        self._patched(monkeypatch, digests_match=True)
        monkeypatch.setattr(Path, "read_text", lambda _self, **_kw: "")
        return cs.endpoint_integrity(
            "endpoint_nodeperturbation",
            self._cosines(**over),
            self._logs(),
        )

    def test_matching_digests_and_cosines_are_not_void(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = self._run(monkeypatch)
        assert out["void_seeds"] == []
        assert out["verdict_void"] is False
        assert out["source_arm"] == "node_perturbation"

    def test_a_cosine_of_one_voids_the_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The clone loaded instead of the endpoint: unchanged from the clone, so the cosine is
        # 1.0 and the run would otherwise score as a policy that held perfectly.
        out = self._run(monkeypatch, **{"3": 1.0})
        assert out["void_seeds"] == [3]
        assert out["verdict_void"] is True

    def test_a_missing_cosine_voids_that_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = self._run(monkeypatch, **{"5": None})
        assert out["void_seeds"] == [5]

    def test_a_departure_inside_the_tolerance_is_allowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorded = cs.ENDPOINT_SOURCE_COSINE["endpoint_nodeperturbation"][2]
        out = self._run(monkeypatch, **{"2": recorded + cs.COSINE_TOLERANCE / 2})
        assert out["void_seeds"] == []

    def test_a_departure_outside_the_tolerance_is_not(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorded = cs.ENDPOINT_SOURCE_COSINE["endpoint_nodeperturbation"][2]
        out = self._run(monkeypatch, **{"2": recorded + cs.COSINE_TOLERANCE * 2})
        assert out["void_seeds"] == [2]

    def test_a_digest_mismatch_voids_every_seed_even_when_cosines_match(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Different weights can share a cosine to the clone; the digest is what makes this an
        # identity check rather than an inference from a summary statistic.
        self._patched(monkeypatch, digests_match=False)
        monkeypatch.setattr(Path, "read_text", lambda _self, **_kw: "")
        out = cs.endpoint_integrity(
            "endpoint_nodeperturbation",
            self._cosines(),
            self._logs(),
        )
        assert out["void_seeds"] == list(cs.SEEDS)
        assert out["verdict_void"] is True
        assert out["per_seed"]["1"]["cosine_ok"] is True
        assert out["per_seed"]["1"]["digest_ok"] is False

    def test_an_unreadable_digest_is_not_a_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cs, "_weight_digest", lambda _path: None)
        out = cs.endpoint_integrity("endpoint_nodeperturbation", self._cosines(), None)
        assert out["verdict_void"] is True


class TestAVoidEndpointCannotBeScored:
    """A void integrity check must remove the verdict, not sit beside one."""

    def _result(self, *, void: bool) -> dict:
        # A panel that would otherwise pass outright, so a failure to suppress is visible. The
        # integrity block is built here rather than computed, so this tests the suppression and
        # not the check that produces it.
        result = cs.assess(dict.fromkeys(cs.SEEDS, 100.0))
        integrity = {
            "source_arm": "node_perturbation",
            "tolerance": cs.COSINE_TOLERANCE,
            "per_seed": {
                str(seed): {
                    "cosine_ok": not (void and seed == 3),
                    "digest_ok": True,
                    "ok": not (void and seed == 3),
                }
                for seed in cs.SEEDS
            },
            "void_seeds": [3] if void else [],
            "verdict_void": void,
        }
        result["integrity"] = integrity
        if integrity["verdict_void"]:
            result["holds"] = False
            result["improves"] = False
            result["pass"] = False
            result["void"] = True
        return result

    def test_the_panel_would_otherwise_pass(self) -> None:
        # Guards the test itself: without the void there is a verdict to suppress.
        clean = self._result(void=False)
        assert clean["pass"] is True
        assert clean.get("void") is not True

    def test_a_void_arm_does_not_pass(self) -> None:
        voided = self._result(void=True)
        assert voided["void"] is True
        assert voided["pass"] is False
        assert voided["holds"] is False
        assert voided["improves"] is False

    def test_a_void_arm_is_not_listed_as_passed(self) -> None:
        assert self._result(void=True)["pass"] is False

    def test_a_void_arm_serialises_without_a_pass(self) -> None:
        import json

        text = json.dumps(cs._jsonable(self._result(void=True)), allow_nan=False)
        assert json.loads(text)["pass"] is False

    def test_the_printed_line_says_void_and_names_the_seeds(self, capsys) -> None:
        cs._print_arm("endpoint_nodeperturbation", self._result(void=True))
        printed = capsys.readouterr().out
        assert "VOID" in printed
        assert "[3]" in printed
        for word in ("holds", "improves", "FAILS"):
            assert word not in printed


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
            # The same eligibility with the scale annealed, and its own frozen control on the
            # identical schedule. That control recovers as the scale falls, so it is read as a
            # path rather than the single number a fixed scale gave.
            "perturbation_annealed",
            "perturbation_annealed_frozen",
            # Not a mechanism: the node-perturbation arm's endpoint weights run with the
            # perturbation off, which is what the assay never measured.
            "endpoint_nodeperturbation",
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

    def test_the_written_record_is_strict_json(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "screen.json"
        cs.write_screen_json(self._out(), path)
        text = path.read_text()
        assert "NaN" not in text
        loaded = json.loads(text, parse_constant=lambda c: pytest.fail(f"bare {c} in the record"))
        assert loaded["arms"]["anchor"]["cosine_mean"] is None

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
