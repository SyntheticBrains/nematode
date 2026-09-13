"""The perturbation-scale sweep's reading, and what it refuses to conclude.

Four failures these pin, each of which would look like a clean result:

* a fit taken over crossing seeds alone, with the non-crossers unreported, turning a censored
  metric into a positive slope;
* a width whose capability control failed being scored as a null, which reads as evidence against a
  mechanism that was never tested there;
* a derived budget for a platform outside the fitted grid presented as a measurement of it;
* a mixed reading resolved toward whichever registered verdict is nearer.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_perturbation_scale as ps  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_rule_positive_control as pc  # noqa: E402  # pyright: ignore[reportMissingImports]

CONFIG_DIR = _root / "configs" / "scenarios" / "foraging"
BASE = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis"


@dataclass
class _Record:
    """The fields of a scanned run this harness reads."""

    success: float
    foods: float


# ═══════════════════════ trials-to-criterion and censoring ═══════════════════


class TestTrialsToCriterion:
    """The registered criterion is the first TRIAL whose trailing 100-trial mean crosses."""

    def test_the_crossing_trial_is_reported_not_the_end_of_a_block(self) -> None:
        # 100 zeros then ones: the trailing 10-trial mean first reaches 0.5 at trial 105, and a
        # non-overlapping block scheme could only have said 110. The distinction is the finding.
        rewards = [0.0] * 100 + [1.0] * 100
        assert ps.trials_to_criterion(rewards, 0.5, window=10) == 105

    def test_the_window_length_is_the_earliest_possible_crossing(self) -> None:
        # No trailing mean exists before the window is full, so nothing can cross before then.
        assert ps.trials_to_criterion([1.0] * 50, 0.5, window=10) == 10

    def test_a_seed_that_never_crosses_has_no_time(self) -> None:
        # Not a large number and not the horizon: a censored seed has no criterion time at all, and
        # substituting the budget would put a fabricated point into the fit.
        assert ps.trials_to_criterion([0.0] * 500, 0.5, window=10) is None

    def test_a_run_shorter_than_the_window_has_no_time(self) -> None:
        assert ps.trials_to_criterion([1.0] * 5, 0.5, window=10) is None

    def test_the_sustained_variant_ignores_a_crossing_that_does_not_hold(self) -> None:
        # A five-trial burst crosses at trial 5 and cannot hold; the real run starts at trial 26.
        rewards = [1.0] * 5 + [0.0] * 20 + [1.0] * 100
        assert ps.trials_to_criterion(rewards, 0.9, window=5) == 5
        assert ps.trials_to_criterion(rewards, 0.9, window=5, sustain=5) == 30

    def test_a_crossing_that_holds_dates_to_the_same_trial_either_way(self) -> None:
        # The sustained variant dates a run from where it BEGAN, so a crossing that holds is not
        # pushed later by the requirement; only a spurious one is discarded.
        rewards = [0.0] * 20 + [1.0] * 100
        assert ps.trials_to_criterion(rewards, 0.9, window=5) == 25
        assert ps.trials_to_criterion(rewards, 0.9, window=5, sustain=5) == 25

    def test_a_sustain_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sustain must be >= 1"):
            ps.trials_to_criterion([1.0] * 200, 0.5, sustain=0)

    def test_a_window_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            ps.trailing_means([1.0, 2.0], window=0)

    def test_the_trailing_mean_is_overlapping_and_one_per_trial(self) -> None:
        means = ps.trailing_means([0.0, 1.0, 2.0, 3.0], window=2)
        assert list(means) == pytest.approx([0.5, 1.5, 2.5])


def _width_cell(
    units: int,
    trials: dict[int, int | None],
    *,
    void: bool = False,
) -> dict[str, Any]:
    return {
        "perturbed_units": units,
        "void": void,
        "per_seed": {
            str(seed): {"trials_to_criterion": value, "trials_to_criterion_sustained": value}
            for seed, value in trials.items()
        },
    }


class TestTheFit:
    def _exact_inverse_n(self) -> dict[int, dict[str, Any]]:
        """Build an exact linear-in-N series: 100 trials per unit, identical across seeds."""
        return {
            width: _width_cell(width, dict.fromkeys(ps.SEEDS, 100 * width))
            for width in ps.S1_WIDTHS
        }

    def test_an_exact_law_recovers_slope_one(self) -> None:
        fit = ps.fit_1n(self._exact_inverse_n())
        assert fit["defined"]
        assert fit["slope"] == pytest.approx(1.0, abs=1e-9)
        assert fit["meets_bar"]

    def test_the_prediction_and_the_bar_travel_with_the_slope(self) -> None:
        fit = ps.fit_1n(self._exact_inverse_n())
        assert fit["prediction"] == 1.0
        assert fit["bar"] == ps.SLOPE_BAR

    def test_a_flat_series_does_not_meet_the_bar(self) -> None:
        flat = {width: _width_cell(width, dict.fromkeys(ps.SEEDS, 5000)) for width in ps.S1_WIDTHS}
        fit = ps.fit_1n(flat)
        assert fit["slope"] == pytest.approx(0.0, abs=1e-9)
        assert not fit["meets_bar"]

    def test_censored_seeds_are_excluded_and_counted(self) -> None:
        widths = self._exact_inverse_n()
        # Half the seeds never cross at the two largest widths, which is the shape censoring takes:
        # it bites hardest where the prediction says learning is slowest.
        for width in (64, 128):
            for seed in (5, 6, 7, 8):
                widths[width]["per_seed"][str(seed)]["trials_to_criterion"] = None
        fit = ps.fit_1n(widths)
        assert fit["n_points"] == len(ps.S1_WIDTHS) * len(ps.SEEDS) - 8
        assert fit["defined"]

    def test_a_void_width_contributes_nothing(self) -> None:
        widths = self._exact_inverse_n()
        widths[128]["void"] = True
        fit = ps.fit_1n(widths)
        assert 128 not in fit["widths_in_fit"]

    def test_crossings_at_one_width_only_leave_the_fit_undefined(self) -> None:
        # A slope through a single x is not a slope; without this it would be fitted anyway.
        widths = {
            width: _width_cell(width, dict.fromkeys(ps.SEEDS, None)) for width in ps.S1_WIDTHS
        }
        widths[8] = _width_cell(8, dict.fromkeys(ps.SEEDS, 800))
        fit = ps.fit_1n(widths)
        assert not fit["defined"]
        assert "single width" in fit["reason"]

    def test_the_interval_is_reproducible(self) -> None:
        first = ps.fit_1n(self._exact_inverse_n())
        second = ps.fit_1n(self._exact_inverse_n())
        assert first["ci_low"] == second["ci_low"]
        assert first["ci_high"] == second["ci_high"]


class TestTheDerivedBudget:
    def test_every_figure_is_labelled_an_extrapolation(self) -> None:
        fit = ps.fit_1n(
            {
                width: _width_cell(width, dict.fromkeys(ps.SEEDS, 100 * width))
                for width in ps.S1_WIDTHS
            },
        )
        for units, label in ((128, "yardstick"), (302, "connectome")):
            row = ps.extrapolate(fit, units, label)
            assert row["extrapolation"] is True
            assert "not a measurement" in row["note"]

    def test_a_platform_beyond_the_grid_is_flagged_as_outside_it(self) -> None:
        fit = ps.fit_1n(
            {
                width: _width_cell(width, dict.fromkeys(ps.SEEDS, 100 * width))
                for width in ps.S1_WIDTHS
            },
        )
        assert ps.extrapolate(fit, ps.CONNECTOME_DRAWS, "draws")["outside_fitted_range"]
        assert not ps.extrapolate(fit, 128, "yardstick")["outside_fitted_range"]

    def test_an_undefined_fit_yields_no_budget(self) -> None:
        row = ps.extrapolate({"defined": False}, 302, "connectome")
        assert row["trials"] is None
        assert row["extrapolation"] is True

    def test_both_readings_of_the_connectomes_dimension_are_reported(self) -> None:
        # 302 units and 1208 draws differ by the settling depth, and nothing in the record says
        # which the arithmetic tracks, so a single figure would pick one silently.
        assert ps.CONNECTOME_DRAWS == ps.CONNECTOME_UNITS * ps.CONNECTOME_SETTLING_STEPS


# ═══════════════════════════ S1's width cells ════════════════════════════════


def _s1_runs(
    rule_score: float,
    reference_score: float,
    shapes: tuple[tuple[int, int], ...] = ps.S1_SHAPES,
) -> list[dict[str, Any]]:
    """Build a complete S1 run list where every shape reads the same, for the clause tested."""
    runs: list[dict[str, Any]] = []
    for hidden, layers in shapes:
        for seed in ps.SEEDS:
            for arm, score in (("node_perturbation", rule_score), ("analytic", reference_score)):
                runs.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "hidden": hidden,
                        "layers": layers,
                        "perturbed_units": hidden * layers,
                        "score": score,
                        "alignment": 0.0,
                        "rewards": [score] * (pc.BLOCK * 4),
                    },
                )
    return runs


class TestReachability:
    """What a width can reach is a property of the width, so the reference is read per width."""

    def _task(self) -> Any:
        return pc.ContextualAssociation.default()

    def test_a_failing_reference_voids_its_width(self) -> None:
        task = self._task()
        floor = task.cue_blind_floor(pc.NOISE)
        runs = _s1_runs(rule_score=floor, reference_score=floor)
        cell = ps.assess_s1_shape(runs, (8, 1), task)
        assert cell["void"]
        assert "analytic reference" in cell["void_reason"]

    def test_a_void_width_is_not_normalised(self) -> None:
        # Dividing by a broken reference would manufacture a number out of a failed control.
        task = self._task()
        floor = task.cue_blind_floor(pc.NOISE)
        cell = ps.assess_s1_shape(_s1_runs(floor, floor), (8, 1), task)
        assert cell["reachability_normalised"] is None

    def test_a_passing_reference_normalises_the_rule(self) -> None:
        task = self._task()
        optimum = task.optimum(pc.NOISE)
        floor = task.cue_blind_floor(pc.NOISE)
        midpoint = floor + 0.5 * (optimum - floor)
        cell = ps.assess_s1_shape(_s1_runs(midpoint, optimum), (8, 1), task)
        assert not cell["void"]
        assert cell["reachability_normalised"] == pytest.approx(0.5, abs=1e-6)

    def test_the_censoring_rate_is_reported(self) -> None:
        task = self._task()
        floor = task.cue_blind_floor(pc.NOISE)
        cell = ps.assess_s1_shape(_s1_runs(floor, task.optimum(pc.NOISE)), (8, 1), task)
        assert cell["censored"] == len(ps.SEEDS)
        assert cell["censoring_rate"] == pytest.approx(1.0)


class TestS1Verdicts:
    def test_a_platform_that_does_not_reproduce_its_pass_is_void(self) -> None:
        task = pc.ContextualAssociation.default()
        floor = task.cue_blind_floor(pc.NOISE)
        s1 = ps.analyse_s1(_s1_runs(floor, task.optimum(pc.NOISE)))
        assert s1["verdict"] == "void"
        assert not s1["baseline_reproduces"]
        assert "drifted" in s1["why"]

    def test_the_largest_widths_pass_is_reported_separately(self) -> None:
        task = pc.ContextualAssociation.default()
        optimum = task.optimum(pc.NOISE)
        s1 = ps.analyse_s1(_s1_runs(optimum, optimum))
        assert s1["baseline_reproduces"]
        assert s1["largest_width_passes"]


# ════════════════════════════ S2's scan and gates ════════════════════════════


class TestScanKeepsTheWidthsApart:
    # A body the plateau-tail parser actually reads, so a dropped log cannot be mistaken for a
    # filtered one: without this every assertion below would pass on `read_log` returning None.
    _BODY = "\n".join(f"Run: {i} Status: SUCCESS Eaten: 20/20" for i in range(1, 41)) + "\n"

    def _dir(self, tmp_path: Path, names: list[str], body: str | None = None) -> Path:
        logs = tmp_path / "logs"
        logs.mkdir()
        for name in names:
            (logs / name).write_text(self._BODY if body is None else body)
        return tmp_path

    def _name(self, arm: str, width: int, seed: int) -> str:
        suffix = "_frozen" if arm == "frozen" else ""
        stem = "ppo" if arm == "ppo" else "nodepert"
        return f"{ps._S2_STEM}_{stem}_w{width:02d}{suffix}-seed{seed}.log"

    def test_the_fixture_body_parses(self, tmp_path: Path) -> None:
        # The guard on every test in this class: a body that does not parse would make them vacuous.
        out = ps.scan_s2(self._dir(tmp_path, [self._name("nodepert", 8, 1)]))
        assert out[8]["learning"][1].foods == pytest.approx(20.0)

    def test_a_label_from_another_campaign_is_skipped(self, tmp_path: Path) -> None:
        out = ps.scan_s2(self._dir(tmp_path, ["something_else-seed1.log"]))
        assert all(not cell["learning"] for cell in out.values())

    def test_an_unregistered_width_is_skipped(self, tmp_path: Path) -> None:
        # Both halves: the width is absent from the result AND no registered cell absorbed its run.
        out = ps.scan_s2(self._dir(tmp_path, [self._name("nodepert", 96, 1)]))
        assert 96 not in out
        assert all(not cell["learning"] for cell in out.values())

    def test_a_duplicate_run_is_refused(self, tmp_path: Path, monkeypatch: Any) -> None:
        # Two logs claiming one cell: the second would otherwise replace the first and the campaign
        # would score as if one run had happened. `seed1` and `seed01` are distinct filenames that
        # parse to the same seed, which is how this arises in practice.
        monkeypatch.setattr(ps, "read_log", lambda *_a, **_k: _Record(0.0, 0.0))
        root = self._dir(
            tmp_path,
            [self._name("nodepert", 8, 1), f"{ps._S2_STEM}_nodepert_w08-seed01.log"],
        )
        with pytest.raises(ValueError, match="duplicates an already-read run"):
            ps.scan_s2(root)

    def test_a_width_written_without_its_padding_is_the_same_cell(self, tmp_path: Path) -> None:
        # `w8` and `w08` name one width; without this the unpadded form would be silently dropped as
        # an unregistered width and its runs would vanish from the campaign.
        out = ps.scan_s2(self._dir(tmp_path, [f"{ps._S2_STEM}_nodepert_w8-seed1.log"]))
        assert 8 in out

    def test_a_frozen_ppo_arm_is_not_a_registered_cell(self, tmp_path: Path) -> None:
        # The body parses, so the skip has to come from the guard rather than from a dropped log.
        name = f"{ps._S2_STEM}_ppo_w08_frozen-seed1.log"
        out = ps.scan_s2(self._dir(tmp_path, [name]))
        assert not out[8]["ppo"]
        assert not out[8]["learning"]
        assert not out[8]["frozen"]

    def test_an_unparseable_log_is_dropped_rather_than_scored(self, tmp_path: Path) -> None:
        out = ps.scan_s2(self._dir(tmp_path, [self._name("nodepert", 8, 1)], body="no run lines\n"))
        assert not out[8]["learning"]


def _width_data(
    learning: float,
    frozen: float,
    ppo: float,
    *,
    ppo_clear: float = ms.COMPETENT_THRESHOLD + 5.0,
    spread: float = 0.0,
) -> dict[str, Any]:
    """One width's three arms. ``spread`` breaks the ties a rank test needs broken."""
    return {
        "learning": {s: _Record(0.0, learning + spread * i) for i, s in enumerate(ps.SEEDS)},
        "frozen": {s: _Record(0.0, frozen) for s in ps.SEEDS},
        "ppo": {s: _Record(ppo_clear, ppo) for s in ps.SEEDS},
        "logs": {},
    }


class TestTheCapabilityGate:
    def test_both_parts_must_hold(self) -> None:
        result = ps.capability(_width_data(1.0, 1.0, 9.0))
        assert result["beats_floor"]
        assert result["competent"]
        assert result["passes"]

    def test_a_width_that_cannot_beat_the_floor_fails_and_says_so(self) -> None:
        result = ps.capability(_width_data(1.0, 9.0, 9.0))
        assert not result["beats_floor"]
        assert "beats the do-nothing floor" in result["why"]

    def test_an_incompetent_width_fails_on_the_committed_threshold(self) -> None:
        result = ps.capability(_width_data(1.0, 1.0, 9.0, ppo_clear=0.0))
        assert not result["competent"]
        assert result["competence_threshold"] == ms.COMPETENT_THRESHOLD

    def test_the_comparator_mismatch_is_recorded(self) -> None:
        # The frozen comparator perturbs and the PPO arm does not, so this is a floor check on the
        # width rather than a matched pair, and the record must not read as the latter.
        assert "not a matched pair" in ps.capability(_width_data(1.0, 1.0, 9.0))["comparator_note"]


class TestBothEffectMinima:
    def _cell(self, effect: float, reachable: float) -> dict[str, Any]:
        return {
            "graded": {"defined": True, "effect": effect},
            "capability": {"reachable_gap_foods": reachable},
        }

    def test_a_shift_below_the_absolute_minimum_fails(self) -> None:
        result = ps._minima(self._cell(0.4, 20.0))
        assert not result["passes"]
        assert f"the {ps.MIN_FOODS} foods minimum" in result["why"]

    def test_a_shift_below_the_reachable_share_fails(self) -> None:
        # 1.5 foods clears the absolute bar but is under a tenth of a 20-food reachable gap.
        result = ps._minima(self._cell(1.5, 20.0))
        assert not result["passes"]
        assert "reachable-gap" in result["why"]

    def test_both_together_pass(self) -> None:
        assert ps._minima(self._cell(3.0, 20.0))["passes"]

    def test_an_unreadable_reachable_gap_does_not_pass_by_default(self) -> None:
        assert not ps._minima(self._cell(5.0, float("nan")))["passes"]


class TestS2Verdicts:
    def _scanned(self, cells: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        return {width: cells[width] for width in ps.S2_WIDTHS}

    def test_an_uninterpretable_width_is_never_a_null(self) -> None:
        # A width whose control failed has not tested the mechanism, so it cannot count against it.
        cells = {
            width: _width_data(1.0, 1.0, 1.0, ppo_clear=0.0, spread=0.1) for width in ps.S2_WIDTHS
        }
        out = ps.analyse_s2(self._scanned(cells))
        assert all(out["widths"][w]["verdict"] == "uninterpretable" for w in ps.S2_WIDTHS)
        assert out["verdict"] == "void"
        assert out["excluded_uninterpretable"] == list(ps.S2_WIDTHS)

    def test_an_uninterpretable_width_is_excluded_from_the_trend(self) -> None:
        cells = {width: _width_data(6.0, 1.0, 12.0, spread=0.1) for width in ps.S2_WIDTHS}
        cells[4] = _width_data(6.0, 1.0, 1.0, ppo_clear=0.0, spread=0.1)
        out = ps.analyse_s2(self._scanned(cells))
        assert 4 not in out["interpretable"]
        assert 4 not in (out["trend"].get("widths") or [])

    def test_a_significant_shift_below_the_minima_is_not_a_win(self) -> None:
        cells = {width: _width_data(1.4, 1.0, 12.0, spread=0.01) for width in ps.S2_WIDTHS}
        out = ps.analyse_s2(self._scanned(cells))
        assert out["verdict"] == "not_rescued"
        assert {out["widths"][w]["verdict"] for w in ps.S2_WIDTHS} <= {
            "below_min_effect",
            "no_improvement",
        }

    def test_a_clear_shift_at_every_width_is_a_rescue(self) -> None:
        cells = {width: _width_data(9.0, 1.0, 12.0, spread=0.1) for width in ps.S2_WIDTHS}
        out = ps.analyse_s2(self._scanned(cells))
        assert out["verdict"] == "rescued"
        assert out["winners"] == list(ps.S2_WIDTHS)

    def test_the_perturbed_unit_count_is_twice_the_width(self) -> None:
        cells = {width: _width_data(1.0, 1.0, 12.0, spread=0.1) for width in ps.S2_WIDTHS}
        out = ps.analyse_s2(self._scanned(cells))
        for width in ps.S2_WIDTHS:
            assert out["widths"][width]["perturbed_units"] == 2 * width

    def test_a_flat_series_is_reported_as_constant_not_as_a_failed_test(self) -> None:
        # Spearman on a constant series returns not-a-number, which in a record reads as a failed
        # test rather than as the flat series it is.
        cells = {width: _width_data(9.0, 1.0, 12.0) for width in ps.S2_WIDTHS}
        trend = ps.analyse_s2(self._scanned(cells))["trend"]
        assert not trend["defined"]
        assert trend["constant_at"] == pytest.approx(8.0)

    def test_a_varying_series_is_correlated(self) -> None:
        cells = {
            width: _width_data(9.0 - 0.5 * index, 1.0, 12.0, spread=0.1)
            for index, width in enumerate(ps.S2_WIDTHS)
        }
        trend = ps.analyse_s2(self._scanned(cells))["trend"]
        assert trend["defined"]
        assert trend["in_predicted_direction"]

    def test_the_trend_is_labelled_descriptive(self) -> None:
        cells = {width: _width_data(9.0, 1.0, 12.0, spread=0.1) for width in ps.S2_WIDTHS}
        out = ps.analyse_s2(self._scanned(cells))
        assert out["trend"]["descriptive_only"]


class TestAnIncompleteCampaignIsNotScored:
    def _complete(self) -> dict[int, dict[str, Any]]:
        return {
            width: {
                "learning": dict.fromkeys(ps.SEEDS, object()),
                "frozen": dict.fromkeys(ps.SEEDS, object()),
                "ppo": dict.fromkeys(ps.SEEDS, object()),
                "logs": {},
            }
            for width in ps.S2_WIDTHS
        }

    def test_a_complete_campaign_passes(self) -> None:
        ps.require_complete_s2(self._complete())

    def test_a_missing_capability_run_refuses_a_verdict(self) -> None:
        # Without this the width would be called interpretable on an arm that never ran.
        scanned = self._complete()
        del scanned[4]["ppo"][3]
        with pytest.raises(ValueError, match=r"w04/ppo seeds \[3\]"):
            ps.require_complete_s2(scanned)

    def test_the_registered_campaign_is_a_hundred_and_twenty_runs(self) -> None:
        assert len(ps.S2_WIDTHS) * 3 * len(ps.SEEDS) == 120


# ═════════════════════════ the combined verdict ══════════════════════════════


class TestTheCombinedVerdict:
    def _s1(self, verdict: str, *, largest_passes: bool) -> dict[str, Any]:
        return {"verdict": verdict, "why": "fixture", "largest_width_passes": largest_passes}

    def _s2(self, verdict: str, *, trend_as_predicted: bool = True) -> dict[str, Any]:
        return {
            "verdict": verdict,
            "trend": {"defined": True, "in_predicted_direction": trend_as_predicted},
        }

    def test_scale_limited_needs_all_three_halves(self) -> None:
        out = ps.combine(
            self._s1("scale_dependent", largest_passes=False),
            self._s2("rescued"),
        )
        assert out["verdict"] == "scale_limited"

    def test_a_confirmed_law_without_a_rescue_is_a_second_defect(self) -> None:
        out = ps.combine(
            self._s1("scale_dependent", largest_passes=False),
            self._s2("not_rescued"),
        )
        assert out["verdict"] == "arithmetic_only"
        assert "second" in out["why"]

    def test_a_flat_sweep_with_the_largest_width_passing_leaves_the_record_standing(self) -> None:
        out = ps.combine(self._s1("flat", largest_passes=True), self._s2("not_rescued"))
        assert out["verdict"] == "not_scale_limited"
        assert "keeps its reading" in out["why"]

    def test_a_mixed_reading_is_recorded_as_mixed(self) -> None:
        # A rescue with no dependence fits no registered outcome; resolving it toward the nearer
        # verdict is the move this phase has repeatedly caught itself making.
        out = ps.combine(self._s1("flat", largest_passes=True), self._s2("rescued"))
        assert out["verdict"] == "mixed"
        assert "both halves stated" in out["why"]

    def test_a_below_bar_slope_with_a_failing_largest_width_is_mixed(self) -> None:
        out = ps.combine(
            self._s1("below_bar", largest_passes=False),
            self._s2("not_rescued"),
        )
        assert out["verdict"] == "mixed"

    def test_a_void_sweep_carries_no_combined_verdict(self) -> None:
        out = ps.combine(self._s1("void", largest_passes=False), self._s2("rescued"))
        assert out["verdict"] == "void"

    def test_s1_alone_is_a_registered_state(self) -> None:
        out = ps.combine(self._s1("scale_dependent", largest_passes=False), None)
        assert out["verdict"] == "s1_only"
        assert "does not depend on S2" in out["why"]


# ═══════════════════════════════ the configs ═════════════════════════════════

# The keys every S2 config is allowed to differ from the committed base by, and nothing else.
_CELL_KEYS = {
    "max_steps": 350,
    "satiety.satiety_gain_per_food": 0.2,
    "environment.foraging.target_foods_to_collect": 20,
}
_WIDTH_KEYS = ("brain.config.actor_hidden_dim", "brain.config.critic_hidden_dim")
_PLASTIC_KEYS = {
    "brain.config.learning_rule": "three_factor",
    "brain.config.enable_activity_traces": True,
    "brain.config.plasticity_normalise_modulator": True,
    "brain.config.plasticity_normalise_trace": True,
    "brain.config.plasticity_homeostasis": True,
    "brain.config.initial_log_std": -1.0,
    "brain.config.plasticity_rate": 0.001,
    "brain.config.plasticity_eligibility": "node_perturbation",
    "brain.config.plasticity_node_noise": 0.2,
    "brain.config.trace_decay": 0.9,
    "brain.config.activation": "tanh",
    "brain.config.plastic_layers": "hidden",
}
_PPO_KEYS = {
    "brain.config.learning_rule": "ppo",
    "brain.config.initial_log_std": -1.0,
    "brain.config.activation": "tanh",
}


def _flat(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, path + "."))
        else:
            out[path] = value
    return out


def _load(name: str) -> dict[str, Any]:
    return _flat(yaml.safe_load((CONFIG_DIR / f"{name}.yml").read_text()))


def _arm_name(arm: str, width: int) -> str:
    suffix = "_frozen" if arm == "frozen" else ""
    stem = "ppo" if arm == "ppo" else "nodepert"
    return f"{BASE}_hard350_{stem}_w{width:02d}{suffix}"


class TestTheConfigsDifferByTheRegisteredKeysOnly:
    """A stray key would be a second manipulation nobody registered."""

    @pytest.mark.parametrize("width", ps.S2_WIDTHS)
    @pytest.mark.parametrize("arm", ["nodepert", "frozen", "ppo"])
    def test_only_the_registered_keys_move(self, arm: str, width: int) -> None:
        base, variant = _load(BASE), _load(_arm_name(arm, width))
        allowed = dict(_CELL_KEYS)
        allowed.update(_PPO_KEYS if arm == "ppo" else _PLASTIC_KEYS)
        if arm == "frozen":
            allowed["brain.config.freeze_updates"] = True
        for key in _WIDTH_KEYS:
            allowed[key] = width
        assert set(base) - set(variant) == set(), "no base key may be dropped"
        for key, value in variant.items():
            if key in base and base[key] == value:
                continue
            assert key in allowed, f"{key} moved but is not a registered key"
            assert value == allowed[key], f"{key} is {value!r}, registered as {allowed[key]!r}"

    @pytest.mark.parametrize("width", ps.S2_WIDTHS)
    def test_the_frozen_arm_still_perturbs(self, width: int) -> None:
        # The perturbation's cost must be present in BOTH arms of the pair, so the contrast measures
        # the update's benefit rather than the net effect of switching perturbation on.
        frozen = _load(_arm_name("frozen", width))
        assert frozen["brain.config.plasticity_node_noise"] == 0.2
        assert frozen["brain.config.freeze_updates"] is True

    @pytest.mark.parametrize("width", ps.S2_WIDTHS)
    def test_the_capability_arm_shares_the_architecture_it_certifies(self, width: int) -> None:
        ppo, plastic = _load(_arm_name("ppo", width)), _load(_arm_name("nodepert", width))
        for key in (
            "brain.config.activation",
            "brain.config.actor_hidden_dim",
            "brain.config.num_hidden_layers",
            "brain.config.initial_log_std",
            "brain.config.entropy_coef",
        ):
            assert ppo[key] == plastic[key], f"{key} differs between the capability and rule arms"

    @pytest.mark.parametrize("width", ps.S2_WIDTHS)
    def test_the_capability_arm_carries_no_plasticity_keys(self, width: int) -> None:
        ppo = _load(_arm_name("ppo", width))
        assert not [k for k in ppo if "plasticity" in k or k.endswith("plastic_layers")]

    def test_the_grid_spans_the_one_step_sweeps_units(self) -> None:
        # Two hidden layers with hidden-only plasticity: the perturbed-unit count is twice the
        # width, so this grid and S1's are the same grid in the dimension under test.
        assert tuple(ps.S2_HIDDEN_LAYERS * w for w in ps.S2_WIDTHS) == ps.S1_WIDTHS

    @pytest.mark.parametrize("width", ps.S2_WIDTHS)
    def test_every_arm_runs_two_hidden_layers(self, width: int) -> None:
        for arm in ("nodepert", "frozen", "ppo"):
            assert _load(_arm_name(arm, width))["brain.config.num_hidden_layers"] == 2


# ═════════════════ the committed control is left where it was ════════════════


class TestTheWidthAxisLeavesTheCommittedControlAlone:
    def test_the_default_width_is_the_pinned_one(self) -> None:
        assert pc.HIDDEN == 8

    def test_the_default_and_the_explicit_pin_are_the_same_run(self) -> None:
        # Every value recorded by I.0-I.3b took the default; if threading the width had shifted the
        # random stream, those records would no longer reproduce.
        task = pc.ContextualAssociation.default()
        default = pc.run_arm("node_perturbation", 1, task, trials=300, node_noise=0.2)
        pinned = pc.run_arm(
            "node_perturbation",
            1,
            task,
            trials=300,
            node_noise=0.2,
            hidden=pc.HIDDEN,
        )
        assert default["score"] == pinned["score"]
        assert default["alignment"] == pinned["alignment"]

    def test_a_different_width_is_a_different_run(self) -> None:
        task = pc.ContextualAssociation.default()
        narrow = pc.run_arm("node_perturbation", 1, task, trials=300, node_noise=0.2, hidden=8)
        wide = pc.run_arm("node_perturbation", 1, task, trials=300, node_noise=0.2, hidden=64)
        assert narrow["score"] != wide["score"]
        assert wide["hidden"] == 64

    def test_a_width_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="hidden must be >= 1"):
            pc._actor(3, __import__("torch").Generator(), 0)

    def test_every_trials_reward_is_carried_without_changing_the_score(self) -> None:
        # Per-trial, not per-block: the registered criterion is a trailing mean at every trial, and
        # block means can only ever report the end of the block a crossing fell inside.
        task = pc.ContextualAssociation.default()
        run = pc.run_arm("node_perturbation", 1, task, trials=300, node_noise=0.2)
        assert len(run["rewards"]) == 300
        assert math.isfinite(run["score"])


class TestTheDepthControl:
    """The yardstick's own shape at a matched unit count, which separates shape from task."""

    def test_the_shape_is_the_yardsticks(self) -> None:
        hidden, layers = ps.S1_DEPTH_CONTROL
        assert (hidden, layers) == (64, 2)
        assert hidden * layers == ps.YARDSTICK_UNITS

    def test_it_is_scored_separately_from_the_width_grid(self) -> None:
        task = pc.ContextualAssociation.default()
        optimum = task.optimum(pc.NOISE)
        runs = _s1_runs(optimum, optimum, (*ps.S1_SHAPES, ps.S1_DEPTH_CONTROL))
        s1 = ps.analyse_s1(runs)
        assert s1["depth_control"]["perturbed_units"] == ps.YARDSTICK_UNITS
        assert s1["depth_control"]["shape"] == "64x2"
        assert s1["depth_control"]["passes"]

    def test_the_matched_one_layer_cell_is_carried_beside_it(self) -> None:
        # 128 units in one layer and in two is the comparison; a number without its match would not
        # separate depth from anything.
        task = pc.ContextualAssociation.default()
        optimum = task.optimum(pc.NOISE)
        runs = _s1_runs(optimum, optimum, (*ps.S1_SHAPES, ps.S1_DEPTH_CONTROL))
        s1 = ps.analyse_s1(runs)
        assert s1["depth_control"]["matched_one_layer_gap_fraction"] == pytest.approx(1.0, abs=1e-6)

    def test_a_sweep_without_it_reports_none_rather_than_a_fabricated_cell(self) -> None:
        task = pc.ContextualAssociation.default()
        optimum = task.optimum(pc.NOISE)
        assert ps.analyse_s1(_s1_runs(optimum, optimum))["depth_control"] is None

    def test_a_shape_is_not_confused_with_a_width_of_the_same_unit_count(self) -> None:
        # 64x2 and 128x1 are both 128 units; keyed on units alone they would pool into one cell.
        task = pc.ContextualAssociation.default()
        optimum, floor = task.optimum(pc.NOISE), task.cue_blind_floor(pc.NOISE)
        runs = _s1_runs(optimum, optimum, ((128, 1),)) + _s1_runs(floor, optimum, ((64, 2),))
        one_layer = ps.assess_s1_shape(runs, (128, 1), task)
        two_layer = ps.assess_s1_shape(runs, (64, 2), task)
        assert one_layer["rule"]["passes"]
        assert not two_layer["rule"]["passes"]


class TestTheDirectionOfTheDependence:
    def _widths(self, trials_at: dict[int, int]) -> dict[int, dict[str, Any]]:
        return {
            width: _width_cell(width, dict.fromkeys(ps.SEEDS, trials_at[width]))
            for width in ps.S1_WIDTHS
        }

    def test_a_negative_slope_is_not_reported_as_below_the_bar(self) -> None:
        # A CI entirely below zero is a dependence running the OTHER WAY, not a weak version of the
        # prediction; "below the bar" would read as a small positive effect.
        task = pc.ContextualAssociation.default()
        optimum = task.optimum(pc.NOISE)
        runs = _s1_runs(optimum, optimum)
        # Make the wider cells cross sooner by shortening their curves' rise.
        for run in runs:
            if run["arm"] != "node_perturbation":
                continue
            lead = {8: 12, 16: 9, 32: 6, 64: 3, 128: 1}[run["hidden"]]
            run["rewards"] = [optimum - 10.0] * (lead * pc.BLOCK) + [optimum] * (pc.BLOCK * 20)
        s1 = ps.analyse_s1(runs)
        assert s1["fit"]["ci_high"] < 0
        assert s1["verdict"] == "opposite_direction"
        assert "opposite to the prediction" in s1["why"]

    def test_an_extrapolation_from_a_flat_slope_is_not_a_budget(self) -> None:
        # A flat series fits a slope of order 1e-17, whose sign is floating-point noise, so the
        # interval and not the point estimate has to decide this.
        fit = ps.fit_1n(self._widths(dict.fromkeys(ps.S1_WIDTHS, 3000)))
        row = ps.extrapolate(fit, ps.CONNECTOME_UNITS, "connectome")
        assert not row["is_a_budget_constraint"]
        assert "not a budget constraint" in row["note"]

    def test_an_exactly_flat_sweep_reads_as_flat_and_not_as_below_the_bar(self) -> None:
        # The degenerate interval a constant series produces sits at ~1e-16 with zero width, so
        # without a tolerance it excludes zero and the sweep would be labelled a dependence.
        task = pc.ContextualAssociation.default()
        optimum = task.optimum(pc.NOISE)
        runs = _s1_runs(optimum, optimum)
        for run in runs:
            if run["arm"] == "node_perturbation":
                run["rewards"] = [optimum - 10.0] * (5 * pc.BLOCK) + [optimum] * (pc.BLOCK * 15)
        s1 = ps.analyse_s1(runs)
        assert s1["verdict"] == "flat"

    def test_a_positive_slope_is_a_budget(self) -> None:
        fit = ps.fit_1n({w: _width_cell(w, dict.fromkeys(ps.SEEDS, 100 * w)) for w in ps.S1_WIDTHS})
        assert ps.extrapolate(fit, ps.CONNECTOME_UNITS, "connectome")["is_a_budget_constraint"]


class TestTheLevelIsReportedSeparatelyFromTheSpeed:
    def test_a_declining_level_is_detected(self) -> None:
        # Speed and asymptote are different claims, and 1/N is about the first: a dimension costing
        # a little final performance while costing no time must not be pooled into one number.
        task = pc.ContextualAssociation.default()
        floor, optimum = task.cue_blind_floor(pc.NOISE), task.optimum(pc.NOISE)
        runs = _s1_runs(optimum, optimum)
        for run in runs:
            if run["arm"] == "node_perturbation":
                step = {8: 0.0, 16: 0.1, 32: 0.2, 64: 0.3, 128: 0.4}[run["hidden"]]
                run["score"] = optimum - step * (optimum - floor)
        level = ps.analyse_s1(runs)["level_trend"]
        assert level["defined"]
        assert level["rho"] == pytest.approx(-1.0)
        assert level["descriptive_only"]

    def test_a_constant_level_is_not_correlated(self) -> None:
        task = pc.ContextualAssociation.default()
        optimum = task.optimum(pc.NOISE)
        level = ps.analyse_s1(_s1_runs(optimum, optimum))["level_trend"]
        assert not level["defined"]
        assert "constant level" in level["reason"]


class TestTheDepthParameterLeavesTheControlAlone:
    def test_the_default_depth_is_the_pinned_one(self) -> None:
        assert pc.HIDDEN_LAYERS == 1

    def test_the_default_and_the_explicit_pins_are_the_same_run(self) -> None:
        task = pc.ContextualAssociation.default()
        default = pc.run_arm("node_perturbation", 1, task, trials=300, node_noise=0.2)
        pinned = pc.run_arm(
            "node_perturbation",
            1,
            task,
            trials=300,
            node_noise=0.2,
            hidden=pc.HIDDEN,
            layers=pc.HIDDEN_LAYERS,
        )
        assert default["score"] == pinned["score"]
        assert default["perturbed_units"] == pc.HIDDEN

    def test_the_dimension_is_width_times_depth(self) -> None:
        task = pc.ContextualAssociation.default()
        run = pc.run_arm(
            "node_perturbation",
            1,
            task,
            trials=300,
            node_noise=0.2,
            hidden=64,
            layers=2,
        )
        assert run["perturbed_units"] == 128

    def test_a_depth_below_one_is_refused(self) -> None:
        import torch

        with pytest.raises(ValueError, match="layers must be >= 1"):
            pc._actor(3, torch.Generator(), 8, 0)

    def test_the_built_stack_is_the_yardsticks_arrangement(self) -> None:
        import torch
        from torch import nn

        actor = pc._actor(3, torch.Generator(), 64, 2)
        linears = [m for m in actor if isinstance(m, nn.Linear)]
        assert [(m.in_features, m.out_features) for m in linears] == [(3, 64), (64, 64), (64, 1)]


class TestDriftReadsTheSeedsItWasGiven:
    """The reused I.3b measurement keyed a module constant; a pilot runs disjoint seeds."""

    def test_the_seed_set_is_a_parameter(self, tmp_path: Path, monkeypatch: Any) -> None:
        logs = {
            "learning": {101: tmp_path / "a.log"},
            "frozen": {101: tmp_path / "b.log"},
        }
        monkeypatch.setattr(
            ps.hm,
            "_weights",
            lambda path, *_a, **_k: (
                __import__("numpy").array([2.0, 0.0])
                if path.name == "a.log"
                else __import__("numpy").array([1.0, 0.0])
            ),
        )
        given = ps.drift(logs, (101,))
        assert given["n_read"] == 1
        assert given["available"]
        assert given["mean_relative"] == pytest.approx(1.0)

    def test_the_wrong_seed_set_is_unavailable_and_not_zero(
        self,
        tmp_path: Path,
        monkeypatch: Any,
    ) -> None:
        # A drift of 0.0 is a claim -- the policy did not move -- and no matching pair is not that
        # claim. Under the default seeds a pilot's runs would read as if nothing had been written.
        logs = {
            "learning": {101: tmp_path / "a.log"},
            "frozen": {101: tmp_path / "b.log"},
        }
        monkeypatch.setattr(ps.hm, "_weights", lambda *_a, **_k: None)
        default = ps.drift(logs)
        assert default["n_read"] == 0
        assert not default["available"]
        assert math.isnan(default["mean_relative"])
        assert default["seeds_expected"] == list(ps.SEEDS)


class TestAnUnderpoweredGateHasNoVerdict:
    """A sample size is not a finding."""

    def test_the_power_floor_is_two_to_the_minus_n(self) -> None:
        assert ps.power_floor(4) == pytest.approx(0.0625)
        assert ps.power_floor(8) == pytest.approx(0.00390625)

    def test_no_pairs_has_an_infinite_floor(self) -> None:
        assert ps.power_floor(0) == float("inf")

    def test_four_pairs_cannot_decide_the_gate(self) -> None:
        # At four pairs the smallest reachable p is 0.0625, above the 0.05 level, so the floor half
        # cannot fire however large the effect is.
        data = _width_data(1.0, 1.0, 19.0, spread=0.1)
        for arm in ("learning", "frozen", "ppo"):
            data[arm] = {s: v for s, v in data[arm].items() if s in (1, 2, 3, 4)}
        result = ps.capability(data)
        assert result["underpowered"]
        assert result["passes"] is None
        assert result["beats_floor"] is None
        assert "cannot fire" in result["why"]

    def test_the_competence_half_still_reads_when_underpowered(self) -> None:
        data = _width_data(1.0, 1.0, 19.0, spread=0.1)
        for arm in ("learning", "frozen", "ppo"):
            data[arm] = {s: v for s, v in data[arm].items() if s in (1, 2, 3, 4)}
        assert ps.capability(data)["competent"]

    def test_eight_pairs_is_powered(self) -> None:
        result = ps.capability(_width_data(1.0, 1.0, 19.0, spread=0.1))
        assert not result["underpowered"]
        assert result["passes"] is True

    def test_an_undecided_gate_is_not_a_null_and_not_uninterpretable(self) -> None:
        cells = {width: _width_data(9.0, 1.0, 19.0, spread=0.1) for width in ps.S2_WIDTHS}
        for cell in cells.values():
            for arm in ("learning", "frozen", "ppo"):
                cell[arm] = {s: v for s, v in cell[arm].items() if s in (1, 2, 3, 4)}
        out = ps.analyse_s2(cells, seeds=(1, 2, 3, 4))
        for width in ps.S2_WIDTHS:
            assert out["widths"][width]["verdict"] == "capability_undecided"
        assert out["interpretable"] == []
        assert out["verdict"] == "void"


class TestTheRegisteredStopClausesGateTheSweep:
    """A void reference at the smallest width voids the sweep, not merely its own cell."""

    def _runs(self, rule: float, reference_at_8: float, reference_elsewhere: float) -> list[Any]:
        runs = _s1_runs(rule, reference_elsewhere)
        for run in runs:
            if run["arm"] == "analytic" and run["hidden"] == 8:
                run["score"] = reference_at_8
                run["rewards"] = [reference_at_8] * (pc.BLOCK * 4)
        return runs

    def test_a_void_reference_at_the_smallest_width_voids_the_sweep(self) -> None:
        # Every other width's reachability is read against a platform the reference has not shown it
        # can reach, so scoring around the broken cell would report a fit nothing underwrites.
        task = pc.ContextualAssociation.default()
        floor, optimum = task.cue_blind_floor(pc.NOISE), task.optimum(pc.NOISE)
        s1 = ps.analyse_s1(self._runs(optimum, floor, optimum))
        assert s1["verdict"] == "void"
        assert not s1["baseline_reproduces"]
        assert s1["baseline_rule_passes"]
        assert not s1["baseline_reference_usable"]
        assert "analytic reference is void" in s1["why"]

    def test_the_two_halves_of_the_gate_are_reported_apart(self) -> None:
        task = pc.ContextualAssociation.default()
        floor, optimum = task.cue_blind_floor(pc.NOISE), task.optimum(pc.NOISE)
        s1 = ps.analyse_s1(_s1_runs(floor, optimum))
        assert not s1["baseline_rule_passes"]
        assert s1["baseline_reference_usable"]
        assert "platform has drifted" in s1["why"]

    def test_a_usable_baseline_lets_the_sweep_be_scored(self) -> None:
        task = pc.ContextualAssociation.default()
        optimum = task.optimum(pc.NOISE)
        s1 = ps.analyse_s1(_s1_runs(optimum, optimum))
        assert s1["baseline_reproduces"]
        assert s1["verdict"] != "void"


class TestAPlatformLimitedS2IsNotANegative:
    def _s1(self, verdict: str, *, largest_passes: bool) -> dict[str, Any]:
        return {"verdict": verdict, "why": "fixture", "largest_width_passes": largest_passes}

    def test_a_void_s2_does_not_become_arithmetic_only(self) -> None:
        # S2 returning void means no width was interpretable, so it cannot supply the "no width
        # rescues the cell" half of arithmetic_only: that would read an absence as a negative.
        out = ps.combine(
            self._s1("scale_dependent", largest_passes=False),
            {"verdict": "void", "trend": {"defined": False}},
        )
        assert out["verdict"] == "s1_only"
        assert "absence of evidence" in out["why"]
        assert out["s2_verdict"] == "void"

    def test_a_void_s2_does_not_become_not_scale_limited_either(self) -> None:
        out = ps.combine(
            self._s1("flat", largest_passes=True),
            {"verdict": "void", "trend": {"defined": False}},
        )
        assert out["verdict"] == "s1_only"

    def test_a_void_s1_still_dominates(self) -> None:
        out = ps.combine(
            self._s1("void", largest_passes=False),
            {"verdict": "void", "trend": {"defined": False}},
        )
        assert out["verdict"] == "void"


class TestScaleLimitedRequiresTheTrend:
    """The design table's row says "trend in the predicted direction"; so does the code."""

    def _s1(self) -> dict[str, Any]:
        return {"verdict": "scale_dependent", "why": "fixture", "largest_width_passes": False}

    def _s2(self, *, trend_as_predicted: bool) -> dict[str, Any]:
        return {
            "verdict": "rescued",
            "trend": {"defined": True, "in_predicted_direction": trend_as_predicted},
        }

    def test_a_rescue_with_the_trend_as_predicted_is_scale_limited(self) -> None:
        assert (
            ps.combine(self._s1(), self._s2(trend_as_predicted=True))["verdict"] == "scale_limited"
        )

    def test_an_isolated_win_against_the_trend_is_not_scale_limited(self) -> None:
        out = ps.combine(self._s1(), self._s2(trend_as_predicted=False))
        assert out["verdict"] == "mixed"
        assert "NOT in the predicted direction" in out["why"]

    def test_an_undefined_trend_does_not_satisfy_the_requirement(self) -> None:
        # A constant series leaves the trend undefined, which is not the same as confirming it.
        out = ps.combine(self._s1(), {"verdict": "rescued", "trend": {"defined": False}})
        assert out["verdict"] == "mixed"


class TestTheCapabilityLabelHasThreeStates:
    def test_an_undecided_gate_is_not_printed_as_a_failure(self) -> None:
        assert ps._capability_label(passes=None) == "undecided"

    def test_a_pass_and_a_failure_keep_their_labels(self) -> None:
        assert ps._capability_label(passes=True) == "pass"
        assert ps._capability_label(passes=False) == "FAIL"
