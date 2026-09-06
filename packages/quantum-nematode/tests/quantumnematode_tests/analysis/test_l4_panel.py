"""L4 panel harness: log parsing, the confirmatory family, the verdict map, the pilot rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
_analysis_dir = _root / "scripts" / "analysis"
if not _analysis_dir.is_dir():
    msg = f"could not locate scripts/analysis walking up from {Path(__file__).resolve()}"
    raise RuntimeError(msg)
sys.path.insert(0, str(_analysis_dir))

import l4_panel as lp  # noqa: E402  # pyright: ignore[reportMissingImports]

_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic"
_SEEDS = tuple(range(1, 9))


def _log(path: Path, statuses: list[str], experiment_id: str | None = None) -> Path:
    lines = [
        f"Run: {i}   Status: {s:<7} Reason: predator_evasion Steps: 2400    Eaten: "
        f"{10 if s == 'SUCCESS' else 3}/10  "
        for i, s in enumerate(statuses, 1)
    ]
    if experiment_id is not None:
        lines.insert(0, f"  Experiment ID: {experiment_id}")
    path.write_text("\n".join(lines))
    return path


def _experiment(
    root: Path,
    experiment_id: str,
    onset: int | None,
    densities: list[float] | None = None,
) -> None:
    folder = root / experiment_id
    folder.mkdir(parents=True)
    exports = root / "exports" / experiment_id
    if densities is not None:
        data = exports / "session" / "data"
        data.mkdir(parents=True)
        (data / "tracking_actions.csv").write_text(
            "run,state,action,probability\n"
            + "".join(f"{i},continuous,,{d}\n" for i, d in enumerate(densities, 1)),
        )
    results = {
        "convergence_run": onset,
        "avg_predator_encounters": 4.0,
        "avg_successful_evasions": 2.0,
        "post_convergence_temperature_comfort_score": 0.7,
    }
    (folder / f"{experiment_id}.json").write_text(
        json.dumps({"results": results, "exports_path": str(exports)}),
    )


def _arm(base: float, seeds: tuple[int, ...] = _SEEDS) -> dict[int, float]:
    # Eight paired seeds with a small seed-locked wobble, enough for a one-sided Wilcoxon
    # to reach q < 0.05 when the bases differ and to give zero deltas when they match.
    return {s: base + 0.5 * ((s % 3) - 1) for s in seeds}


def _values(  # noqa: PLR0913 -- one keyword per arm reads better than a dict
    *,
    wt_plastic: float = 70.0,
    rn_plastic: float = 45.0,
    wt_frozen: float = 20.0,
    wt_hebbian: float = 30.0,
    rn_frozen: float = 20.0,
    rn_hebbian: float = 30.0,
    mlp_plastic: float = 65.0,
    seeds: tuple[int, ...] = _SEEDS,
) -> dict[str, dict[int, float]]:
    return {
        "wt_plastic": _arm(wt_plastic, seeds),
        "rn_plastic": _arm(rn_plastic, seeds),
        "wt_frozen": _arm(wt_frozen, seeds),
        "wt_hebbian": _arm(wt_hebbian, seeds),
        "rn_frozen": _arm(rn_frozen, seeds),
        "rn_hebbian": _arm(rn_hebbian, seeds),
        "mlp_plastic": _arm(mlp_plastic, seeds),
    }


# --- parsing ----------------------------------------------------------------------------


class TestReadLog:
    def test_scores_success_curve_and_experiment_record(self, tmp_path: Path) -> None:
        experiments = tmp_path / "experiments"
        _experiment(experiments, "exp1", onset=40)
        log = _log(tmp_path / "run.log", ["FAILED"] * 200 + ["SUCCESS"] * 100, "exp1")
        record = lp.read_log(log, experiments)
        assert record is not None
        assert record.success == 100.0  # final quarter (75 episodes) all SUCCESS
        assert record.foods == 10.0
        assert record.episodes == 300
        assert record.converged is True
        assert record.onset == 40
        assert record.evasion_rate == 50.0
        assert record.temp_comfort == 0.7
        # Blocks of CURVE_WINDOW: first 250 hold 50 successes; the last 50 are all successes.
        assert record.curve == [20.0, 100.0]

    def test_peak_action_density_is_read_from_the_export(self, tmp_path: Path) -> None:
        experiments = tmp_path / "experiments"
        _experiment(experiments, "exp-d", onset=10, densities=[2.5, 18.4, 1e17, 7.0])
        log = _log(tmp_path / "run.log", ["SUCCESS"] * 40, "exp-d")
        record = lp.read_log(log, experiments)
        assert record is not None
        assert record.peak_action_density == 1e17

    def test_peak_action_density_is_none_without_an_export(self, tmp_path: Path) -> None:
        experiments = tmp_path / "experiments"
        _experiment(experiments, "exp-n", onset=10)
        log = _log(tmp_path / "run.log", ["SUCCESS"] * 40, "exp-n")
        record = lp.read_log(log, experiments)
        assert record is not None
        assert record.peak_action_density is None

    def test_missing_experiment_record_reports_convergence_unknown(self, tmp_path: Path) -> None:
        log = _log(tmp_path / "run.log", ["SUCCESS"] * 40, "exp-missing")
        record = lp.read_log(log, tmp_path / "experiments")
        assert record is not None
        assert record.success == 100.0
        assert record.converged is None
        assert record.onset is None

    def test_non_converged_record_is_false_not_none(self, tmp_path: Path) -> None:
        experiments = tmp_path / "experiments"
        _experiment(experiments, "exp2", onset=None)
        log = _log(tmp_path / "run.log", ["SUCCESS"] * 40, "exp2")
        record = lp.read_log(log, experiments)
        assert record is not None
        assert record.converged is False

    def test_log_without_run_lines_is_none(self, tmp_path: Path) -> None:
        log = tmp_path / "empty.log"
        log.write_text("nothing here\n")
        assert lp.read_log(log, tmp_path) is None


class TestScanCampaign:
    def test_maps_registered_stems_rates_and_seeds(self, tmp_path: Path, capsys) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        _log(logs / f"{_STEM}-seed1.log", ["SUCCESS"] * 40)
        _log(logs / f"{_STEM}_rewired_null__rate_0p0003-seed101.log", ["FAILED"] * 40)
        _log(logs / "not_an_arm-seed1.log", ["SUCCESS"] * 40)
        _log(logs / "weird.log", ["SUCCESS"] * 40)
        scanned = lp.scan_campaign(tmp_path, tmp_path / "experiments")
        assert [(a, r, s) for a, r, s, _ in scanned] == [
            ("wt_plastic", None, 1),
            ("rn_plastic", 0.0003, 101),
        ]
        warned = capsys.readouterr().out
        assert "not a registered arm" in warned
        assert "unrecognised label" in warned

    def test_every_registered_stem_round_trips_through_the_label(self) -> None:
        for stem, arm in lp.ARMS.items():
            match = lp._LABEL.match(f"{stem}-seed7.log")
            assert match is not None
            assert lp.ARMS[match.group("stem")] == arm
            assert match.group("seed") == "7"


class TestSeedGuard:
    def test_confirmatory_mode_refuses_pilot_seeds(self, tmp_path: Path) -> None:
        log = _log(tmp_path / "run.log", ["SUCCESS"] * 40)
        record = lp.read_log(log, tmp_path)
        assert record is not None
        with pytest.raises(ValueError, match="not a panel seed"):
            lp.group_panel([("wt_plastic", None, 101, record)])

    def test_panel_seeds_are_accepted(self, tmp_path: Path) -> None:
        log = _log(tmp_path / "run.log", ["SUCCESS"] * 40)
        record = lp.read_log(log, tmp_path)
        assert record is not None
        panel = lp.group_panel([("wt_plastic", None, s, record) for s in lp.PANEL_SEEDS])
        assert sorted(panel["wt_plastic"]) == list(lp.PANEL_SEEDS)


def test_rate_encoding_round_trips_the_grid() -> None:
    """The filename-safe rate encoding is lossless on every grid point."""
    for rate in lp.RATE_GRID:
        assert lp.decode_rate(lp.encode_rate(rate)) == rate
    assert lp.encode_rate(0.0003) == "0p0003"


# --- the confirmatory family and the verdict map ----------------------------------------


class TestFamily:
    def test_has_exactly_four_members_each_corrected(self) -> None:
        out = lp.analyse(_values(), {})
        assert tuple(out["family"]) == lp.FAMILY
        for test in lp.FAMILY:
            assert "bh_q" in out["family"][test]

    def test_descriptive_pairs_are_all_twenty_one_and_uncorrected(self) -> None:
        out = lp.analyse(_values(), {})
        pairs = out["descriptive_pairs"]
        assert len(pairs) == 21
        assert all(p["descriptive"] for p in pairs)
        assert all("bh_q" not in p for p in pairs)

    def test_t4_is_the_gain_contrast(self) -> None:
        # Gains: wild-type 70-20 = 50, rewired 45-20 = 25 -> +25.
        out = lp.analyse(_values(), {})
        assert out["family"]["T4"]["mean_delta"] == pytest.approx(25.0)
        assert out["family"]["T4"]["pass"]


class TestVerdictMap:
    def test_recovery(self) -> None:
        out = lp.analyse(_values(), {})
        assert out["verdict"]["verdict"] == "recovery"
        assert out["band"]["pass"]
        assert out["verdict"]["gain_agrees_with_primary"]
        assert out["verdict"]["ensemble_invariance"]["T1"] == {"positive_seeds": 8, "n": 8}

    def test_structure_only_when_the_band_fails(self) -> None:
        out = lp.analyse(_values(mlp_plastic=95.0), {})
        assert out["family"]["T1"]["pass"]
        assert not out["band"]["pass"]
        assert out["band"]["ci_hi"] < 0
        assert out["verdict"]["verdict"] == "structure_only"

    def test_band_passes_when_the_interval_spans_zero(self) -> None:
        values = _values()
        # Noisy MLP around the wild-type level: the paired delta's interval spans zero.
        wobble = [-6.0, 5.0, -4.0, 7.0, -5.0, 6.0, -7.0, 4.0]
        values["mlp_plastic"] = {
            s: values["wt_plastic"][s] + w for s, w in zip(_SEEDS, wobble, strict=True)
        }
        out = lp.analyse(values, {})
        band = out["band"]
        assert band["ci_lo"] <= 0.0 <= band["ci_hi"]
        assert band["pass"]
        assert band["ci_width"] > 0

    def test_sanity_floor_fail_decides_first(self) -> None:
        # The frozen floor equals the plastic arm: nothing was learned, whatever T1 says.
        out = lp.analyse(_values(wt_frozen=70.0), {})
        assert out["family"]["T1"]["pass"]
        assert out["verdict"]["verdict"] == "sanity_floor_fail"

    def test_hebbian_floor_alone_fails_the_floors(self) -> None:
        out = lp.analyse(_values(wt_hebbian=70.0), {})
        assert out["verdict"]["verdict"] == "sanity_floor_fail"

    def test_robustness_when_the_primary_contrast_is_null(self) -> None:
        out = lp.analyse(_values(rn_plastic=70.0), {})
        t1 = out["family"]["T1"]
        assert not t1["pass"]
        assert t1["ci_lo"] <= 0.0 <= t1["ci_hi"]
        assert out["verdict"]["verdict"] == "robustness"

    def test_rewired_beats_wild_type_is_named_not_lost(self) -> None:
        out = lp.analyse(_values(wt_plastic=45.0, rn_plastic=70.0), {})
        assert out["family"]["T1"]["ci_hi"] < 0
        assert out["verdict"]["verdict"] == "rewired_beats_wild_type"

    def test_inconclusive_when_positive_but_underpowered(self) -> None:
        # Only the rewired arm has four seeds, so T1 pairs on four: the one-sided signed-rank
        # floor is 1/16 and a consistent positive delta cannot reach q < 0.05, while its
        # bootstrap interval sits above zero. The floors keep eight seeds and pass.
        values = _values()
        values["rn_plastic"] = _arm(45.0, (1, 2, 3, 4))
        out = lp.analyse(values, {})
        t1 = out["family"]["T1"]
        assert t1["n"] == 4
        assert not t1["pass"]
        assert t1["ci_lo"] > 0
        assert out["verdict"]["verdict"] == "inconclusive"

    def test_gain_disagreement_is_recorded_but_does_not_move_the_verdict(self) -> None:
        # Raw plateau favours wild-type (70 vs 45) but the rewired wiring gained more from
        # learning (45-5 = 40 vs 70-40 = 30).
        out = lp.analyse(_values(wt_frozen=40.0, rn_frozen=5.0), {})
        assert out["verdict"]["verdict"] == "recovery"
        assert not out["family"]["T4"]["pass"]
        assert not out["verdict"]["gain_agrees_with_primary"]

    def test_structure_only_needs_a_sufficient_band(self) -> None:
        # T1 passes but the MLP arm has one seed: neither MLP-dependent verdict is supportable.
        values = _values()
        values["mlp_plastic"] = {1: 95.0}
        out = lp.analyse(values, {})
        assert out["family"]["T1"]["pass"]
        assert not out["band"]["sufficient"]
        assert out["verdict"]["verdict"] == "insufficient_seeds"

    def test_insufficient_seeds(self) -> None:
        values = _values()
        values["wt_plastic"] = {1: 70.0}
        out = lp.analyse(values, {})
        assert out["verdict"]["verdict"] == "insufficient_seeds"


# --- the pilot rules --------------------------------------------------------------------


class TestBudgetRule:
    @pytest.mark.parametrize(
        ("onset", "budget"),
        [
            (100, 2000),
            (1000, 2000),
            (1600, 2000),
            (1601, 2500),
            (2000, 2500),
            (2400, 3000),
            (2401, 3500),
        ],
    )
    def test_rounds_up_with_headroom_and_floors(self, onset: int, budget: int) -> None:
        assert lp.budget_from_onset(onset) == budget


class TestSelectRate:
    def test_highest_pooled_mean_wins(self) -> None:
        assert lp.select_rate({0.0003: 40.0, 0.001: 55.0, 0.003: 50.0}) == 0.001
        assert lp.select_rate({0.0003: 60.0, 0.001: 55.0, 0.003: 50.0}) == 0.0003

    def test_tie_goes_to_the_default(self) -> None:
        assert lp.select_rate({0.0003: 55.0, 0.001: 55.0, 0.003: 55.0}) == lp.DEFAULT_RATE
        assert lp.select_rate({0.0003: 55.0, 0.003: 55.0}) == 0.0003  # no default among the tied

    def test_empty_is_none(self) -> None:
        assert lp.select_rate({}) is None


def _record(success: float, onset: int | None, episodes: int = 3000) -> lp.SeedRecord:
    return lp.SeedRecord(
        success=success,
        foods=8.0,
        episodes=episodes,
        converged=onset is not None,
        onset=onset,
        evasion_rate=None,
        temp_comfort=None,
        curve=[],
    )


def _pilot(
    per_rate: dict[float, dict[str, tuple[float, int | None]]],
    episodes: int = 3000,
) -> lp.Scanned:
    scanned: lp.Scanned = []
    for rate, arms in per_rate.items():
        for arm, (success, onset) in arms.items():
            for seed in lp.PILOT_SEEDS:
                scanned.append((arm, rate, seed, _record(success, onset, episodes)))
    for arm in lp.FROZEN_ARMS:
        for seed in lp.PILOT_SEEDS:
            scanned.append((arm, None, seed, _record(15.0, 5, episodes)))
    return scanned


def _grid(
    three_factor: dict[float, tuple[float, int | None]],
) -> dict[float, dict[str, tuple[float, int | None]]]:
    return {
        rate: {
            "wt_plastic": (s, onset),
            "rn_plastic": (s - 5.0, onset),
            "mlp_plastic": (s + 5.0, onset),
            "wt_hebbian": (20.0, 300),
            "rn_hebbian": (20.0, 300),
        }
        for rate, (s, onset) in three_factor.items()
    }


class TestAnalysePilot:
    def test_pooled_selection_and_budget(self) -> None:
        out = lp.analyse_pilot(
            _pilot(_grid({0.0003: (40.0, 900), 0.001: (55.0, 1800), 0.003: (50.0, 1200)})),
            {},
        )
        assert out["selected_rate"] == 0.001
        assert out["pooled"]["0p001"] == pytest.approx(55.0)
        # Latest onset at the selected rate is 1800 (three-factor arms); 1.25 x 1800 = 2250 -> 2500.
        assert out["budget"] == 2500
        assert out["budget_basis"]["latest_onset"] == 1800
        assert out["non_converged_three_factor_arms"] == []
        assert out["action_required"] == []
        assert set(out["floors"]) == set(lp.FROZEN_ARMS)

    def test_non_converged_arm_owes_an_extension(self) -> None:
        grid = _grid({0.0003: (40.0, 900), 0.001: (55.0, 1800), 0.003: (50.0, 1200)})
        grid[0.001]["mlp_plastic"] = (60.0, None)
        out = lp.analyse_pilot(_pilot(grid), {})
        assert out["non_converged_three_factor_arms"] == ["mlp_plastic"]
        assert any("extend it once to 6000" in a for a in out["action_required"])
        assert out["budget"] == 2500  # the converged arms still give the rule an input

    def test_still_non_converged_at_the_extended_budget_pins_it(self) -> None:
        grid = _grid({0.0003: (40.0, 900), 0.001: (55.0, 1800), 0.003: (50.0, 1200)})
        grid[0.001]["mlp_plastic"] = (60.0, None)
        out = lp.analyse_pilot(_pilot(grid, episodes=6000), {})
        assert out["budget"] == lp.PILOT_EXTENDED_BUDGET
        assert out["budget_basis"]["pinned_at_extended_budget_for"] == ["mlp_plastic"]

    def test_rate_missing_a_three_factor_arm_is_ineligible(self, capsys) -> None:
        grid = _grid({0.0003: (90.0, 900), 0.001: (55.0, 1800)})
        del grid[0.0003]["mlp_plastic"]
        out = lp.analyse_pilot(_pilot(grid), {})
        assert out["selected_rate"] == 0.001
        assert "ineligible" in capsys.readouterr().out

    def test_rate_with_a_missing_pilot_seed_is_ineligible(self, capsys) -> None:
        grid = _grid({0.0003: (90.0, 900), 0.001: (55.0, 1800)})
        scanned = [
            row
            for row in _pilot(grid)
            if not (row[0] == "mlp_plastic" and row[1] == 0.0003 and row[2] == 102)
        ]
        out = lp.analyse_pilot(scanned, {})
        assert out["selected_rate"] == 0.001
        assert "ineligible" in capsys.readouterr().out

    def test_budget_basis_ignores_the_floors(self) -> None:
        scanned = _pilot(_grid({0.001: (55.0, 1000)}))
        for row in scanned:
            if row[0] in lp.FROZEN_ARMS:
                row[3].onset = 2900  # a late floor onset must not set the budget
        out = lp.analyse_pilot(scanned, {})
        assert out["budget_basis"]["latest_onset"] == 1000
        assert out["budget"] == 2000

    def test_summary_is_stamped_unpinned(self) -> None:
        out = lp.analyse_pilot(_pilot(_grid({0.001: (55.0, 1800)})), {})
        assert out["pinned"] is False
        assert "dated amendment" in out["pin_note"]
        none = lp.analyse_pilot([], {})
        assert none["pinned"] is False

    def test_panel_seed_in_a_pilot_is_ignored(self, capsys) -> None:
        scanned = _pilot(_grid({0.001: (55.0, 1800)}))
        scanned.append(("wt_plastic", 0.001, 3, _record(99.0, 10)))
        out = lp.analyse_pilot(scanned, {})
        assert out["grid"]["0p001"]["wt_plastic"]["per_seed"] == {101: 55.0, 102: 55.0}
        assert "not a pilot seed" in capsys.readouterr().out


# --- the sensitivity pass ---------------------------------------------------------------


def test_sensitivity_pass_is_descriptive_and_cannot_change_the_verdict(tmp_path: Path) -> None:
    """A second campaign at another rate is reported beside the verdict, never folded into it."""
    panel = tmp_path / "panel" / "logs"
    other = tmp_path / "other" / "logs"
    panel.mkdir(parents=True)
    other.mkdir(parents=True)
    for seed in lp.PANEL_SEEDS:
        for arm in (
            "wt_plastic",
            "rn_plastic",
            "wt_frozen",
            "wt_hebbian",
            "rn_frozen",
            "mlp_plastic",
        ):
            stem = lp.STEM_OF[arm]
            # A robustness-shaped panel: floors cleared, the primary contrast null.
            wins = {"wt_plastic": 20, "rn_plastic": 20, "wt_frozen": 2, "wt_hebbian": 4}.get(arm, 1)
            _log(panel / f"{stem}-seed{seed}.log", ["FAILED"] * (40 - wins) + ["SUCCESS"] * wins)
            if arm in ("wt_plastic", "rn_plastic"):
                # At the other rate the wild-type arm clearly wins: the verdict must not move.
                wins_other = 30 if arm == "wt_plastic" else 5
                _log(
                    other / f"{stem}-seed{seed}.log",
                    ["FAILED"] * (40 - wins_other) + ["SUCCESS"] * wins_other,
                )
    out = tmp_path / "panel.json"
    code = lp.main(
        [
            "--campaign-dir",
            str(tmp_path / "panel"),
            "--experiments-dir",
            str(tmp_path / "experiments"),
            "--sensitivity",
            f"0.0003={tmp_path / 'other'}",
            "--out",
            str(out),
        ],
    )
    assert code == 0
    result = json.loads(out.read_text())
    assert result["verdict"]["verdict"] == "robustness"
    assert len(result["sensitivity"]) == 1
    assert result["sensitivity"][0]["descriptive"] is True
    assert result["sensitivity"][0]["rate"] == 0.0003
    assert (
        result["sensitivity"][0]["mean_delta"] > 0
    )  # the other rate favoured wild-type and still changed nothing


# --- exports ----------------------------------------------------------------------------


def test_csv_exports(tmp_path: Path) -> None:
    """The per-seed table and the learning curves carry one row per seed and per window."""
    experiments = tmp_path / "experiments"
    _experiment(experiments, "e", onset=10)
    log = _log(tmp_path / "run.log", ["FAILED"] * 250 + ["SUCCESS"] * 250, "e")
    record = lp.read_log(log, experiments)
    assert record is not None
    panel = {"wt_plastic": {1: record}, "rn_plastic": {1: record}}
    lp.write_per_seed_csv(panel, tmp_path / "per-seed.csv")
    lp.write_curves_csv(panel, tmp_path / "curves.csv")
    rows = (tmp_path / "per-seed.csv").read_text().splitlines()
    assert rows[0].startswith("arm,seed,success")
    assert rows[0].endswith("peak_action_density")
    assert len(rows) == 3
    curves = (tmp_path / "curves.csv").read_text().splitlines()
    assert curves[0] == "arm,seed,window_end,success"
    assert curves[1:] == [
        "wt_plastic,1,250,0.00",
        "wt_plastic,1,500,100.00",
        "rn_plastic,1,250,0.00",
        "rn_plastic,1,500,100.00",
    ]
