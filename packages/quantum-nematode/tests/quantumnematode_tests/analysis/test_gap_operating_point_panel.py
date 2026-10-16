"""G.0's panel: its arms, its seeds, D23's minimum, the verdict map and its checks.

Covers the registration of the gap model's operating point (roadmap D23, D24): every cell of the
verdict table, ``no_gap_effect``, an unreadable level, the anchor's three minimum cases, the
per-level seed sets, the completeness check, the predictor's states and the matched statistics.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import gap_operating_point as g  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]

_THERMAL = _root / "configs" / "scenarios" / "thermal_foraging"
_M = 0.06


# ── Arms and seeds ───────────────────────────────────────────────────────────────────────────


class TestArms:
    def test_every_level_has_the_preflight_s_four_arms(self) -> None:
        assert gp.panel_stems("gap_operating_point") == g.STEMS
        for arms in g.STEMS.values():
            assert set(arms) == set(gp.ARMS)

    def test_every_arm_is_a_committed_config(self) -> None:
        for arms in g.STEMS.values():
            for stem in arms.values():
                assert (_THERMAL / f"{stem}.yml").is_file(), stem

    def test_the_single_null_levels_share_the_count_wild_type(self) -> None:
        for level in (g.SCATTERED, g.HELD, g.GAP_OFF_VS_WT):
            assert g.STEMS[level]["wt_learn"] == g.STEMS[g.COUNT]["wt_learn"]
            assert g.STEMS[level]["wt_frozen"] == g.STEMS[g.COUNT]["wt_frozen"]

    def test_gap_off_is_one_brain_read_against_both_count_arms(self) -> None:
        off = g.STEMS[g.GAP_OFF_VS_WT]["rn_learn"]
        assert g.STEMS[g.GAP_OFF_VS_NULL]["wt_learn"] == off
        assert g.STEMS[g.GAP_OFF_VS_NULL]["rn_learn"] == g.STEMS[g.COUNT]["rn_learn"]

    def test_the_profile_borrows_the_count_floors(self) -> None:
        for level in g.FLOOR_BORROWED:
            assert g.STEMS[level]["wt_frozen"] == g.STEMS[g.COUNT]["wt_frozen"]
            assert g.STEMS[level]["rn_frozen"] == g.STEMS[g.COUNT]["rn_frozen"]


class TestSeeds:
    def test_the_panel_is_896_runs_and_the_anchor_64(self) -> None:
        assert sum(len(s) for s in g.runs().values()) == 896
        assert sum(len(s) for s in g.runs(g.ANCHOR).values()) == 64

    def test_the_deciding_levels_carry_floors_on_every_seed(self) -> None:
        for level in (g.COUNT, *g.DECIDING):
            assert g.FLOOR_SEEDS[level] == g.GAP_SEEDS[level]
        assert len(g.GAP_SEEDS[g.UNIT]) == 64
        assert len(g.GAP_SEEDS[g.SCATTERED]) == 32

    def test_floors_never_cover_seeds_the_gap_is_not_read_on(self) -> None:
        for level in g.LEVELS:
            assert set(g.FLOOR_SEEDS[level]) <= set(g.GAP_SEEDS[level])

    def test_the_bands_are_disjoint(self) -> None:
        bands = [
            set(g.SEEDS),
            set(g.ANCHOR_SEEDS),
            set(g.SMOKE_SEEDS),
            set(g.GATE_PILOT_SEEDS),
            set(ts.SEEDS),
        ]
        for i, a in enumerate(bands):
            for b in bands[i + 1 :]:
                assert not a & b

    def test_the_gate_pilot_preflights_both_deciding_levels(self) -> None:
        assert sum(len(s) for s in g.runs(g.GATE_PILOT).values()) == 64
        assert set(g.PREFLIGHT_LEVELS) == {g.COUNT, g.UNIT, g.SCATTERED}
        assert not set(g.GATE_PILOT_SEEDS) & (
            set(g.SEEDS) | set(g.ANCHOR_SEEDS) | set(g.SMOKE_SEEDS)
        )

    def test_the_plan_groups_the_configs_by_band(self) -> None:
        plan = g.plan()
        assert plan["n_runs"] == 896
        assert [grp["n_seeds"] for grp in plan["groups"]] == [64, 32, 16]
        assert [len(grp["configs"]) for grp in plan["groups"]] == [8, 4, 16]


# ── D23's minimum ────────────────────────────────────────────────────────────────────────────


class TestAnchorMinimum:
    def test_a_lead_in_range_carries_the_committed_minimum(self) -> None:
        rule = g.anchor_minimum(0.09, 0.05, 0.12, readable=True)
        assert rule == {"case": "carried", "minimum": pytest.approx(0.0576527777)}

    @pytest.mark.parametrize("lead", [0.058, 0.13])
    def test_the_range_is_closed(self, lead: float) -> None:
        assert g.anchor_minimum(lead, 0.01, 0.2, readable=True)["case"] == "carried"

    @pytest.mark.parametrize("lead", [0.03, 0.2])
    def test_a_lead_outside_the_range_refixes_two_thirds_of_it(self, lead: float) -> None:
        rule = g.anchor_minimum(lead, lead - 0.02, lead + 0.02, readable=True)
        assert rule["case"] == "refixed"
        assert rule["minimum"] == pytest.approx(2 / 3 * abs(lead))

    def test_an_interval_including_zero_carries_the_minimum(self) -> None:
        rule = g.anchor_minimum(0.2, -0.01, 0.4, readable=True)
        assert rule == {"case": "interval_includes_zero", "minimum": g.CARRIED_MINIMUM}

    def test_a_reversed_anchor_fixes_nothing(self) -> None:
        """The null ahead on Emmons: no minimum, so the panel waits on a dated decision."""
        assert g.anchor_minimum(-0.1, -0.15, -0.05, readable=True) == {
            "case": "reversed",
            "minimum": None,
        }

    def test_the_committed_anchor_refixes_the_minimum(self) -> None:
        """Seeds 9801-9816 led by 0.1341 [0.101, 0.168]: above the range, so m is 2/3 of it."""
        rule = g.anchor_minimum(0.1341458, 0.1009, 0.1675, readable=True)
        assert rule["case"] == "refixed"
        assert rule["minimum"] == pytest.approx(0.0894, abs=1e-4)

    def test_an_unreadable_anchor_fixes_nothing(self) -> None:
        assert g.anchor_minimum(0.09, 0.05, 0.12, readable=False)["minimum"] is None


# ── The verdict map ──────────────────────────────────────────────────────────────────────────

# A reading at each classify state, at the minimum _M.
_READINGS: dict[str, tuple[float, float, float, float]] = {
    "move_wt": (0.1, 0.05, 0.15, 0.01),
    "below": (0.03, 0.01, 0.05, 0.01),
    "no_move": (0.0, -0.02, 0.02, 0.5),
    "move_null": (-0.1, -0.15, -0.05, 0.01),
    "unresolved": (0.05, 0.01, 0.09, 0.2),
}


def _gap(state: str) -> dict[str, Any]:
    mean, lo, hi, q = _READINGS[state]
    return {"gap_mean": mean, "test": {"ci_lo": lo, "ci_hi": hi, "bh_q": q}}


def _gate(readable: bool = True) -> dict[str, Any]:  # noqa: FBT001, FBT002
    return {"gate_passes": readable, "saturated": False}


def _panel(
    unit: str,
    scattered: str,
    *,
    count: str = "move_wt",
    readable: dict[str, bool] | None = None,
) -> dict[str, Any]:
    readable = readable or {}
    gaps = {g.COUNT: _gap(count), g.UNIT: _gap(unit), g.SCATTERED: _gap(scattered)}
    gaps[g.HELD] = _gap("no_move")
    gates = {level: _gate(readable.get(level, True)) for level in g.FAMILY}
    return g.read_panel(gates, gaps, _M)


_FAILS = ("no_move", "below", "move_null")
_DOES_NOT = ("no_move", "below", "move_null")


class TestVerdictMap:
    def test_each_reading_lands_in_its_state(self) -> None:
        for state in _READINGS:
            assert _panel(state, "no_move", count="move_wt")["states"][g.UNIT] == state

    @pytest.mark.parametrize("scattered", _DOES_NOT)
    def test_holds_and_does_not_survives(self, scattered: str) -> None:
        assert _panel("move_wt", scattered)["verdict"] == "survives"

    def test_holds_and_reproduces_is_both(self) -> None:
        assert _panel("move_wt", "move_wt")["verdict"] == "both"

    def test_holds_and_open_is_unresolved(self) -> None:
        assert _panel("move_wt", "unresolved")["verdict"] == "unresolved"

    @pytest.mark.parametrize("unit", _FAILS)
    @pytest.mark.parametrize("scattered", [*_DOES_NOT, "move_wt", "unresolved"])
    def test_fails_is_artefact_whatever_the_outliers_do(self, unit: str, scattered: str) -> None:
        assert _panel(unit, scattered)["verdict"] == "artefact"

    @pytest.mark.parametrize("scattered", [*_DOES_NOT, "move_wt", "unresolved"])
    def test_open_is_unresolved(self, scattered: str) -> None:
        assert _panel("unresolved", scattered)["verdict"] == "unresolved"

    def test_every_combination_is_named(self) -> None:
        assert set(g.VERDICTS) == {
            (u, s) for u in (g.HOLDS, g.FAILS, g.OPEN) for s in (g.DOES_NOT, g.REPRODUCES, g.OPEN)
        }

    def test_no_count_lead_is_no_gap_effect(self) -> None:
        assert _panel("move_wt", "no_move", count="no_move")["verdict"] == "no_gap_effect"
        assert _panel("move_wt", "no_move", count="move_null")["verdict"] == "no_gap_effect"

    def test_an_unreadable_count_level_is_unreadable(self) -> None:
        reading = _panel("move_wt", "no_move", readable={g.COUNT: False})
        assert reading["verdict"] == "unreadable"

    def test_an_unreadable_deciding_level_is_open(self) -> None:
        assert _panel("move_wt", "no_move", readable={g.UNIT: False})["verdict"] == "unresolved"
        reading = _panel("move_wt", "move_wt", readable={g.SCATTERED: False})
        assert reading["classes"][g.SCATTERED] == g.OPEN
        assert reading["verdict"] == "unresolved"
        assert _panel("no_move", "no_move", readable={g.SCATTERED: False})["verdict"] == "artefact"

    def test_a_saturated_level_is_unreadable(self) -> None:
        gaps = {lv: _gap("move_wt") for lv in g.FAMILY}
        gates = {lv: _gate() for lv in g.FAMILY}
        gates[g.UNIT] = {"gate_passes": True, "saturated": True}
        assert g.read_panel(gates, gaps, _M)["classes"][g.UNIT] == g.OPEN


# ── The manifest and its completeness ────────────────────────────────────────────────────────


def _write_logs(where: Path, requirement: g.Requirement, extra_seed: int | None = None) -> None:
    for stem, seeds in g.runs(requirement).items():
        for seed in (*seeds, *((extra_seed,) if extra_seed else ())):
            (where / f"{stem}-seed{seed}.log").write_text("")
            (where / f"{stem}-seed{seed}.exit").write_text("0\n")


class TestManifest:
    def test_a_complete_panel_passes_and_reads_each_level_on_its_seeds(
        self,
        tmp_path: Path,
    ) -> None:
        _write_logs(tmp_path, g.PANEL)
        manifest = g.build_manifest([tmp_path], tmp_path / "m.txt", g.PANEL)
        g.require_complete(manifest, g.PANEL)
        rows = [line.split() for line in manifest.read_text().splitlines()]
        held_frozen = {int(s) for arm, lv, s, _ in rows if lv == g.HELD and arm == "rn_frozen"}
        assert held_frozen == set(g.SEEDS_16)
        scattered_wt = {int(s) for arm, lv, s, _ in rows if lv == g.SCATTERED and arm == "wt_learn"}
        assert scattered_wt == set(g.SEEDS_32)
        assert len(rows) == sum(len(s) for arms in g.PANEL.values() for s in arms.values())

    def test_a_missing_run_is_refused(self, tmp_path: Path) -> None:
        _write_logs(tmp_path, g.PANEL)
        (tmp_path / f"{g.STEMS[g.HELD]['rn_frozen']}-seed2401.log").unlink()
        manifest = g.build_manifest([tmp_path], tmp_path / "m.txt", g.PANEL)
        with pytest.raises(g.GapOperatingPointError, match="outliers_held/rn_frozen: \\[2401\\]"):
            g.require_complete(manifest, g.PANEL)

    def test_a_run_that_did_not_exit_cleanly_is_not_evidence(self, tmp_path: Path) -> None:
        _write_logs(tmp_path, g.ANCHOR)
        (tmp_path / f"{g.STEMS[g.COUNT]['rn_learn']}-seed9805.exit").write_text("1\n")
        (tmp_path / f"{g.STEMS[g.COUNT]['rn_frozen']}-seed9806.exit").unlink()
        manifest = g.build_manifest([tmp_path], tmp_path / "m.txt", g.ANCHOR)
        with pytest.raises(g.GapOperatingPointError, match="rn_learn: \\[9805\\]"):
            g.require_complete(manifest, g.ANCHOR)

    def test_runs_off_the_band_are_ignored(self, tmp_path: Path) -> None:
        _write_logs(tmp_path, g.ANCHOR, extra_seed=g.SEEDS[0])
        manifest = g.build_manifest([tmp_path], tmp_path / "m.txt", g.ANCHOR)
        assert {int(line.split()[2]) for line in manifest.read_text().splitlines()} == set(
            g.ANCHOR_SEEDS,
        )

    def test_two_runs_of_one_arm_are_refused(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        _write_logs(first, g.ANCHOR)
        (second / f"{g.STEMS[g.COUNT]['wt_learn']}-seed9801.log").write_text("")
        (second / f"{g.STEMS[g.COUNT]['wt_learn']}-seed9801.exit").write_text("0\n")
        with pytest.raises(g.GapOperatingPointError, match="has two runs"):
            g.build_manifest([first, second], tmp_path / "m.txt", g.ANCHOR)


class TestCorrection:
    def test_the_family_is_corrected_together_and_nothing_else(self) -> None:
        def reading(p: float) -> dict[str, Any]:
            return {"gap_mean": 0.1, "test": {"wilcoxon_p": p, "ci_lo": 0.0, "ci_hi": 0.2}}

        gaps = {
            m: {lv: reading(0.01 * (k + 1)) for k, lv in enumerate(g.LEVELS)} for m in g.METRICS
        }
        g.correct(gaps)
        for m in g.METRICS:
            assert all("bh_q" in gaps[m][lv]["test"] for lv in g.FAMILY)
            assert not any("bh_q" in gaps[m][lv]["test"] for lv in g.LEVELS if lv not in g.FAMILY)


# ── The predictor and the table ──────────────────────────────────────────────────────────────


class TestPredictor:
    @pytest.mark.parametrize(
        ("rho", "p", "ci", "state"),
        [
            (0.4, 0.01, (0.2, 0.6), "predicts"),
            (0.1, 0.4, (-0.2, 0.25), "no_prediction"),
            (-0.4, 0.01, (-0.6, -0.2), "unresolved"),
            (0.2, 0.04, (0.05, 0.35), "unresolved"),
            (0.35, 0.2, (-0.1, 0.6), "unresolved"),
        ],
    )
    def test_the_registered_states(
        self,
        rho: float,
        p: float,
        ci: tuple[float, float],
        state: str,
    ) -> None:
        assert g.predictor_class(rho, p, ci) == state

    def test_the_committed_outcomes_are_the_082_gaps(self) -> None:
        spec = g.PREDICTOR_SOURCES["cook"]
        gaps = g.outcomes(spec["outcome_csv"], spec["outcome_column"])
        assert set(gaps) == set(spec["seeds"])

    def test_the_path_set_is_sensors_commands_and_motors(self) -> None:
        topology = g._brain(g.STEMS[g.COUNT]["wt_learn"], g.SMOKE_SEEDS[0]).topology
        names = list(topology.neuron_names)
        path = {names[i] for i in g.path_indices(topology)}
        assert {"AFDL", "AFDR", "ASEL", "AVAL", "PVCR"} <= path
        assert len(path) == 8 + len(g.COMMAND_NEURONS) + topology._motor_flat_indices.numel()


class TestTable:
    def test_the_wild_type_s_outliers_couple_far_above_one(self) -> None:
        stats = g.matched_statistics(g._brain(g.STEMS[g.COUNT]["wt_learn"], g.SMOKE_SEEDS[0]))
        assert stats["max"] == pytest.approx(89.67, abs=0.01)
        assert stats["n_pairs"] == 1095
        assert stats["count_degree_spearman"] is not None

    def test_unit_weights_have_no_count_to_correlate(self) -> None:
        stats = g.matched_statistics(g._brain(g.STEMS[g.UNIT]["wt_learn"], g.SMOKE_SEEDS[0]))
        assert stats["count_degree_spearman"] is None
        # A pair of two degree-one neurons couples at exactly 1; no pair can exceed it.
        assert stats["max"] <= 1.0

    def test_the_counts_read_back_alike_under_every_weighting(self) -> None:
        values = {
            level: g.matched_statistics(g._brain(g.STEMS[level]["wt_learn"], g.SMOKE_SEEDS[0]))[
                "count_degree_spearman"
            ]
            for level in (g.COUNT, "log_count", "scale0p5")
        }
        assert len(set(values.values())) == 1

    def test_the_depth_levels_and_gap_off_are_not_tabled(self) -> None:
        tabled = set(g.TABLE_STEMS)
        for level in ("depth2", "depth3", "depth6", g.GAP_OFF_VS_WT):
            assert g.STEMS[level]["rn_learn"] not in tabled


def test_the_csv_has_a_row_per_seed(tmp_path: Path) -> None:
    """The per-seed CSV writes every level's columns, blank where a level does not read a seed."""
    gates = {
        lv: {w: {"per_seed": {2401: {"learn": 50.0, "floor": 1.0}}} for w in ("wt", "rn")}
        for lv in g.LEVELS
    }
    gaps = {m: {lv: {"per_seed": {2401: 0.1}} for lv in g.LEVELS} for m in g.METRICS}
    path = g.write_csv({"seeds": [2401, 2402], "gates": gates, "gaps": gaps}, tmp_path / "s.csv")
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert lines[0].split(",")[0] == "seed"
    assert len(lines[0].split(",")) == 1 + 6 * len(g.LEVELS)
    assert set(lines[2].split(",")[1:]) == {""}
