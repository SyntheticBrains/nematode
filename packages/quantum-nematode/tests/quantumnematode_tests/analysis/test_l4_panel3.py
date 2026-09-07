"""L4 panel-3 harness: the replication seed range, the two-test family, the verdict, the pooling."""

from __future__ import annotations

import csv
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

import l4_panel3 as p3  # noqa: E402  # pyright: ignore[reportMissingImports]

_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic"
_REP = tuple(range(17, 65))
_P2 = tuple(range(1, 17))


def _log(path: Path, statuses: list[str]) -> Path:
    path.write_text(
        "\n".join(
            f"Run: {i}   Status: {s:<7} Reason: predator_evasion Steps: 2400    Eaten: "
            f"{10 if s == 'SUCCESS' else 3}/10  "
            for i, s in enumerate(statuses, 1)
        ),
    )
    return path


def _arm(base: float, seeds: tuple[int, ...]) -> dict[int, float]:
    return {s: base + 0.5 * ((s % 3) - 1) for s in seeds}


def _values(wt: float = 40.0, rn: float = 15.0) -> dict[str, dict[int, float]]:
    return {"wt_hebbian": _arm(wt, _REP), "rn_hebbian": _arm(rn, _REP)}


def _record(
    success: float,
    episodes: int = 1000,
    *,
    converged: bool | None = True,
) -> p3.SeedRecord:
    return p3.SeedRecord(
        success=success,
        foods=5.0,
        episodes=episodes,
        converged=converged,
        onset=100 if converged else None,
        evasion_rate=None,
        temp_comfort=None,
        curve=[success],
    )


def _panel(values: dict[str, dict[int, float]]) -> dict[str, dict[int, p3.SeedRecord]]:
    return {arm: {s: _record(v) for s, v in seeds.items()} for arm, seeds in values.items()}


def _panel2(wt_heb: float = 30.0, rn_heb: float = 15.0) -> dict[str, dict[int, float]]:
    return {
        "wt_hebbian": _arm(wt_heb, _P2),
        "rn_hebbian": _arm(rn_heb, _P2),
        "wt_frozen": _arm(8.0, _P2 + _REP),
        "rn_frozen": _arm(8.0, _P2 + _REP),
    }


def _write_panel2_csv(path: Path, values: dict[str, dict[int, float]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "success"])
        for arm, seeds in values.items():
            for seed, value in sorted(seeds.items()):
                writer.writerow([arm, seed, value])
    return path


class TestSeedRanges:
    def test_constants(self) -> None:
        assert p3.REPLICATION_SEEDS == _REP
        assert p3.POOLED_SEEDS == _P2 + _REP
        assert p3.ARMS3 == ("wt_hebbian", "rn_hebbian")
        assert p3.PANEL2_CSV.is_file()

    def test_panel2_seed_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the replication seeds 17-64"):
            p3.group_panel([("wt_hebbian", 16, _record(1.0))])

    def test_other_arm_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a panel-3 arm"):
            p3.group_panel([("wt_hebbian_count", 20, _record(1.0))])

    def test_replication_bounds_are_accepted(self) -> None:
        panel = p3.group_panel([("wt_hebbian", 17, _record(1.0)), ("rn_hebbian", 64, _record(1.0))])
        assert set(panel["wt_hebbian"]) == {17}
        assert set(panel["rn_hebbian"]) == {64}


class TestDiscordance:
    def test_exact_p_on_known_counts(self) -> None:
        # 9 wild-type-only, 3 rewired-only: P(X >= 9 | Bin(12, 1/2)) = 299/4096.
        wt = {s: (50.0 if s <= 26 else 0.0) for s in _REP}  # 17..26 competent (10 seeds)
        rn = {s: (50.0 if s in (17, 27, 28, 29) else 0.0) for s in _REP}  # 17 both; 27-29 rn-only
        d = p3.discordance(wt, rn, _REP)
        assert (d["b_wild_type_only"], d["c_rewired_only"], d["both"]) == (9, 3, 1)
        assert d["wilcoxon_p"] == pytest.approx(299 / 4096)

    def test_no_discordant_pairs_reports_p_one(self) -> None:
        wt = dict.fromkeys(_REP, 50.0)
        d = p3.discordance(wt, dict(wt), _REP)
        assert (d["b_wild_type_only"], d["c_rewired_only"]) == (0, 0)
        assert d["wilcoxon_p"] == 1.0
        tests = p3.family_tests({"wt_hebbian": wt, "rn_hebbian": dict(wt)})
        assert tests["R2"]["pass"] is False

    def test_threshold_is_inclusive(self) -> None:
        d = p3.discordance({17: 20.0}, {17: 19.9}, (17,))
        assert d["b_wild_type_only"] == 1


class TestFamilyAndVerdict:
    def test_two_tests_corrected_together(self) -> None:
        tests = p3.family_tests(_values())
        assert set(tests) == {"R1", "R2"}
        assert tests["R1"]["n"] == 48
        assert tests["R1"]["complete"] is True
        assert tests["R2"]["complete"] is True
        assert tests["R1"]["pass"]
        assert tests["R2"]["pass"]
        assert p3.verdict(tests) == "specific_wiring"

    def test_panel2_seeds_never_enter_the_family(self) -> None:
        values = _values()
        values["wt_hebbian"][1] = 99.0
        values["rn_hebbian"][1] = 0.0
        tests = p3.family_tests(values)
        assert 1 not in tests["R1"]["seeds"]
        assert 1 not in tests["R2"]["seeds"]

    def test_incomplete_is_flagged_not_fatal(self) -> None:
        values = _values()
        del values["wt_hebbian"][64]
        tests = p3.family_tests(values)
        assert tests["R1"]["complete"] is False
        assert tests["R1"]["sufficient"] is True

    def test_reverse(self) -> None:
        tests = p3.family_tests(_values(wt=15.0, rn=40.0))
        assert tests["R1"]["reverse"]
        assert p3.verdict(tests) == "rewired_beats_wild_type"

    def test_degree_statistics(self) -> None:
        assert p3.verdict(p3.family_tests(_values(wt=30.0, rn=30.0))) == "degree_statistics"

    def test_inconclusive(self) -> None:
        tests = p3.family_tests(_values())
        tests["R1"].update({"pass": False, "ci_lo": 1.0, "ci_hi": 5.0})
        assert p3.verdict(tests) == "inconclusive"

    def test_insufficient_seeds(self) -> None:
        tests = p3.family_tests({"wt_hebbian": {17: 90.0}, "rn_hebbian": {17: 0.0}})
        assert p3.verdict(tests) == "insufficient_seeds"

    def test_r2_passes_r1_fails_is_annotated_not_promoted(self) -> None:
        # Wild-type competent on 30 seeds at a bare 21% with the rewired just under the threshold;
        # on the other 18 the wild-type is dead and the rewired sits at 19.9: the discordance is
        # 30-0 while the ranked deltas are +2 on 30 seeds and -19.9 on 18, a negative mean.
        wt = {s: (21.0 if s <= 46 else 0.0) for s in _REP}
        rn = {s: (19.0 if s <= 46 else 19.9) for s in _REP}
        tests = p3.family_tests({"wt_hebbian": wt, "rn_hebbian": rn})
        assert tests["R2"]["pass"] is True
        assert tests["R1"]["pass"] is False
        result = p3.verdict(tests)
        assert result != "specific_wiring"
        notes = p3.annotate(result, tests)
        assert notes["competent_fraction_confirms"] is True
        assert "reported, not promoted" in notes["reading"]

    def test_r2_annotation_when_r1_passes(self) -> None:
        tests = p3.family_tests(_values())
        notes = p3.annotate(p3.verdict(tests), tests)
        assert "and on the competent fraction" in notes["reading"]


class TestDescriptive:
    def test_learning_gains_use_panel2_floors_on_both_seed_sets(self) -> None:
        gains = p3.learning_gains(_values(), _panel2())
        assert (
            gains["floors_from"]
            == "docs/experiments/logbooks/supporting/041-l4-panel2/per-seed.csv"
        )
        assert gains["floors_campaign"] == "campaigns/l4-panel2-sweep"
        assert gains["wt_hebbian_replication"]["n"] == 48
        assert gains["wt_hebbian_replication"]["mean"] == pytest.approx(32.0)
        # Pooled gains need the Hebbian values on 1-16 too, which analyse() merges in.
        assert gains["wt_hebbian_pooled"]["n"] == 48

    def test_analyse_pools_panel2_seeds_descriptively(self) -> None:
        out = p3.analyse(_panel(_values()), _panel2(), {})
        pooled = out["pooled_descriptive"]
        assert pooled["contrast"]["n"] == 64
        assert pooled["contrast"]["descriptive"] is True
        assert pooled["wt_hebbian"]["n"] == 64
        assert out["learning_gains"]["wt_hebbian_pooled"]["n"] == 64
        assert out["per_arm"]["wt_hebbian"]["n"] == 48  # confirmatory arm summary stays at 17-64
        assert out["verdict"]["verdict"] == "specific_wiring"

    def test_extensions(self) -> None:
        panel = _panel(_values())
        panel["rn_hebbian"][20] = _record(5.0, converged=False)
        panel["wt_hebbian"][21] = _record(5.0, episodes=1500, converged=False)
        assert p3.extensions_needed(panel) == [{"arm": "rn_hebbian", "seed": 20, "episodes": 1000}]

    def test_main_end_to_end(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        for seed in _REP:
            _log(logs / f"{_STEM}_hebbian-seed{seed}.log", ["SUCCESS"] * 40)
            _log(
                logs / f"{_STEM}_hebbian_rewired_null-seed{seed}.log",
                ["SUCCESS" if seed % 5 == 0 else "FAILED"] * 40,
            )
        p2csv = _write_panel2_csv(tmp_path / "p2.csv", _panel2())
        out = tmp_path / "panel3.json"
        code = p3.main(
            [
                "--campaign-dir",
                str(tmp_path),
                "--panel2-csv",
                str(p2csv),
                "--out",
                str(out),
                "--csv",
                str(tmp_path / "s.csv"),
            ],
        )
        assert code == 0
        data = json.loads(out.read_text())
        assert data["verdict"]["verdict"] == "specific_wiring"
        assert data["family"]["R2"]["b_wild_type_only"] > data["family"]["R2"]["c_rewired_only"]
        with (tmp_path / "s.csv").open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 96
        assert "VERDICT: specific_wiring" in capsys.readouterr().out

    def test_main_rejects_a_panel2_seed(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        _log(logs / f"{_STEM}_hebbian-seed3.log", ["SUCCESS"] * 40)
        assert p3.main(["--campaign-dir", str(tmp_path)]) == 2
