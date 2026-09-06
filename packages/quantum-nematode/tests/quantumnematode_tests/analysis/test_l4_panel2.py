"""L4 panel-2 harness: the registry, seed ranges, the family, the verdict map, the sweep."""

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

import l4_panel2 as p2  # noqa: E402  # pyright: ignore[reportMissingImports]

_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic"
_HEB = tuple(range(1, 17))
_SWEEP = tuple(range(1, 65))


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


def _experiment(root: Path, experiment_id: str, onset: int | None) -> None:
    folder = root / experiment_id
    folder.mkdir(parents=True)
    results = {"convergence_run": onset}
    (folder / f"{experiment_id}.json").write_text(
        json.dumps({"results": results, "exports_path": str(root / "exports" / experiment_id)}),
    )


def _arm(base: float, seeds: tuple[int, ...]) -> dict[int, float]:
    return {s: base + 0.5 * ((s % 3) - 1) for s in seeds}


def _values(  # noqa: PLR0913 -- one keyword per arm reads better than a dict
    *,
    wt_hebbian: float = 40.0,
    rn_hebbian: float = 15.0,
    wt_hebbian_count: float = 45.0,
    rn_hebbian_count: float = 15.0,
    wt_frozen: float = 8.0,
    rn_frozen: float = 8.0,
    wt_frozen_count: float = 10.0,
    rn_frozen_count: float = 8.0,
) -> dict[str, dict[int, float]]:
    return {
        "wt_hebbian": _arm(wt_hebbian, _HEB),
        "rn_hebbian": _arm(rn_hebbian, _HEB),
        "wt_hebbian_count": _arm(wt_hebbian_count, _HEB),
        "rn_hebbian_count": _arm(rn_hebbian_count, _HEB),
        "wt_frozen": _arm(wt_frozen, _SWEEP),
        "rn_frozen": _arm(rn_frozen, _SWEEP),
        "wt_frozen_count": _arm(wt_frozen_count, _SWEEP),
        "rn_frozen_count": _arm(rn_frozen_count, _SWEEP),
    }


def _record(
    success: float,
    episodes: int = 1000,
    *,
    converged: bool | None = True,
) -> p2.SeedRecord:
    return p2.SeedRecord(
        success=success,
        foods=5.0,
        episodes=episodes,
        converged=converged,
        onset=100 if converged else None,
        evasion_rate=None,
        temp_comfort=None,
        curve=[success],
    )


def _panel(values: dict[str, dict[int, float]]) -> dict[str, dict[int, p2.SeedRecord]]:
    return {arm: {s: _record(v) for s, v in seeds.items()} for arm, seeds in values.items()}


# --- registry and seed ranges ------------------------------------------------------------


class TestRegistry:
    def test_eight_arms_with_the_count_init_stems(self) -> None:
        assert len(p2.ARMS) == 8
        assert set(p2.ARM_KEYS) == set(p2.FROZEN_ARMS) | set(p2.HEBBIAN_ARMS)
        assert p2.ARMS[f"{_STEM}_hebbian_countinit"] == "wt_hebbian_count"
        assert p2.ARMS[f"{_STEM}_frozen_rewired_null_countinit"] == "rn_frozen_count"
        for stem in p2.ARMS:
            assert (
                p2.REPO / "configs" / "scenarios" / "foraging_predator_thermal" / f"{stem}.yml"
            ).is_file()

    def test_seed_ranges(self) -> None:
        for arm in p2.HEBBIAN_ARMS:
            assert p2.SEEDS_OF[arm] == _HEB
        for arm in p2.FROZEN_ARMS:
            assert p2.SEEDS_OF[arm] == _SWEEP
        assert p2.HEBBIAN_BUDGET == 1000
        assert p2.SWEEP_BUDGET == 600
        assert p2.EXTENSION_BUDGET == 1500
        assert p2.COMPETENT_THRESHOLD == 20.0

    def test_scan_reads_registered_stems_only(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        _log(logs / f"{_STEM}_hebbian_countinit-seed3.log", ["SUCCESS"] * 40)
        _log(logs / f"{_STEM}-seed3.log", ["SUCCESS"] * 40)  # the plastic arm is not in panel 2
        _log(logs / f"{_STEM}_hebbian__rate_0p001-seed3.log", ["SUCCESS"] * 40)  # pilot label
        scanned = p2.scan_campaign(tmp_path, tmp_path / "experiments")
        assert [(arm, seed) for arm, seed, _ in scanned] == [("wt_hebbian_count", 3)]

    def test_hebbian_seed_outside_1_16_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the arm's registered seeds 1-16"):
            p2.group_panel([("wt_hebbian", 17, _record(10.0))])

    def test_frozen_seed_outside_1_64_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the arm's registered seeds 1-64"):
            p2.group_panel([("rn_frozen_count", 65, _record(10.0))])

    def test_frozen_seed_up_to_64_is_accepted(self) -> None:
        panel = p2.group_panel([("wt_frozen", 64, _record(10.0)), ("wt_hebbian", 16, _record(1.0))])
        assert set(panel["wt_frozen"]) == {64}
        assert set(panel["wt_hebbian"]) == {16}


# --- family and verdict ------------------------------------------------------------------


class TestFamily:
    def test_exactly_four_tests_with_their_directions(self) -> None:
        tests = p2.family_tests(_values())
        assert set(tests) == {"P1", "P2", "P3", "P4"}
        assert tests["P1"]["mean_delta"] == pytest.approx(25.0)
        assert tests["P2"]["mean_delta"] == pytest.approx(30.0)
        assert tests["P3"]["mean_delta"] == pytest.approx(5.0)
        assert tests["P4"]["mean_delta"] == pytest.approx(0.0)
        assert tests["P1"]["n"] == 16
        assert tests["P4"]["n"] == 64
        assert all("bh_q" in t for t in tests.values())

    def test_hebbian_contrasts_ignore_sweep_only_seeds(self) -> None:
        values = _values()
        # A frozen arm's sweep seeds must not leak into a Hebbian contrast even if a Hebbian
        # arm somehow carried them.
        values["wt_hebbian"][40] = 99.0
        values["rn_hebbian"][40] = 0.0
        tests = p2.family_tests(values)
        assert 40 not in tests["P1"]["seeds"]

    def test_pass_requires_positive_direction(self) -> None:
        tests = p2.family_tests(_values(wt_hebbian=15.0, rn_hebbian=40.0))
        assert not tests["P1"]["pass"]
        assert tests["P1"]["reverse"]


class TestVerdict:
    def test_specific_wiring(self) -> None:
        tests = p2.family_tests(_values())
        assert p2.verdict(tests) == "specific_wiring"

    def test_rewired_beats_wild_type(self) -> None:
        tests = p2.family_tests(_values(wt_hebbian=15.0, rn_hebbian=40.0))
        assert p2.verdict(tests) == "rewired_beats_wild_type"

    def test_degree_statistics(self) -> None:
        tests = p2.family_tests(_values(wt_hebbian=30.0, rn_hebbian=30.0))
        assert p2.verdict(tests) == "degree_statistics"

    def test_inconclusive(self) -> None:
        # A positive interval clear of zero but the corrected q above alpha.
        tests = p2.family_tests(_values())
        tests["P1"].update({"pass": False, "ci_lo": 1.0, "ci_hi": 5.0})
        assert p2.verdict(tests) == "inconclusive"

    def test_insufficient_seeds_takes_precedence(self) -> None:
        values = _values()
        values["wt_hebbian"] = {1: 90.0}
        values["rn_hebbian"] = {1: 0.0}
        tests = p2.family_tests(values)
        assert p2.verdict(tests) == "insufficient_seeds"

    def test_secondary_tests_never_change_the_verdict(self) -> None:
        # P2 reversed, P3 reversed, P4 reversed: the verdict still follows P1.
        values = _values(
            wt_hebbian_count=5.0,
            rn_hebbian_count=40.0,
            wt_frozen=2.0,
            rn_frozen=30.0,
        )
        tests = p2.family_tests(values)
        assert p2.verdict(tests) == "specific_wiring"
        notes = p2.annotate("specific_wiring", tests)
        assert notes["count_preserves_contrast"] is False
        assert notes["count_improves_wild_type"] is False
        assert notes["prior_differs"] is False
        assert "created by Hebbian alignment" in notes["reading"]

    def test_p4_annotation_when_the_prior_already_differs(self) -> None:
        tests = p2.family_tests(_values(wt_frozen=30.0, rn_frozen=5.0))
        notes = p2.annotate(p2.verdict(tests), tests)
        assert notes["prior_differs"] is True
        assert "already present in the untrained prior" in notes["reading"]


# --- descriptive -------------------------------------------------------------------------


class TestPriorSweep:
    def test_competent_fraction_and_distribution(self) -> None:
        values = _values()
        values["wt_frozen"] = {s: (50.0 if s <= 16 else 5.0) for s in _SWEEP}
        sweep = p2.prior_sweep(values)
        d = sweep["arms"]["wt_frozen"]
        assert d["n"] == 64
        assert d["competent_fraction"] == pytest.approx(0.25)
        assert d["min"] == 5.0
        assert d["max"] == 50.0
        assert d["sorted"] == sorted(d["sorted"])
        assert {(p["a"], p["b"]) for p in sweep["pairs"]} == {
            ("wt_frozen_count", "wt_frozen"),
            ("rn_frozen_count", "rn_frozen"),
        }
        assert all(p["descriptive"] for p in sweep["pairs"])

    def test_threshold_is_inclusive(self) -> None:
        d = p2.distribution({1: 20.0, 2: 19.9})
        assert d["competent_fraction"] == pytest.approx(0.5)

    def test_learning_gains_use_the_hebbian_seeds_and_own_floor(self) -> None:
        gains = p2.learning_gains(_values())
        assert set(gains) == set(p2.HEBBIAN_ARMS)
        assert gains["wt_hebbian"]["floor"] == "wt_frozen"
        assert gains["wt_hebbian"]["n"] == 16
        assert gains["wt_hebbian"]["mean"] == pytest.approx(32.0)
        assert gains["rn_hebbian_count"]["floor"] == "rn_frozen_count"

    def test_descriptive_pairs_exclude_the_family(self) -> None:
        rows = p2.descriptive_pairs(_values())
        pairs = {(r["a"], r["b"]) for r in rows}
        for pair in p2.FAMILY_PAIRS.values():
            assert pair not in pairs
            assert pair[::-1] not in pairs
        assert len(rows) == 28 - 4


class TestExtensionsAndOutput:
    def test_non_converged_hebbian_runs_need_an_extension(self) -> None:
        panel = _panel(_values())
        panel["wt_hebbian"][5] = _record(10.0, episodes=1000, converged=False)
        panel["wt_frozen"][5] = _record(10.0, episodes=600, converged=False)  # frozen: never
        panel["rn_hebbian"][2] = _record(10.0, episodes=1500, converged=False)  # already extended
        assert p2.extensions_needed(panel) == [{"arm": "wt_hebbian", "seed": 5, "episodes": 1000}]

    def test_analyse_and_csv_shape(self, tmp_path: Path) -> None:
        panel = _panel(_values())
        out = p2.analyse(panel, {})
        assert out["verdict"]["verdict"] == "specific_wiring"
        assert set(out["family"]) == set(p2.FAMILY)
        assert out["per_arm"]["wt_frozen"]["n"] == 64
        assert out["prior_sweep"]["arms"]["rn_frozen_count"]["n"] == 64
        p2.write_per_seed_csv(panel, tmp_path / "per-seed.csv")
        with (tmp_path / "per-seed.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4 * 16 + 4 * 64
        assert rows[0]["arm"] == "wt_frozen"
        assert set(rows[0]) == set(p2._CSV_FIELDS)
        p2.write_curves_csv(panel, tmp_path / "curves.csv")
        assert (tmp_path / "curves.csv").read_text().startswith("arm,seed,window_end,success\n")

    def test_main_reads_both_campaigns(self, tmp_path: Path) -> None:
        heb, sweep, experiments = (
            tmp_path / "heb" / "logs",
            tmp_path / "sweep" / "logs",
            tmp_path / "exp",
        )
        heb.mkdir(parents=True)
        sweep.mkdir(parents=True)
        for arm_stem in (
            "hebbian",
            "hebbian_rewired_null",
            "hebbian_countinit",
            "hebbian_rewired_null_countinit",
        ):
            for seed in _HEB:
                _experiment(experiments, f"{arm_stem}-{seed}", onset=10)
                good = "rewired" not in arm_stem
                _log(
                    heb / f"{_STEM}_{arm_stem}-seed{seed}.log",
                    ["SUCCESS" if good or seed % 4 == 0 else "FAILED"] * 40,
                    f"{arm_stem}-{seed}",
                )
        for arm_stem in (
            "frozen",
            "frozen_rewired_null",
            "frozen_countinit",
            "frozen_rewired_null_countinit",
        ):
            for seed in _SWEEP:
                _log(sweep / f"{_STEM}_{arm_stem}-seed{seed}.log", ["FAILED"] * 40)
        out = tmp_path / "panel2.json"
        code = p2.main(
            [
                "--campaign-dir",
                str(tmp_path / "heb"),
                "--sweep-dir",
                str(tmp_path / "sweep"),
                "--experiments-dir",
                str(experiments),
                "--out",
                str(out),
            ],
        )
        assert code == 0
        data = json.loads(out.read_text())
        assert data["verdict"]["verdict"] == "specific_wiring"
        assert data["family"]["P4"]["n"] == 64
        assert data["prior_sweep"]["arms"]["wt_frozen"]["competent_fraction"] == 0.0

    def test_main_rejects_an_out_of_range_seed(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        _log(logs / f"{_STEM}_hebbian-seed17.log", ["SUCCESS"] * 40)
        assert p2.main(["--campaign-dir", str(tmp_path)]) == 2
