"""Sign-grounding harness: registry, seed ranges, the family, the substrate gate, the verdict."""

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

import l4_atlas_signs as gs  # noqa: E402  # pyright: ignore[reportMissingImports]

_SWEEP = tuple(range(1, 65))
_HEB = tuple(range(1, 17))


def _arm(base: float, seeds: tuple[int, ...]) -> dict[int, float]:
    return {s: base + 0.5 * ((s % 3) - 1) for s in seeds}


def _values(**over: float) -> dict[str, dict[int, float]]:
    base = {
        "wt_frozen_atlas": 12.0,
        "rn_frozen_atlas": 9.0,
        "wt_hebbian_atlas": 45.0,
        "rn_hebbian_atlas": 20.0,
        "wt_hebbian_dale": 50.0,
        "rn_hebbian_dale": 22.0,
    }
    base.update(over)
    return {arm: _arm(v, gs.SEEDS_OF[arm]) for arm, v in base.items()}


def _random_signs(wt_frozen: float = 8.0, **over: float) -> dict[str, dict[int, float]]:
    base = {"wt_frozen": wt_frozen, "rn_frozen": 8.0, "wt_hebbian": 30.0, "rn_hebbian": 18.0}
    base.update(over)
    seeds = {"wt_frozen": _SWEEP, "rn_frozen": _SWEEP, "wt_hebbian": _HEB, "rn_hebbian": _HEB}
    return {arm: _arm(v, seeds[arm]) for arm, v in base.items()}


def _record(success: float, episodes: int = 600, *, converged: bool | None = True) -> gs.SeedRecord:
    return gs.SeedRecord(
        success=success,
        foods=5.0,
        episodes=episodes,
        converged=converged,
        onset=50 if converged else None,
        evasion_rate=None,
        temp_comfort=None,
        curve=[success],
    )


def _panel(values: dict[str, dict[int, float]]) -> dict[str, dict[int, gs.SeedRecord]]:
    return {
        arm: {s: _record(v, episodes=gs.BUDGETS[arm]) for s, v in seeds.items()}
        for arm, seeds in values.items()
    }


class TestRegistry:
    def test_six_grounded_arms_whose_configs_exist(self) -> None:
        assert len(gs.ARMS) == 6
        assert set(gs.ARM_KEYS) == set(gs.FROZEN_ARMS) | set(gs.HEBBIAN_ARMS)
        configs = gs.REPO / "configs" / "scenarios" / "foraging_predator_thermal"
        for stem in gs.ARMS:
            assert (configs / f"{stem}.yml").is_file(), stem

    def test_seed_ranges_and_budgets(self) -> None:
        for arm in gs.FROZEN_ARMS:
            assert gs.SEEDS_OF[arm] == _SWEEP
            assert gs.BUDGETS[arm] == 600
        for arm in gs.HEBBIAN_ARMS:
            assert gs.SEEDS_OF[arm] == _HEB
            assert gs.BUDGETS[arm] == 1000

    def test_every_arm_has_a_random_sign_comparator(self) -> None:
        assert set(gs.RANDOM_OF) == set(gs.ARM_KEYS)
        committed = gs.read_panel2()
        for comparator in set(gs.RANDOM_OF.values()):
            assert comparator in committed

    def test_out_of_range_seed_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the arm's registered seeds 1-16"):
            gs.group_panel([("wt_hebbian_atlas", 17, _record(1.0, episodes=1000))])
        with pytest.raises(ValueError, match="outside the arm's registered seeds 1-64"):
            gs.group_panel([("wt_frozen_atlas", 65, _record(1.0))])

    def test_unregistered_run_length_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="neither the budget nor its registered extension"):
            gs.group_panel([("wt_frozen_atlas", 1, _record(1.0, episodes=2000))])

    def test_the_extension_wins_a_duplicate_either_order(self) -> None:
        base = _record(1.0, episodes=1000, converged=False)
        extended = _record(9.0, episodes=1500)
        for order in ([base, extended], [extended, base]):
            panel = gs.group_panel([("wt_hebbian_atlas", 4, r) for r in order])
            assert panel["wt_hebbian_atlas"][4].episodes == 1500


class TestFamily:
    def test_four_tests_with_their_directions(self) -> None:
        tests = gs.family_tests(_values(), _random_signs())
        assert set(tests) == set(gs.FAMILY)
        assert tests["G1"]["mean_delta"] == pytest.approx(4.0)
        assert tests["G2"]["mean_delta"] == pytest.approx(25.0)
        assert tests["G3"]["mean_delta"] == pytest.approx(28.0)
        assert tests["G4"]["mean_delta"] == pytest.approx(5.0)
        assert tests["G1"]["n"] == 64
        assert tests["G2"]["n"] == 16
        assert all(t["complete"] for t in tests.values())

    def test_g1_compares_against_the_committed_table(self) -> None:
        """G1's comparator is panel 2's value, never a re-run arm."""
        tests = gs.family_tests(_values(), _random_signs(wt_frozen=20.0))
        assert tests["G1"]["mean_delta"] == pytest.approx(-8.0)
        assert tests["G1"]["reverse"]


class TestSubstrateGate:
    def test_intact_substrate(self) -> None:
        substrate = gs.substrate_broken(_values(), _random_signs())
        assert substrate["broken"] is False

    def test_a_collapsed_prior_is_named(self) -> None:
        # Grounded frozen arms competent nowhere; the committed ones competent everywhere.
        values = _values(wt_frozen_atlas=1.0, rn_frozen_atlas=1.0)
        substrate = gs.substrate_broken(values, _random_signs(wt_frozen=50.0, rn_frozen=50.0))
        assert substrate["broken"] is True
        assert substrate["arms"]["wt_frozen_atlas"]["grounded_competent_fraction"] == 0.0
        assert (
            gs.verdict(
                gs.family_tests(values, _random_signs(wt_frozen=50.0, rn_frozen=50.0)),
                substrate,
            )
            == "substrate_fail"
        )

    def test_a_drop_that_stays_above_the_threshold_is_not_a_break(self) -> None:
        """A real drop that is not a collapse.

        Grounded competent on half the seeds against committed on three quarters.
        """
        values = _values()
        values["wt_frozen_atlas"] = {s: (50.0 if s % 2 == 0 else 1.0) for s in _SWEEP}
        values["rn_frozen_atlas"] = dict(values["wt_frozen_atlas"])
        random_signs = _random_signs()
        for arm in ("wt_frozen", "rn_frozen"):
            random_signs[arm] = {s: (50.0 if s % 4 else 1.0) for s in _SWEEP}
        substrate = gs.substrate_broken(values, random_signs)
        assert substrate["arms"]["wt_frozen_atlas"]["grounded_competent_fraction"] == 0.5
        assert substrate["arms"]["wt_frozen_atlas"]["random_competent_fraction"] == 0.75
        assert substrate["broken"] is False


class TestVerdict:
    def test_specific_wiring(self) -> None:
        tests = gs.family_tests(_values(), _random_signs())
        assert (
            gs.verdict(tests, gs.substrate_broken(_values(), _random_signs())) == "specific_wiring"
        )

    def test_substrate_fail_outranks_the_wiring_verdicts(self) -> None:
        values = _values(wt_frozen_atlas=0.0, rn_frozen_atlas=0.0)
        random_signs = _random_signs(wt_frozen=60.0, rn_frozen=60.0)
        tests = gs.family_tests(values, random_signs)
        assert tests["G2"]["pass"] is True  # the wiring contrast would otherwise pass
        assert gs.verdict(tests, gs.substrate_broken(values, random_signs)) == "substrate_fail"

    def test_rewired_beats_wild_type(self) -> None:
        values = _values(wt_hebbian_atlas=20.0, rn_hebbian_atlas=50.0)
        tests = gs.family_tests(values, _random_signs())
        assert (
            gs.verdict(tests, gs.substrate_broken(values, _random_signs()))
            == "rewired_beats_wild_type"
        )

    def test_degree_statistics(self) -> None:
        values = _values(rn_hebbian_atlas=45.0)
        tests = gs.family_tests(values, _random_signs())
        assert (
            gs.verdict(tests, gs.substrate_broken(values, _random_signs())) == "degree_statistics"
        )

    def test_inconclusive(self) -> None:
        tests = gs.family_tests(_values(), _random_signs())
        tests["G2"].update({"pass": False, "ci_lo": 1.0, "ci_hi": 5.0})
        assert gs.verdict(tests, gs.substrate_broken(_values(), _random_signs())) == "inconclusive"

    def test_insufficient_seeds_outranks_everything(self) -> None:
        values = _values()
        values["wt_hebbian_atlas"] = {1: 90.0}
        values["rn_hebbian_atlas"] = {1: 0.0}
        tests = gs.family_tests(values, _random_signs())
        assert (
            gs.verdict(tests, gs.substrate_broken(values, _random_signs())) == "insufficient_seeds"
        )

    def test_annotations_never_change_the_verdict(self) -> None:
        # G3 and G4 reverse, and G1 reverses on a prior that is still competent either way,
        # so the substrate gate stays quiet and only G2 decides.
        values = _values(
            wt_hebbian_dale=10.0,
            rn_hebbian_dale=40.0,
            wt_frozen_atlas=30.0,
            rn_frozen_atlas=30.0,
        )
        random_signs = _random_signs(wt_frozen=40.0, rn_frozen=40.0)
        tests = gs.family_tests(values, random_signs)
        assert gs.verdict(tests, gs.substrate_broken(values, random_signs)) == "specific_wiring"
        notes = gs.annotate(tests)
        assert notes["prior_changed"] is False
        assert notes["prior_worsened"] is True
        assert notes["contrast_holds_under_dale"] is False
        assert notes["enforcement_helps"] is False


class TestDescriptiveAndOutput:
    def test_every_arm_is_compared_against_its_committed_counterpart(self) -> None:
        rows = gs.against_random(_values(), _random_signs())
        assert len(rows) == len(gs.ARM_KEYS)
        assert all(r["descriptive"] for r in rows)
        assert {r["a"] for r in rows} == set(gs.ARM_KEYS)
        dale = next(r for r in rows if r["a"] == "wt_hebbian_dale")
        assert "wt_hebbian" in dale["b"]

    def test_descriptive_pairs_exclude_the_family(self) -> None:
        rows = gs.descriptive_pairs(_values())
        pairs = {(r["a"], r["b"]) for r in rows}
        for family_pair in (
            ("wt_hebbian_atlas", "rn_hebbian_atlas"),
            ("wt_hebbian_dale", "rn_hebbian_dale"),
            ("wt_hebbian_dale", "wt_hebbian_atlas"),
        ):
            assert family_pair not in pairs
            assert family_pair[::-1] not in pairs

    def test_extensions_use_each_arm_budget(self) -> None:
        panel = _panel(_values())
        panel["wt_frozen_atlas"][3] = _record(1.0, episodes=600, converged=False)
        panel["wt_hebbian_atlas"][2] = _record(1.0, episodes=1000, converged=False)
        rows = gs.extensions_needed(panel)
        assert {(r["arm"], r["seed"], r["extend_to"]) for r in rows} == {
            ("wt_frozen_atlas", 3, 900),
            ("wt_hebbian_atlas", 2, 1500),
        }

    def test_analyse_and_csv_shape(self, tmp_path: Path) -> None:
        out = gs.analyse(_panel(_values()), _random_signs(), {})
        assert out["verdict"]["verdict"] == "specific_wiring"
        assert set(out["family"]) == set(gs.FAMILY)
        assert out["substrate"]["broken"] is False
        assert out["per_arm"]["wt_frozen_atlas"]["n"] == 64
        gs.write_per_seed_csv(_panel(_values()), tmp_path / "per-seed.csv")
        with (tmp_path / "per-seed.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2 * 64 + 4 * 16

    def test_main_end_to_end(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        for stem, arm in gs.ARMS.items():
            good = arm.startswith("wt_")
            for seed in gs.SEEDS_OF[arm]:
                statuses = ["SUCCESS" if good else "FAILED"] * gs.BUDGETS[arm]
                (logs / f"{stem}-seed{seed}.log").write_text(
                    "\n".join(
                        f"Run: {i}   Status: {s:<7} Reason: x Steps: 10    "
                        f"Eaten: {10 if s == 'SUCCESS' else 1}/10  "
                        for i, s in enumerate(statuses, 1)
                    ),
                )
        out = tmp_path / "panel.json"
        assert gs.main(["--campaign-dir", str(tmp_path), "--out", str(out)]) == 0
        data = json.loads(out.read_text())
        assert set(data["family"]) == set(gs.FAMILY)
        # The fixture's rewired arms clear nothing, so its grounded rewired prior collapses
        # against panel 2's committed one: the gate fires, which is the behaviour under test.
        assert data["substrate"]["arms"]["rn_frozen_atlas"]["broken"] is True
        assert data["verdict"]["verdict"] == "substrate_fail"

    def test_main_rejects_an_out_of_range_seed(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        stem = next(s for s, a in gs.ARMS.items() if a == "wt_hebbian_atlas")
        (logs / f"{stem}-seed40.log").write_text(
            "\n".join(
                f"Run: {i}   Status: SUCCESS Reason: x Steps: 10    Eaten: 10/10  "
                for i in range(1, 1001)
            ),
        )
        assert gs.main(["--campaign-dir", str(tmp_path)]) == 2


class TestSignFlipTelemetry:
    """What each run did to its grounded sign structure, read from its own endpoint."""

    @staticmethod
    def _endpoint(tmp_path: Path, weights: list[float], signs: list[int]) -> tuple[Path, Path]:
        """Write a log naming an experiment whose export holds a topology with these values."""
        import torch

        experiments = tmp_path / "experiments"
        exports = tmp_path / "exports" / "e1" / "weights"
        exports.mkdir(parents=True)
        torch.save(
            {
                "topology": {
                    "w_chem": torch.tensor(weights, dtype=torch.float32),
                    "chem_sign": torch.tensor(signs, dtype=torch.int8),
                },
            },
            exports / "final.pt",
        )
        (experiments / "e1").mkdir(parents=True)
        (experiments / "e1" / "e1.json").write_text(
            json.dumps(
                {
                    "exports_path": str(exports.parent.relative_to(gs.REPO))
                    if exports.is_relative_to(gs.REPO)
                    else str(exports.parent),
                },
            ),
        )
        log = tmp_path / "run.log"
        log.write_text(
            "  Experiment ID: e1\nRun: 1   Status: SUCCESS Reason: x Steps: 1    Eaten: 10/10  ",
        )
        return log, experiments

    def test_violations_and_silences_are_counted_separately(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(gs, "REPO", tmp_path)
        # four grounded synapses: one kept, one violated, one silenced, one kept; one ungrounded.
        log, experiments = self._endpoint(
            tmp_path,
            weights=[0.4, -0.3, 0.0, -0.5, 0.9],
            signs=[1, 1, 1, -1, 0],
        )
        measured = gs.run_sign_flips(log, experiments)
        assert measured is not None
        assert measured["violated"] == pytest.approx(0.25)
        assert measured["silenced"] == pytest.approx(0.25)

    def test_a_missing_endpoint_reads_as_none(self, tmp_path: Path) -> None:
        log = tmp_path / "run.log"
        log.write_text("Run: 1   Status: SUCCESS Reason: x Steps: 1    Eaten: 10/10  ")
        assert gs.run_sign_flips(log, tmp_path / "experiments") is None

    def test_per_arm_summary_reports_both_and_tolerates_gaps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(gs, "REPO", tmp_path)
        log, experiments = self._endpoint(tmp_path, weights=[0.4, -0.3], signs=[1, 1])
        panel = _panel(_values())
        logs = {("wt_hebbian_atlas", 1): log}  # one run readable, the rest absent
        summary = gs.sign_flips(panel, logs, experiments)
        assert summary["wt_hebbian_atlas"]["n_read"] == 1
        assert summary["wt_hebbian_atlas"]["violated_mean"] == pytest.approx(0.5)
        assert summary["wt_frozen_atlas"]["n_read"] == 0
        assert summary["wt_frozen_atlas"]["violated_mean"] is None
