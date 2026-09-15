"""V.4's reading, and the four ways a replication can quietly stop being one.

* **The instrument must not change.** A replication varies the evidence and holds the reading
  fixed. If `wiring_premise` or `connectome_structure_efficiency` were edited here, "the instrument
  changed" would compete with "the effect is not there" -- not separable after the fact.
* **The vocabulary must not fork.** The harness emits `specific_wiring` / `below_min_effect` /
  `degree_statistics`; V.1 registered prose branches. Running both in parallel is how two records
  come to disagree about one run, so the prose is a mapping and the harness's names are reported.
* **Not every non-positive verdict is a failure to replicate.** `saturated` is what the klinotaxis
  cell returned in V.1's own pilot, `no_learning` means a gate failed, and a materially censored
  contrast is the case L.0 met on `hard350`. None is evidence against the original result.
* **A split must stay a split.** One cell replicating and the other not is evidence about the
  scope of block V's generalisation, which it stays only if the pooled reading is withheld.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import wiring_fresh_rewiring as fr  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

# The commits each panel ran at: the thermal four at V.1's, the hard350 four at V.3's.
_PANEL_COMMITS = {
    "431a4689": [
        f"connectomeppo_small_continuous2d_thermal_klinotaxis{s}.yml"
        for s in ("_t20", "_rewired_null_t20", "_frozen_t20", "_rewired_null_frozen_t20")
    ],
    "48ba778c": [
        f"connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350{s}.yml"
        for s in ("", "_rewired_null", "_frozen", "_rewired_null_frozen")
    ],
}


class TestTheSeedsAreFresh:
    def test_they_are_disjoint_from_both_prior_panels(self) -> None:
        # V.1 ran 1-64 on the thermal cell and V.3 ran 1-32 on hard_food, so V.3's rewirings are a
        # SUBSET of V.1's -- the two positives share their nulls, which is what this panel fixes.
        for cell, prior in fr.PRIOR_SEEDS.items():
            assert not set(fr.SEEDS) & set(prior), f"{cell} overlaps a prior panel"
        assert set(fr.PRIOR_SEEDS["hard_food"]) < set(fr.PRIOR_SEEDS["thermal"])

    def test_they_are_disjoint_from_the_pilot_seeds(self) -> None:
        # Every pilot in this programme uses 101-104.
        assert not set(fr.SEEDS) & set(range(101, 105))

    def test_there_are_thirty_two(self) -> None:
        assert len(fr.SEEDS) == 32


class TestTheRewiringIsCoupledToTheRunSeed:
    """The coupling this panel works around, and must not remove."""

    @pytest.mark.parametrize(
        "name",
        [n for names in _PANEL_COMMITS.values() for n in names if "rewired_null" in n],
    )
    def test_rewire_seed_is_unset(self, name: str) -> None:
        # With `rewire_seed` unset, each seed's rewiring derives from its run seed, which is what
        # pairs a rewired arm with its wild-type partner. Pinning it would decouple rewiring from
        # initialisation -- a different, unregistered question.
        path = next(_root.glob(f"configs/scenarios/*/{name}"), None)
        assert path is not None, f"{name} not found"
        keys = [
            line
            for line in path.read_text().splitlines()
            if line.split("#")[0].strip().startswith("rewire_seed")
        ]
        assert not keys, f"{name} pins rewire_seed: {keys}"

    @pytest.mark.parametrize(
        "name",
        [n for names in _PANEL_COMMITS.values() for n in names if "rewired_null" in n],
    )
    def test_the_null_is_degree_preserving(self, name: str) -> None:
        path = next(_root.glob(f"configs/scenarios/*/{name}"), None)
        assert path is not None, f"{name} not found"
        assert any(
            line.split("#")[0].strip() == "wiring: rewired_degree_preserving"
            for line in path.read_text().splitlines()
        ), f"{name} is not the degree-preserving null"


class TestTheCommittedHarnessesAreUnmodified:
    """The replication's central property, asserted rather than trusted."""

    @pytest.mark.parametrize(
        ("commit", "names"),
        list(_PANEL_COMMITS.items()),
        ids=list(_PANEL_COMMITS),
    )
    def test_each_panels_configs_are_unchanged_since_it_ran(
        self,
        commit: str,
        names: list[str],
    ) -> None:
        for name in names:
            path = next(_root.glob(f"configs/scenarios/*/{name}"), None)
            assert path is not None, f"{name} not found"
            rel = path.relative_to(_root)
            diff = subprocess.run(  # noqa: S603 — fixed argv, no shell
                ["git", "diff", "--quiet", commit, "--", str(rel)],  # noqa: S607
                cwd=_root,
                check=False,
                capture_output=True,
            )
            assert diff.returncode == 0, (
                f"{name} changed since {commit}; this would measure something else"
            )

    @pytest.mark.parametrize("module", ["wiring_premise.py", "connectome_structure_efficiency.py"])
    def test_the_scoring_modules_are_untouched_by_this_change(self, module: str) -> None:
        # Both must be byte-identical to main. A replication that edits its own instrument cannot
        # distinguish a changed reading from a changed world.
        diff = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "diff", "--quiet", "origin/main", "--", f"scripts/analysis/{module}"],  # noqa: S607
            cwd=_root,
            check=False,
            capture_output=True,
        )
        assert diff.returncode == 0, f"{module} differs from main; the instrument must not change"

    def test_the_driver_reuses_the_harness_constants(self) -> None:
        # Not re-declared here: the minimum, the crossing floor and the significance level all come
        # from the harness, so the two cannot drift apart.
        assert wp.MIN_EFFICIENCY_GAIN == 0.20
        assert wp.CROSSING_FLOOR == 0.8
        assert not hasattr(fr, "MIN_EFFICIENCY_GAIN")
        assert not hasattr(fr, "CROSSING_FLOOR")
        assert not hasattr(fr, "SIG_Q")


class TestTheVocabularyDoesNotFork:
    def test_every_prose_branch_maps_a_real_harness_verdict(self) -> None:
        for name in fr.PROSE_BRANCH:
            assert name in {"specific_wiring", "below_min_effect", "degree_statistics"}

    def test_the_harness_verdicts_are_all_accounted_for(self) -> None:
        # Anything the harness can return has either a prose branch or an explicit not-a-failure
        # reading; an unhandled verdict would fall through to "unrecognised".
        emitted = {
            "specific_wiring",
            "below_min_effect",
            "degree_statistics",
            "saturated",
            "no_learning",
            "insufficient_seeds",
        }
        assert emitted == set(fr.PROSE_BRANCH) | set(fr.NOT_A_FAILURE)

    @pytest.mark.parametrize(
        ("verdict", "prose", "failure"),
        [
            pytest.param("specific_wiring", "replicates", False, id="replicates"),
            pytest.param(
                "below_min_effect",
                "same direction, below the minimum",
                False,
                id="below",
            ),
            pytest.param("degree_statistics", "does not replicate", True, id="fails"),
        ],
    )
    def test_each_verdict_reads_as_its_registered_branch(
        self,
        verdict: str,
        prose: str,
        *,
        failure: bool,
    ) -> None:
        out = fr.branch("hard_food", {"verdict": verdict, "axis": "efficiency"}, None)
        assert out["prose_branch"] == prose
        assert out["is_replication_failure"] is failure

    def test_only_degree_statistics_withdraws_the_positive(self) -> None:
        out = fr.branch("thermal", {"verdict": "degree_statistics"}, None)
        assert "WITHDRAWN" in out["why"]


class TestNotEveryNonPositiveIsAFailure:
    @pytest.mark.parametrize("verdict", ["saturated", "no_learning", "insufficient_seeds"])
    def test_it_is_not_recorded_as_a_replication_failure(self, verdict: str) -> None:
        out = fr.branch("thermal", {"verdict": verdict}, None)
        assert out["is_replication_failure"] is False
        assert out["prose_branch"] is None
        assert out["why"]

    def test_saturation_names_v1s_own_pilot(self) -> None:
        out = fr.branch("thermal", {"verdict": "saturated"}, None)
        assert "klinotaxis" in out["why"]

    def test_censoring_is_computed_through_the_harness(self) -> None:
        # The harness computes this for a printed flag and does not store it, so reading a
        # `materially_censored` key would silently always be False.
        report = {
            "horizon_episodes": 3000,
            "per_seed": {
                wp.efficiency._WILD: {
                    str(s): {"episodes_to_30pct_success": 3000 if s < 10 else 500}
                    for s in range(32)
                },
                wp.efficiency._REWIRED: {
                    str(s): {"episodes_to_30pct_success": 400} for s in range(32)
                },
            },
        }
        out = fr.branch("hard_food", {"verdict": "degree_statistics"}, report)
        assert out["materially_censored"] is True
        assert out["crossing_rates"][wp.efficiency._WILD] < wp.CROSSING_FLOOR

    def test_a_censored_non_failure_says_it_is_not_evidence_against(self) -> None:
        report = {
            "horizon_episodes": 3000,
            "per_seed": {
                arm: {str(s): {"episodes_to_30pct_success": 3000} for s in range(32)}
                for arm in (wp.efficiency._WILD, wp.efficiency._REWIRED)
            },
        }
        out = fr.branch("thermal", {"verdict": "saturated"}, report)
        assert "not evidence against the original result" in out["why"]


class TestASplitStaysASplit:
    @staticmethod
    def _analysed(thermal: str, hard: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        def fake(_cells: object, out: dict, _manifest: object = None) -> None:
            out["verdicts"] = {
                "thermal": {"verdict": thermal, "axis": "efficiency"},
                "hard_food": {"verdict": hard, "axis": "efficiency"},
            }

        monkeypatch.setattr(wp, "load", lambda _m: {})
        monkeypatch.setattr(wp, "analyse", fake)
        return fr.analyse(Path("unused.txt"))

    def test_one_replicating_one_failing_is_a_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = self._analysed("specific_wiring", "degree_statistics", monkeypatch)
        assert out["split"] is True
        assert out["pooled_reading_withheld"] is True
        assert "SCOPE" in out["split_note"]
        assert out["replicating_cells"] == ["thermal"]
        assert out["failing_cells"] == ["hard_food"]

    def test_both_replicating_is_not_a_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = self._analysed("specific_wiring", "specific_wiring", monkeypatch)
        assert out["split"] is False
        assert out["split_note"] is None
        assert out["failing_cells"] == []

    def test_both_failing_is_not_a_split_either(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two failures agree; there is nothing to withhold.
        out = self._analysed("degree_statistics", "degree_statistics", monkeypatch)
        assert out["split"] is False
        assert len(out["failing_cells"]) == 2

    def test_a_saturated_cell_does_not_make_a_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Saturation is not a failure, so pairing it with a replication is not a disagreement.
        out = self._analysed("specific_wiring", "saturated", monkeypatch)
        assert out["split"] is False
        assert out["failing_cells"] == []


class TestTheManifestAndCompleteness:
    @staticmethod
    def _logs(tmp_path: Path, seeds: tuple[int, ...], stems: list[str]) -> Path:
        logs = tmp_path / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            for s in seeds:
                (logs / f"{stem}-seed{s}.log").write_text("")
        return tmp_path

    def test_it_keys_cells_and_arms_the_harness_knows(self, tmp_path: Path) -> None:
        campaign = self._logs(tmp_path, (65,), list(fr.ARM_BY_STEM))
        manifest = fr.build_manifest(campaign, tmp_path / "m.txt", seeds=(65,))
        rows = [line.split() for line in manifest.read_text().splitlines()]
        assert {r[0] for r in rows} <= set(wp.CELLS)
        assert {r[1] for r in rows} <= set(wp.ARMS)
        assert len(rows) == 8

    def test_an_unrecognised_log_raises_rather_than_being_skipped(self, tmp_path: Path) -> None:
        # A silently dropped arm removes one side of a paired test.
        campaign = self._logs(tmp_path, (65,), ["some_other_config"])
        with pytest.raises(ValueError, match="does not have"):
            fr.build_manifest(campaign, tmp_path / "m.txt", seeds=(65,))

    def test_seeds_outside_the_panel_are_left_out(self, tmp_path: Path) -> None:
        campaign = self._logs(tmp_path, (65, 101), list(fr.ARM_BY_STEM))
        manifest = fr.build_manifest(campaign, tmp_path / "m.txt", seeds=(65,))
        assert {int(line.split()[2]) for line in manifest.read_text().splitlines()} == {65}

    def test_an_incomplete_panel_refuses_a_branch(self, tmp_path: Path) -> None:
        campaign = self._logs(tmp_path, (65,), list(fr.ARM_BY_STEM))
        manifest = fr.build_manifest(campaign, tmp_path / "m.txt", seeds=(65, 66))
        with pytest.raises(ValueError, match="panel is incomplete"):
            fr.require_complete(manifest, seeds=(65, 66))


class TestTheComparatorsAndPower:
    def test_both_committed_figures_are_carried(self) -> None:
        assert fr.COMPARATORS["thermal"]["gain_fraction"] == pytest.approx(0.354)
        assert fr.COMPARATORS["hard_food"]["gain_fraction"] == pytest.approx(0.235)

    def test_v1s_per_panel_spread_travels_with_it(self) -> None:
        # Its pooled +35.4% came from 64 seeds and its own panels spread 15 points, so a near-miss
        # on this cell is read against that rather than as a clean failure.
        gains = fr.COMPARATORS["thermal"]["per_panel_gains"]
        assert gains is not None
        assert max(gains) - min(gains) > 0.14

    def test_the_power_note_disclaims_the_registered_procedure(self) -> None:
        power = fr._power(32)
        assert power["k_needed"] == 22
        assert "planning figures" in power["note"]
