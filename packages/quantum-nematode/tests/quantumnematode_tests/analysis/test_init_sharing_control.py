"""A.1's driver: the manifest mapping, the seed bands, and the instrument it must not touch.

The control's credibility rests on three things this module can check without running a campaign:
the stem mapping is complete and unambiguous, the seeds have never been used, and the committed
instrument is byte-identical to ``main``. A replication that edits its own instrument cannot
distinguish a changed reading from a changed world.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[5]
_ANALYSIS = _REPO / "scripts" / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import init_sharing_control as isc  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIGS = _REPO / "configs" / "scenarios"


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )


class TestTheStemMappingIsComplete:
    """Every arm of every cell at every mode, and every stem a real config."""

    def test_every_stem_names_a_config_that_exists(self) -> None:
        missing = [s for s in isc.ARM_BY_STEM if not list(_CONFIGS.rglob(f"{s}.yml"))]
        assert not missing, f"stems with no config: {missing}"

    def test_the_panel_is_the_full_cross_product(self) -> None:
        # 2 cells x 4 arms x 3 modes. A missing key silently drops one side of a paired test.
        assert len(isc.ARM_BY_STEM) == len(isc.CELLS) * 4 * len(isc.MODES)
        seen = set(isc.ARM_BY_STEM.values())
        assert len(seen) == len(isc.ARM_BY_STEM), "two stems map to the same (cell, arm, mode)"
        for cell in isc.CELLS:
            for arm in ("wt_ppo", "rn_ppo", "wt_frozen", "rn_frozen"):
                for mode in isc.MODES:
                    assert (cell, arm, mode) in seen, f"missing {cell}/{arm}/{mode}"

    def test_the_baseline_mode_uses_the_committed_configs_unchanged(self) -> None:
        # The edge_order level must be the block-V configs themselves, or the baseline is not the
        # baseline. Their stems carry no draw-mode suffix.
        baseline = [s for s, (_, _, m) in isc.ARM_BY_STEM.items() if m == isc.BASELINE_MODE]
        assert len(baseline) == 8
        for stem in baseline:
            assert not stem.endswith(("_densemask", "_fanin"))


class TestTheSeedsAreFresh:
    """Every campaign in the repository has used 1-96, and 101-104 is the pilot band."""

    def test_panel_seeds_are_unburnt(self) -> None:
        assert not set(isc.SEEDS) & isc.BURNT_SEEDS

    def test_pilot_seeds_are_unburnt_and_disjoint_from_the_panel(self) -> None:
        assert not set(isc.PILOT_SEEDS) & isc.BURNT_SEEDS
        assert not set(isc.PILOT_SEEDS) & set(isc.SEEDS)

    def test_the_panel_meets_the_registered_seed_count(self) -> None:
        # Raised from D15's floor of 16 after the pilot found no across-mode correlation: at 16 the
        # censored metric could not have detected a total dissolution of the effect.
        assert len(isc.SEEDS) == 32


class TestTheNewConfigsAreOneKeyDeltas:
    """`rewire_seed` stays unset, as two committed tests already assert for the block-V eight."""

    @pytest.mark.parametrize(
        "stem",
        [s for s, (_, _, m) in isc.ARM_BY_STEM.items() if m != isc.BASELINE_MODE],
    )
    def test_rewire_seed_is_unset(self, stem: str) -> None:
        # Unset means the rewiring RNG derives from the run seed, so each seed's wild-type and
        # rewired arms pair. Pinning it is a different experiment.
        path = next(iter(_CONFIGS.rglob(f"{stem}.yml")))
        body = "\n".join(
            line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
        )
        assert "rewire_seed:" not in body

    @pytest.mark.parametrize(
        "stem",
        [s for s, (_, _, m) in isc.ARM_BY_STEM.items() if m != isc.BASELINE_MODE],
    )
    def test_the_draw_mode_is_declared(self, stem: str) -> None:
        path = next(iter(_CONFIGS.rglob(f"{stem}.yml")))
        expected = isc.ARM_BY_STEM[stem][2]
        assert f"weight_draw: {expected}" in path.read_text()


class TestTheInstrumentIsUntouched:
    """Both committed harnesses must be byte-identical to main."""

    @pytest.mark.parametrize(
        "module",
        ["wiring_premise.py", "connectome_structure_efficiency.py"],
    )
    def test_the_scoring_modules_are_untouched_by_this_change(self, module: str) -> None:
        if _git(["rev-parse", "--verify", "--quiet", "origin/main"]).returncode != 0:
            pytest.skip("origin/main is not available in this checkout")
        diff = _git(["diff", "--quiet", "origin/main", "--", f"scripts/analysis/{module}"])
        assert diff.returncode in (0, 1), f"git could not compare {module}: {diff.stderr!r}"
        assert diff.returncode == 0, f"{module} differs from main; the instrument must not change"

    def test_the_driver_reuses_the_harness_constants(self) -> None:
        # Orientation and censoring come from the instrument, never from a second copy here.
        assert isc.CENSORED_METRIC in isc.wp.efficiency._METRICS
        assert isc.UNCENSORED_METRIC in isc.wp.efficiency._METRICS
        assert isc.wp.efficiency._METRICS[isc.CENSORED_METRIC] is False
        assert isc.wp.efficiency._METRICS[isc.UNCENSORED_METRIC] is True


class TestTheCensoringRuleIsFixedInAdvance:
    """The rule picks the metric; the data never picks it after the fact."""

    def test_comparable_censoring_keeps_the_registered_metric(self) -> None:
        rates = {m: {"wild_type": 0.95, "rewired_null": 0.93} for m in isc.MODES}
        choice = isc.choose_metric(rates)
        assert choice["censoring_comparable"] is True
        assert choice["primary_metric"] == isc.CENSORED_METRIC
        assert choice["reported_beside"] == isc.UNCENSORED_METRIC

    def test_divergent_censoring_moves_the_primary_to_the_uncensored_metric(self) -> None:
        rates = {m: {"wild_type": 0.98, "rewired_null": 0.55} for m in isc.MODES}
        choice = isc.choose_metric(rates)
        assert choice["censoring_comparable"] is False
        assert choice["primary_metric"] == isc.UNCENSORED_METRIC
        assert choice["reported_beside"] == isc.CENSORED_METRIC

    def test_both_metrics_are_reported_either_way(self) -> None:
        for rates in (
            {m: {"wild_type": 0.95, "rewired_null": 0.93} for m in isc.MODES},
            {m: {"wild_type": 0.98, "rewired_null": 0.55} for m in isc.MODES},
        ):
            choice = isc.choose_metric(rates)
            assert {choice["primary_metric"], choice["reported_beside"]} == {
                isc.CENSORED_METRIC,
                isc.UNCENSORED_METRIC,
            }


class TestTheManifestRefusesAPartialPanel:
    def test_an_incomplete_panel_raises(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.txt"
        manifest.write_text("thermal wt_ppo 129 a.log\n")
        with pytest.raises(ValueError, match="incomplete"):
            isc.require_complete(manifest, "edge_order", (129, 130))

    def test_an_unknown_config_stem_raises(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "not_a_panel_config-seed129.log").write_text("")
        with pytest.raises(ValueError, match="does not have"):
            isc.build_manifest(tmp_path, tmp_path / "m.txt", "edge_order", (129,))
