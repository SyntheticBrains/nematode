"""G.0-A's selection: the pilot's arms, the calibrated ohmic scale, and every branch of the rule.

Covers the selection rule of `add-ohmic-gap-coupling` (design Decision C, as amended by the probe's
calibration in Decision D): readability, eligibility, the shown-slower test against each cell's
margin, and the four outcomes; the rule never reads the wiring gap.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import gap_substrate_selection as gs  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_gap_substrate_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]

_PROBE = _root / "docs/experiments/logbooks/supporting/097-gap-substrate/probe.json"


class TestArms:
    def test_352_runs_on_16_disjoint_seeds(self) -> None:
        assert sum(len(s) for s in gs.runs().values()) == 352
        assert tuple(range(9871, 9887)) == gs.SEEDS
        assert not set(gs.SEEDS) & set(gs.SMOKE_SEEDS)

    def test_every_level_has_the_preflight_s_four_arms(self) -> None:
        for cell in gs.CELLS:
            for arms in gs.levels(cell).values():
                assert set(arms) == set(gp.ARMS)

    def test_the_speed_level_compares_the_two_wild_types(self) -> None:
        for cell in gs.CELLS:
            speed = gs.levels(cell)["speed"]
            assert "_g0a_ohmic_wt" in speed["wt_learn"]
            assert "_g0a_log_wt" in speed["rn_learn"]

    def test_the_ohmic_configs_carry_the_committed_calibration(self) -> None:
        """The scale in every ohmic config is the one the probe chose before any run."""
        chosen = json.loads(_PROBE.read_text())["chosen_scale"]
        assert chosen == gen.OHMIC_SCALE
        for stem, (_, _, changes) in gen.CHILDREN.items():
            if "_g0a_ohmic_" in stem:
                assert changes["gap_weight_scale"] == chosen


def _test(lo: float, hi: float) -> dict[str, float]:
    return {"ci_lo": lo, "ci_hi": hi}


class TestRule:
    @pytest.mark.parametrize(
        ("lo", "hi", "margin", "slower"),
        [
            (-0.10, -0.03, 0.0239, True),
            (-0.10, -0.02, 0.0239, False),
            (-0.05, 0.02, 0.0143, False),
            (-0.20, -0.0144, 0.0143, True),
        ],
    )
    def test_shown_slower_needs_the_whole_interval_below_the_margin(
        self,
        lo: float,
        hi: float,
        margin: float,
        slower: bool,  # noqa: FBT001
    ) -> None:
        assert gs.shown_slower(_test(lo, hi), margin) is slower

    def test_ohmic_when_eligible_and_not_shown_slower(self) -> None:
        both = {"thermal": True, "hard350": True}
        assert gs.decide(
            both,
            {"ohmic": both, "log": both},
            {"thermal": False, "hard350": False},
        ) == ("ohmic")

    def test_log_when_ohmic_is_shown_slower_on_either_cell(self) -> None:
        both = {"thermal": True, "hard350": True}
        slower = {"thermal": False, "hard350": True}
        assert gs.decide(both, {"ohmic": both, "log": both}, slower) == "log"

    def test_log_when_ohmic_is_ineligible(self) -> None:
        both = {"thermal": True, "hard350": True}
        ohmic = {"thermal": True, "hard350": False}
        assert gs.decide(
            both,
            {"ohmic": ohmic, "log": both},
            {"thermal": False, "hard350": False},
        ) == ("log")

    def test_neither_when_no_model_is_eligible_on_both(self) -> None:
        both = {"thermal": True, "hard350": True}
        one = {"thermal": False, "hard350": True}
        assert gs.decide(
            both,
            {"ohmic": one, "log": one},
            {"thermal": False, "hard350": False},
        ) == ("neither")

    def test_unreadable_when_the_mlp_fails_a_cell(self) -> None:
        both = {"thermal": True, "hard350": True}
        assert gs.decide(
            {"thermal": True, "hard350": False},
            {"ohmic": both, "log": both},
            {"thermal": False, "hard350": False},
        ) == ("unreadable")


class TestManifest:
    def _write(
        self,
        where: Path,
        cell: str,
        *,
        skip: str | None = None,
        fail: str | None = None,
        mislabel: str | None = None,
    ) -> None:
        for stem in set(gen.ARMS[cell].values()):
            for seed in gs.SEEDS:
                name = f"{stem}-seed{seed}"
                if name == skip:
                    continue
                ran = "other.yml" if name == mislabel else f"{stem}.yml"
                (where / f"{name}.log").write_text(f"  Config: /session/{ran}\n")
                (where / f"{name}.exit").write_text("1\n" if name == fail else "0\n")

    def test_a_complete_cell_passes(self, tmp_path: Path) -> None:
        self._write(tmp_path, "thermal")
        manifest = gs.build_manifest([tmp_path], tmp_path / "m.txt", "thermal")
        gs.require_complete(manifest, "thermal")

    def test_a_missing_or_failed_run_is_refused(self, tmp_path: Path) -> None:
        mlp = gen.ARMS["thermal"]["mlp_learn"]
        self._write(tmp_path, "thermal", fail=f"{mlp}-seed9871")
        manifest = gs.build_manifest([tmp_path], tmp_path / "m.txt", "thermal")
        with pytest.raises(gs.SelectionError, match="9871"):
            gs.require_complete(manifest, "thermal")

    def test_a_log_recording_another_config_is_refused(self, tmp_path: Path) -> None:
        mlp = gen.ARMS["thermal"]["mlp_learn"]
        self._write(tmp_path, "thermal", mislabel=f"{mlp}-seed9871")
        with pytest.raises(gs.SelectionError, match=r"other\.yml"):
            gs.build_manifest([tmp_path], tmp_path / "m.txt", "thermal")

    def test_a_log_recording_no_config_is_refused(self, tmp_path: Path) -> None:
        self._write(tmp_path, "thermal")
        mlp = gen.ARMS["thermal"]["mlp_learn"]
        (tmp_path / f"{mlp}-seed9871.log").write_text("")
        with pytest.raises(gs.SelectionError, match="None"):
            gs.build_manifest([tmp_path], tmp_path / "m.txt", "thermal")
