"""M.8's panel: its configs, its constants, its verdict map and its checks.

Covers the connectome-ppo-brain requirements "A gap-only rewired null" and "Plastic gap junctions
under PPO" through the configs that run them, and the registration's readings.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest
from quantumnematode.utils.config_loader import load_simulation_config

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_plastic_gap_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import plastic_gaps as pg  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]

_THERMAL = _root / "configs" / "scenarios" / "thermal_foraging"


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    return load_simulation_config(str(_THERMAL / f"{stem}.yml")).model_dump()


class TestConfigs:
    def test_every_arm_is_a_committed_config(self) -> None:
        for arms in pg.STEMS.values():
            for stem in arms.values():
                assert (_THERMAL / f"{stem}.yml").is_file(), stem

    @pytest.mark.parametrize("child", sorted(gen.CHILDREN))
    def test_each_new_arm_differs_from_its_parent_in_the_listed_keys_alone(
        self,
        child: str,
    ) -> None:
        parent, changes = gen.CHILDREN[child]
        got, expected = _loaded(child), dict(_loaded(parent))
        expected["brain"] = {**expected["brain"]}
        expected["brain"]["config"] = {**expected["brain"]["config"], **changes}
        assert got == expected

    def test_the_table_fits_the_gate_preflight(self) -> None:
        for arms in pg.STEMS.values():
            assert set(arms) == set(gp.ARMS)

    def test_one_frozen_floor_per_wiring_serves_both_levels(self) -> None:
        assert pg.STEMS[pg.FIXED]["wt_frozen"] == pg.STEMS[pg.PLASTIC]["wt_frozen"]
        assert pg.STEMS[pg.FIXED]["rn_frozen"] == pg.STEMS[pg.PLASTIC]["rn_frozen"]


class TestRegistration:
    def test_the_seeds_are_the_first_64_of_the_thermal_split(self) -> None:
        assert ts.SEEDS[:64] == pg.SEEDS

    def test_the_minimum_is_two_thirds_of_the_thermal_split(self) -> None:
        assert pytest.approx(2 / 3 * 0.08647916666666666) == pg.MINIMUM

    def test_the_pilot_seeds_are_disjoint(self) -> None:
        assert not set(pg.PILOT_SEEDS) & set(ts.SEEDS)

    def test_the_identity_runs_are_six_wild_type_runs(self) -> None:
        assert len(pg.IDENTITY_RUNS) == 6
        assert {stem for stem, _ in pg.IDENTITY_RUNS} == {
            pg.STEMS[pg.FIXED]["wt_learn"],
            pg.STEMS[pg.FIXED]["wt_frozen"],
        }


def _t(lo: float) -> dict[str, float]:
    return {"ci_lo": lo}


class TestVerdict:
    @pytest.mark.parametrize(
        ("interaction", "lead", "expected"),
        [
            ("move_null", "no_move", "strength"),
            ("move_null", "below", "strength"),
            ("no_move", "move_wt", "placement"),
            ("move_null", "move_wt", "partly_strength"),
            ("move_wt", "move_wt", "placement_amplified"),
            ("no_move", "below", "mixed"),
            ("below", "move_wt", "mixed"),
            ("unresolved", "move_wt", "unresolved"),
        ],
    )
    def test_the_map(self, interaction: str, lead: str, expected: str) -> None:
        assert pg.verdict(_t(0.01), interaction, lead) == expected

    def test_no_base_withholds_attribution(self) -> None:
        assert pg.verdict(_t(-0.01), "move_null", "no_move") == "no_gap_effect"

    def test_an_unresolved_base_leaves_the_verdict_unresolved(self) -> None:
        assert pg.verdict(_t(0.01), "no_move", "move_wt", "unresolved") == "unresolved"


class TestChecks:
    def test_missing_identity_runs_are_not_identical(self, tmp_path: Path) -> None:
        assert pg.identity(tmp_path)["all_identical"] is False

    @staticmethod
    def _write_twins(fixed_dir: Path, plastic_dir: Path) -> None:
        for seed in pg.PILOT_SEEDS:
            for level, where in ((pg.FIXED, fixed_dir), (pg.PLASTIC, plastic_dir)):
                for arm in ("wt_learn", "rn_learn"):
                    path = where / f"{pg.STEMS[level][arm]}-seed{seed}.log"
                    path.write_text("Run: 1 Status: SUCCESS\n")

    @staticmethod
    def _reader(moved_stems: set[str]) -> Any:
        """Fake exports: a run's multipliers moved iff its stem is listed."""
        import torch

        def read(log: Path) -> dict[str, Any]:
            stem = log.stem.rpartition("-seed")[0]
            gap = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
            moved = torch.tensor([[0.0, 0.3], [0.1, 0.0]])
            return {"g_gap": gap, "gap_log_multiplier": moved if stem in moved_stems else 0 * gap}

        return read

    def test_unmoved_multipliers_mean_inert_plasticity(self, tmp_path: Path) -> None:
        self._write_twins(tmp_path, tmp_path)
        reader = self._reader({pg.STEMS[pg.PLASTIC]["rn_learn"]})
        result = pg.plasticity([tmp_path], read_topology=reader)
        assert result["complete"] is True
        assert result["plasticity_acts"] is False
        assert all(result["per_seed"]["gap_only_null"].values())
        assert not any(result["per_seed"]["wild_type"].values())

    def test_twins_in_different_campaigns_are_paired(self, tmp_path: Path) -> None:
        reused, panel = tmp_path / "reused", tmp_path / "panel"
        reused.mkdir()
        panel.mkdir()
        self._write_twins(reused, panel)
        reader = self._reader({pg.STEMS[pg.PLASTIC][a] for a in ("wt_learn", "rn_learn")})
        result = pg.plasticity([panel, reused], read_topology=reader)
        assert result["complete"] is True
        assert result["plasticity_acts"] is True

    def test_a_missing_twin_or_export_reads_none(self, tmp_path: Path) -> None:
        self._write_twins(tmp_path, tmp_path)
        (tmp_path / f"{pg.STEMS[pg.FIXED]['wt_learn']}-seed{pg.PILOT_SEEDS[0]}.log").unlink()
        result = pg.plasticity([tmp_path], read_topology=lambda _log: None)
        assert result["complete"] is False
        assert set(result["per_seed"]["wild_type"].values()) == {None}
