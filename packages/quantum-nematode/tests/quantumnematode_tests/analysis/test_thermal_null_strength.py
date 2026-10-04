"""A.6t: the null-strength control and its split on block V's thermal cell.

Covers the architecture-comparison-protocol requirement "A control repeated on another cell is sized
and judged on that cell's own effect": the panel's minimum is the thermal cell's committed effect,
never hard350's or this campaign's own; and the configs, stems, verdict maps and manifest the
registration names.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import null_strength_control as nsc  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_null_strength as tns  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIGS = _root / "configs" / "scenarios" / "thermal_foraging"
_SEED = 4


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    """Load the whole simulation config as a run gets it, through the real loader."""
    return load_simulation_config(str(_CONFIGS / f"{stem}.yml")).model_dump()


@functools.cache
def _topology(stem: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one arm at one seed and return its chemical mask and gap buffer."""
    container = load_simulation_config(str(_CONFIGS / f"{stem}.yml")).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    cfg = container.config.model_copy(update={"seed": _SEED})
    top = ConnectomePPOBrain(config=cfg, device=DeviceType.CPU).topology
    return top.m_chem.clone(), top.g_gap.clone()


class TestTheConfigs:
    def test_every_arm_is_a_committed_config(self) -> None:
        """Three levels, the wild-type arms shared, every stem on disk."""
        missing = [s for s in tns.LEVELS_BY_STEM if not (_CONFIGS / f"{s}.yml").is_file()]
        assert not missing, missing
        assert len(tns.LEVELS_BY_STEM) == 8
        assert len(tns.NEW_ARMS) == 4

    @pytest.mark.parametrize("stem", sorted(tns.NEW_ARMS))
    def test_each_new_arm_differs_from_its_parent_in_wiring_alone(self, stem: str) -> None:
        """The delta, re-checked through the real loader."""
        _half, parent = tns.NEW_ARMS[stem]
        child, expected = _loaded(stem), dict(_loaded(parent))
        assert child["brain"]["config"]["wiring"] == tns.NEW_WIRING[stem]
        assert expected["brain"]["config"]["wiring"] == "rewired_degree_preserving"
        expected = {**expected, "brain": {**expected["brain"]}}
        expected["brain"]["config"] = {
            **expected["brain"]["config"],
            "wiring": tns.NEW_WIRING[stem],
        }
        assert child == expected

    def test_the_wild_type_and_current_null_are_block_v_thermal_configs(self) -> None:
        """The reused arms are block V's thermal cell, target 20, unchanged."""
        assert tns.STEMS[tns.FULL]["wt_learn"].endswith("_thermal_klinotaxis_t20")
        assert _loaded(tns.STEMS[tns.FULL]["wt_learn"])["brain"]["config"]["wiring"] == "wild_type"

    def test_the_gap_held_null_has_the_current_nulls_chemical_graph(self) -> None:
        """Exactly paired: the same chemical mask, gap junctions at the wild type's."""
        full_mask, full_gap = _topology(tns.STEMS[tns.FULL]["rn_learn"])
        held_mask, held_gap = _topology(tns.STEMS[tns.GAP_HELD]["rn_learn"])
        _wild_mask, wild_gap = _topology(tns.STEMS[tns.FULL]["wt_learn"])
        assert torch.equal(full_mask, held_mask)
        assert torch.equal(held_gap, wild_gap)
        assert not torch.equal(full_gap, wild_gap)


class TestRegistration:
    def test_seeds_are_fresh_and_number_128(self) -> None:
        """Disjoint from A.6 and its split, which ran 305-384."""
        assert len(tns.SEEDS) == 128
        used = set(nsc.SEEDS_BY_HALF["ppo"]) | set(nsc.SEEDS_BY_HALF["reading"])
        assert not used & set(tns.SEEDS)

    def test_the_minimum_is_the_thermal_cells_own(self) -> None:
        """2/3 of A.1's thermal effect, not hard350's and not this campaign's."""
        assert pytest.approx(0.0829, abs=1e-4) == tns.REFERENCE_EFFECT
        assert pytest.approx(2 / 3 * 0.08289583333333334) == tns.MINIMUM
        assert nsc.REFERENCE_EFFECT["ppo"] != tns.REFERENCE_EFFECT


def _test(lo: float, hi: float) -> dict[str, float]:
    return {"ci_lo": lo, "ci_hi": hi, "bh_q": 0.01}


class TestVerdicts:
    def test_combined_maps_as_a6(self) -> None:
        assert tns.verdict("combined", "move_null", _test(0.01, 0.1)) == "gap_or_autapse"
        assert tns.verdict("combined", "no_move", _test(0.01, 0.1)) == "chemical"
        assert tns.verdict("combined", "no_move", _test(-0.01, 0.1)) == "no_gap_to_attribute"
        assert tns.verdict("combined", "below", _test(0.01, 0.1)) == "below_minimum"

    def test_split_maps_as_the_gap_only_split(self) -> None:
        assert tns.verdict("split", "move_null", _test(0.01, 0.1)) == "gap_junctions"
        assert tns.verdict("split", "below", _test(0.01, 0.1)) == "partial"
        assert tns.verdict("split", "no_move", _test(0.01, 0.1)) == "not_gap_junctions"
        assert tns.verdict("split", "move_wt", _test(0.01, 0.1)) == "opposite"

    def test_an_unreadable_level_stops_the_reading(self) -> None:
        gates = {
            level: {
                "gate_passes": level != tns.CHEMICAL,
                "saturated": False,
                "wt": {"vs_floor": {"ci_lo": 0.1}},
                "rn": {"vs_floor": {"ci_lo": 0.1}},
            }
            for level in tns.LEVELS
        }
        got = tns.read_panel(gates, {}, _test(0.01, 0.1))
        assert got["verdict"] == "unreadable"

    def test_the_split_share_is_described_only_when_defined(self) -> None:
        both = {"combined": {"interaction_mean": -0.04}, "split": {"interaction_mean": -0.03}}
        assert tns.split_share(both) == pytest.approx(0.75)
        zero = {"combined": {"interaction_mean": 0.0}, "split": {"interaction_mean": -0.03}}
        assert tns.split_share(zero) is None


class TestManifest:
    def test_a_wild_type_run_serves_every_level(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        for stem in tns.LEVELS_BY_STEM:
            (logs / f"{stem}-seed385.log").write_text("x")
        manifest = tns.build_manifest(tmp_path, tmp_path / "m.txt", seeds=(385,))
        rows = [line.split()[:3] for line in manifest.read_text().splitlines()]
        assert len(rows) == 4 * 3
        assert sum(1 for arm, _lv, _s in rows if arm == "wt_learn") == 3

    def test_a_foreign_log_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "something_else-seed385.log").write_text("x")
        with pytest.raises(tns.ThermalControlError, match="does not have"):
            tns.build_manifest(tmp_path, tmp_path / "m.txt", seeds=(385,))
