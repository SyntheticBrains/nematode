"""A.6t's follow-up: the gap-only split on block V's thermal cell at target 35.

Covers the architecture-comparison-protocol requirement "A difficulty pin for a follow-up is
chosen by the gates alone, under a rule fixed before the pilot that decides it", and the configs,
constants, verdict maps and manifest the registration names.
"""

from __future__ import annotations

import functools
import json
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
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import generate_thermal_target_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_null_strength as tns  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIGS = _root / "configs" / "scenarios" / "thermal_foraging"
_PILOT = _root / "docs/experiments/logbooks/supporting/078-thermal-split/pilot.json"


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    """Load the whole simulation config as a run gets it, through the real loader."""
    return load_simulation_config(str(_CONFIGS / f"{stem}.yml")).model_dump()


@functools.cache
def _topology(stem: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one arm at seed 4 and return its chemical mask and gap buffer."""
    container = load_simulation_config(str(_CONFIGS / f"{stem}.yml")).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    top = ConnectomePPOBrain(
        config=container.config.model_copy(update={"seed": 4}),
        device=DeviceType.CPU,
    ).topology
    return top.m_chem.clone(), top.g_gap.clone()


class TestTheConfigs:
    def test_every_arm_is_a_committed_config(self) -> None:
        missing = [s for s in ts.LEVELS_BY_STEM if not (_CONFIGS / f"{s}.yml").is_file()]
        assert not missing, missing
        assert len(ts.LEVELS_BY_STEM) == 6

    @pytest.mark.parametrize("parent", gen.PARENTS)
    def test_each_differs_from_its_target_20_parent_in_the_target_alone(self, parent: str) -> None:
        child = _loaded(gen.child_stem(parent, ts.TARGET))
        expected = dict(_loaded(parent))
        env = {**expected["environment"]}
        env["foraging"] = {**env["foraging"], "target_foods_to_collect": ts.TARGET}
        expected["environment"] = env
        assert child == expected

    def test_the_gap_held_null_keeps_the_current_nulls_chemical_graph(self) -> None:
        full_mask, full_gap = _topology(ts.STEMS[ts.FULL]["rn_learn"])
        held_mask, held_gap = _topology(ts.STEMS[ts.GAP_HELD]["rn_learn"])
        _wild_mask, wild_gap = _topology(ts.STEMS[ts.FULL]["wt_learn"])
        assert torch.equal(full_mask, held_mask)
        assert torch.equal(held_gap, wild_gap)
        assert not torch.equal(full_gap, wild_gap)


class TestRegistration:
    def test_seeds_are_fresh(self) -> None:
        assert len(ts.SEEDS) == 128
        assert not set(ts.SEEDS) & set(tns.SEEDS)
        assert not set(ts.SEEDS) & {1001, 1002, 1003, 1004}

    def test_the_target_is_the_lowest_the_pilot_found_readable(self) -> None:
        pilot = json.loads(_PILOT.read_text())
        readable = sorted(int(t) for t, v in pilot["targets"].items() if v["preflight"]["launch"])
        assert readable[0] == ts.TARGET == pilot["chosen_target"]

    def test_the_minimum_scales_a1_by_the_wild_types_own_auc(self) -> None:
        pilot = json.loads(_PILOT.read_text())
        t20 = pilot["wild_type_auc_success_t20_a1_seeds_129_160"]
        t35 = pilot["targets"]["35"]["wild_type_auc_success_mean"]
        assert pytest.approx(t20, abs=1e-6) == ts.WILD_AUC_T20
        assert pytest.approx(t35, abs=1e-6) == ts.WILD_AUC_T35
        assert pytest.approx(2 / 3 * ts.A1_EFFECT_T20 * t35 / t20) == ts.MINIMUM
        assert pytest.approx(0.0239, abs=1e-4) == ts.MINIMUM


def _gates(*, passes: bool = True, saturated: bool = False) -> dict[str, Any]:
    return {
        "gate_passes": passes,
        "saturated": saturated,
        "wt": {"vs_floor": {"ci_lo": 0.1}},
        "rn": {"vs_floor": {"ci_lo": 0.1}},
    }


def _test(lo: float, hi: float, q: float = 0.01) -> dict[str, float]:
    return {"ci_lo": lo, "ci_hi": hi, "bh_q": q}


class TestReading:
    def test_a_saturated_level_stops_the_reading(self) -> None:
        gates = {ts.FULL: _gates(), ts.GAP_HELD: _gates(saturated=True)}
        got = ts.read_panel(gates, {}, {})
        assert got["verdict"] == "unreadable"

    def test_both_readings_map_through_their_tables(self) -> None:
        gates = {level: _gates() for level in ts.LEVELS}
        split = {"interaction_mean": -0.05, "test": _test(-0.07, -0.03)}
        lead = {"gap_mean": 0.001, "test": _test(-0.01, 0.012, q=0.6)}
        got = ts.read_panel(gates, split, lead)
        assert got["split"] == {"state": "move_null", "verdict": "gap_junctions"}
        assert got["lead"] == {"state": "no_move", "verdict": "no_lead"}

    def test_a_lead_that_remains(self) -> None:
        gates = {level: _gates() for level in ts.LEVELS}
        split = {"interaction_mean": 0.0, "test": _test(-0.005, 0.005, q=0.9)}
        lead = {"gap_mean": 0.05, "test": _test(0.03, 0.07)}
        got = ts.read_panel(gates, split, lead)
        assert got["lead"]["verdict"] == "lead_remains"


class TestManifest:
    def test_a_wild_type_run_serves_both_levels(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        for stem in ts.LEVELS_BY_STEM:
            (logs / f"{stem}-seed513.log").write_text("x")
        rows = (
            ts.build_manifest(tmp_path, tmp_path / "m.txt", seeds=(513,)).read_text().splitlines()
        )
        assert len(rows) == 4 * 2

    def test_a_foreign_log_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "other-seed513.log").write_text("x")
        with pytest.raises(ts.ThermalSplitError, match="does not have"):
            ts.build_manifest(tmp_path, tmp_path / "m.txt", seeds=(513,))
