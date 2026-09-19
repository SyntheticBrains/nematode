"""The L.1b arms: the pooled readout at 0.0001, one key from L.0/L.1's pooled arms.

* **One key from the pooled parent.** Each 0.0001 pooled arm differs from its committed parent in
  `plasticity_rate` alone.
* **One key from the wide arm at the same rate.** The pooled and wide 0.0001 arms differ in
  `readout_width` alone, so the 2x2 at 0.0001 is L.1's 2x2 with one key moved on every learning
  cell.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrainConfig
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_DIR = _REPO_ROOT / "configs" / "scenarios" / "foraging"
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop"
_RATE = 0.0001


def _cfg(name: str) -> ConnectomePPOBrainConfig:
    config = load_simulation_config(str(_DIR / f"{name}.yml")).brain
    assert config is not None
    assert isinstance(config.config, ConnectomePPOBrainConfig)
    return config.config


def _diff(a: str, b: str) -> set[str]:
    da, db = _cfg(a).model_dump(), _cfg(b).model_dump()
    return {k for k in set(da) | set(db) if da.get(k) != db.get(k)}


class TestOneKeyEach:
    @pytest.mark.parametrize("suffix", ["", "_rewired_null"])
    def test_one_key_from_the_pooled_parent(self, suffix: str) -> None:
        assert _diff(f"{_STEM}_readout_only{suffix}", f"{_STEM}_readout_only_r1e4{suffix}") == {
            "plasticity_rate",
        }

    @pytest.mark.parametrize("suffix", ["", "_rewired_null"])
    def test_one_key_from_the_wide_arm_at_the_same_rate(self, suffix: str) -> None:
        assert _diff(
            f"{_STEM}_readout_only_r1e4{suffix}",
            f"{_STEM}_readout_only_wide_r1e4{suffix}",
        ) == {"readout_width"}

    def test_the_values(self) -> None:
        for suffix in ("", "_rewired_null"):
            cfg = _cfg(f"{_STEM}_readout_only_r1e4{suffix}")
            assert cfg.plasticity_rate == pytest.approx(_RATE)
            assert cfg.readout_width == "pooled"
        assert _cfg(f"{_STEM}_readout_only_r1e4_rewired_null").wiring == "rewired_degree_preserving"
        assert _cfg(f"{_STEM}_readout_only_r1e4").wiring == "wild_type"
