"""Tests for the L4 pilot runner: derived grid configs and the campaign plan it hands over."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from types import ModuleType

_REPO = Path(__file__).resolve().parents[4].parent
_SCRIPT = _REPO / "scripts" / "campaigns" / "l4_panel_pilot.py"


@pytest.fixture(scope="module")
def pilot() -> ModuleType:
    """Load ``scripts/campaigns/l4_panel_pilot.py`` by path."""
    spec = importlib.util.spec_from_file_location("l4_panel_pilot", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flatten(data: object, prefix: str = "") -> dict[str, object]:
    if isinstance(data, dict):
        out: dict[str, object] = {}
        for key, value in data.items():
            out.update(_flatten(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): data}


class TestDerivedConfigs:
    @pytest.mark.parametrize("arm", ["wt_plastic", "rn_hebbian", "mlp_plastic"])
    def test_one_key_off_the_committed_arm(
        self,
        pilot: ModuleType,
        tmp_path: Path,
        arm: str,
    ) -> None:
        lp = pilot.l4_panel
        derived = pilot.derive_config(arm, 0.0003, tmp_path)
        parent = lp.CONFIG_DIR / f"{lp.STEM_OF[arm]}.yml"
        base = _flatten(yaml.safe_load(parent.read_text()))
        variant = _flatten(yaml.safe_load(derived.read_text()))
        touched = (set(variant) - set(base)) | {
            k for k in set(base) & set(variant) if base[k] != variant[k]
        }
        assert touched == {"brain.config.plasticity_rate"}
        assert set(base) - set(variant) == set()
        assert variant["brain.config.plasticity_rate"] == 0.0003
        assert derived.name == f"{lp.STEM_OF[arm]}__rate_0p0003.yml"

    def test_derived_config_loads(self, pilot: ModuleType, tmp_path: Path) -> None:
        from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrainConfig
        from quantumnematode.utils.config_loader import load_simulation_config

        derived = pilot.derive_config("wt_plastic", 0.003, tmp_path)
        config = load_simulation_config(str(derived))
        assert config.brain is not None
        brain_config = config.brain.config
        assert isinstance(brain_config, ConnectomePPOBrainConfig)
        assert brain_config.plasticity_rate == 0.003
        assert brain_config.learning_rule == "three_factor"


class TestPlan:
    def test_default_plan_is_rule_bearing_arms_times_grid_plus_frozen_once(
        self,
        pilot: ModuleType,
        tmp_path: Path,
    ) -> None:
        lp = pilot.l4_panel
        configs = pilot.plan_configs(tmp_path)
        assert len(configs) == len(lp.RULE_BEARING_ARMS) * len(lp.RATE_GRID) + len(lp.FROZEN_ARMS)
        assert len(lp.RULE_BEARING_ARMS) == 5
        derived = [c for c in configs if c.parent == tmp_path / "configs"]
        committed = [c for c in configs if c.parent == lp.CONFIG_DIR]
        assert len(derived) == 15
        assert sorted(c.name for c in committed) == sorted(
            f"{lp.STEM_OF[arm]}.yml" for arm in lp.FROZEN_ARMS
        )
        assert all(c.is_file() for c in configs)

    def test_only_and_rate_restrict_the_plan(self, pilot: ModuleType, tmp_path: Path) -> None:
        configs = pilot.plan_configs(tmp_path, arms=("wt_plastic",), rates=(0.001,))
        assert [c.name for c in configs] == [
            "connectomeppo_small_continuous2d_combined_klinotaxis_plastic__rate_0p001.yml",
        ]

    def test_campaign_argv_carries_seeds_budget_and_passthrough(
        self,
        pilot: ModuleType,
        tmp_path: Path,
    ) -> None:
        args = pilot.parse_arguments(["--out", str(tmp_path), "--dry-run"])
        assert args.seeds == "101-102"
        assert args.runs == pilot.l4_panel.PILOT_BUDGET
        argv = pilot.campaign_argv([tmp_path / "a.yml"], args)
        assert argv[0].endswith("run_campaign.py")
        assert argv[1:3] == ["--config", str(tmp_path / "a.yml")]
        assert "--dry-run" in argv
        assert argv[argv.index("--seeds") + 1] == "101-102"
        assert argv[argv.index("--runs") + 1] == "3000"
        assert argv[argv.index("--") + 1 :] == list(pilot.PASSTHROUGH)

    def test_main_hands_the_plan_to_the_campaign_runner(
        self,
        pilot: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            seen.append(command)
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(pilot.subprocess, "run", fake_run)
        code = pilot.main(
            ["--out", str(tmp_path), "--only", "wt_plastic", "--rate", "0.001", "--dry-run"],
        )
        assert code == 0
        assert len(seen) == 1
        command = seen[0]
        assert command[0] == sys.executable
        assert command.count("--config") == 1
        assert "--track-experiment" in command
