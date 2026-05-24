"""Regression: archived configs that name legacy nociception_* modules still load.

The biology-driven predator sensor modules
(``predator_mechanosensation_*`` / ``predator_chemosensation_*``) ship
alongside the legacy ``nociception`` / ``nociception_temporal`` /
``nociception_klinotaxis`` modules — which stay in the registry forever
as frozen historical record.

This regression test guards the frozen-legacy invariant: every archived
config under ``configs/evolution/`` that names any ``nociception*``
module in its ``brain.sensory_modules`` list must continue to LOAD
cleanly via YAML parsing, and every legacy module name it references
must still be present in the SENSORY_MODULES registry. The earlier
predator-evasion-config reproducibility guarantee — every archived
logbook can re-run its evolution arms byte-identically — depends on
this.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from quantumnematode.brain.modules import SENSORY_MODULES, ModuleName

# Repo root inferred from this test file's path (two parents up from the
# packages/quantum-nematode/tests/quantumnematode_tests/utils/ subtree).
_REPO_ROOT = Path(__file__).resolve().parents[5]
_EVOLUTION_CONFIGS_DIR = _REPO_ROOT / "configs" / "evolution"


def _discover_legacy_configs() -> list[Path]:
    """Discover archived configs that name a legacy nociception module.

    Scans every `*.yml` under `configs/evolution/` and returns the
    subset whose `brain.config.sensory_modules` list contains a string
    starting with `nociception`.
    """
    if not _EVOLUTION_CONFIGS_DIR.exists():
        return []
    matches: list[Path] = []
    for path in sorted(_EVOLUTION_CONFIGS_DIR.rglob("*.yml")):
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        brain = data.get("brain") or {}
        config = brain.get("config") or {}
        sensory_modules = config.get("sensory_modules") or []
        if any(
            isinstance(name, str) and name.startswith("nociception") for name in sensory_modules
        ):
            matches.append(path)
    return matches


_LEGACY_CONFIGS = _discover_legacy_configs()


class TestLegacyNociceptionConfigsLoad:
    """Every archived `nociception*`-naming config must still load."""

    def test_at_least_one_legacy_config_discovered(self) -> None:
        # If this fails, either configs/evolution/ moved, or every archived
        # config has been migrated off the legacy modules (in which case
        # the legacy modules themselves should be considered for retirement
        # in a future change). Until then this guards against an empty
        # discovery set silently passing the parametrised test below.
        assert _LEGACY_CONFIGS, (
            "No archived evolution configs found that reference legacy "
            "nociception_* sensor modules. Verify configs/evolution/ exists "
            "and contains the archived Phase 5 predator-evasion configs."
        )

    @pytest.mark.parametrize(
        "config_path",
        _LEGACY_CONFIGS,
        ids=lambda p: p.name,
    )
    def test_legacy_config_yaml_parses(self, config_path: Path) -> None:
        """Each archived config must parse as valid YAML."""
        data = yaml.safe_load(config_path.read_text())
        assert isinstance(data, dict), f"{config_path.name}: top-level YAML is not a mapping"

    @pytest.mark.parametrize(
        "config_path",
        _LEGACY_CONFIGS,
        ids=lambda p: p.name,
    )
    def test_legacy_nociception_modules_still_registered(
        self,
        config_path: Path,
    ) -> None:
        """Legacy module names must still be registered in SENSORY_MODULES.

        Every nociception* module name referenced by the discovered
        config must round-trip through `ModuleName(name)` and appear in
        the SENSORY_MODULES dict — guarding against accidental retirement
        of the frozen-legacy modules.
        """
        data = yaml.safe_load(config_path.read_text())
        brain = data.get("brain") or {}
        config = brain.get("config") or {}
        sensory_modules = config.get("sensory_modules") or []
        legacy_names = [
            name
            for name in sensory_modules
            if isinstance(name, str) and name.startswith("nociception")
        ]
        assert legacy_names, (
            f"{config_path.name}: expected at least one nociception* "
            f"module in sensory_modules, got {sensory_modules}"
        )
        for name in legacy_names:
            # The registry is keyed by ModuleName enum values, but lookups
            # accept the string form because ModuleName is a StrEnum.
            try:
                module_name = ModuleName(name)
            except ValueError as exc:
                pytest.fail(
                    f"{config_path.name}: legacy module name '{name}' is no "
                    f"longer a ModuleName enum value — the frozen-legacy "
                    f"invariant is broken. ({exc})",
                )
            assert module_name in SENSORY_MODULES, (
                f"{config_path.name}: legacy module {name} is no longer "
                f"registered in SENSORY_MODULES"
            )
