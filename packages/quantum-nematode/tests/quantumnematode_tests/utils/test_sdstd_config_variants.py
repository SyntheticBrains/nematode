"""The `_sdstd` config variants differ from their parents by exactly one key.

Covers the add-state-dependent-action-std change's config-hygiene task 4.1:
each state-dependent-std variant is its parent plus
``brain.config.continuous_std_mode: state_dependent`` and nothing else — the
parents are the 029/036 records and must stay authoritative. The variants
must also load through the real config loader (validator accepts the
combination; #253 warning does not fire on the new key).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[4].parent
_CONFIGS = _REPO_ROOT / "configs" / "scenarios"

_C3_STEM = "small_continuous2d_combined_klinotaxis_sdstd.yml"
_GATE_STEM = "mlpppo_small_continuous2d_thermotaxis_seeking_{}_sdstd.yml"
_VARIANTS = [
    _CONFIGS / "foraging_predator_thermal" / f"{arch}_{_C3_STEM}"
    for arch in ("mlpppo", "cfcppo", "transformerppo", "connectomeppo")
] + [
    _CONFIGS / "thermal_foraging" / _GATE_STEM.format(mode) for mode in ("klinotaxis", "derivative")
]


def _flatten(data: object, prefix: str = "") -> dict[str, object]:
    if isinstance(data, dict):
        out: dict[str, object] = {}
        for key, value in data.items():
            out.update(_flatten(value, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: data}


@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda p: p.stem)
class TestSdstdVariants:
    def test_single_key_delta_from_parent(self, variant: Path) -> None:
        parent = variant.with_name(variant.name.replace("_sdstd", ""))
        assert parent.exists(), parent
        flat_parent = _flatten(yaml.safe_load(parent.read_text()))
        flat_variant = _flatten(yaml.safe_load(variant.read_text()))
        added = set(flat_variant) - set(flat_parent)
        removed = set(flat_parent) - set(flat_variant)
        changed = {
            k for k in set(flat_parent) & set(flat_variant) if flat_parent[k] != flat_variant[k]
        }
        assert removed == set()
        assert changed == set()
        assert added == {"brain.config.continuous_std_mode"}
        assert flat_variant["brain.config.continuous_std_mode"] == "state_dependent"

    def test_variant_loads_through_the_real_loader(self, variant: Path) -> None:
        config = load_simulation_config(str(variant))
        assert config.brain is not None
        brain_config = config.brain.config
        assert brain_config is not None
        assert brain_config.continuous_std_mode == "state_dependent"
