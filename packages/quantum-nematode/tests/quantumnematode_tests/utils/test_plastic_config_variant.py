"""The plastic arm config differs from its PPO parent by exactly two keys.

The panel's arms are only comparable if they differ where they claim to and
nowhere else. This pins that for the plastic wild-type arm: any environment,
reward or sensing drift between it and its parent would become a rival
explanation for whatever the arm measures.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrainConfig
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[4].parent
_VARIANT = (
    _REPO_ROOT
    / "configs"
    / "scenarios"
    / "foraging_predator_thermal"
    / "connectomeppo_small_continuous2d_combined_klinotaxis_plastic.yml"
)
_PARENT = _VARIANT.with_name(_VARIANT.name.replace("_plastic", ""))

_EXPECTED_ADDED = {
    "brain.config.learning_rule",
    "brain.config.enable_activity_traces",
}


def _flatten(data: object, prefix: str = "") -> dict[str, object]:
    if isinstance(data, dict):
        out: dict[str, object] = {}
        for key, value in data.items():
            out.update(_flatten(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): data}


class TestPlasticVariantIsAMinimalDelta:
    """Exactly the rule selection and the trace it reads."""

    def test_only_the_rule_keys_differ(self) -> None:
        parent = _flatten(yaml.safe_load(_PARENT.read_text()))
        variant = _flatten(yaml.safe_load(_VARIANT.read_text()))

        added = set(variant) - set(parent)
        removed = set(parent) - set(variant)
        changed = {key for key in set(parent) & set(variant) if parent[key] != variant[key]}

        assert added == _EXPECTED_ADDED
        assert removed == set()
        assert changed == set()

    def test_variant_loads_and_selects_the_rule(self) -> None:
        config = load_simulation_config(str(_VARIANT))
        assert config.brain is not None
        brain_config = config.brain.config
        assert isinstance(brain_config, ConnectomePPOBrainConfig)
        assert brain_config.learning_rule == "three_factor"
        assert brain_config.enable_activity_traces is True

    def test_parent_is_unchanged(self) -> None:
        """The PPO record the plastic arm derives from stays on the PPO rule."""
        config = load_simulation_config(str(_PARENT))
        assert config.brain is not None
        brain_config = config.brain.config
        assert isinstance(brain_config, ConnectomePPOBrainConfig)
        assert brain_config.learning_rule == "ppo"
