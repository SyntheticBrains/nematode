"""The plastic arm config differs from its PPO parent by exactly the rule keys.

The panel's arms are only comparable if they differ where they claim to and
nowhere else. This pins that for the plastic wild-type arm: any environment,
reward or sensing drift between it and its parent would become a rival
explanation for whatever the arm measures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrainConfig
from quantumnematode.brain.arch.mlpppo import MLPPPOBrainConfig
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
_FROZEN = _VARIANT.with_name(_VARIANT.name.replace(".yml", "_frozen.yml"))
_HEBBIAN = _VARIANT.with_name(_VARIANT.name.replace(".yml", "_hebbian.yml"))

# Every key the plastic wild-type arm adds to its PPO parent: the rule, the trace it
# reads, the scaling switches, and the panel's shared recipe. This set IS the contract;
# the floors and the rewired arms inherit it from this arm, and the MLP adds its own.
_EXPECTED_ADDED = {
    "brain.config.learning_rule",
    "brain.config.enable_activity_traces",
    "brain.config.plasticity_normalise_modulator",
    "brain.config.plasticity_normalise_trace",
    "brain.config.plasticity_homeostasis",
    "brain.config.initial_log_std",
    "brain.config.plasticity_rate",
}
# The MLP yardstick additionally swaps its hidden non-linearity for bounded units.
_EXPECTED_ADDED_MLP = _EXPECTED_ADDED | {"brain.config.activation", "brain.config.plastic_layers"}


def _flatten(data: object, prefix: str = "") -> dict[str, object]:
    if isinstance(data, dict):
        out: dict[str, object] = {}
        for key, value in data.items():
            out.update(_flatten(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): data}


class TestPlasticVariantIsAMinimalDelta:
    """Exactly the rule selection, the trace it reads, and the two scaling switches."""

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
        assert brain_config.plasticity_normalise_modulator is True
        assert brain_config.plasticity_normalise_trace is True
        assert brain_config.plasticity_homeostasis is True
        assert brain_config.initial_log_std == -1.0
        assert brain_config.plasticity_rate == 0.001

    def test_parent_is_unchanged(self) -> None:
        """The PPO record the plastic arm derives from stays on the PPO rule."""
        config = load_simulation_config(str(_PARENT))
        assert config.brain is not None
        brain_config = config.brain.config
        assert isinstance(brain_config, ConnectomePPOBrainConfig)
        assert brain_config.learning_rule == "ppo"


class TestSanityFloorConfigs:
    """Each floor differs from the plastic arm only where it claims to."""

    def test_frozen_floor_changes_only_the_freeze(self) -> None:
        """The parent already declares the flag, so the floor flips its value."""
        plastic = _flatten(yaml.safe_load(_VARIANT.read_text()))
        frozen = _flatten(yaml.safe_load(_FROZEN.read_text()))

        assert set(frozen) - set(plastic) == set()
        assert set(plastic) - set(frozen) == set()
        assert {k for k in set(plastic) & set(frozen) if plastic[k] != frozen[k]} == {
            "brain.config.freeze_updates",
        }

    def test_hebbian_floor_changes_only_the_rule(self) -> None:
        plastic = _flatten(yaml.safe_load(_VARIANT.read_text()))
        hebbian = _flatten(yaml.safe_load(_HEBBIAN.read_text()))

        assert set(hebbian) - set(plastic) == set()
        assert set(plastic) - set(hebbian) == set()
        assert {k for k in set(plastic) & set(hebbian) if plastic[k] != hebbian[k]} == {
            "brain.config.learning_rule",
        }

    def test_frozen_floor_loads_as_a_frozen_plastic_arm(self) -> None:
        config = load_simulation_config(str(_FROZEN))
        assert config.brain is not None
        brain_config = config.brain.config
        assert isinstance(brain_config, ConnectomePPOBrainConfig)
        # The plasticity rule, not the gradient rule -- the floor must decode
        # like the arm it bounds.
        assert brain_config.learning_rule == "three_factor"
        assert brain_config.freeze_updates is True

    def test_hebbian_floor_loads_as_the_unmodulated_arm(self) -> None:
        config = load_simulation_config(str(_HEBBIAN))
        assert config.brain is not None
        brain_config = config.brain.config
        assert isinstance(brain_config, ConnectomePPOBrainConfig)
        assert brain_config.learning_rule == "hebbian"
        assert brain_config.freeze_updates is False

    def test_floor_names_keep_the_parent_as_a_prefix(self) -> None:
        """So an arm and its floors sort together in the scenario directory."""
        stem = _VARIANT.name.removesuffix(".yml")
        assert _FROZEN.name.startswith(stem)
        assert _HEBBIAN.name.startswith(stem)


_MLP_VARIANT = _VARIANT.with_name(_VARIANT.name.replace("connectomeppo", "mlpppo"))
_MLP_PARENT = _MLP_VARIANT.with_name(_MLP_VARIANT.name.replace("_plastic", ""))


class TestMatchedRuleMLPConfig:
    """The yardstick arm differs from its PPO parent only by the rule keys."""

    def test_mlp_plastic_is_a_minimal_delta(self) -> None:
        parent = _flatten(yaml.safe_load(_MLP_PARENT.read_text()))
        variant = _flatten(yaml.safe_load(_MLP_VARIANT.read_text()))
        assert set(variant) - set(parent) == _EXPECTED_ADDED_MLP
        assert set(parent) - set(variant) == set()
        assert {k for k in set(parent) & set(variant) if parent[k] != variant[k]} == set()

    def test_mlp_plastic_loads_and_selects_the_rule(self) -> None:
        config = load_simulation_config(str(_MLP_VARIANT))
        assert config.brain is not None
        brain_config = config.brain.config
        assert isinstance(brain_config, MLPPPOBrainConfig)
        assert brain_config.learning_rule == "three_factor"
        assert brain_config.enable_activity_traces is True
        assert brain_config.activation == "tanh"
        assert brain_config.plastic_layers == "hidden"

    def test_mlp_and_connectome_plastic_arms_share_every_plasticity_value(self) -> None:
        """Matched means the same numbers, read from the actual arm configs."""
        mlp = load_simulation_config(str(_MLP_VARIANT)).brain
        conn = load_simulation_config(str(_VARIANT)).brain
        assert mlp is not None
        assert conn is not None
        for field in (
            "learning_rule",
            "plasticity_rate",
            "plasticity_weight_decay",
            "plasticity_weight_bound",
            "plasticity_baseline_rate",
            "trace_decay",
            "enable_activity_traces",
            "plasticity_normalise_modulator",
            "plasticity_normalise_trace",
            "plasticity_scale_rate",
            "plasticity_scale_floor",
            "plasticity_homeostasis",
            "initial_log_std",
        ):
            assert getattr(mlp.config, field) == getattr(conn.config, field), field

    def test_mlp_parent_is_unchanged(self) -> None:
        config = load_simulation_config(str(_MLP_PARENT))
        assert config.brain is not None
        assert isinstance(config.brain.config, MLPPPOBrainConfig)
        assert config.brain.config.learning_rule == "ppo"


_REWIRED = _VARIANT.with_name(_VARIANT.name.replace(".yml", "_rewired_null.yml"))


class TestPlasticRewiredNullConfig:
    """The primary contrast's other cell: one key different from the plastic arm."""

    def test_rewired_null_adds_only_the_wiring_key(self) -> None:
        plastic = _flatten(yaml.safe_load(_VARIANT.read_text()))
        rewired = _flatten(yaml.safe_load(_REWIRED.read_text()))
        assert set(rewired) - set(plastic) == {"brain.config.wiring"}
        assert set(plastic) - set(rewired) == set()
        assert {k for k in set(plastic) & set(rewired) if plastic[k] != rewired[k]} == set()

    def test_rewired_null_loads_as_the_plastic_null_arm(self) -> None:
        config = load_simulation_config(str(_REWIRED))
        assert config.brain is not None
        brain_config = config.brain.config
        assert isinstance(brain_config, ConnectomePPOBrainConfig)
        assert brain_config.wiring == "rewired_degree_preserving"
        assert brain_config.learning_rule == "three_factor"
        assert brain_config.enable_activity_traces is True
        assert brain_config.rewire_seed is None

    def test_plastic_parent_stays_wild_type(self) -> None:
        config = load_simulation_config(str(_VARIANT))
        assert config.brain is not None
        assert isinstance(config.brain.config, ConnectomePPOBrainConfig)
        assert config.brain.config.wiring == "wild_type"

    def test_name_keeps_the_plastic_parent_as_a_prefix(self) -> None:
        assert _REWIRED.name.startswith(_VARIANT.name.removesuffix(".yml"))


_FROZEN_REWIRED = _FROZEN.with_name(_FROZEN.name.replace(".yml", "_rewired_null.yml"))
_HEBBIAN_REWIRED = _HEBBIAN.with_name(_HEBBIAN.name.replace(".yml", "_rewired_null.yml"))


class TestRewiredFloorConfigs:
    """One key off its wild-type floor, and one key off the plastic rewired-null arm."""

    @pytest.mark.parametrize(
        ("parent", "derived"),
        [(_FROZEN, _FROZEN_REWIRED), (_HEBBIAN, _HEBBIAN_REWIRED)],
        ids=["frozen", "hebbian"],
    )
    def test_adds_only_the_wiring_key(self, parent: Path, derived: Path) -> None:
        base = _flatten(yaml.safe_load(parent.read_text()))
        variant = _flatten(yaml.safe_load(derived.read_text()))
        assert set(variant) - set(base) == {"brain.config.wiring"}
        assert set(base) - set(variant) == set()
        assert {k for k in set(base) & set(variant) if base[k] != variant[k]} == set()

    def test_frozen_rewired_is_one_key_off_the_plastic_null_arm(self) -> None:
        null_arm = _flatten(yaml.safe_load(_REWIRED.read_text()))
        floor = _flatten(yaml.safe_load(_FROZEN_REWIRED.read_text()))
        assert set(floor) == set(null_arm)
        assert {k for k in null_arm if null_arm[k] != floor[k]} == {"brain.config.freeze_updates"}

    def test_hebbian_rewired_is_one_key_off_the_plastic_null_arm(self) -> None:
        null_arm = _flatten(yaml.safe_load(_REWIRED.read_text()))
        floor = _flatten(yaml.safe_load(_HEBBIAN_REWIRED.read_text()))
        assert set(floor) == set(null_arm)
        assert {k for k in null_arm if null_arm[k] != floor[k]} == {"brain.config.learning_rule"}

    def test_frozen_rewired_loads_as_a_frozen_null_arm(self) -> None:
        config = load_simulation_config(str(_FROZEN_REWIRED))
        assert config.brain is not None
        brain_config = config.brain.config
        assert isinstance(brain_config, ConnectomePPOBrainConfig)
        assert brain_config.wiring == "rewired_degree_preserving"
        assert brain_config.rewire_seed is None
        assert brain_config.learning_rule == "three_factor"
        assert brain_config.freeze_updates is True
        assert brain_config.enable_activity_traces is True
        assert brain_config.chemical_mask_mode == "strict"

    def test_hebbian_rewired_loads_as_the_unmodulated_null_arm(self) -> None:
        config = load_simulation_config(str(_HEBBIAN_REWIRED))
        assert config.brain is not None
        brain_config = config.brain.config
        assert isinstance(brain_config, ConnectomePPOBrainConfig)
        assert brain_config.wiring == "rewired_degree_preserving"
        assert brain_config.rewire_seed is None
        assert brain_config.learning_rule == "hebbian"
        assert brain_config.freeze_updates is False
        assert brain_config.enable_activity_traces is True

    def test_names_keep_the_wild_type_floor_as_a_prefix(self) -> None:
        assert _FROZEN_REWIRED.name.startswith(_FROZEN.name.removesuffix(".yml"))
        assert _HEBBIAN_REWIRED.name.startswith(_HEBBIAN.name.removesuffix(".yml"))


_COUNT_INIT = [
    (path, path.with_name(path.name.replace(".yml", "_countinit.yml")))
    for path in (_FROZEN, _FROZEN_REWIRED, _HEBBIAN, _HEBBIAN_REWIRED)
]


class TestCountInitConfigs:
    """Each count-initialised floor is one key off its degree-scaled parent."""

    @pytest.mark.parametrize(
        ("parent", "derived"),
        _COUNT_INIT,
        ids=["frozen", "frozen_rewired", "hebbian", "hebbian_rewired"],
    )
    def test_adds_only_the_init_key(self, parent: Path, derived: Path) -> None:
        base = _flatten(yaml.safe_load(parent.read_text()))
        variant = _flatten(yaml.safe_load(derived.read_text()))
        assert set(variant) - set(base) == {"brain.config.weight_init"}
        assert set(base) - set(variant) == set()
        assert {k for k in set(base) & set(variant) if base[k] != variant[k]} == set()
        assert variant["brain.config.weight_init"] == "count_scaled"

    @pytest.mark.parametrize(
        ("parent", "derived"),
        _COUNT_INIT,
        ids=["frozen", "frozen_rewired", "hebbian", "hebbian_rewired"],
    )
    def test_loads_with_the_parent_rule_and_wiring(self, parent: Path, derived: Path) -> None:
        base = load_simulation_config(str(parent)).brain
        variant = load_simulation_config(str(derived)).brain
        assert base is not None
        assert variant is not None
        assert isinstance(base.config, ConnectomePPOBrainConfig)
        assert isinstance(variant.config, ConnectomePPOBrainConfig)
        assert variant.config.weight_init == "count_scaled"
        assert base.config.weight_init == "degree_scaled"
        assert variant.config.learning_rule == base.config.learning_rule
        assert variant.config.wiring == base.config.wiring
        assert variant.config.freeze_updates == base.config.freeze_updates

    def test_parents_keep_the_degree_scaled_default(self) -> None:
        for parent, _ in _COUNT_INIT:
            assert "weight_init" not in parent.read_text()


_WARM = {
    "clone": [
        (_FROZEN, "_clone"),
        (_FROZEN_REWIRED, "_clone"),
        (_HEBBIAN, "_clone"),
        (_HEBBIAN_REWIRED, "_clone"),
        (_VARIANT, "_clone"),
        (_REWIRED, "_clone"),
        (_FROZEN, "_fullclone"),
        (_FROZEN_REWIRED, "_fullclone"),
    ],
}
_LOWSTD = _PARENT.with_name(_PARENT.name.replace(".yml", "_lowstd.yml"))
_LOWSTD_REWIRED = _PARENT.with_name(_PARENT.name.replace(".yml", "_rewired_null_lowstd.yml"))


class TestWarmStartConfigs:
    """Each warm-started arm is one key off its parent.

    The clone arms add `weights_path`; the low-noise PPO arms add `initial_log_std`; the
    fine-tune arms add `weights_path` to the low-noise arms.
    """

    @pytest.mark.parametrize(
        ("parent", "suffix"),
        _WARM["clone"],
        ids=lambda x: getattr(x, "name", x),
    )
    def test_clone_arms_add_only_the_weights_path(self, parent: Path, suffix: str) -> None:
        derived = parent.with_name(parent.name.replace(".yml", f"{suffix}.yml"))
        base = _flatten(yaml.safe_load(parent.read_text()))
        variant = _flatten(yaml.safe_load(derived.read_text()))
        assert set(variant) - set(base) == {"brain.config.weights_path"}
        assert set(base) - set(variant) == set()
        assert {k for k in set(base) & set(variant) if base[k] != variant[k]} == set()
        expected_set = "full" if suffix == "_fullclone" else "plastic"
        wiring = "rn" if "rewired_null" in parent.name else "wt"
        assert (
            variant["brain.config.weights_path"]
            == f"campaigns/l4-warm-start/clones/{expected_set}_{wiring}_seed{{seed}}.pt"
        )

    @pytest.mark.parametrize(
        ("parent", "derived"),
        [
            (_PARENT, _LOWSTD),
            (_PARENT.with_name(_PARENT.name.replace(".yml", "_rewired_null.yml")), _LOWSTD_REWIRED),
        ],
        ids=["wt", "rn"],
    )
    def test_lowstd_arms_add_only_the_noise_key(self, parent: Path, derived: Path) -> None:
        base = _flatten(yaml.safe_load(parent.read_text()))
        variant = _flatten(yaml.safe_load(derived.read_text()))
        assert set(variant) - set(base) == {"brain.config.initial_log_std"}
        assert variant["brain.config.initial_log_std"] == -1.0
        assert {k for k in set(base) & set(variant) if base[k] != variant[k]} == set()

    @pytest.mark.parametrize("parent", [_LOWSTD, _LOWSTD_REWIRED], ids=["wt", "rn"])
    def test_fine_tune_arms_add_only_the_weights_path(self, parent: Path) -> None:
        derived = parent.with_name(parent.name.replace(".yml", "_fullclone.yml"))
        base = _flatten(yaml.safe_load(parent.read_text()))
        variant = _flatten(yaml.safe_load(derived.read_text()))
        assert set(variant) - set(base) == {"brain.config.weights_path"}
        assert {k for k in set(base) & set(variant) if base[k] != variant[k]} == set()

    def test_every_warm_started_config_loads(self) -> None:
        for path in _VARIANT.parent.glob("connectomeppo_*clone*.yml"):
            config = load_simulation_config(str(path)).brain
            assert config is not None
            assert isinstance(config.config, ConnectomePPOBrainConfig)
            assert config.config.weights_path is not None


_ATLAS_SIGNS = [
    (_FROZEN, "_atlassigns"),
    (_FROZEN_REWIRED, "_atlassigns"),
    (_HEBBIAN, "_atlassigns"),
    (_HEBBIAN_REWIRED, "_atlassigns"),
]


class TestAtlasSignConfigs:
    """Grounded arms are one `synapse_signs` key off their parent; Dale arms one key off those."""

    @pytest.mark.parametrize(
        ("parent", "suffix"),
        _ATLAS_SIGNS,
        ids=["frozen", "frozen_rewired", "hebbian", "hebbian_rewired"],
    )
    def test_grounded_arms_add_only_the_sign_key(self, parent: Path, suffix: str) -> None:
        derived = parent.with_name(parent.name.replace(".yml", f"{suffix}.yml"))
        base = _flatten(yaml.safe_load(parent.read_text()))
        variant = _flatten(yaml.safe_load(derived.read_text()))
        assert set(variant) - set(base) == {"brain.config.synapse_signs"}
        assert set(base) - set(variant) == set()
        assert {k for k in set(base) & set(variant) if base[k] != variant[k]} == set()
        assert variant["brain.config.synapse_signs"] == "atlas"

    @pytest.mark.parametrize(
        "parent",
        [
            _HEBBIAN.with_name(_HEBBIAN.name.replace(".yml", "_atlassigns.yml")),
            _HEBBIAN_REWIRED.with_name(_HEBBIAN_REWIRED.name.replace(".yml", "_atlassigns.yml")),
        ],
        ids=["wt", "rn"],
    )
    def test_dale_arms_add_only_the_enforcement_key(self, parent: Path) -> None:
        derived = parent.with_name(parent.name.replace(".yml", "_dale.yml"))
        base = _flatten(yaml.safe_load(parent.read_text()))
        variant = _flatten(yaml.safe_load(derived.read_text()))
        assert set(variant) - set(base) == {"brain.config.enforce_synapse_signs"}
        assert variant["brain.config.enforce_synapse_signs"] is True
        assert {k for k in set(base) & set(variant) if base[k] != variant[k]} == set()

    def test_every_grounded_config_loads_with_grounded_signs(self) -> None:
        paths = sorted(_VARIANT.parent.glob("connectomeppo_*atlassigns*.yml"))
        assert len(paths) == 6
        for path in paths:
            config = load_simulation_config(str(path)).brain
            assert config is not None
            assert isinstance(config.config, ConnectomePPOBrainConfig)
            assert config.config.synapse_signs == "atlas"
            assert config.config.enforce_synapse_signs == path.name.endswith("_dale.yml")

    def test_parents_keep_random_signs(self) -> None:
        for parent, _suffix in _ATLAS_SIGNS:
            assert "synapse_signs" not in parent.read_text()
