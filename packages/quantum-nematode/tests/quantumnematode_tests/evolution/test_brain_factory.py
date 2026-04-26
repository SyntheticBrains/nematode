"""Tests for :mod:`quantumnematode.evolution.brain_factory`."""

from __future__ import annotations

from pathlib import Path

import pytest
from quantumnematode.brain.arch.lstmppo import LSTMPPOBrain
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain
from quantumnematode.evolution.brain_factory import instantiate_brain_from_sim_config
from quantumnematode.utils.config_loader import load_simulation_config

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"
LSTMPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/lstmppo_small_klinotaxis.yml"


def test_instantiate_mlpppo_from_sim_config() -> None:
    """The wrapper SHALL return an MLPPPOBrain for an mlpppo config."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    brain = instantiate_brain_from_sim_config(sim_config)
    assert isinstance(brain, MLPPPOBrain)
    assert brain._episode_count == 0


def test_instantiate_lstmppo_from_sim_config() -> None:
    """The wrapper SHALL return an LSTMPPOBrain for an lstmppo config."""
    sim_config = load_simulation_config(str(LSTMPPO_CONFIG))
    brain = instantiate_brain_from_sim_config(sim_config)
    assert isinstance(brain, LSTMPPOBrain)
    assert brain._episode_count == 0


def test_instantiate_brain_seed_patches_brain_config_not_sim_config() -> None:
    """``seed`` SHALL patch ``BrainConfig.seed``, not ``SimulationConfig.seed``.

    The brain reads its seed from BrainConfig.seed via
    ``ensure_seed(config.seed)``; if the wrapper had patched
    SimulationConfig.seed instead, ``brain.seed`` would be either None
    (auto-generated) or whatever the YAML had.
    """
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    # The pilot config's brain.config.seed is None, so without patching the
    # brain would auto-generate a random seed.  With patching, brain.seed == 42.
    brain = instantiate_brain_from_sim_config(sim_config, seed=42)
    assert isinstance(brain, MLPPPOBrain)
    assert brain.seed == 42


def test_instantiate_brain_forces_weights_path_none(tmp_path: Path) -> None:
    """The wrapper SHALL force ``BrainConfig.weights_path = None``.

    Loading pretrained weights from disk would conflict with the genome
    (which is the sole weight source for evolution).  We patch
    ``brain.config.weights_path`` to a non-existent path before
    instantiation; if the wrapper failed to force it to None, the brain's
    constructor would attempt to load from disk and raise FileNotFoundError.
    """
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    # Inject a bogus weights_path into the brain config.  Use a tmp_path child
    # rather than /tmp/ to keep ruff S108 happy.
    bogus_path = tmp_path / "does-not-exist.pt"
    bogus = sim_config.brain.config.model_copy(update={"weights_path": str(bogus_path)})  # type: ignore[union-attr]
    sim_config = sim_config.model_copy(
        update={"brain": sim_config.brain.model_copy(update={"config": bogus})},  # type: ignore[union-attr]
    )

    # Should not raise — the wrapper overrides weights_path to None.
    brain = instantiate_brain_from_sim_config(sim_config, seed=7)
    assert isinstance(brain, MLPPPOBrain)
    assert brain.seed == 7


def test_instantiate_without_seed_uses_yaml_or_auto() -> None:
    """When ``seed=None``, the wrapper SHALL leave ``BrainConfig.seed`` alone.

    If the YAML did not set a seed, the brain auto-generates one (via
    ``ensure_seed``); if the YAML did set one, the brain uses it.  Either
    way, ``brain.seed`` SHALL be a valid int.
    """
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    brain = instantiate_brain_from_sim_config(sim_config, seed=None)
    assert isinstance(brain, MLPPPOBrain)
    assert isinstance(brain.seed, int)


def test_instantiate_brain_requires_brain_config() -> None:
    """A SimulationConfig with no brain SHALL raise ValueError."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    sim_config = sim_config.model_copy(update={"brain": None})
    with pytest.raises(ValueError, match=r"brain\.name"):
        instantiate_brain_from_sim_config(sim_config)
