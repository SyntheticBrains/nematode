"""Atlas-grounded synapse signs and Dale's law enforcement on the connectome brain.

Grounding replaces the sign a weight drew with the sign its pre-synaptic neuron's released
transmitter implies, and must change nothing else: the same draws, the same magnitudes, the same
per-neuron incoming norms, the same readout and gains. Enforcement then holds those signs through
plasticity while leaving ungrounded synapses free.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.neurotransmitters import read_atlas_transmitters, sign_for
from quantumnematode.learning_rules.three_factor import ThreeFactorRule
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_ARMS = _REPO_ROOT / "configs" / "scenarios" / "foraging_predator_thermal"
_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis"
_FROZEN = _ARMS / f"{_STEM}_plastic_frozen.yml"
_PLASTIC = _ARMS / f"{_STEM}_plastic.yml"
_SEED = 5
_GROUNDED = 3176
_EXCITATORY = 2962
_INHIBITORY = 214


def _cfg(path: Path, **overrides: object) -> ConnectomePPOBrainConfig:
    config = load_simulation_config(str(path)).brain
    assert config is not None
    assert isinstance(config.config, ConnectomePPOBrainConfig)
    return config.config.model_copy(update={"seed": _SEED, **overrides})


def _brain(path: Path, **overrides: object) -> ConnectomePPOBrain:
    return ConnectomePPOBrain(config=_cfg(path, **overrides), device=DeviceType.CPU)


def _buffer(brain: ConnectomePPOBrain, name: str) -> torch.Tensor:
    """Return a registered buffer, narrowed — ``nn.Module`` types these as ``Tensor | Module``."""
    value = getattr(brain.topology, name)
    assert isinstance(value, torch.Tensor)
    return value


@pytest.fixture(scope="module")
def pair() -> tuple[ConnectomePPOBrain, ConnectomePPOBrain]:
    """Build the same arm at the same seed, random-sign and atlas-grounded."""
    return _brain(_FROZEN), _brain(_FROZEN, synapse_signs="atlas")


class TestGroundingChangesOnlySigns:
    def test_default_is_bit_identical(self) -> None:
        assert _cfg(_FROZEN).synapse_signs == "random"
        default = _brain(_FROZEN)
        explicit = _brain(_FROZEN, synapse_signs="random")
        assert torch.equal(default.topology.w_chem, explicit.topology.w_chem)
        assert int((_buffer(default, "chem_sign") != 0).sum()) == 0

    def test_magnitudes_and_norms_are_untouched(
        self,
        pair: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        random_brain, atlas = pair
        assert torch.equal(random_brain.topology.w_chem.abs(), atlas.topology.w_chem.abs())
        mask = _buffer(random_brain, "m_chem")
        norms = [
            torch.sqrt(((b.topology.w_chem * mask) ** 2).sum(0)) for b in (random_brain, atlas)
        ]
        assert torch.allclose(norms[0], norms[1])

    def test_periphery_and_wiring_are_untouched(
        self,
        pair: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        random_brain, atlas = pair
        assert torch.equal(_buffer(random_brain, "m_chem"), _buffer(atlas, "m_chem"))
        assert torch.equal(_buffer(random_brain, "g_gap"), _buffer(atlas, "g_gap"))
        assert torch.equal(random_brain.topology.readout, atlas.topology.readout)
        assert torch.equal(random_brain.topology.food_gains, atlas.topology.food_gains)
        assert torch.equal(random_brain.topology.log_std, atlas.topology.log_std)

    def test_grounded_signs_match_the_atlas(
        self,
        pair: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        _random_brain, atlas = pair
        signs = _buffer(atlas, "chem_sign")
        assert int((signs != 0).sum()) == _GROUNDED
        assert int((signs > 0).sum()) == _EXCITATORY
        assert int((signs < 0).sum()) == _INHIBITORY
        weights = atlas.topology.w_chem
        assert torch.all(torch.sign(weights[signs != 0]) == signs[signs != 0].to(weights.dtype))

    def test_ungrounded_synapses_keep_the_sign_they_drew(
        self,
        pair: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        random_brain, atlas = pair
        free = (_buffer(atlas, "chem_sign") == 0) & _buffer(atlas, "m_chem")
        assert int(free.sum()) == 3709 - _GROUNDED
        assert torch.equal(random_brain.topology.w_chem[free], atlas.topology.w_chem[free])

    def test_grounding_makes_the_network_mostly_excitatory(
        self,
        pair: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        """The point of the change: random signs are a coin flip, the animal's are not."""
        random_brain, atlas = pair
        mask = _buffer(random_brain, "m_chem")
        assert 0.45 < float((random_brain.topology.w_chem[mask] < 0).float().mean()) < 0.55
        assert float((atlas.topology.w_chem[mask] < 0).float().mean()) < 0.2

    def test_signs_follow_the_pre_synaptic_neuron(
        self,
        pair: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        _random_brain, atlas = pair
        atlas_transmitters = read_atlas_transmitters()
        index = atlas.topology._idx
        signs = _buffer(atlas, "chem_sign")
        for name in ("AVAL", "AIYL", "RIML", "ADEL"):
            row = signs[index[name]]
            expected = sign_for(atlas_transmitters[name])
            outgoing = row[row != 0]
            if expected is None:
                assert int(outgoing.numel()) == 0
            else:
                assert torch.all(outgoing == expected)


class TestDalesLaw:
    def test_enforcement_requires_grounding(self) -> None:
        """Refused when a config is validated, and again when a brain is built from one."""
        base = _cfg(_PLASTIC).model_dump()
        with pytest.raises(ValueError, match="requires synapse_signs='atlas'"):
            ConnectomePPOBrainConfig.model_validate(base | {"enforce_synapse_signs": True})
        # `model_copy` skips validators, so the brain guards construction itself.
        unchecked = _cfg(_PLASTIC, enforce_synapse_signs=True)
        with pytest.raises(ValueError, match="requires synapse_signs='atlas'"):
            ConnectomePPOBrain(config=unchecked, device=DeviceType.CPU)

    def test_off_by_default_and_the_signs_constrain_nothing(self) -> None:
        # The rule holds the grounded signs whenever the substrate has them, because a
        # decorrelating variant reads the same identities without constraining any weight.
        # Enforcement is the separate switch, and with it off no projection happens.
        brain = _brain(_PLASTIC, synapse_signs="atlas")
        rule = brain._rule
        assert isinstance(rule, ThreeFactorRule)
        assert len(rule._synapse_signs) == 1
        assert rule.enforce_signs is False
        weights = brain.topology.w_chem
        signs = _buffer(brain, "chem_sign")
        with torch.no_grad():
            weights.data.copy_(-torch.sign(signs).to(weights.dtype))  # every grounded sign violated
        violated = weights.detach().clone()
        rule._project_signs(0, weights)
        assert torch.equal(weights.detach(), violated)

    def test_an_ungrounded_brain_hands_the_rule_no_signs(self) -> None:
        rule = _brain(_PLASTIC)._rule
        assert isinstance(rule, ThreeFactorRule)
        assert rule._synapse_signs == []

    def test_the_projection_clamps_each_grounded_synapse(self) -> None:
        brain = _brain(_PLASTIC, synapse_signs="atlas", enforce_synapse_signs=True)
        rule = brain._rule
        assert isinstance(rule, ThreeFactorRule)
        assert len(rule._synapse_signs) == 1
        weights = brain.topology.w_chem
        signs = _buffer(brain, "chem_sign")
        with torch.no_grad():
            weights.data.copy_(-torch.sign(signs).to(weights.dtype))  # every grounded sign violated
        assert int(((signs > 0) & (weights < 0)).sum()) == _EXCITATORY
        rule._project_signs(0, weights)
        assert int(((signs > 0) & (weights < 0)).sum()) == 0
        assert int(((signs < 0) & (weights > 0)).sum()) == 0

    def test_the_projection_leaves_ungrounded_synapses_free(self) -> None:
        brain = _brain(_PLASTIC, synapse_signs="atlas", enforce_synapse_signs=True)
        rule = brain._rule
        assert isinstance(rule, ThreeFactorRule)
        weights = brain.topology.w_chem
        free = (_buffer(brain, "chem_sign") == 0) & _buffer(brain, "m_chem")
        before = weights[free].clone()
        rule._project_signs(0, weights)
        assert torch.equal(weights[free], before)

    def test_no_violation_survives_a_real_update(self) -> None:
        """Wired into the update path, ahead of homeostasis and the clamp."""
        brain = _brain(_PLASTIC, synapse_signs="atlas", enforce_synapse_signs=True)
        signs = _buffer(brain, "chem_sign")
        with torch.no_grad():
            brain.topology.w_chem.data.copy_(-torch.sign(signs).to(brain.topology.w_chem.dtype))
        brain.prepare_episode()
        torch.manual_seed(1)
        for step in range(4):
            brain.run_brain(
                BrainParams(food_gradient_strength=0.3, food_gradient_direction=0.2 * step),
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            brain.learn(BrainParams(), reward=0.5, episode_done=(step == 3))
        weights = brain.topology.w_chem
        assert int(((signs > 0) & (weights < 0)).sum()) == 0
        assert int(((signs < 0) & (weights > 0)).sum()) == 0
        assert torch.all(weights.abs() <= brain.config.plasticity_weight_bound + 1e-6)
