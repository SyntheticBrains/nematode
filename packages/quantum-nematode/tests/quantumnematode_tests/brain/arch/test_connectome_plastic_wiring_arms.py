"""The two plastic wiring arms differ in wiring and in nothing else.

The panel's primary contrast -- plastic wild-type against plastic
degree-preserving rewired-null -- is worth exactly as much as this list is
true. Every assertion here is at the brain level on the real C3 plastic
configs; the rewiring function itself is proved separately in
``connectome/test_rewiring.py`` and is not re-tested.
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
from quantumnematode.utils.config_loader import load_simulation_config

# This file sits in tests/quantumnematode_tests/brain/arch -- one level deeper
# than the utils/ and campaigns/ tests, whose parents[4].parent idiom would
# resolve to packages/ from here.
_REPO_ROOT = Path(__file__).resolve().parents[6]
_ARMS = _REPO_ROOT / "configs" / "scenarios" / "foraging_predator_thermal"
_WILD = _ARMS / "connectomeppo_small_continuous2d_combined_klinotaxis_plastic.yml"
_REWIRED = _ARMS / "connectomeppo_small_continuous2d_combined_klinotaxis_plastic_rewired_null.yml"

_SEED = 23
# Verified to yield a different rewiring from _SEED; "differs" is probabilistic
# in principle, so the pair is pinned rather than drawn.
_OTHER_SEED = 24
_STEPS = 6

_GAINS = (
    "food_gains",
    "predator_distal_gains",
    "predator_anterior_gains",
    "predator_posterior_gains",
    "thermotaxis_gains",
)
_PLASTICITY_FIELDS = (
    "learning_rule",
    "plasticity_rate",
    "plasticity_weight_decay",
    "plasticity_weight_bound",
    "plasticity_baseline_rate",
    "enable_activity_traces",
    "trace_decay",
    "freeze_updates",
    "chemical_mask_mode",
)


def _config(path: Path) -> ConnectomePPOBrainConfig:
    config = load_simulation_config(str(path)).brain
    assert config is not None
    assert isinstance(config.config, ConnectomePPOBrainConfig)
    return config.config


def _brain(path: Path, seed: int = _SEED, **overrides: object) -> ConnectomePPOBrain:
    cfg = _config(path).model_copy(update={"seed": seed, **overrides})
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _drive(brain: ConnectomePPOBrain) -> None:
    brain.prepare_episode()
    torch.manual_seed(_SEED + 1)
    for step in range(_STEPS):
        brain.run_brain(
            BrainParams(
                food_gradient_strength=0.3 + 0.05 * step,
                food_gradient_direction=0.1 * step - 0.2,
            ),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        brain.learn(BrainParams(), reward=0.4 * (step % 3) - 0.2, episode_done=(step == _STEPS - 1))


@pytest.fixture(scope="module")
def arms() -> tuple[ConnectomePPOBrain, ConnectomePPOBrain]:
    """Build the plastic wild-type and plastic rewired-null brains at one seed."""
    return _brain(_WILD), _brain(_REWIRED)


class TestPeripheryIsIdentical:
    """Everything that is not wiring is bit-identical at one seed."""

    def test_readout_and_log_std(self, arms: tuple[ConnectomePPOBrain, ConnectomePPOBrain]) -> None:
        wild, rewired = arms
        assert torch.equal(wild.topology.readout, rewired.topology.readout)
        assert torch.equal(wild.topology.log_std, rewired.topology.log_std)

    @pytest.mark.parametrize("gain", _GAINS)
    def test_every_sensory_gain(
        self,
        arms: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
        gain: str,
    ) -> None:
        """The C3 cell enables all three projections; every gain must match."""
        wild, rewired = arms
        assert torch.equal(getattr(wild.topology, gain), getattr(rewired.topology, gain))

    def test_plasticity_and_trace_configuration(self) -> None:
        wild, rewired = _config(_WILD), _config(_REWIRED)
        for field in _PLASTICITY_FIELDS:
            assert getattr(wild, field) == getattr(rewired, field), field
        # Inherited, not restated: the null must be enforced in the forward too.
        assert rewired.chemical_mask_mode == "strict"


class TestWiringDiffersDegreesDoNot:
    def test_wiring_differs(self, arms: tuple[ConnectomePPOBrain, ConnectomePPOBrain]) -> None:
        wild, rewired = arms
        assert not torch.equal(wild.topology.m_chem, rewired.topology.m_chem)
        assert not torch.equal(wild.topology.w_chem, rewired.topology.w_chem)
        assert not torch.equal(wild.topology.g_gap, rewired.topology.g_gap)

    def test_degrees_and_edge_count_preserved(
        self,
        arms: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        wild, rewired = arms
        mw, mr = wild.topology.m_chem, rewired.topology.m_chem
        assert int(mw.sum()) == int(mr.sum())
        assert torch.equal(mw.sum(0), mr.sum(0))  # chemical in-degree
        assert torch.equal(mw.sum(1), mr.sum(1))  # chemical out-degree
        gw, gr = wild.topology.g_gap != 0, rewired.topology.g_gap != 0
        assert torch.equal(gw.sum(0), gr.sum(0))  # gap degree

    def test_initialisation_scale_is_preserved_but_energy_is_not_claimed(
        self,
        arms: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        """Scale follows in-degree and is equal. Per-neuron ENERGY is not asserted.

        The same sequence of normal draws lands on different pre/post pairs
        under rewiring, so each neuron's realised sum of squared incoming
        weights differs between the arms even though its initialisation
        scale, 1/sqrt(in-degree), is identical. A later reader who "fixes"
        this test by asserting energy equality would be asserting something
        false: with identical in-degree the two arms draw the same number of
        values at the same scale for each neuron, but not the same values,
        because the draw order follows the edge list and the edge list differs.
        """
        wild, rewired = arms
        in_degree_wild = wild.topology.m_chem.sum(0)
        in_degree_rewired = rewired.topology.m_chem.sum(0)
        assert torch.equal(in_degree_wild, in_degree_rewired)
        # The scale is a function of in-degree alone, so equal in-degree means
        # equal scale; nothing about realised energy is claimed here.


class TestPlasticityIsConfinedToTheNullWiring:
    def test_updates_and_traces_stay_on_the_rewired_edge_set(self) -> None:
        brain = _brain(_REWIRED)
        mask = brain.topology.m_chem
        before = brain.topology.w_chem.detach().clone()

        _drive(brain)

        changed = brain.topology.w_chem != before
        assert bool(changed.any())
        assert bool((changed <= mask).all())
        assert bool(((brain.topology.activity_traces != 0) <= mask).all())

    def test_null_arm_learns_and_freezes(self) -> None:
        learning = _brain(_REWIRED)
        before = learning.topology.w_chem.detach().clone()
        _drive(learning)
        assert not torch.equal(before, learning.topology.w_chem)

        frozen = _brain(_REWIRED, freeze_updates=True)
        before = frozen.topology.w_chem.detach().clone()
        _drive(frozen)
        assert torch.equal(before, frozen.topology.w_chem)


class TestPairingBySeed:
    def test_deterministic_at_one_seed_and_different_at_another(self) -> None:
        first = _brain(_REWIRED).topology.m_chem
        again = _brain(_REWIRED).topology.m_chem
        other = _brain(_REWIRED, seed=_OTHER_SEED).topology.m_chem
        assert torch.equal(first, again)
        assert not torch.equal(first, other)

    def test_rewire_seed_derives_from_the_run_seed(self) -> None:
        """Unset in the config, so seed k of each arm is a matched pair."""
        assert _config(_REWIRED).rewire_seed is None
