"""L.0's arms, and the four things that would make the contrast about something else.

The learner writes the 2x4 readout and nothing else, so the connectome enters as a **fixed feature
map** and rewiring changes that map. Four properties have to hold for the pair to differ by the
wiring alone, and each is a way the result could be wrong rather than absent:

* the feedback projection `B` must be **identical across wirings at a seed**. It is, by construction
  -- drawn from a generator seeded with the run seed, where the rewiring draws from a separate numpy
  one -- but that is a reading of the code, and the alternative is each wiring exploring through a
  different random projection, which confounds the wiring with the feedback path invisibly;
* the rewiring must preserve what the arms are matched on -- neuron set, per-post fan-in, edge
  count, motor pool -- since those set the weight-init scale, the strict mask's shape, the
  gap-junction normalisation and the readout's inputs;
* it must nonetheless actually rewire, or the null is the wild type under another name;
* the substrate must stay frozen, or this is not a fixed-features contrast at all.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
    _reject_unsupported_plasticity_modes,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "configs").is_dir():
    _root = _root.parent
_CONFIGS = _root / "configs" / "scenarios" / "foraging"
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
_LEARNING = f"{_STEM}_eprop_readout_only"
_FROZEN = f"{_STEM}_eprop_frozen"
_SEEDS = (1, 2, 7)


def _topology(name: str, seed: int) -> Any:
    simulation = load_simulation_config(str(_CONFIGS / f"{name}.yml"))
    assert simulation.brain is not None
    config = simulation.brain.config
    assert isinstance(config, ConnectomePPOBrainConfig)
    config.seed = seed
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU).topology


def _brain_keys(name: str) -> dict[str, Any]:
    return yaml.safe_load((_CONFIGS / f"{name}.yml").read_text())["brain"]["config"]


class TestTheArmsDifferByTheWiringAlone:
    @pytest.mark.parametrize("stem", [_LEARNING, _FROZEN])
    def test_the_rewired_config_differs_in_the_wiring_key_alone(self, stem: str) -> None:
        wild, rewired = _brain_keys(stem), _brain_keys(f"{stem}_rewired_null")
        differing = {k for k in set(wild) | set(rewired) if wild.get(k) != rewired.get(k)}
        assert differing == {"wiring"}
        assert rewired["wiring"] == "rewired_degree_preserving"

    def test_the_floors_differ_in_the_freeze_and_nothing_behavioural(self) -> None:
        # Three keys, not one, and the two extra are REQUIRED to differ: a frozen arm may not
        # declare a plastic readout, because "a plastic-readout floor" is not a thing -- with no
        # update no tensor moves. The next test asserts the guard that enforces it. So this is
        # R.2's shared floor, and it is behaviourally the right null for these arms.
        expected = {"freeze_updates", "plasticity_plastic_readout", "plasticity_plastic_tensors"}
        for learning, frozen in (
            (_LEARNING, _FROZEN),
            (f"{_LEARNING}_rewired_null", f"{_FROZEN}_rewired_null"),
        ):
            a, b = _brain_keys(learning), _brain_keys(frozen)
            differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
            assert differing == expected, f"{learning} vs {frozen}: {differing}"
            assert b["freeze_updates"] is True
            assert "plasticity_plastic_readout" not in b

    def test_a_frozen_arm_declaring_a_plastic_readout_is_refused(self) -> None:
        # The reason the floor cannot simply carry the learning arm's keys, asserted rather than
        # left as a comment: it would be reported as a plastic-readout floor, which does not exist.
        simulation = load_simulation_config(str(_CONFIGS / f"{_LEARNING}.yml"))
        assert simulation.brain is not None
        config = simulation.brain.config
        assert isinstance(config, ConnectomePPOBrainConfig)
        with pytest.raises(ValueError, match="no such arm"):
            _reject_unsupported_plasticity_modes(
                config.model_copy(update={"freeze_updates": True}),
            )

    def test_rewire_seed_is_unset_so_the_arms_pair(self) -> None:
        # Each seed's rewiring derives from its run seed, the discipline V.1 and V.3 ran under; a
        # pinned rewire_seed would give every seed the same null graph.
        for stem in (f"{_LEARNING}_rewired_null", f"{_FROZEN}_rewired_null"):
            assert "rewire_seed" not in _brain_keys(stem)


class TestTheProjectionIsMatchedAcrossWirings:
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_b_is_identical_at_a_seed(self, seed: int) -> None:
        wild = _topology(_LEARNING, seed).learning_signal_projection()
        rewired = _topology(f"{_LEARNING}_rewired_null", seed).learning_signal_projection()
        assert wild is not None
        assert rewired is not None
        assert torch.equal(wild, rewired)

    def test_b_still_differs_between_seeds(self) -> None:
        # Matched across wirings, not constant: a projection identical at every seed would make the
        # 32 runs one run repeated.
        first = _topology(_LEARNING, 1).learning_signal_projection()
        second = _topology(_LEARNING, 2).learning_signal_projection()
        assert first is not None
        assert second is not None
        assert not torch.equal(first, second)

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_the_readout_starts_identical(self, seed: int) -> None:
        # The anatomical contrast is derived from the motor classes, which rewiring preserves.
        wild = _topology(_LEARNING, seed)
        rewired = _topology(f"{_LEARNING}_rewired_null", seed)
        assert torch.equal(wild.readout.detach(), rewired.readout.detach())


class TestTheRewiringPreservesWhatTheArmsAreMatchedOn:
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_neuron_set_fan_in_and_edge_count(self, seed: int) -> None:
        wild = _topology(_LEARNING, seed)
        rewired = _topology(f"{_LEARNING}_rewired_null", seed)
        assert wild.n_neurons == rewired.n_neurons
        # Per-post fan-in sets the weight-init scale 1/sqrt(in-degree) and the gap-junction
        # normalisation; the motor pool is what the readout reads.
        assert torch.equal(wild.m_chem.sum(dim=0), rewired.m_chem.sum(dim=0))
        assert int(wild.m_chem.sum()) == int(rewired.m_chem.sum())
        assert torch.equal(wild._motor_flat_indices, rewired._motor_flat_indices)

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_it_nonetheless_rewires(self, seed: int) -> None:
        wild = _topology(_LEARNING, seed)
        rewired = _topology(f"{_LEARNING}_rewired_null", seed)
        assert not torch.equal(wild.m_chem, rewired.m_chem)


class TestTheSubstrateStaysFrozen:
    @pytest.mark.parametrize(
        "stem",
        [_LEARNING, f"{_LEARNING}_rewired_null"],
    )
    def test_only_the_readout_is_plastic(self, stem: str) -> None:
        # If `w_chem` were exposed this would not be a fixed-features contrast at all.
        topology = _topology(stem, 1)
        assert [tuple(w.shape) for w in topology.plastic_weights] == [(2, 4)]
        assert topology.plastic_homeostasis == [False]

    @pytest.mark.parametrize("stem", [_LEARNING, f"{_LEARNING}_rewired_null"])
    def test_the_operating_point_is_r2s(self, stem: str) -> None:
        keys = _brain_keys(stem)
        assert keys["plasticity_eligibility"] == "eprop"
        assert keys["plasticity_learning_signal"] == "random"
        assert keys["plasticity_plastic_readout"] is True
        assert keys["plasticity_plastic_tensors"] == "readout_only"
        assert keys["plasticity_node_noise"] == 0.0
        assert keys["plasticity_rate"] == 0.001
        assert keys["initial_log_std"] == -1.0
        assert keys["forward_pass_depth"] == 4
