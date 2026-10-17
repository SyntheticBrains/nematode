"""Ohmic gap-junction coupling in the settling map.

Covers the connectome-ppo-brain requirement "Ohmic gap-junction coupling in the settling map":
the operator is the gap Laplacian's implicit step; the gap coupling adds no gain; sensors relay
across gap junctions; single and batched settling agree; the default is byte-identical.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[6]
_T35 = (
    _REPO
    / "configs/scenarios/thermal_foraging"
    / "connectomeppo_small_continuous2d_thermal_klinotaxis_t35.yml"
)
_SEED = 11


def _config(**overrides: object) -> ConnectomePPOBrainConfig:
    container = load_simulation_config(str(_T35)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    return container.config.model_copy(
        update={"seed": _SEED, "connectome_source": "emmons_2024_hermaphrodite", **overrides},
    )


def _brain(**overrides: object) -> ConnectomePPOBrain:
    torch.manual_seed(0)
    return ConnectomePPOBrain(config=_config(**overrides), device=DeviceType.CPU)


def _inputs(batch: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(3)
    food = torch.tensor(rng.uniform(-1, 1, (batch, 3)), dtype=torch.float32)
    thermal = torch.tensor(rng.uniform(-1, 1, (batch, 3)), dtype=torch.float32)
    return food, thermal


@pytest.fixture(scope="module")
def ohmic() -> ConnectomePPOBrain:
    """Build the ohmic brain once for the module."""
    return _brain(gap_coupling="ohmic")


class TestOperator:
    def test_it_is_the_gap_laplacian_s_implicit_step(self, ohmic: ConnectomePPOBrain) -> None:
        """Scenario "The operator is the gap Laplacian's implicit step"."""
        gap = ohmic.topology.g_gap.double()
        expected = torch.linalg.inv(
            torch.eye(gap.shape[0], dtype=torch.float64) + torch.diag(gap.sum(dim=1)) - gap,
        )
        operator = ohmic.topology.ohmic_inverse.double()
        assert torch.allclose(operator, expected, atol=1e-6)
        assert torch.allclose(operator.sum(dim=1), torch.ones(gap.shape[0], dtype=torch.float64))
        assert torch.allclose(operator, operator.T, atol=1e-6)
        assert bool((operator > -1e-7).all())

    def test_it_is_not_saved_with_the_weights(self, ohmic: ConnectomePPOBrain) -> None:
        assert "ohmic_inverse" not in ohmic.topology.state_dict()
        assert set(ohmic.topology.state_dict()) == set(_brain().topology.state_dict())


class TestSettling:
    def test_the_gap_coupling_adds_no_gain(self) -> None:
        """Scenario "The gap coupling adds no gain".

        With every chemical weight zeroed, nothing but the gap coupling can carry activity past the
        first iteration: additively the injected sensors keep reverberating through their gap
        partners; ohmically the settled state is exactly zero, since an average of no drive is none.
        """
        food, thermal = _inputs()
        settled = {}
        for coupling in ("additive", "ohmic"):
            topology = _brain(gap_coupling=coupling).topology
            with torch.no_grad():
                topology.w_chem.zero_()
                _, settled[coupling] = topology.forward_with_hidden_batched(
                    food,
                    thermotaxis_features=thermal,
                )
        assert float(settled["additive"].abs().max()) > 0.1
        assert float(settled["ohmic"].abs().max()) == 0.0

    def test_sensors_relay_across_gap_junctions(self) -> None:
        """Scenario "Sensors relay across gap junctions": the state starts at ``M^-1 s``."""
        topology = _brain(gap_coupling="ohmic", forward_pass_depth=1).topology
        food, thermal = _inputs()
        topology.capture_penultimate = True
        with torch.no_grad():
            topology.forward_with_hidden_batched(food, thermotaxis_features=thermal)
            sensors = topology._sensor_current_batched(food, None, None, None, thermal)
        assert topology.penultimate is not None
        assert torch.allclose(topology.penultimate, sensors @ topology.ohmic_inverse, atol=1e-6)
        injected = sensors.abs().sum(dim=0) > 0
        relayed = topology.penultimate.abs().sum(dim=0) > 0
        assert int(relayed.sum()) > int(injected.sum())

    def test_single_and_batched_agree(self, ohmic: ConnectomePPOBrain) -> None:
        """Scenario "Single and batched settling agree"."""
        food, thermal = _inputs()
        topology = ohmic.topology
        with torch.no_grad():
            _, batched = topology.forward_with_hidden_batched(food, thermotaxis_features=thermal)
            for i in range(food.shape[0]):
                _, single = topology.forward_with_hidden(food[i], thermotaxis_features=thermal[i])
                assert torch.allclose(single, batched[i], atol=1e-5)

    def test_ohmic_differs_from_additive(self, ohmic: ConnectomePPOBrain) -> None:
        food, thermal = _inputs()
        with torch.no_grad():
            _, a = _brain().topology.forward_with_hidden_batched(food, thermotaxis_features=thermal)
            _, o = ohmic.topology.forward_with_hidden_batched(food, thermotaxis_features=thermal)
        assert not torch.allclose(a, o)


def test_ppo_learns_under_ohmic_coupling() -> None:
    """The update runs through the ohmic settling, and the chemical weights receive a gradient."""
    brain = _brain(gap_coupling="ohmic", rollout_buffer_size=16, num_minibatches=2, num_epochs=2)
    before = brain.topology.w_chem.detach().clone()
    torch.manual_seed(0)
    for step in range(24):
        params = BrainParams(
            temperature=20.0 + step % 4,
            temperature_lateral_gradient=0.1,
            temperature_ddt=0.01,
            cultivation_temperature=20.0,
        )
        brain.run_brain(params, reward=None, input_data=None, top_only=False, top_randomize=False)
        brain.learn(params, reward=0.1 * (step % 3), episode_done=step % 11 == 10)
    assert not torch.equal(before, brain.topology.w_chem.detach())
    assert bool(torch.isfinite(brain.topology.w_chem).all())


class TestCheckpoints:
    def test_weights_from_the_other_coupling_are_refused(self, ohmic: ConnectomePPOBrain) -> None:
        with pytest.raises(ValueError, match="gap_coupling='additive'"):
            ohmic.load_weight_components(_brain().get_weight_components())
        with pytest.raises(ValueError, match="gap_coupling='ohmic'"):
            _brain().load_weight_components(ohmic.get_weight_components())

    def test_a_file_silent_about_coupling_is_additive(self, ohmic: ConnectomePPOBrain) -> None:
        components = _brain().get_weight_components()
        state = dict(components["training_state"].state)
        state.pop("gap_coupling")
        components["training_state"] = type(components["training_state"])(
            name="training_state",
            state=state,
        )
        _brain().load_weight_components(components)
        with pytest.raises(ValueError, match="gap_coupling"):
            ohmic.load_weight_components(components)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"enable_gap_junctions": False}, "no gap junctions"),
        ({"dynamics": "leaky", "input_gain": 512.0}, "already couples"),
        ({"plastic_gaps": True}, "plastic_gaps"),
        ({"learning_rule": "three_factor", "enable_activity_traces": True}, "not PPO"),
        ({"plasticity_node_noise": 0.1}, "node perturbation"),
        ({"enable_activity_traces": True}, "activity traces"),
        ({"gap_coupling": "resistive"}, "Unsupported gap_coupling"),
    ],
)
def test_unsupported_combinations_are_refused(overrides: dict[str, Any], match: str) -> None:
    """Refused by the build, which repeats the validator's checks for copied configs."""
    with pytest.raises(ValueError, match=match):
        _brain(**{"gap_coupling": "ohmic", **overrides})


def test_the_validator_refuses_too() -> None:
    """A config written with ohmic coupling and no gap junctions never validates."""
    with pytest.raises(ValueError, match="no gap junctions"):
        ConnectomePPOBrainConfig(gap_coupling="ohmic", enable_gap_junctions=False)


def test_the_default_is_byte_identical() -> None:
    """Scenario "The default is byte-identical": explicit additive is the implicit build."""
    implicit, explicit = _brain(), _brain(gap_coupling="additive")
    for name, value in implicit.topology.state_dict().items():
        assert torch.equal(value, explicit.topology.state_dict()[name]), name
    food, thermal = _inputs()
    with torch.no_grad():
        a = implicit.topology.forward_with_hidden_batched(food, thermotaxis_features=thermal)[1]
        b = explicit.topology.forward_with_hidden_batched(food, thermotaxis_features=thermal)[1]
    assert torch.equal(a, b)
    assert not hasattr(implicit.topology, "ohmic_inverse")


def test_the_modulator_current_passes_through_the_operator() -> None:
    """The held current enters each iteration's drive, so ohmic coupling averages it too."""
    brain = _brain(gap_coupling="ohmic", modulators=True)
    topology = brain.topology
    food, thermal = _inputs(1)
    with torch.no_grad():
        topology.w_chem.zero_()
        modulators = torch.tensor([[1.0, 0.5, 0.2]])
        _, settled = topology.forward_with_hidden_batched(
            food,
            thermotaxis_features=thermal,
            modulators=modulators,
        )
        held = topology._held_current(modulators)
    assert held is not None
    assert torch.allclose(settled, torch.tanh(held @ topology.ohmic_inverse), atol=1e-6)
