"""Unit tests for the ConnectomePPOBrain thermotaxis-sensor projection.

Covers:

- Thermo features route to AFDL + AFDR (gain shape (3, 2))
- Bilateral broadcast convention (one column per L/R member)
- No thermo inputs → projection inactive
- Thermo gains independent of food gains (disjoint parameter tensors)
- Strict-mask invariant preserved
- ``learnable_parameters`` exposes the thermo matrix only when enabled
- Foraging-only configs allocate zero thermo parameters
- preprocess pack + ``_unpack_state`` round-trip (thermo-only and combined)
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import (
    _N_THERMOTAXIS_FEATURES,
    _SENSOR_NEURONS_THERMOTAXIS,
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.env.env import ContactZone

_SEED = 2026


def _make_thermo_brain(**cfg_overrides: object) -> ConnectomePPOBrain:
    """Construct a ConnectomePPOBrain with the thermotaxis projection enabled."""
    cfg = ConnectomePPOBrainConfig(
        seed=_SEED,
        sensing_mode="klinotaxis",
        enable_thermotaxis_projection=True,
        **cfg_overrides,  # type: ignore[arg-type]
    )
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _baseline_food_features() -> torch.Tensor:
    """Klinotaxis-mode food input [concentration, lateral, dC/dt]."""
    return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


def _raw_injection(
    brain: ConnectomePPOBrain,
    *,
    thermo: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the pre-recurrence injected hidden vector (302-dim).

    The recurrence loop overwrites injected sensory-neuron activations, so
    to spot-check injection targets we inspect ``h`` BEFORE the recurrence
    starts, reconstructing the ``forward_with_hidden`` pre-recurrence path
    via the same private ``_inject_thermotaxis`` helper.
    """
    food = _baseline_food_features()
    h = torch.zeros(brain.topology.n_neurons, dtype=torch.float32)
    food_inj = food @ brain.topology.food_gains
    h = h.index_add(0, brain.topology._food_neuron_indices, food_inj)
    if brain.topology.enable_thermotaxis_projection:
        h = brain.topology._inject_thermotaxis(h, thermo)
    return h


# ────────────────────────────────────────────────────────────────────────────
# Spec scenarios
# ────────────────────────────────────────────────────────────────────────────


class TestThermotaxisRouting:
    """Spec scenario: Thermo features route to AFDL + AFDR."""

    def test_gain_matrix_shape_is_3_by_2(self) -> None:
        brain = _make_thermo_brain()
        assert brain.topology.thermotaxis_gains.shape == (
            _N_THERMOTAXIS_FEATURES,
            len(_SENSOR_NEURONS_THERMOTAXIS),
        )

    def test_thermo_features_inject_only_into_afd(self) -> None:
        brain = _make_thermo_brain(forward_pass_depth=1)
        thermo = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)

        h_with = _raw_injection(brain, thermo=thermo)
        h_without = _raw_injection(brain)

        for name in _SENSOR_NEURONS_THERMOTAXIS:
            idx = brain.topology._idx[name]
            assert h_with[idx] != h_without[idx], f"Thermo injection should affect {name}."

    def test_thermo_does_not_affect_food_targets(self) -> None:
        brain = _make_thermo_brain(forward_pass_depth=1)
        thermo = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)

        h_with = _raw_injection(brain, thermo=thermo)
        h_without = _raw_injection(brain)

        # Food sensory neurons (ASE/AWC/AWA) must be untouched by thermo
        # injection (AFD is disjoint from the food target set).
        for name in ("ASEL", "ASER", "AWCL", "AWCR", "AWAL", "AWAR"):
            idx = brain.topology._idx[name]
            assert h_with[idx] == h_without[idx], (
                f"Thermo injection should NOT reach food target {name}."
            )


class TestBilateralBroadcastConvention:
    """Spec scenario: Bilateral broadcast convention (thermotaxis)."""

    def test_afdl_and_afdr_are_separate_columns(self) -> None:
        brain = _make_thermo_brain()
        # 2 columns, one per AFD member.
        assert brain.topology.thermotaxis_gains.shape[1] == 2
        # AFDL and AFDR map to distinct neuron indices.
        afdl = brain.topology._idx["AFDL"]
        afdr = brain.topology._idx["AFDR"]
        assert afdl != afdr
        assert brain.topology._thermotaxis_neuron_indices.tolist() == [afdl, afdr]


class TestNoThermoInputsLeavesProjectionInactive:
    """Spec scenario: No thermo inputs → projection inactive."""

    def test_none_thermo_is_byte_identical_to_food_only(self) -> None:
        brain = _make_thermo_brain(forward_pass_depth=4)
        food = _baseline_food_features()

        _, hidden_no_thermo = brain.topology.forward_with_hidden(food)
        _, hidden_explicit_none = brain.topology.forward_with_hidden(
            food,
            thermotaxis_features=None,
        )
        assert torch.equal(hidden_no_thermo, hidden_explicit_none)

    def test_zero_filled_thermo_matches_no_thermo(self) -> None:
        """The real preprocess pipeline emits zero-filled thermo when temperature is None."""
        brain = _make_thermo_brain(forward_pass_depth=4)
        food = _baseline_food_features()
        zero_thermo = torch.zeros(_N_THERMOTAXIS_FEATURES, dtype=torch.float32)

        _, hidden_zero = brain.topology.forward_with_hidden(food, thermotaxis_features=zero_thermo)
        _, hidden_none = brain.topology.forward_with_hidden(food)
        assert torch.equal(hidden_zero, hidden_none), (
            "Zero-filled thermo features should produce byte-identical activations "
            "to the no-thermo path; any drift indicates the thermo path is silently "
            "mutating the baseline."
        )

    def test_temperature_none_yields_zero_thermo_block(self) -> None:
        brain = _make_thermo_brain()
        params = BrainParams(food_concentration=0.5, lateral_scale=1.0, derivative_scale=1.0)
        state = brain.preprocess(params)
        # 3 food + 3 thermo = 6; thermo block is all zeros.
        assert state.shape == (6,)
        assert np.allclose(state[3:], 0.0)


class TestThermoAndFoodGainsIndependent:
    """Spec scenario: thermo projection leaves food + chemical-synapse weights untouched."""

    def test_thermo_gains_disjoint_from_food_gains(self) -> None:
        brain = _make_thermo_brain()
        assert brain.topology.food_gains.data_ptr() != brain.topology.thermotaxis_gains.data_ptr()

    def test_strict_mask_holds_after_thermo_construction(self) -> None:
        brain = _make_thermo_brain(chemical_mask_mode="strict")
        masked_off = brain.topology.w_chem.data * (~brain.topology.m_chem)
        assert torch.all(masked_off == 0)

    def test_gradient_flows_to_thermo_gains(self) -> None:
        """End-to-end: a backward pass populates ``thermotaxis_gains.grad``.

        Proves the differentiable chain thermo features → AFD injection →
        K-step recurrence → motor readout → logits actually carries gradient
        back to the thermo gain matrix (not just that the matrix is a
        registered ``nn.Parameter``). Uses the canonical K=4 depth so the
        AFD signal reaches the motor neurons through the connectome.
        """
        brain = _make_thermo_brain(forward_pass_depth=4)
        food = _baseline_food_features()
        thermo = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float32)

        logits = brain.topology.forward(food, thermotaxis_features=thermo)
        logits.sum().backward()

        grad = brain.topology.thermotaxis_gains.grad
        assert grad is not None
        assert grad.shape == (_N_THERMOTAXIS_FEATURES, 2)
        assert grad.abs().sum().item() > 0.0, (
            "Gradient must reach thermotaxis_gains through the AFD → motor "
            "pathway; an all-zero gradient means the thermo injection is "
            "disconnected from the action logits."
        )


class TestLearnableParametersExposesThermoGains:
    """Spec scenario: thermo gains are PPO-learnable separately from the other projections."""

    def test_learnable_parameters_includes_thermo_matrix_when_enabled(self) -> None:
        brain = _make_thermo_brain()
        params = brain.topology.learnable_parameters
        # Four tensors: w_chem, food_gains, readout, thermotaxis_gains.
        assert len(params) == 4
        assert id(brain.topology.thermotaxis_gains) in {id(p) for p in params}

    def test_learnable_parameters_omits_thermo_matrix_when_disabled(self) -> None:
        cfg = ConnectomePPOBrainConfig(
            seed=_SEED,
            sensing_mode="klinotaxis",
            enable_thermotaxis_projection=False,
        )
        brain = ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)
        assert len(brain.topology.learnable_parameters) == 3
        assert not hasattr(brain.topology, "thermotaxis_gains")


class TestForagingOnlyByteIdentity:
    """Spec scenario: foraging-only configs preserve the pre-projection food path."""

    def test_thermo_off_topology_has_no_thermo_attributes(self) -> None:
        cfg = ConnectomePPOBrainConfig(
            seed=_SEED,
            sensing_mode="klinotaxis",
            enable_thermotaxis_projection=False,
        )
        brain = ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)
        assert not hasattr(brain.topology, "thermotaxis_gains")
        assert not hasattr(brain.topology, "_thermotaxis_neuron_indices")

    def test_thermo_off_preprocess_returns_food_only_state(self) -> None:
        cfg = ConnectomePPOBrainConfig(
            seed=_SEED,
            sensing_mode="klinotaxis",
            enable_thermotaxis_projection=False,
        )
        brain = ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)
        params = BrainParams(food_concentration=0.5, lateral_scale=1.0, derivative_scale=1.0)
        state = brain.preprocess(params)
        assert state.shape == (3,)


class TestPreprocessPacksThermoSlotsCorrectly:
    """The state-vector layout matches ``_unpack_state``'s expectations."""

    def test_preprocess_packs_food_then_thermo(self) -> None:
        brain = _make_thermo_brain()
        params = BrainParams(
            food_concentration=0.5,
            food_lateral_gradient=0.1,
            food_dconcentration_dt=0.05,
            lateral_scale=1.0,
            derivative_scale=1.0,
            temperature=27.5,
            cultivation_temperature=20.0,
            temperature_lateral_gradient=0.3,
            temperature_ddt=0.1,
        )
        state = brain.preprocess(params)
        # 3 food + 3 thermo = 6.
        assert state.shape == (6,)
        state_t = torch.from_numpy(state)
        _food, distal, mechano, zone, thermo = brain._unpack_state(state_t)
        assert distal is None
        assert mechano is None
        assert zone is None
        assert thermo is not None
        assert thermo.shape == (3,)
        # temp_deviation = (27.5 - 20) / 15 = 0.5.
        assert thermo[0].item() == pytest.approx(0.5)

    def test_combined_predator_and_thermo_layout(self) -> None:
        """Predator + thermo enabled together: [food(3) + predator(8) + thermo(3)] = 14."""
        cfg = ConnectomePPOBrainConfig(
            seed=_SEED,
            sensing_mode="klinotaxis",
            enable_predator_projection=True,
            enable_thermotaxis_projection=True,
        )
        brain = ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)
        params = BrainParams(
            food_concentration=0.5,
            lateral_scale=1.0,
            derivative_scale=1.0,
            predator_distal_concentration=0.7,
            predator_contact_intensity=0.4,
            predator_contact_zone=ContactZone.ANTERIOR,
            temperature=27.5,
            cultivation_temperature=20.0,
        )
        state = brain.preprocess(params)
        assert state.shape == (14,)
        state_t = torch.from_numpy(state)
        food, distal, mechano, zone, thermo = brain._unpack_state(state_t)
        assert food.shape == (3,)
        assert distal is not None
        assert distal.shape == (2,)
        assert mechano is not None
        assert zone == ContactZone.ANTERIOR
        assert thermo is not None
        assert thermo.shape == (3,)
        assert thermo[0].item() == pytest.approx(0.5)
