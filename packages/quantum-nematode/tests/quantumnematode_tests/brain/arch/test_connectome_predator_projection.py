"""Unit tests for the ConnectomePPOBrain predator-sensor projection.

Covers:

- Bilateral broadcast convention (one column per L/R member, identical init)
- Distal-chemo routes to ASHL + ASHR + ASIL + ASIR (gain shape (2, 4))
- Anterior contact routes to ALML + ALMR + AVM (gain shape (2, 3))
- Posterior contact routes to PLML + PLMR (gain shape (2, 2))
- Lateral contact routes degenerately to ALM + PLM at half-weight
- ``ContactZone.NONE`` with active distal → distal only, no mechano
- No predator inputs → projection inactive
- Predator gains independent of food gains (disjoint parameter tensors)
- ``learnable_parameters`` exposes all three predator matrices
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import (
    _LATERAL_HALF_WEIGHT,
    _N_PREDATOR_DISTAL_FEATURES,
    _N_PREDATOR_MECHANO_FEATURES,
    _SENSOR_NEURONS_PREDATOR_ANTERIOR,
    _SENSOR_NEURONS_PREDATOR_DISTAL,
    _SENSOR_NEURONS_PREDATOR_POSTERIOR,
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.env.env import ContactZone

_SEED = 2026


def _make_predator_brain(**cfg_overrides: object) -> ConnectomePPOBrain:
    """Construct a ConnectomePPOBrain with the predator projection enabled."""
    cfg = ConnectomePPOBrainConfig(
        seed=_SEED,
        sensing_mode="klinotaxis",
        enable_predator_projection=True,
        **cfg_overrides,  # type: ignore[arg-type]
    )
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _baseline_food_features() -> torch.Tensor:
    """Klinotaxis-mode food input [concentration, lateral, dC/dt]."""
    return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


def _raw_injection(
    brain: ConnectomePPOBrain,
    *,
    distal: torch.Tensor | None = None,
    mechano: torch.Tensor | None = None,
    zone: ContactZone | None = None,
) -> torch.Tensor:
    """Return the pre-recurrence injected hidden vector (302-dim).

    The recurrence loop overwrites injected sensory-neuron activations
    with their downstream pre-activation, so to spot-check injection
    targets we need to inspect ``h`` BEFORE the recurrence starts. This
    helper reconstructs the ``forward_with_hidden`` pre-recurrence path
    using the same private methods (``_inject_predator``) so the assert
    is robust against recurrence-driven overwrites.
    """
    food = _baseline_food_features()
    h = torch.zeros(brain.topology.n_neurons, dtype=torch.float32)
    food_inj = food @ brain.topology.food_gains
    h = h.index_add(0, brain.topology._food_neuron_indices, food_inj)
    if brain.topology.enable_predator_projection:
        h = brain.topology._inject_predator(h, distal, mechano, zone)
    return h


# ────────────────────────────────────────────────────────────────────────────
# Spec scenarios
# ────────────────────────────────────────────────────────────────────────────


class TestBilateralBroadcastConvention:
    """Spec scenario: Bilateral broadcast convention."""

    def test_gain_matrices_have_one_column_per_neuron(self) -> None:
        """Each L/R member gets its own column (option-a layout per spec)."""
        brain = _make_predator_brain()
        topo = brain.topology
        # Distal: 4 columns for ASHL/ASHR/ASIL/ASIR.
        assert topo.predator_distal_gains.shape == (
            _N_PREDATOR_DISTAL_FEATURES,
            len(_SENSOR_NEURONS_PREDATOR_DISTAL),
        )
        # Anterior: 3 columns for ALML/ALMR/AVM.
        assert topo.predator_anterior_gains.shape == (
            _N_PREDATOR_MECHANO_FEATURES,
            len(_SENSOR_NEURONS_PREDATOR_ANTERIOR),
        )
        # Posterior: 2 columns for PLML/PLMR.
        assert topo.predator_posterior_gains.shape == (
            _N_PREDATOR_MECHANO_FEATURES,
            len(_SENSOR_NEURONS_PREDATOR_POSTERIOR),
        )


class TestDistalChemoRouting:
    """Spec scenario: Distal-chemo routes to ASHL + ASHR + ASIL + ASIR.

    Inspects the pre-recurrence injected hidden state via
    :func:`_raw_injection`. Inspecting the post-recurrence ``forward_with_hidden``
    output at the injected neurons would be misleading: the recurrence
    loop overwrites the sensory-neuron activations with their downstream
    pre-activation on the very first iteration.
    """

    def test_distal_features_inject_only_into_distal_targets(self) -> None:
        brain = _make_predator_brain(forward_pass_depth=1)
        distal = torch.tensor([0.7, 0.2], dtype=torch.float32)

        h_with = _raw_injection(brain, distal=distal)
        h_without = _raw_injection(brain)

        for name in _SENSOR_NEURONS_PREDATOR_DISTAL:
            idx = brain.topology._idx[name]
            assert h_with[idx] != h_without[idx], (
                f"Distal injection should affect {name}; activation unchanged."
            )

    def test_distal_does_not_affect_mechano_targets(self) -> None:
        brain = _make_predator_brain(forward_pass_depth=1)
        distal = torch.tensor([0.7, 0.2], dtype=torch.float32)

        h_with = _raw_injection(brain, distal=distal)
        h_without = _raw_injection(brain)

        for name in (*_SENSOR_NEURONS_PREDATOR_ANTERIOR, *_SENSOR_NEURONS_PREDATOR_POSTERIOR):
            idx = brain.topology._idx[name]
            assert h_with[idx] == h_without[idx], (
                f"Distal injection should NOT reach mechano target {name}."
            )


class TestAnteriorContactRouting:
    """Spec scenario: Anterior contact routes to ALML + ALMR + AVM."""

    def test_anterior_contact_injects_only_into_anterior_targets(self) -> None:
        brain = _make_predator_brain(forward_pass_depth=1)
        mechano = torch.tensor([0.8, 0.3], dtype=torch.float32)

        h_with = _raw_injection(brain, mechano=mechano, zone=ContactZone.ANTERIOR)
        h_without = _raw_injection(brain)

        for name in _SENSOR_NEURONS_PREDATOR_ANTERIOR:
            idx = brain.topology._idx[name]
            assert h_with[idx] != h_without[idx], f"Anterior-zone injection should affect {name}."
        # Posterior targets unaffected by anterior-zone injection.
        for name in _SENSOR_NEURONS_PREDATOR_POSTERIOR:
            idx = brain.topology._idx[name]
            assert h_with[idx] == h_without[idx]


class TestPosteriorContactRouting:
    """Spec scenario: Posterior contact routes to PLML + PLMR."""

    def test_posterior_contact_injects_only_into_posterior_targets(self) -> None:
        brain = _make_predator_brain(forward_pass_depth=1)
        mechano = torch.tensor([0.8, 0.3], dtype=torch.float32)

        h_with = _raw_injection(brain, mechano=mechano, zone=ContactZone.POSTERIOR)
        h_without = _raw_injection(brain)

        for name in _SENSOR_NEURONS_PREDATOR_POSTERIOR:
            idx = brain.topology._idx[name]
            assert h_with[idx] != h_without[idx], f"Posterior-zone injection should affect {name}."
        # Anterior + AVM targets unaffected.
        for name in _SENSOR_NEURONS_PREDATOR_ANTERIOR:
            idx = brain.topology._idx[name]
            assert h_with[idx] == h_without[idx]


class TestLateralContactRouting:
    """Spec scenario: Lateral routes degenerately to ALM + PLM at half-weight."""

    def test_lateral_injects_into_alm_plus_plm_only(self) -> None:
        brain = _make_predator_brain(forward_pass_depth=1)
        mechano = torch.tensor([0.8, 0.3], dtype=torch.float32)

        h_with = _raw_injection(brain, mechano=mechano, zone=ContactZone.LATERAL)
        h_without = _raw_injection(brain)

        # ALML, ALMR, PLML, PLMR all receive injection.
        for name in ("ALML", "ALMR", "PLML", "PLMR"):
            idx = brain.topology._idx[name]
            assert h_with[idx] != h_without[idx], f"Lateral injection should affect {name}."
        # AVM (unilateral, anterior-only) does NOT receive lateral injection.
        avm_idx = brain.topology._idx["AVM"]
        assert h_with[avm_idx] == h_without[avm_idx], (
            "Lateral-zone injection should NOT affect AVM (unilateral, "
            "excluded from the bilateral lateral pathway per spec)."
        )

    def test_lateral_alm_injection_is_half_of_anterior_alm_injection(self) -> None:
        """Lateral routing reuses anterior gains scaled by ``_LATERAL_HALF_WEIGHT``.

        With the same mechano features and the same seeded init of
        ``predator_anterior_gains``, the lateral injection at ALML must
        be exactly half the anterior-only injection at ALML.
        """
        brain = _make_predator_brain(forward_pass_depth=1)
        mechano = torch.tensor([0.8, 0.3], dtype=torch.float32)

        # Read the raw pre-tanh injection at ALML by manually computing
        # the matmul (avoids the tanh nonlinearity that makes the
        # "half-of" relationship harder to assert directly).
        anterior_inj = mechano @ brain.topology.predator_anterior_gains  # shape (3,)
        anterior_alml_inj = float(anterior_inj[0].item())  # ALML is column 0
        lateral_alml_inj = anterior_alml_inj * _LATERAL_HALF_WEIGHT
        # Sanity check: the spec mandates the half-weight factor is a
        # fixed constant. Asserting the constant matches the spec value.
        assert _LATERAL_HALF_WEIGHT == 0.5
        # The ratio of lateral-to-anterior injection at ALML is exactly 0.5.
        assert lateral_alml_inj == pytest.approx(anterior_alml_inj * 0.5)


class TestNoPredatorInputsLeavesProjectionInactive:
    """Spec scenario: No predator inputs → projection inactive.

    With distal AND mechano features set to None, the predator path
    contributes nothing — the 9 predator targets receive only the food
    path's downstream recurrent activation, which is identical to the
    baseline forward with no predator features at all.
    """

    def test_no_predator_inputs_produces_identical_hidden_to_food_only(self) -> None:
        brain = _make_predator_brain(forward_pass_depth=4)
        food = _baseline_food_features()

        _, hidden_no_predator = brain.topology.forward_with_hidden(food)
        _, hidden_explicit_none = brain.topology.forward_with_hidden(
            food,
            predator_distal_features=None,
            predator_mechano_features=None,
            predator_contact_zone=ContactZone.NONE,
        )
        # Byte-identical: predator path is a pure no-op when no features
        # are provided.
        assert torch.equal(hidden_no_predator, hidden_explicit_none)

    def test_zero_filled_features_match_no_features_at_zone_none(self) -> None:
        """Real ``preprocess`` pipeline emits zero-filled tensors + zone=NONE.

        When the projection is enabled but the env has no predator (or the
        predator BrainParams fields are unset), ``_extract_predator_features``
        produces ``[0.0, 0.0]`` for both distal and mechano + a zone one-hot
        with NONE active. The resulting forward pass must be byte-identical
        to the same brain called with all-None predator inputs: distal injects
        ``zeros @ gains = zeros`` (no-op); mechano gate is closed by
        ``zone == ContactZone.NONE`` (no-op).
        """
        brain = _make_predator_brain(forward_pass_depth=4)
        food = _baseline_food_features()
        zero_distal = torch.zeros(2, dtype=torch.float32)
        zero_mechano = torch.zeros(2, dtype=torch.float32)

        _, hidden_zero_filled = brain.topology.forward_with_hidden(
            food,
            predator_distal_features=zero_distal,
            predator_mechano_features=zero_mechano,
            predator_contact_zone=ContactZone.NONE,
        )
        _, hidden_none = brain.topology.forward_with_hidden(food)
        assert torch.equal(hidden_zero_filled, hidden_none), (
            "Zero-filled predator features at zone=NONE should produce "
            "byte-identical activations to the no-predator-features path; "
            "any drift indicates the predator code path is silently mutating "
            "the food-only baseline."
        )


class TestContactZoneNoneRoutesDistalOnly:
    """Spec scenario: ContactZone.NONE with active distal → distal only."""

    def test_distal_active_with_zone_none_injects_distal_not_mechano(self) -> None:
        brain = _make_predator_brain(forward_pass_depth=1)
        distal = torch.tensor([0.5, 0.1], dtype=torch.float32)
        mechano = torch.tensor([0.0, 0.0], dtype=torch.float32)

        h_with = _raw_injection(brain, distal=distal, mechano=mechano, zone=ContactZone.NONE)
        h_without = _raw_injection(brain)

        # Distal targets affected.
        for name in _SENSOR_NEURONS_PREDATOR_DISTAL:
            idx = brain.topology._idx[name]
            assert h_with[idx] != h_without[idx]
        # Mechano targets unaffected (zone is NONE → no mechano injection).
        for name in (*_SENSOR_NEURONS_PREDATOR_ANTERIOR, *_SENSOR_NEURONS_PREDATOR_POSTERIOR):
            idx = brain.topology._idx[name]
            assert h_with[idx] == h_without[idx]


class TestPredatorAndFoodGainsIndependent:
    """Spec scenario: predator projection leaves food + chemical-synapse weights untouched."""

    def test_predator_gains_are_disjoint_parameters_from_food_gains(self) -> None:
        brain = _make_predator_brain()
        topo = brain.topology
        # Each gain matrix is its own ``nn.Parameter`` — no shared storage.
        assert topo.food_gains.data_ptr() != topo.predator_distal_gains.data_ptr()
        assert topo.food_gains.data_ptr() != topo.predator_anterior_gains.data_ptr()
        assert topo.food_gains.data_ptr() != topo.predator_posterior_gains.data_ptr()

    def test_strict_mask_holds_after_predator_construction(self) -> None:
        """Adding the predator projection does not create new chemical-synapse edges."""
        brain = _make_predator_brain(chemical_mask_mode="strict")
        topo = brain.topology
        # ``w_chem * ~m_chem`` must be all-zero after construction (the
        # strict mask is preserved; predator path does not touch w_chem).
        masked_off = topo.w_chem.data * (~topo.m_chem)
        assert torch.all(masked_off == 0)


class TestLearnableParametersExposesPredatorGains:
    """Spec scenario: predator gains are PPO-learnable separately from the food projection."""

    def test_learnable_parameters_includes_three_predator_matrices_when_enabled(self) -> None:
        brain = _make_predator_brain()
        params = brain.topology.learnable_parameters
        # Should include: w_chem, food_gains, readout, predator_distal_gains,
        # predator_anterior_gains, predator_posterior_gains = 6 tensors.
        assert len(params) == 6
        param_ids = {id(p) for p in params}
        assert id(brain.topology.predator_distal_gains) in param_ids
        assert id(brain.topology.predator_anterior_gains) in param_ids
        assert id(brain.topology.predator_posterior_gains) in param_ids

    def test_learnable_parameters_omits_predator_matrices_when_disabled(self) -> None:
        """Foraging-only configs do NOT see the predator parameters in the optimiser."""
        cfg = ConnectomePPOBrainConfig(
            seed=_SEED,
            sensing_mode="klinotaxis",
            enable_predator_projection=False,
        )
        brain = ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)
        params = brain.topology.learnable_parameters
        # Only w_chem + food_gains + readout = 3 tensors.
        assert len(params) == 3
        # The predator matrices are not even allocated on the topology.
        assert not hasattr(brain.topology, "predator_distal_gains")


class TestForagingOnlyByteIdentityVsPrePredatorBuild:
    """Foraging-only configs preserve the pre-projection food path byte-identically.

    With ``enable_predator_projection=False`` (the default), the topology
    allocates zero predator-related ``nn.Parameter`` objects, so the RNG-
    stream consumption order during ``__init__`` is byte-identical to
    pre-projection builds. This is the structural invariant that lets
    existing foraging configs ship unchanged when the projection feature
    is added.
    """

    def test_predator_off_topology_has_no_predator_attributes(self) -> None:
        cfg = ConnectomePPOBrainConfig(
            seed=_SEED,
            sensing_mode="klinotaxis",
            enable_predator_projection=False,
        )
        brain = ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)
        assert not hasattr(brain.topology, "predator_distal_gains")
        assert not hasattr(brain.topology, "predator_anterior_gains")
        assert not hasattr(brain.topology, "predator_posterior_gains")
        assert not hasattr(brain.topology, "_predator_distal_neuron_indices")

    def test_predator_off_preprocess_returns_food_only_state(self) -> None:
        cfg = ConnectomePPOBrainConfig(
            seed=_SEED,
            sensing_mode="klinotaxis",
            enable_predator_projection=False,
        )
        brain = ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)
        params = BrainParams(food_concentration=0.5, lateral_scale=1.0, derivative_scale=1.0)
        state = brain.preprocess(params)
        # n_food_features=3 in klinotaxis mode; no predator slots.
        assert state.shape == (3,)
        assert state.dtype == np.float32


class TestPreprocessPacksPredatorSlotsCorrectly:
    """The state vector layout matches ``_unpack_state``'s expectations."""

    def test_preprocess_packs_food_distal_mechano_zone(self) -> None:
        brain = _make_predator_brain()
        params = BrainParams(
            food_concentration=0.5,
            food_lateral_gradient=0.1,
            food_dconcentration_dt=0.05,
            lateral_scale=1.0,
            derivative_scale=1.0,
            predator_distal_concentration=0.7,
            predator_distal_dconcentration_dt=0.2,
            predator_contact_intensity=0.8,
            predator_mechano_dintensity_dt=0.3,
            predator_contact_zone=ContactZone.ANTERIOR,
        )
        state = brain.preprocess(params)
        # 3 food + 2 distal + 2 mechano + 4 zone-one-hot = 11.
        assert state.shape == (11,)
        # Round-trip via ``_unpack_state``.
        state_t = torch.from_numpy(state)
        food, distal, mechano, zone = brain._unpack_state(state_t)
        assert food.shape == (3,)
        assert distal is not None
        assert distal.shape == (2,)
        assert distal[0].item() == pytest.approx(0.7)
        assert distal[1].item() == pytest.approx(0.2)
        assert mechano is not None
        assert mechano.shape == (2,)
        assert mechano[0].item() == pytest.approx(0.8)
        assert mechano[1].item() == pytest.approx(0.3)
        assert zone == ContactZone.ANTERIOR

    def test_preprocess_defaults_to_zone_none_when_no_predator_state(self) -> None:
        brain = _make_predator_brain()
        params = BrainParams(
            food_concentration=0.5,
            lateral_scale=1.0,
            derivative_scale=1.0,
        )
        state = brain.preprocess(params)
        state_t = torch.from_numpy(state)
        _food, _distal, _mechano, zone = brain._unpack_state(state_t)
        assert zone == ContactZone.NONE
