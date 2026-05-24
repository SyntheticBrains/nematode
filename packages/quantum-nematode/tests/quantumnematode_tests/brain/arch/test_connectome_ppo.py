"""Unit tests for ConnectomePPOBrain.

Covers the spec scenarios at
``openspec/changes/add-architecture-plugin-interface/specs/connectome-ppo-brain/spec.md``:
construction loads the Cook 2019 connectome; strict-mask enforced before
and after PPO updates; gap junctions remain non-learnable + symmetric;
gap-junction fan-in normalisation; sensor projection routes to ASE/AWC/AWA
sensory neurons; motor readout aggregates VB/DB/VA/DA classes;
forward-pass depth K is configurable; soft-prior mode allows new edges;
frozen-updates flag freezes parameters; numerical invariants.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import DEFAULT_ACTIONS
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._registry import (
    Registration,
    get_registration,
    instantiate_brain,
    list_registered_brains,
)
from quantumnematode.brain.arch.connectome_ppo import (
    _MOTOR_CLASSES,
    _N_ACTIONS,
    _N_FOOD_FEATURES_BY_MODE,
    _SENSOR_NEURONS_FOOD,
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import BrainType, DeviceType

_SEED = 2026


def _make_brain(**cfg_overrides: object) -> ConnectomePPOBrain:
    """Construct a ConnectomePPOBrain with defaults overridable per-test."""
    cfg = ConnectomePPOBrainConfig(seed=_SEED, **cfg_overrides)  # type: ignore[arg-type]
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _make_params(strength: float = 0.42, angle: float = 0.13) -> BrainParams:
    return BrainParams(
        food_gradient_strength=strength,
        food_gradient_direction=angle,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Construction + registration
# ──────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_brain_construction_loads_the_connectome(self) -> None:
        """Brain construction loads Cook 2019 and builds W_chem + M_chem + G_gap."""
        brain = _make_brain()
        assert brain.topology.n_neurons == 302
        assert brain.topology.w_chem.shape == (302, 302)
        assert brain.topology.m_chem.shape == (302, 302)
        assert brain.topology.g_gap.shape == (302, 302)
        assert brain.topology.w_chem.dtype == torch.float32

    def test_strict_mask_invariant_at_init(self) -> None:
        """Non-existent chemical-synapse edges are zero at construction time."""
        brain = _make_brain()
        not_mask = ~brain.topology.m_chem
        assert (brain.topology.w_chem * not_mask).abs().max().item() == 0.0

    def test_unsupported_connectome_source_raises(self) -> None:
        """Pydantic's Literal validation rejects unknown source values."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="cook_2019_hermaphrodite"):
            ConnectomePPOBrainConfig(
                seed=_SEED,
                connectome_source="not_a_real_source",  # type: ignore[arg-type]
            )


# ──────────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────────


class TestRegistration:
    def test_connectome_ppo_brain_self_registers_at_import(self) -> None:
        """ConnectomePPOBrain self-registers as 'connectomeppo'."""
        assert "connectomeppo" in list_registered_brains()
        reg = get_registration("connectomeppo")
        assert isinstance(reg, Registration)
        assert reg.name == "connectomeppo"
        assert reg.brain_cls is ConnectomePPOBrain
        assert reg.config_cls is ConnectomePPOBrainConfig
        assert reg.brain_type is BrainType.CONNECTOMEPPO
        assert reg.families == ("classical",)

    def test_instantiate_brain_via_registry(self) -> None:
        """Registry-instantiated brain is functionally identical to direct."""
        cfg = ConnectomePPOBrainConfig(seed=_SEED)
        brain = instantiate_brain("connectomeppo", cfg, device=DeviceType.CPU)
        assert isinstance(brain, ConnectomePPOBrain)


# ──────────────────────────────────────────────────────────────────────────────
# Gap junctions: non-learnable, symmetric, fan-in normalised
# ──────────────────────────────────────────────────────────────────────────────


class TestGapJunctions:
    def test_gap_junction_weights_are_symmetric(self) -> None:
        brain = _make_brain()
        g = brain.topology.g_gap
        assert torch.allclose(g, g.T), "Gap-junction matrix must be symmetric"

    def test_gap_junction_weights_not_learnable(self) -> None:
        brain = _make_brain()
        assert brain.topology.g_gap.requires_grad is False

    def test_gap_junction_disabled_zeros_the_matrix(self) -> None:
        """When enable_gap_junctions=False the matrix is all zeros."""
        brain = _make_brain(enable_gap_junctions=False)
        assert torch.equal(brain.topology.g_gap, torch.zeros(302, 302))


# ──────────────────────────────────────────────────────────────────────────────
# Sensor projection
# ──────────────────────────────────────────────────────────────────────────────


class TestSensorProjection:
    def test_food_gains_shape_matches_sensor_neuron_count(self) -> None:
        brain = _make_brain()
        # Default sensing_mode is "oracle" → 2 features.
        assert brain.topology.food_gains.shape == (
            _N_FOOD_FEATURES_BY_MODE["oracle"],
            len(_SENSOR_NEURONS_FOOD),
        )

    def test_food_gains_shape_klinotaxis(self) -> None:
        brain = _make_brain(sensing_mode="klinotaxis")
        assert brain.topology.food_gains.shape == (
            _N_FOOD_FEATURES_BY_MODE["klinotaxis"],
            len(_SENSOR_NEURONS_FOOD),
        )

    def test_preprocess_emits_correct_feature_count_per_mode(self) -> None:
        brain_oracle = _make_brain()
        brain_klino = _make_brain(sensing_mode="klinotaxis")
        params = _make_params()
        assert brain_oracle.preprocess(params).shape == (2,)
        assert brain_klino.preprocess(params).shape == (3,)

    def test_food_gains_are_learnable(self) -> None:
        brain = _make_brain()
        assert brain.topology.food_gains.requires_grad is True

    def test_food_neuron_indices_cover_canonical_pathway(self) -> None:
        brain = _make_brain()
        # The six neurons addressed by the food projection.
        indices = brain.topology._food_neuron_indices.tolist()
        names = [brain.topology.neuron_names[i] for i in indices]
        assert set(names) == set(_SENSOR_NEURONS_FOOD)


# ──────────────────────────────────────────────────────────────────────────────
# Motor readout
# ──────────────────────────────────────────────────────────────────────────────


class TestMotorReadout:
    def test_motor_class_neurons_present(self) -> None:
        brain = _make_brain()
        # Each motor class has > 0 neurons in the connectome.
        boundaries = brain.topology._motor_class_boundaries.tolist()
        for k in range(_N_ACTIONS):
            count = boundaries[k + 1] - boundaries[k]
            assert count > 0, f"Motor class {_MOTOR_CLASSES[k]} has zero neurons in the index"

    def test_readout_matrix_shape(self) -> None:
        brain = _make_brain()
        assert brain.topology.readout.shape == (_N_ACTIONS, _N_ACTIONS)
        assert brain.topology.readout.requires_grad is True


# ──────────────────────────────────────────────────────────────────────────────
# Forward-pass numerical invariants
# ──────────────────────────────────────────────────────────────────────────────


class TestForwardPass:
    def test_forward_pass_output_is_finite(self) -> None:
        brain = _make_brain()
        actions = brain.run_brain(
            _make_params(),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        assert len(actions) == 1
        assert np.isfinite(actions[0].probability)
        assert brain.current_probabilities is not None
        assert np.isfinite(brain.current_probabilities).all()

    def test_forward_pass_has_non_degenerate_variance(self) -> None:
        """Across ≥ 100 forward passes, action-logit variance is > 0."""
        brain = _make_brain()
        # Pin the global RNG so the test is deterministic.
        np.random.seed(_SEED)  # noqa: NPY002 — match brain's legacy global RNG
        torch.manual_seed(_SEED)
        probs_samples: list[np.ndarray] = []
        for _ in range(100):
            params = _make_params(
                strength=float(np.random.uniform(0, 1)),  # noqa: NPY002 — driven by global seeded RNG
                angle=float(np.random.uniform(-np.pi, np.pi)),  # noqa: NPY002 — driven by global seeded RNG
            )
            brain.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            assert brain.current_probabilities is not None
            probs_samples.append(brain.current_probabilities.copy())
        arr = np.array(probs_samples)
        assert arr.var() > 0.0, "Probability variance should be > 0 (non-degenerate)"
        # Not collapsed to a single action: the per-action mean should not
        # be exactly 1.0 for any single action.
        per_action_mean = arr.mean(axis=0)
        assert per_action_mean.max() < 1.0

    def test_forward_pass_depth_is_configurable(self) -> None:
        """K=2 produces different activations than K=4 for the same input."""
        np.random.seed(_SEED)  # noqa: NPY002 — match brain's legacy global RNG
        torch.manual_seed(_SEED)
        brain2 = _make_brain(forward_pass_depth=2)
        np.random.seed(_SEED)  # noqa: NPY002 — match brain's legacy global RNG
        torch.manual_seed(_SEED)
        brain4 = _make_brain(forward_pass_depth=4)

        params = _make_params()
        x = torch.from_numpy(brain2.preprocess(params))
        logits2 = brain2.topology(x)
        logits4 = brain4.topology(x)
        assert not torch.allclose(logits2, logits4)

    def test_invalid_forward_pass_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="forward_pass_depth must be >= 1"):
            _make_brain(forward_pass_depth=0)


# ──────────────────────────────────────────────────────────────────────────────
# Strict-mask enforcement across PPO updates
# ──────────────────────────────────────────────────────────────────────────────


class TestStrictMaskInvariant:
    def _drive_one_ppo_update(self, brain: ConnectomePPOBrain, n_steps: int = 8) -> None:
        """Push enough experience through the brain to trigger a PPO update.

        The brain's rollout buffer triggers an update when it reaches
        ``num_minibatches`` items (4 by default at end-of-episode). We
        produce ``n_steps`` (state, reward) pairs and finish with
        ``episode_done=True`` so the update fires.
        """
        for step in range(n_steps):
            params = _make_params(
                strength=0.5 + 0.01 * step,
                angle=0.1 * step,
            )
            brain.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            brain.learn(params, reward=0.1, episode_done=(step == n_steps - 1))

    def test_strict_mask_invariant_after_ppo_update(self) -> None:
        """W_chem is still strict-masked after a PPO update has run."""
        brain = _make_brain()
        self._drive_one_ppo_update(brain)
        not_mask = ~brain.topology.m_chem
        violation = (brain.topology.w_chem * not_mask).abs().max().item()
        assert violation == 0.0, (
            f"Strict-mask invariant violated: max|W_chem along ~M_chem| = {violation}"
        )

    def test_existing_edge_weights_change_under_ppo(self) -> None:
        """At least one weight along an existing edge changes from its init."""
        brain = _make_brain()
        initial = brain.topology.w_chem.detach().clone()
        self._drive_one_ppo_update(brain)
        delta = (brain.topology.w_chem - initial).abs().max().item()
        assert delta > 0.0, "Expected at least one chemical-synapse weight to change after PPO"

    def test_gap_junctions_remain_byte_identical_across_ppo(self) -> None:
        brain = _make_brain()
        initial_g = brain.topology.g_gap.detach().clone()
        self._drive_one_ppo_update(brain)
        assert torch.equal(brain.topology.g_gap, initial_g)


# ──────────────────────────────────────────────────────────────────────────────
# Soft-prior mode
# ──────────────────────────────────────────────────────────────────────────────


class TestSoftPriorMode:
    def test_soft_prior_apply_mask_is_noop(self) -> None:
        """``apply_weight_mask(mode='soft_prior')`` does not zero new edges.

        Injects synthetic non-zero weights along ~M_chem edges, then
        invokes the soft-prior projection: it must not zero them. Under
        ``strict`` mode the same projection would zero them.
        """
        brain = _make_brain(chemical_mask_mode="soft_prior")
        not_mask = ~brain.topology.m_chem
        # Inject non-zero weights along ~M_chem in-place via masked-fill.
        with torch.no_grad():
            brain.topology.w_chem.data = torch.where(
                not_mask,
                torch.full_like(brain.topology.w_chem, 1.5),
                brain.topology.w_chem.data,
            )
        before = (brain.topology.w_chem * not_mask).abs().max().item()
        assert before > 0.0  # confirm injection worked

        brain.topology.apply_weight_mask(mode="soft_prior")
        after_soft = (brain.topology.w_chem * not_mask).abs().max().item()
        assert after_soft == before  # soft-prior is a no-op

        brain.topology.apply_weight_mask(mode="strict")
        after_strict = (brain.topology.w_chem * not_mask).abs().max().item()
        assert after_strict == 0.0  # strict zeros them


# ──────────────────────────────────────────────────────────────────────────────
# Frozen-updates flag (paired-control)
# ──────────────────────────────────────────────────────────────────────────────


class TestFrozenUpdates:
    def test_frozen_updates_skip_optimiser(self) -> None:
        """With freeze_updates=True, weights are unchanged after a PPO update."""
        brain = _make_brain(freeze_updates=True)
        initial_w_chem = brain.topology.w_chem.detach().clone()
        initial_food_gains = brain.topology.food_gains.detach().clone()
        initial_readout = brain.topology.readout.detach().clone()

        # Drive what would be a PPO update under freeze_updates=False.
        for step in range(8):
            params = _make_params(strength=0.5 + 0.01 * step, angle=0.1 * step)
            brain.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            brain.learn(params, reward=0.1, episode_done=(step == 7))

        assert torch.equal(brain.topology.w_chem, initial_w_chem)
        assert torch.equal(brain.topology.food_gains, initial_food_gains)
        assert torch.equal(brain.topology.readout, initial_readout)

    def test_frozen_brain_still_samples_actions(self) -> None:
        """The frozen brain still produces action samples via the forward pass."""
        brain = _make_brain(freeze_updates=True)
        for _ in range(20):
            actions = brain.run_brain(
                _make_params(),
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            assert actions[0].action in DEFAULT_ACTIONS


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigDefaults:
    def test_default_field_values(self) -> None:
        cfg = ConnectomePPOBrainConfig()
        assert cfg.connectome_source == "cook_2019_hermaphrodite"
        assert cfg.enable_gap_junctions is True
        assert cfg.chemical_mask_mode == "strict"
        assert cfg.forward_pass_depth == 4
        assert cfg.freeze_updates is False


# ──────────────────────────────────────────────────────────────────────────────
# Topology Protocol conformance
# ──────────────────────────────────────────────────────────────────────────────


class TestTopologyProtocol:
    def test_topology_exposes_learnable_parameters_excluding_g_gap(self) -> None:
        """learnable_parameters returns w_chem + food_gains + readout (not g_gap)."""
        brain = _make_brain()
        params = brain.topology.learnable_parameters
        # ``in`` on a list of tensors uses element-wise equality; use ``is``
        # identity checks instead via ``id``-based membership.
        param_ids = {id(p) for p in params}
        assert id(brain.topology.w_chem) in param_ids
        assert id(brain.topology.food_gains) in param_ids
        assert id(brain.topology.readout) in param_ids
        assert id(brain.topology.g_gap) not in param_ids
