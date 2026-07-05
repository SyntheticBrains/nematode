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
        """Across >= 100 forward passes, action-logit variance is > 0."""
        brain = _make_brain()
        # Pin the global RNG so the test is deterministic.
        np.random.seed(_SEED)  # noqa: NPY002 - match brain's legacy global RNG
        torch.manual_seed(_SEED)
        probs_samples: list[np.ndarray] = []
        for _ in range(100):
            params = _make_params(
                strength=float(np.random.uniform(0, 1)),  # noqa: NPY002 - driven by global seeded RNG
                angle=float(np.random.uniform(-np.pi, np.pi)),  # noqa: NPY002 - driven by global seeded RNG
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
        np.random.seed(_SEED)  # noqa: NPY002 - match brain's legacy global RNG
        torch.manual_seed(_SEED)
        brain2 = _make_brain(forward_pass_depth=2)
        np.random.seed(_SEED)  # noqa: NPY002 - match brain's legacy global RNG
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
    def test_apply_weight_mask_is_pure_projector(self) -> None:
        """``ConnectomeTopology.apply_weight_mask`` is a stateless projector.

        Returns ``weights * M_chem`` without mutating ``self``. Caller
        owns the mode policy and the parameter-storage write-back.
        """
        brain = _make_brain()
        not_mask = ~brain.topology.m_chem
        candidate = torch.full_like(brain.topology.w_chem, 1.5)
        # Snapshot the topology's own weights before invocation.
        original_w = brain.topology.w_chem.detach().clone()

        masked = brain.topology.apply_weight_mask(candidate)

        # Pure projector: ``self.w_chem`` is unchanged by the call.
        assert torch.equal(brain.topology.w_chem.detach(), original_w)
        # Non-existent edges are zeroed in the returned tensor.
        assert (masked * not_mask).abs().max().item() == 0.0
        # Existing edges pass through unchanged.
        assert torch.equal(masked * brain.topology.m_chem, candidate * brain.topology.m_chem)

    def test_soft_prior_mode_does_not_project_after_ppo_step(self) -> None:
        """Soft-prior skips post-step projection so ~M_chem weights survive.

        Under ``chemical_mask_mode="soft_prior"`` the brain's update
        loop does not project ``w_chem`` after the optimiser step, so
        weights the optimiser placed along ~M_chem edges remain
        non-zero. Strict mode would zero them.
        """
        brain = _make_brain(chemical_mask_mode="soft_prior")
        not_mask = ~brain.topology.m_chem
        # Inject non-zero weights along ~M_chem to simulate optimiser drift.
        with torch.no_grad():
            brain.topology.w_chem.data = torch.where(
                not_mask,
                torch.full_like(brain.topology.w_chem, 1.5),
                brain.topology.w_chem.data,
            )
        before = (brain.topology.w_chem * not_mask).abs().max().item()
        assert before > 0.0

        # Drive a PPO update: under soft-prior, the brain skips the
        # post-step projection, so ~M_chem weights stay non-zero.
        self._drive_one_ppo_update(brain)
        after = (brain.topology.w_chem * not_mask).abs().max().item()
        assert after > 0.0

    def test_soft_prior_gradients_flow_through_non_wild_type_edges(self) -> None:
        """Soft-prior must let gradients reach ~M_chem entries.

        The forward pass uses raw ``w_chem`` (not ``w_chem * m_chem``) so
        backprop produces non-zero gradients on edges outside the
        wild-type adjacency - that's what lets the optimiser grow new
        edges from a zero initialisation.
        """
        brain = _make_brain(chemical_mask_mode="soft_prior")
        params = _make_params(strength=0.5, angle=0.3)
        food_features = torch.from_numpy(brain.preprocess(params)).to(brain.device)

        logits = brain.topology.forward(food_features)
        # Scalar loss so backward populates ``w_chem.grad``.
        logits.sum().backward()

        not_mask = ~brain.topology.m_chem
        grad = brain.topology.w_chem.grad
        assert grad is not None
        max_grad_off_mask = (grad * not_mask).abs().max().item()
        assert max_grad_off_mask > 0.0, (
            "Soft-prior must let gradients reach ~M_chem entries so the "
            "optimiser can grow new edges; observed all-zero gradient there."
        )

    def test_strict_mode_pins_gradients_to_wild_type_edges(self) -> None:
        """Strict mode pins gradients on ~M_chem to zero.

        The forward pass uses ``w_chem * m_chem`` so backprop's chain
        rule multiplies the upstream gradient by ``m_chem`` - zero on
        non-wild-type entries. This is what guarantees ``w_chem`` data
        on those entries never moves from the zero initialisation.
        """
        brain = _make_brain(chemical_mask_mode="strict")
        params = _make_params(strength=0.5, angle=0.3)
        food_features = torch.from_numpy(brain.preprocess(params)).to(brain.device)

        logits = brain.topology.forward(food_features)
        logits.sum().backward()

        not_mask = ~brain.topology.m_chem
        grad = brain.topology.w_chem.grad
        assert grad is not None
        max_grad_off_mask = (grad * not_mask).abs().max().item()
        assert max_grad_off_mask == 0.0, (
            "Strict mode must pin ~M_chem gradients to zero; "
            f"observed max|grad| = {max_grad_off_mask}"
        )

    @staticmethod
    def _drive_one_ppo_update(brain: ConnectomePPOBrain, n_steps: int = 8) -> None:
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


# ──────────────────────────────────────────────────────────────────────────────
# Degree-preserving rewired-null wiring control
# ──────────────────────────────────────────────────────────────────────────────
class TestWiringControl:
    """The ``wiring: wild_type | rewired_degree_preserving`` connectome-structure control."""

    def test_wild_type_matches_loaded_connectome(self) -> None:
        """wiring: wild_type builds the mask from the real Cook adjacency (rewiring not applied)."""
        from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite

        brain = _make_brain(wiring="wild_type")
        connectome = load_cook_2019_hermaphrodite()
        idx = {n: i for i, n in enumerate(brain.topology.neuron_names)}
        expected = torch.zeros(302, 302, dtype=torch.bool)
        for s in connectome.chemical_synapses:
            expected[idx[s.pre], idx[s.post]] = True
        assert torch.equal(brain.topology.m_chem, expected)

    def test_wild_type_is_byte_identical_to_default(self) -> None:
        """The new ``wiring`` field defaults to wild_type and perturbs no built tensor."""
        a = _make_brain()  # default wiring
        b = _make_brain(wiring="wild_type")
        assert torch.equal(a.topology.m_chem, b.topology.m_chem)
        assert torch.equal(a.topology.w_chem, b.topology.w_chem)
        assert torch.equal(a.topology.g_gap, b.topology.g_gap)

    def test_rewired_preserves_degree_but_changes_edges(self) -> None:
        """rewired_degree_preserving changes which neurons connect but keeps every degree."""
        wild = _make_brain(wiring="wild_type")
        rewired = _make_brain(wiring="rewired_degree_preserving")
        assert not torch.equal(rewired.topology.m_chem, wild.topology.m_chem)
        assert int(rewired.topology.m_chem.sum()) == int(wild.topology.m_chem.sum())
        # m_chem is [pre, post]: sum over dim 0 = per-post in-degree, dim 1 = per-pre out-degree.
        assert torch.equal(rewired.topology.m_chem.sum(0), wild.topology.m_chem.sum(0))
        assert torch.equal(rewired.topology.m_chem.sum(1), wild.topology.m_chem.sum(1))

    def test_rewired_gap_junctions_stay_symmetric(self) -> None:
        """The rewired gap-junction coupling remains symmetric (gap junctions are undirected)."""
        g = _make_brain(wiring="rewired_degree_preserving").topology.g_gap
        assert torch.allclose(g, g.T), "Rewired gap-junction matrix must stay symmetric"

    def test_rewired_topology_forward_pass_is_finite(self) -> None:
        """A rewired brain runs a forward pass without error and produces finite logits."""
        rewired = _make_brain(wiring="rewired_degree_preserving")
        logits = rewired.topology(torch.from_numpy(rewired.preprocess(_make_params())))
        assert torch.isfinite(logits).all()

    def test_rewire_seed_controls_the_draw(self) -> None:
        """Same rewire_seed -> identical null; different rewire_seed -> different null."""
        a = _make_brain(wiring="rewired_degree_preserving", rewire_seed=11)
        b = _make_brain(wiring="rewired_degree_preserving", rewire_seed=11)
        c = _make_brain(wiring="rewired_degree_preserving", rewire_seed=12)
        assert torch.equal(a.topology.m_chem, b.topology.m_chem)
        assert not torch.equal(a.topology.m_chem, c.topology.m_chem)

    def test_rewired_matches_wild_type_init_at_same_seed(self) -> None:
        """Matched init: the rewiring's dedicated RNG leaves food_gains/readout byte-identical."""
        wild = _make_brain(wiring="wild_type")
        rewired = _make_brain(wiring="rewired_degree_preserving")
        assert torch.equal(wild.topology.food_gains, rewired.topology.food_gains)
        assert torch.equal(wild.topology.readout, rewired.topology.readout)

    def test_rewired_brain_trains_and_keeps_strict_mask(self) -> None:
        """A rewired brain survives a PPO update: finite weights, strict-mask still holds."""
        brain = _make_brain(wiring="rewired_degree_preserving")
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
        assert torch.isfinite(brain.topology.w_chem).all()
        violation = (brain.topology.w_chem * ~brain.topology.m_chem).abs().max().item()
        assert violation == 0.0, (
            f"strict-mask violated on rewired m_chem: max|W along ~M| = {violation}"
        )
