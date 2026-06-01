"""Unit tests for the Spiking (recurrent adaptive-LIF) PPO brain architecture.

Mirrors ``test_cfcppo.py``. Covers: registry / Protocol conformance;
forward-pass shape + finite logits/value over the 4-action set; recurrent
neuron-state carry + reset and genuine recurrence (identical inputs at
different in-episode steps can differ); non-degenerate logit variance; a
truncated-BPTT PPO update with gradient flowing through the surrogate, leaving
params finite; WeightPersistence Protocol conformance + round-trip identical
logits; determinism under a fixed seed; substrate-level recurrent adaptive-LIF /
leaky-integrator-readout dynamics (decay, soft reset, adaptation, recurrent
current, surrogate gradient finite/non-zero) plus the carried-over
``SurrogateGradientSpike`` / ``LIFLayer`` tests; and the entropy + surrogate-slope
schedule validators.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._registry import (
    Registration,
    get_registration,
    instantiate_brain,
    list_registered_brains,
)
from quantumnematode.brain.arch._spiking_layers import (
    LeakyIntegratorReadout,
    LIFLayer,
    RecurrentAdaptiveLIFCell,
    SurrogateGradientSpike,
)
from quantumnematode.brain.arch.dtypes import BrainType, DeviceType
from quantumnematode.brain.arch.spiking_ppo import (
    NeuronState,
    SpikingPPOBrain,
    SpikingPPOBrainConfig,
    SpikingPPORolloutBuffer,
)
from quantumnematode.brain.modules import ModuleName
from quantumnematode.brain.weights import WeightPersistence
from quantumnematode.env import Direction

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SENSORY_MODULES = [ModuleName.FOOD_CHEMOTAXIS, ModuleName.PROPRIOCEPTION]
INPUT_DIM = 4  # 2 modules * 2 features each
_SEED = 42


def _make_config(**overrides: object) -> SpikingPPOBrainConfig:
    """Create a config with sensory modules and small defaults for fast tests."""
    defaults: dict[str, object] = {
        "sensory_modules": SENSORY_MODULES,
        "hidden_size": 16,
        "rollout_buffer_size": 32,
        "bptt_chunk_length": 8,
        "critic_hidden_dim": 16,
        "num_epochs": 2,
        # A lower membrane decay makes the carried state move faster, so the
        # recurrence and variance tests exercise genuine temporal dynamics
        # within the short test horizons.
        "membrane_decay_init": 0.5,
        "seed": _SEED,
    }
    defaults.update(overrides)
    return SpikingPPOBrainConfig(**defaults)  # type: ignore[arg-type]


def _make_brain(**overrides: object) -> SpikingPPOBrain:
    """Create a brain with small defaults for fast tests."""
    config = _make_config(**overrides)
    return SpikingPPOBrain(config=config, num_actions=4, device=DeviceType.CPU)


def _make_params(grad_strength: float = 0.5, direction: Direction = Direction.UP) -> BrainParams:
    """Create a BrainParams for testing."""
    return BrainParams(
        food_gradient_strength=grad_strength,
        food_gradient_direction=np.pi / 2,
        agent_direction=direction,
    )


def _logits(brain: SpikingPPOBrain, params: BrainParams) -> torch.Tensor:
    """Compute action logits for ``params`` under the brain's current neuron state."""
    feat = torch.tensor(brain.preprocess(params), dtype=torch.float32)
    slope = brain._get_surrogate_slope()
    with torch.no_grad():
        logits, _ = brain._core_forward(feat, brain.neuron_state, slope)
    return logits


def _value(brain: SpikingPPOBrain, params: BrainParams) -> float:
    """Compute the critic value for ``params`` under the brain's current neuron state."""
    feat = torch.tensor(brain.preprocess(params), dtype=torch.float32)
    slope = brain._get_surrogate_slope()
    with torch.no_grad():
        _, state = brain._core_forward(feat, brain.neuron_state, slope)
        return brain.critic(brain._hidden_membrane(state).detach()).item()


def _first_cell(brain: SpikingPPOBrain) -> RecurrentAdaptiveLIFCell:
    """Return the first recurrent adaptive-LIF hidden cell (typed for the checker)."""
    layer = brain.hidden_layers[0]
    assert isinstance(layer, RecurrentAdaptiveLIFCell)
    return layer


# ──────────────────────────────────────────────────────────────────────────────
# Config Validation
# ──────────────────────────────────────────────────────────────────────────────


class TestSpikingPPOBrainConfig:
    """Test cases for Spiking PPO brain configuration."""

    def test_default_config(self) -> None:
        """Default configuration values match the spec."""
        config = SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES)
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 1
        assert config.timesteps_per_step == 1
        assert config.v_threshold == 1.0
        assert config.membrane_decay_init == 0.9
        assert config.adaptation_decay_init == 0.9
        assert config.adapt_scale_init == 0.1
        assert config.readout_decay_init == 0.9
        assert config.surrogate_slope == 2.0
        assert config.surrogate_slope_end is None
        assert config.surrogate_slope_anneal_episodes is None
        assert config.critic_hidden_dim == 64
        assert config.critic_num_layers == 2
        assert config.actor_lr == 0.0003
        assert config.critic_lr == 0.0003
        assert config.clip_epsilon == 0.2
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.value_loss_coef == 0.5
        assert config.entropy_coef == 0.01
        assert config.rollout_buffer_size == 512
        assert config.num_epochs == 4
        assert config.max_grad_norm == 0.5
        assert config.bptt_chunk_length == 64

    def test_custom_config(self) -> None:
        """Custom configuration values are accepted."""
        config = SpikingPPOBrainConfig(
            sensory_modules=SENSORY_MODULES,
            hidden_size=128,
            num_hidden_layers=2,
            timesteps_per_step=2,
            surrogate_slope=5.0,
        )
        assert config.hidden_size == 128
        assert config.num_hidden_layers == 2
        assert config.timesteps_per_step == 2
        assert config.surrogate_slope == 5.0

    def test_sensory_modules_required(self) -> None:
        """sensory_modules=None is rejected."""
        with pytest.raises(ValueError, match="sensory_modules is required"):
            SpikingPPOBrainConfig()

    def test_sensory_modules_non_empty(self) -> None:
        """Empty sensory_modules is rejected."""
        with pytest.raises(ValueError, match="sensory_modules is required"):
            SpikingPPOBrainConfig(sensory_modules=[])

    def test_invalid_hidden_size(self) -> None:
        """hidden_size < 1 is rejected."""
        with pytest.raises(ValueError, match="hidden_size must be >= 1"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, hidden_size=0)

    def test_invalid_timesteps_per_step(self) -> None:
        """timesteps_per_step < 1 is rejected."""
        with pytest.raises(ValueError, match="timesteps_per_step must be >= 1"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, timesteps_per_step=0)

    def test_invalid_bptt_chunk_length(self) -> None:
        """bptt_chunk_length < 4 is rejected."""
        with pytest.raises(ValueError, match="bptt_chunk_length must be >= 4"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, bptt_chunk_length=2)

    def test_invalid_rollout_buffer_size(self) -> None:
        """rollout_buffer_size < bptt_chunk_length is rejected."""
        with pytest.raises(ValueError, match="rollout_buffer_size"):
            SpikingPPOBrainConfig(
                sensory_modules=SENSORY_MODULES,
                rollout_buffer_size=8,
                bptt_chunk_length=16,
            )

    def test_invalid_decays(self) -> None:
        """Decays outside [0, 1) are rejected."""
        with pytest.raises(ValueError, match="membrane_decay_init"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, membrane_decay_init=1.0)
        with pytest.raises(ValueError, match="adaptation_decay_init"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, adaptation_decay_init=1.5)
        with pytest.raises(ValueError, match="readout_decay_init"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, readout_decay_init=-0.1)

    def test_invalid_surrogate_slope(self) -> None:
        """A non-positive base surrogate slope is rejected."""
        with pytest.raises(ValueError, match="surrogate_slope must be > 0"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, surrogate_slope=0.0)

    def test_invalid_network_and_lr_params(self) -> None:
        """Non-positive critic sizes, layer counts, and learning rates are rejected."""
        with pytest.raises(ValueError, match="critic_hidden_dim"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, critic_hidden_dim=0)
        with pytest.raises(ValueError, match="critic_num_layers"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, critic_num_layers=0)
        with pytest.raises(ValueError, match="num_hidden_layers"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, num_hidden_layers=0)
        with pytest.raises(ValueError, match="actor_lr"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, actor_lr=0.0)
        with pytest.raises(ValueError, match="critic_lr"):
            SpikingPPOBrainConfig(sensory_modules=SENSORY_MODULES, critic_lr=-0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Registry + BrainType + Protocol conformance (task 5.1)
# ──────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    """Registry / Protocol conformance: brain builds via the registry."""

    def test_brain_self_registers_at_import(self) -> None:
        """The @register_brain decorator places spikingppo in the registry."""
        assert "spikingppo" in list_registered_brains()

    def test_registration_metadata(self) -> None:
        """The Registration carries the expected fields."""
        reg = get_registration("spikingppo")
        assert isinstance(reg, Registration)
        assert reg.name == "spikingppo"
        assert reg.config_cls is SpikingPPOBrainConfig
        assert reg.brain_cls is SpikingPPOBrain
        assert reg.brain_type is BrainType.SPIKING_PPO
        assert reg.families == ("spiking",)

    def test_brain_type_enum_value(self) -> None:
        """BrainType.SPIKING_PPO value matches the registered name."""
        assert BrainType.SPIKING_PPO.value == "spikingppo"

    def test_in_spiking_family_set(self) -> None:
        """The brain type is auto-populated into SPIKING_BRAIN_TYPES via families."""
        from quantumnematode.brain.arch import dtypes

        assert BrainType.SPIKING_PPO in dtypes.SPIKING_BRAIN_TYPES

    def test_instantiate_via_registry(self) -> None:
        """instantiate_brain() dispatches via the registry without per-arch branches."""
        brain = instantiate_brain("spikingppo", _make_config(), device=DeviceType.CPU)
        assert isinstance(brain, SpikingPPOBrain)

    def test_satisfies_classical_brain_protocol(self) -> None:
        """Brain is a runtime ClassicalBrain (has action_set + learn)."""
        from quantumnematode.brain.arch._brain import ClassicalBrain

        brain = _make_brain()
        assert isinstance(brain, ClassicalBrain)


# ──────────────────────────────────────────────────────────────────────────────
# Construction + topology
# ──────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    """Brain construction builds the spiking core + readout + critic."""

    def test_constructs_core_readout_and_critic(self) -> None:
        """Encoder + recurrent core + readout + critic built; input_dim derived from modules."""
        brain = _make_brain()
        assert brain.input_dim == INPUT_DIM
        assert brain.num_actions == 4
        assert brain.hidden_size == 16
        assert len(brain.hidden_layers) == 1
        assert isinstance(brain.hidden_layers[0], RecurrentAdaptiveLIFCell)
        assert isinstance(brain.readout, LeakyIntegratorReadout)
        assert hasattr(brain, "critic")
        # Encoder maps input_dim -> hidden_size; readout maps hidden_size -> num_actions.
        assert brain.encoder.in_features == INPUT_DIM
        assert brain.encoder.out_features == 16
        assert brain.readout.fc.out_features == 4

    def test_multiple_hidden_layers(self) -> None:
        """num_hidden_layers > 1 stacks multiple recurrent LIF cells."""
        brain = _make_brain(num_hidden_layers=3)
        assert len(brain.hidden_layers) == 3
        for layer in brain.hidden_layers:
            assert isinstance(layer, RecurrentAdaptiveLIFCell)

    def test_neuron_state_initialized(self) -> None:
        """The carried neuron state has per-layer (v, a, s) plus a readout membrane m."""
        brain = _make_brain(num_hidden_layers=2)
        state = brain.neuron_state
        assert len(state["v"]) == 2
        assert len(state["a"]) == 2
        assert len(state["s"]) == 2
        assert state["v"][0].shape == (1, 16)
        assert state["m"].shape == (1, 4)

    def test_separate_optimizers_no_overlap(self) -> None:
        """Actor and critic have separate optimizers covering disjoint params."""
        brain = _make_brain()
        actor_params = {id(p) for p in brain.actor_optimizer.param_groups[0]["params"]}
        critic_params = {id(p) for p in brain.critic_optimizer.param_groups[0]["params"]}
        assert brain.actor_optimizer is not brain.critic_optimizer
        assert len(actor_params & critic_params) == 0
        # Actor optimizer covers encoder + feature_norm + core + readout.
        for p in brain.encoder.parameters():
            assert id(p) in actor_params
        for p in brain.feature_norm.parameters():
            assert id(p) in actor_params
        for p in brain.hidden_layers.parameters():
            assert id(p) in actor_params
        for p in brain.readout.parameters():
            assert id(p) in actor_params

    def test_action_set_length_must_match_num_actions(self) -> None:
        """Action-set length mismatch with num_actions raises at construction."""
        with pytest.raises(ValueError, match="action_set must have exactly"):
            SpikingPPOBrain(
                config=_make_config(),
                num_actions=4,
                device=DeviceType.CPU,
                action_set=DEFAULT_ACTIONS[:2],
            )


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass: finite logits + value over the action set (task 5.2)
# ──────────────────────────────────────────────────────────────────────────────


class TestForwardPass:
    """Forward pass produces finite logits and a value over the action set."""

    def test_logits_shape_and_finite(self) -> None:
        """Logits are a finite num_actions-vector; value is a finite scalar."""
        brain = _make_brain()
        brain.prepare_episode()
        params = _make_params()
        logits = _logits(brain, params)
        value = _value(brain, params)
        assert logits.shape == (brain.num_actions,)
        assert torch.isfinite(logits).all()
        assert np.isfinite(value)

    def test_run_brain_returns_action_data(self) -> None:
        """run_brain returns a single ActionData drawn from DEFAULT_ACTIONS."""
        brain = _make_brain()
        brain.prepare_episode()
        results = brain.run_brain(_make_params(), top_only=False, top_randomize=False)
        assert len(results) == 1
        assert isinstance(results[0], ActionData)
        assert results[0].action in DEFAULT_ACTIONS
        assert 0.0 <= results[0].probability <= 1.0

    def test_output_dim_equals_num_actions(self) -> None:
        """The readout output dimensionality equals num_actions."""
        brain = _make_brain()
        brain.prepare_episode()
        logits = _logits(brain, _make_params())
        assert logits.shape == (brain.num_actions,)

    def test_timesteps_per_step_runs(self) -> None:
        """With timesteps_per_step > 1 the forward pass still yields finite logits."""
        brain = _make_brain(timesteps_per_step=3)
        brain.prepare_episode()
        logits = _logits(brain, _make_params())
        assert logits.shape == (4,)
        assert torch.isfinite(logits).all()


# ──────────────────────────────────────────────────────────────────────────────
# Neuron-state carry + reset + genuine recurrence (task 5.3)
# ──────────────────────────────────────────────────────────────────────────────


class TestRecurrence:
    """The carried neuron state evolves within an episode and resets at boundaries."""

    def test_state_carries_within_episode(self) -> None:
        """The hidden membrane changes across steps (carried into the recurrence)."""
        brain = _make_brain()
        brain.prepare_episode()
        v_before = brain.neuron_state["v"][-1].clone()
        brain.run_brain(_make_params(), top_only=False, top_randomize=False)
        assert not torch.equal(v_before, brain.neuron_state["v"][-1])

    def test_prepare_episode_resets_state_to_zero(self) -> None:
        """prepare_episode() resets the neuron state to zeros."""
        brain = _make_brain()
        brain.prepare_episode()
        for _ in range(5):
            brain.run_brain(_make_params(), top_only=False, top_randomize=False)
            brain.learn(_make_params(), reward=0.1, episode_done=False)
        assert not torch.all(brain.neuron_state["v"][-1] == 0)

        brain.prepare_episode()
        assert torch.all(brain.neuron_state["v"][-1] == 0)
        assert torch.all(brain.neuron_state["a"][-1] == 0)
        assert torch.all(brain.neuron_state["s"][-1] == 0)
        assert torch.all(brain.neuron_state["m"] == 0)

    def test_prepare_episode_resets_pending_and_step_count(self) -> None:
        """prepare_episode() resets pending features and step count."""
        brain = _make_brain()
        brain.prepare_episode()
        brain.run_brain(_make_params(), top_only=False, top_randomize=False)
        assert brain._pending_features is not None
        assert brain._step_count == 1

        brain.prepare_episode()
        assert brain._pending_features is None
        assert brain._step_count == 0

    def test_identical_inputs_differ_across_steps(self) -> None:
        """Two identical inputs at different in-episode steps differ — genuine recurrence.

        With a zero state the first logits are deterministic; after advancing
        the carried state with several steps of the same input, the same input
        yields different logits.
        """
        brain = _make_brain()
        brain.prepare_episode()
        params = _make_params(0.6)
        logits_step0 = _logits(brain, params)
        for _ in range(6):
            brain.run_brain(params, top_only=False, top_randomize=False)
        logits_step_n = _logits(brain, params)
        assert not torch.allclose(logits_step0, logits_step_n, atol=1e-5), (
            "Identical inputs at different steps produced identical logits — "
            "the core is not genuinely recurrent."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Non-degenerate variance (spec variance scenario, unit level) (task 5.9)
# ──────────────────────────────────────────────────────────────────────────────


class TestVariance:
    """Forward pass produces non-degenerate variance across a sample of passes."""

    def test_logit_variance_strictly_positive(self) -> None:
        """Across >= 100 varied forward passes, per-action logit variance > 0.

        The policy SHALL not collapse to a constant action across the sample
        (interpreted at the categorical-sampling output surface).
        """
        brain = _make_brain()
        brain.prepare_episode()
        rng = np.random.default_rng(0)
        all_logits: list[torch.Tensor] = []
        sampled_actions: list[Action] = []
        for _ in range(128):
            params = BrainParams(
                food_gradient_strength=float(rng.uniform(0.0, 1.0)),
                food_gradient_direction=float(rng.uniform(-np.pi, np.pi)),
                agent_direction=Direction.UP,
            )
            all_logits.append(_logits(brain, params))
            result = brain.run_brain(params, top_only=False, top_randomize=False)
            sampled_actions.append(result[0].action)

        stacked = torch.stack(all_logits)  # (128, num_actions)
        variance = stacked.var(dim=0)
        assert (variance > 0).all(), f"Some action logits collapsed: variance={variance}"
        assert len(set(sampled_actions)) > 1, (
            f"All forward passes sampled the same action: {set(sampled_actions)}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# PPO update over truncated-BPTT chunks (task 5.4)
# ──────────────────────────────────────────────────────────────────────────────


class TestPPOUpdate:
    """PPO update runs over truncated-BPTT sequence chunks and leaves params finite."""

    def test_buffer_fills_and_ppo_update_runs(self) -> None:
        """Buffer fills and a PPO update runs without error."""
        brain = _make_brain(rollout_buffer_size=16, bptt_chunk_length=4)
        brain.prepare_episode()
        params = _make_params()
        for step in range(20):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=(step == 19))
        assert brain.history_data.losses, "Expected at least one loss recorded"

    def test_ppo_leaves_params_finite(self) -> None:
        """After PPO, all network parameters remain finite."""
        brain = _make_brain(rollout_buffer_size=16, bptt_chunk_length=4)
        brain.prepare_episode()
        params = _make_params()
        for step in range(20):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=(step == 19))

        for loss in brain.history_data.losses:
            assert np.isfinite(loss), f"Non-finite loss: {loss}"
        for p in brain.hidden_layers.parameters():
            assert torch.isfinite(p).all()
        for p in brain.encoder.parameters():
            assert torch.isfinite(p).all()
        for p in brain.readout.parameters():
            assert torch.isfinite(p).all()
        for p in brain.critic.parameters():
            assert torch.isfinite(p).all()

    def test_gradient_flows_through_surrogate(self) -> None:
        """A PPO chunk replay flows a non-zero gradient into the recurrent core.

        Building gradients through the spiking core requires the surrogate
        backward to be exercised — a zero/None grad on the recurrent weights
        would mean the surrogate did not carry signal.
        """
        brain = _make_brain(rollout_buffer_size=16, bptt_chunk_length=8)
        brain.prepare_episode()
        rng = np.random.default_rng(1)
        # Fill a rollout with varied inputs and non-trivial rewards.
        for step in range(16):
            params = _make_params(float(rng.uniform(0.1, 0.9)))
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=float(rng.uniform(-1.0, 1.0)), episode_done=(step == 15))

        # The recurrent spike-feedback weights are the surrogate-only path:
        # they receive gradient ONLY through spikes (surrogate backward).
        cell = _first_cell(brain)
        rec_weight = cell.recurrent.weight
        assert rec_weight.grad is not None
        assert torch.isfinite(rec_weight.grad).all()
        assert torch.any(rec_weight.grad != 0), (
            "Recurrent (surrogate-only) weights got zero gradient — the surrogate "
            "did not carry signal through the spike."
        )
        # The learnable membrane decay also gets gradient through the surrogate.
        raw_decay = cell.raw_membrane_decay
        assert raw_decay.grad is not None
        assert torch.isfinite(raw_decay.grad).all()

    def test_multilayer_ppo_update_trains_every_layer(self) -> None:
        """With num_hidden_layers=2, a PPO update flows gradient into EVERY stacked cell.

        Guards the multi-layer BPTT replay (the nested tick/layer loop): a regression
        that drove only the first layer would leave the second layer's surrogate-only
        recurrent weights at zero/None gradient while every other test still passed.
        """
        brain = _make_brain(num_hidden_layers=2, rollout_buffer_size=16, bptt_chunk_length=8)
        brain.prepare_episode()
        rng = np.random.default_rng(2)
        for step in range(16):
            params = _make_params(float(rng.uniform(0.1, 0.9)))
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=float(rng.uniform(-1.0, 1.0)), episode_done=(step == 15))

        assert len(brain.hidden_layers) == 2
        for layer_idx, layer in enumerate(brain.hidden_layers):
            assert isinstance(layer, RecurrentAdaptiveLIFCell)
            # raw_membrane_decay receives gradient through the (smooth) surrogate whenever
            # the cell participates in the forward/backward — a spike-density-independent
            # signal that the multi-layer replay genuinely DRIVES this stacked cell (a
            # regression that skipped the second layer would leave it None/zero here).
            beta_grad = layer.raw_membrane_decay.grad
            assert beta_grad is not None, f"Layer {layer_idx} was not driven (no gradient)"
            assert torch.isfinite(beta_grad).all()
            assert torch.any(beta_grad != 0), f"Layer {layer_idx} got zero gradient — not trained"
            # The recurrent spike-feedback weights are wired into the graph and stay finite
            # (their magnitude depends on this layer's spike density, so non-zero is not
            # asserted for deeper, sparser-firing layers).
            rec_grad = layer.recurrent.weight.grad
            assert rec_grad is not None
            assert torch.isfinite(rec_grad).all()

    def test_multitick_ppo_update_leaves_params_finite(self) -> None:
        """With timesteps_per_step=2, the inner-tick unroll trains without blowing up.

        Exercises the inner-tick loop (constant input current held across ticks, the
        recurrence advancing each tick) through a full PPO update.
        """
        brain = _make_brain(timesteps_per_step=2, rollout_buffer_size=16, bptt_chunk_length=8)
        brain.prepare_episode()
        rng = np.random.default_rng(3)
        for step in range(16):
            params = _make_params(float(rng.uniform(0.1, 0.9)))
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=float(rng.uniform(-1.0, 1.0)), episode_done=(step == 15))

        assert brain.history_data.losses, "Expected a PPO update to have run"
        cell = _first_cell(brain)
        assert cell.recurrent.weight.grad is not None
        assert torch.isfinite(cell.recurrent.weight.grad).all()
        for p in brain.hidden_layers.parameters():
            assert torch.isfinite(p).all()
        for p in brain.readout.parameters():
            assert torch.isfinite(p).all()

    def test_ppo_replays_over_chunks(self) -> None:
        """A buffer larger than the chunk length yields multiple BPTT chunks."""
        buf = SpikingPPORolloutBuffer(buffer_size=16, device=torch.device("cpu"))
        for i in range(16):
            buf.add(
                features=np.array([float(i)] * INPUT_DIM, dtype=np.float32),
                action=i % 4,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=False,
                state=_dummy_state(),
            )
        returns = torch.ones(16)
        advantages = torch.ones(16)
        chunks = list(buf.get_sequential_chunks(4, returns, advantages))
        assert len(chunks) == 4  # 16 / 4
        for chunk in chunks:
            assert chunk["end"] - chunk["start"] == 4


# ──────────────────────────────────────────────────────────────────────────────
# Weight persistence round-trip (task 5.5)
# ──────────────────────────────────────────────────────────────────────────────


class TestWeightPersistence:
    """WeightPersistence Protocol conformance + round-trip identical logits."""

    def test_satisfies_weight_persistence_protocol(self) -> None:
        """Brain conforms to the runtime-checkable WeightPersistence Protocol."""
        assert isinstance(_make_brain(), WeightPersistence)

    def test_get_weight_components_exposes_required(self) -> None:
        """The core, readout, critic, encoder, and feature_norm are exposed."""
        brain = _make_brain()
        comps = brain.get_weight_components()
        for required in ("encoder", "feature_norm", "hidden_layers", "readout", "critic"):
            assert required in comps
            assert comps[required].name == required

    def test_get_weight_components_filter(self) -> None:
        """Filtering by name returns only the requested components."""
        brain = _make_brain()
        filtered = brain.get_weight_components(components={"hidden_layers", "critic"})
        assert set(filtered.keys()) == {"hidden_layers", "critic"}

    def test_get_weight_components_unknown_raises(self) -> None:
        """Requesting an unknown component name raises ValueError."""
        brain = _make_brain()
        with pytest.raises(ValueError, match="Unknown weight components"):
            brain.get_weight_components(components={"nonexistent"})

    def test_round_trip_identical_logits(self) -> None:
        """Two brains produce identical logits after a weight round-trip."""
        brain1 = _make_brain()
        brain1.prepare_episode()
        params = _make_params()
        for step in range(32):
            brain1.run_brain(params, top_only=False, top_randomize=False)
            brain1.learn(params, reward=0.1, episode_done=(step == 31))
        brain1.post_process_episode()

        components = brain1.get_weight_components()
        brain2 = _make_brain()
        brain2.load_weight_components(components)

        assert brain2._episode_count == brain1._episode_count
        # Same input + same (zeroed) neuron state -> identical logits.
        brain1.prepare_episode()
        brain2.prepare_episode()
        p = _make_params(0.7)
        torch.testing.assert_close(_logits(brain1, p), _logits(brain2, p))

    def test_round_trip_after_advancing_state(self) -> None:
        """Round-trip identity holds after advancing both brains' state identically."""
        brain1 = _make_brain()
        components = brain1.get_weight_components()
        # brain2 starts from DIFFERENT weights (seed 999) so the round-trip identity
        # genuinely depends on load_weight_components (not on a shared init seed).
        brain2 = _make_brain(seed=999)
        brain2.load_weight_components(components)

        brain1.prepare_episode()
        brain2.prepare_episode()
        # Advance both with the same inputs; the carried state should track.
        for _ in range(5):
            p = _make_params(0.3)
            l1 = _logits(brain1, p)
            l2 = _logits(brain2, p)
            torch.testing.assert_close(l1, l2)
            brain1.run_brain(p, top_only=False, top_randomize=False)
            brain2.run_brain(p, top_only=False, top_randomize=False)


# ──────────────────────────────────────────────────────────────────────────────
# Determinism (task 5.6)
# ──────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    """Determinism under a fixed seed."""

    def test_same_seed_same_initial_logits(self) -> None:
        """Two brains with the same seed produce identical initial logits."""
        b1 = _make_brain(seed=123)
        b2 = _make_brain(seed=123)
        b1.prepare_episode()
        b2.prepare_episode()
        torch.testing.assert_close(_logits(b1, _make_params(0.6)), _logits(b2, _make_params(0.6)))

    def test_same_seed_same_sampled_trajectory(self) -> None:
        """Two brains with the same seed sample the same action sequence."""
        b1 = _make_brain(seed=7)
        b2 = _make_brain(seed=7)
        b1.prepare_episode()
        b2.prepare_episode()
        params = _make_params(0.5)
        actions1 = [
            b1.run_brain(params, top_only=False, top_randomize=False)[0].action for _ in range(10)
        ]
        actions2 = [
            b2.run_brain(params, top_only=False, top_randomize=False)[0].action for _ in range(10)
        ]
        assert actions1 == actions2


# ──────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ──────────────────────────────────────────────────────────────────────────────


def _dummy_state(hidden_size: int = 16, num_actions: int = 4) -> NeuronState:
    """Build a dummy single-layer neuron state for buffer tests."""
    return {
        "v": [torch.randn(1, hidden_size)],
        "a": [torch.zeros(1, hidden_size)],
        "s": [torch.zeros(1, hidden_size)],
        "m": torch.zeros(1, num_actions),
    }


class TestRolloutBuffer:
    """Rollout buffer stores a single neuron state per step."""

    @pytest.fixture
    def buffer(self) -> SpikingPPORolloutBuffer:
        return SpikingPPORolloutBuffer(buffer_size=10, device=torch.device("cpu"))

    def test_add_and_full(self, buffer: SpikingPPORolloutBuffer) -> None:
        """Adding fills the buffer; is_full flips at capacity."""
        assert len(buffer) == 0
        for i in range(10):
            buffer.add(
                features=np.array([0.1 * i] * INPUT_DIM, dtype=np.float32),
                action=i % 4,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=(i == 9),
                state=_dummy_state(),
            )
        assert buffer.is_full()
        assert len(buffer) == 10

    def test_reset(self, buffer: SpikingPPORolloutBuffer) -> None:
        """Reset clears the buffer."""
        for _ in range(5):
            buffer.add(
                features=np.zeros(INPUT_DIM, dtype=np.float32),
                action=0,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=False,
                state=_dummy_state(),
            )
        buffer.reset()
        assert len(buffer) == 0
        assert not buffer.is_full()

    def test_state_stored_detached(self, buffer: SpikingPPORolloutBuffer) -> None:
        """The stored neuron state is detached (no grad ties back to the live graph)."""
        live = _dummy_state()
        live["v"][0].requires_grad_(True)  # noqa: FBT003
        buffer.add(
            features=np.zeros(INPUT_DIM, dtype=np.float32),
            action=0,
            log_prob=-0.5,
            value=0.5,
            reward=0.1,
            done=False,
            state=live,
        )
        assert not buffer.states[0]["v"][0].requires_grad

    def test_gae_computation_positive_returns(self, buffer: SpikingPPORolloutBuffer) -> None:
        """GAE returns/advantages have the right shape; positive rewards -> positive returns."""
        for i in range(5):
            buffer.add(
                features=np.zeros(INPUT_DIM, dtype=np.float32),
                action=0,
                log_prob=-1.0,
                value=float(i) * 0.1,
                reward=1.0,
                done=(i == 4),
                state=_dummy_state(),
            )
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )
        assert returns.shape == (5,)
        assert advantages.shape == (5,)
        assert returns.sum().item() > 0

    def test_episode_boundary_in_chunks(self) -> None:
        """Episode boundaries (done flags) are preserved within chunks."""
        buf = SpikingPPORolloutBuffer(buffer_size=8, device=torch.device("cpu"))
        for i in range(8):
            buf.add(
                features=np.zeros(INPUT_DIM, dtype=np.float32),
                action=0,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=(i == 3),
                state=_dummy_state(),
            )
        chunks = list(buf.get_sequential_chunks(8, torch.ones(8), torch.ones(8)))
        assert len(chunks) == 1
        assert chunks[0]["dones"][3] is True


# ──────────────────────────────────────────────────────────────────────────────
# Protocol compliance (trivial methods)
# ──────────────────────────────────────────────────────────────────────────────


class TestProtocolMethods:
    """Trivial / unsupported Protocol methods behave as the base requires."""

    def test_update_memory_noop(self) -> None:
        """update_memory is a no-op (does not raise)."""
        _make_brain().update_memory(reward=1.0)

    def test_copy_raises(self) -> None:
        """Copy raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _make_brain().copy()

    def test_build_brain_raises(self) -> None:
        """build_brain raises NotImplementedError (no quantum circuit)."""
        with pytest.raises(NotImplementedError):
            _make_brain().build_brain()

    def test_action_set_getter_setter(self) -> None:
        """action_set getter returns a copy; setter validates length."""
        brain = _make_brain()
        original = brain.action_set
        assert len(original) == 4
        # Getter returns a defensive copy.
        original.clear()
        assert len(brain.action_set) == 4
        # Setter rejects a wrong-length list.
        with pytest.raises(ValueError, match="action_set must have exactly"):
            brain.action_set = DEFAULT_ACTIONS[:2]

    def test_post_process_increments_episode_count(self) -> None:
        """post_process_episode increments the schedule-driving episode counter."""
        brain = _make_brain()
        assert brain._episode_count == 0
        brain.post_process_episode()
        assert brain._episode_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# Schedule validators (task 5.8)
# ──────────────────────────────────────────────────────────────────────────────


class TestEntropySchedule:
    """The entropy coefficient is flat by default and linearly annealed when configured."""

    def test_flat_when_no_end(self) -> None:
        """With entropy_coef_end unset, the coefficient stays flat at entropy_coef."""
        brain = _make_brain(entropy_coef=0.05)
        assert brain._get_entropy_coef() == pytest.approx(0.05)
        brain._episode_count = 500
        assert brain._get_entropy_coef() == pytest.approx(0.05)

    def test_linear_anneal_then_holds(self) -> None:
        """entropy_coef_end + entropy_decay_episodes -> linear anneal, then hold at the floor."""
        brain = _make_brain(entropy_coef=0.08, entropy_coef_end=0.02, entropy_decay_episodes=800)
        brain._episode_count = 0
        assert brain._get_entropy_coef() == pytest.approx(0.08)
        brain._episode_count = 400
        assert brain._get_entropy_coef() == pytest.approx(0.05)
        brain._episode_count = 800
        assert brain._get_entropy_coef() == pytest.approx(0.02)
        brain._episode_count = 1200  # past the decay window -> clamped at the floor
        assert brain._get_entropy_coef() == pytest.approx(0.02)

    def test_end_without_decay_raises(self) -> None:
        """entropy_coef_end without entropy_decay_episodes is rejected (no silent flat fallback)."""
        with pytest.raises(ValueError, match="must be set together"):
            _make_brain(entropy_coef_end=0.02)

    def test_decay_without_end_raises(self) -> None:
        """entropy_decay_episodes without entropy_coef_end is rejected (no silent flat fallback)."""
        with pytest.raises(ValueError, match="must be set together"):
            _make_brain(entropy_decay_episodes=800)

    def test_non_positive_decay_raises(self) -> None:
        """A non-positive decay window with a set floor is rejected (0 and negative)."""
        with pytest.raises(ValueError, match="when annealing"):
            _make_brain(entropy_coef_end=0.02, entropy_decay_episodes=0)
        with pytest.raises(ValueError, match="when annealing"):
            _make_brain(entropy_coef_end=0.02, entropy_decay_episodes=-1)

    def test_negative_end_raises(self) -> None:
        """A negative entropy floor is rejected."""
        with pytest.raises(ValueError, match="entropy_coef_end must be >="):
            _make_brain(entropy_coef_end=-0.01, entropy_decay_episodes=800)


class TestSurrogateSlopeSchedule:
    """The surrogate slope is flat by default and linearly annealed when configured."""

    def test_flat_when_unset(self) -> None:
        """With the schedule unset, the slope stays flat at surrogate_slope."""
        brain = _make_brain(surrogate_slope=3.0)
        assert brain._get_surrogate_slope() == pytest.approx(3.0)
        brain._episode_count = 999
        assert brain._get_surrogate_slope() == pytest.approx(3.0)

    def test_linear_anneal_then_holds(self) -> None:
        """surrogate_slope_end + anneal_episodes -> linear anneal, then hold."""
        brain = _make_brain(
            surrogate_slope=2.0,
            surrogate_slope_end=10.0,
            surrogate_slope_anneal_episodes=400,
        )
        brain._episode_count = 0
        assert brain._get_surrogate_slope() == pytest.approx(2.0)
        brain._episode_count = 200
        assert brain._get_surrogate_slope() == pytest.approx(6.0)
        brain._episode_count = 400
        assert brain._get_surrogate_slope() == pytest.approx(10.0)
        brain._episode_count = 800  # past the window -> held
        assert brain._get_surrogate_slope() == pytest.approx(10.0)

    def test_end_without_anneal_raises(self) -> None:
        """surrogate_slope_end without anneal_episodes is rejected."""
        with pytest.raises(ValueError, match="must be set "):
            _make_brain(surrogate_slope_end=10.0)

    def test_anneal_without_end_raises(self) -> None:
        """surrogate_slope_anneal_episodes without end is rejected."""
        with pytest.raises(ValueError, match="must be set "):
            _make_brain(surrogate_slope_anneal_episodes=400)

    def test_non_positive_anneal_raises(self) -> None:
        """A non-positive anneal window with a set end is rejected."""
        with pytest.raises(ValueError, match="when annealing"):
            _make_brain(surrogate_slope_end=10.0, surrogate_slope_anneal_episodes=0)

    def test_non_positive_end_raises(self) -> None:
        """A non-positive surrogate-slope end is rejected."""
        with pytest.raises(ValueError, match="surrogate_slope_end must be > 0"):
            _make_brain(surrogate_slope_end=0.0, surrogate_slope_anneal_episodes=400)


# ──────────────────────────────────────────────────────────────────────────────
# Substrate-level cell tests (task 5.7)
# ──────────────────────────────────────────────────────────────────────────────


class TestSurrogateGradientSpike:
    """Carried-over tests for the surrogate gradient spike function."""

    def test_forward_pass(self) -> None:
        """Forward pass produces binary spikes."""
        v = torch.tensor([[0.5, 1.2, 0.8, 1.5]])
        spikes = SurrogateGradientSpike.apply(v, 1.0, 10.0)
        assert isinstance(spikes, torch.Tensor)
        assert torch.all((spikes == 0) | (spikes == 1))
        assert spikes[0, 0] == 0
        assert spikes[0, 1] == 1
        assert spikes[0, 2] == 0
        assert spikes[0, 3] == 1

    def test_gradient_flow(self) -> None:
        """Gradients flow through the surrogate function."""
        v = torch.tensor([[0.5, 1.2]], requires_grad=True)
        spikes = SurrogateGradientSpike.apply(v, 1.0, 10.0)
        assert isinstance(spikes, torch.Tensor)
        spikes.sum().backward()
        assert v.grad is not None
        assert torch.any(v.grad != 0)
        assert torch.isfinite(v.grad).all()


class TestLIFLayer:
    """Carried-over tests for the (feedforward) LIF layer."""

    def test_forward_single_timestep(self) -> None:
        """Forward pass for a single timestep yields binary spikes + state."""
        layer = LIFLayer(input_dim=2, output_dim=4)
        x = torch.randn(1, 2)
        spikes, state = layer(x, state=None)
        assert spikes.shape == (1, 4)
        assert isinstance(state, tuple)
        v_membrane, _ = state
        assert v_membrane.shape == (1, 4)
        assert torch.all((spikes == 0) | (spikes == 1))

    def test_stateful_dynamics(self) -> None:
        """Membrane state persists across timesteps."""
        layer = LIFLayer(input_dim=2, output_dim=4, tau_m=20.0)
        x = torch.ones(1, 2) * 0.1
        _spikes1, state1 = layer(x, state=None)
        v1, _ = state1
        _spikes2, state2 = layer(x, state=state1)
        v2, _ = state2
        assert not torch.allclose(v1, v2)


class TestRecurrentAdaptiveLIFCell:
    """Substrate-level tests for the recurrent adaptive-LIF cell."""

    def test_membrane_decay_in_unit_interval(self) -> None:
        """The effective membrane decay beta = sigmoid(raw) lies in (0, 1)."""
        cell = RecurrentAdaptiveLIFCell(input_dim=4, num_neurons=4, membrane_decay_init=0.9)
        beta = cell.membrane_decay
        assert torch.all(beta > 0)
        assert torch.all(beta < 1)
        # Init reproduces the requested decay.
        assert torch.allclose(beta, torch.full((4,), 0.9), atol=1e-4)

    def test_forward_shapes_and_binary_spikes(self) -> None:
        """Forward returns binary spikes and a same-shaped (v, a, s) state."""
        cell = RecurrentAdaptiveLIFCell(input_dim=4, num_neurons=4)
        state = cell.init_state(1, torch.device("cpu"))
        spikes, new_state = cell(torch.ones(1, 4), state, slope=2.0)
        assert spikes.shape == (1, 4)
        assert torch.all((spikes == 0) | (spikes == 1))
        assert new_state.v.shape == (1, 4)
        assert new_state.a.shape == (1, 4)
        assert new_state.s.shape == (1, 4)

    def test_decay_integration(self) -> None:
        """Membrane integrates v <- beta*v + (1-beta)*I (no spike at sub-threshold)."""
        # beta init 0.5; sub-threshold input keeps v below threshold so no spike.
        cell = RecurrentAdaptiveLIFCell(
            input_dim=1,
            num_neurons=1,
            membrane_decay_init=0.5,
            adapt_scale_init=0.0,
            v_threshold=10.0,
        )
        with torch.no_grad():
            cell.recurrent.weight.zero_()
        state = cell.init_state(1, torch.device("cpu"))
        current = torch.tensor([[2.0]])
        _, s1 = cell(current, state, slope=2.0)
        # v = 0.5*0 + 0.5*2.0 = 1.0
        assert s1.v.item() == pytest.approx(1.0, abs=1e-5)
        _, s2 = cell(current, s1, slope=2.0)
        # v = 0.5*1.0 + 0.5*2.0 = 1.5
        assert s2.v.item() == pytest.approx(1.5, abs=1e-5)

    def test_soft_reset_subtractive(self) -> None:
        """A spiking neuron's membrane is reduced by the threshold (subtractive reset)."""
        # beta init ~0 so v ~ current; strong current drives a spike.
        cell = RecurrentAdaptiveLIFCell(
            input_dim=1,
            num_neurons=1,
            membrane_decay_init=1e-6,
            adapt_scale_init=0.0,
            v_threshold=1.0,
        )
        with torch.no_grad():
            cell.recurrent.weight.zero_()
        state = cell.init_state(1, torch.device("cpu"))
        spikes, new_state = cell(torch.tensor([[5.0]]), state, slope=2.0)
        assert spikes.item() == 1.0
        # v ~ 5.0 pre-reset; subtractive reset: 5.0 - 1*theta(=1.0) = 4.0.
        assert new_state.v.item() == pytest.approx(4.0, abs=1e-3)

    def test_adaptation_accumulates_and_raises_threshold(self) -> None:
        """Adaptation increments on a spike and the adaptive threshold uses it."""
        cell = RecurrentAdaptiveLIFCell(
            input_dim=1,
            num_neurons=1,
            membrane_decay_init=1e-6,
            adaptation_decay=0.9,
            adapt_scale_init=0.5,
            v_threshold=1.0,
        )
        with torch.no_grad():
            cell.recurrent.weight.zero_()
        state = cell.init_state(1, torch.device("cpu"))
        _, s1 = cell(torch.tensor([[5.0]]), state, slope=2.0)
        # a <- 0.9*0 + 1 = 1.0 after a spike.
        assert s1.a.item() == pytest.approx(1.0, abs=1e-5)
        _, s2 = cell(torch.tensor([[5.0]]), s1, slope=2.0)
        # a <- 0.9*1.0 + 1 = 1.9 after a second spike.
        assert s2.a.item() == pytest.approx(1.9, abs=1e-5)

    def test_recurrent_current_changes_trajectory(self) -> None:
        """Non-zero recurrent weights change the membrane trajectory vs zero weights."""
        cell_a = RecurrentAdaptiveLIFCell(input_dim=3, num_neurons=3, membrane_decay_init=0.3)
        cell_b = RecurrentAdaptiveLIFCell(input_dim=3, num_neurons=3, membrane_decay_init=0.3)
        with torch.no_grad():
            cell_a.recurrent.weight.zero_()
            cell_b.recurrent.weight.fill_(0.5)
            # Match all non-recurrent params so the recurrent current is the
            # only difference.
            cell_b.raw_membrane_decay.copy_(cell_a.raw_membrane_decay)
            cell_b.adapt_scale.copy_(cell_a.adapt_scale)
        sa = cell_a.init_state(1, torch.device("cpu"))
        sb = cell_b.init_state(1, torch.device("cpu"))
        current = torch.ones(1, 3) * 3.0
        for _ in range(4):
            _, sa = cell_a(current, sa, slope=2.0)
            _, sb = cell_b(current, sb, slope=2.0)
        assert not torch.allclose(sa.v, sb.v)

    def test_surrogate_gradient_finite_and_nonzero(self) -> None:
        """Gradient flows through the spike (surrogate) into the recurrent + decay params."""
        cell = RecurrentAdaptiveLIFCell(input_dim=3, num_neurons=3, membrane_decay_init=0.3)
        state = cell.init_state(1, torch.device("cpu"))
        current = torch.ones(1, 3) * 3.0
        outs: list[torch.Tensor] = []
        for _ in range(3):
            spikes, state = cell(current, state, slope=2.0)
            outs.append(spikes)
        loss = torch.stack(outs).sum() + state.v.sum()
        loss.backward()
        assert cell.recurrent.weight.grad is not None
        assert torch.isfinite(cell.recurrent.weight.grad).all()
        assert torch.any(cell.recurrent.weight.grad != 0)
        assert cell.raw_membrane_decay.grad is not None
        assert torch.isfinite(cell.raw_membrane_decay.grad).all()

    def test_slope_runtime_argument(self) -> None:
        """The slope is a per-forward argument and affects the surrogate gradient magnitude."""

        def grad_for_slope(slope: float) -> float:
            cell = RecurrentAdaptiveLIFCell(
                input_dim=1,
                num_neurons=1,
                membrane_decay_init=0.5,
                adapt_scale_init=0.0,
                v_threshold=1.0,
            )
            with torch.no_grad():
                cell.recurrent.weight.zero_()
            state = cell.init_state(1, torch.device("cpu"))
            # Drive the membrane right to the threshold where the surrogate
            # derivative is most slope-sensitive: v = (1-beta)*I = 0.5*2 = 1.0.
            current = torch.tensor([[2.0]], requires_grad=True)
            spikes, _ = cell(current, state, slope=slope)
            spikes.sum().backward()
            assert current.grad is not None
            return float(current.grad.abs().sum())

        g_shallow = grad_for_slope(1.0)
        g_sharp = grad_for_slope(8.0)
        assert g_shallow > 0
        assert g_sharp > 0
        # At the threshold the sharper slope yields a larger surrogate derivative.
        assert g_sharp > g_shallow


class TestLeakyIntegratorReadout:
    """Substrate-level tests for the non-spiking leaky-integrator readout."""

    def test_output_is_membrane(self) -> None:
        """The readout output IS the new membrane."""
        ro = LeakyIntegratorReadout(input_dim=4, output_dim=4, readout_decay_init=0.9)
        m = ro.init_state(1, torch.device("cpu"))
        out, new_m = ro(torch.ones(1, 4), m)
        assert torch.allclose(out, new_m)
        assert out.shape == (1, 4)

    def test_membrane_carries(self) -> None:
        """The output membrane integrates spikes across calls (carried state)."""
        ro = LeakyIntegratorReadout(input_dim=4, output_dim=4, readout_decay_init=0.9)
        m = ro.init_state(1, torch.device("cpu"))
        _, m1 = ro(torch.ones(1, 4), m)
        _, m2 = ro(torch.ones(1, 4), m1)
        assert not torch.allclose(m1, m2)

    def test_decay_in_unit_interval(self) -> None:
        """The effective readout decay lies in (0, 1) and reproduces the init."""
        ro = LeakyIntegratorReadout(input_dim=2, output_dim=2, readout_decay_init=0.8)
        decay = float(ro.readout_decay.detach())
        assert 0.0 < decay < 1.0
        assert decay == pytest.approx(0.8, abs=1e-4)
