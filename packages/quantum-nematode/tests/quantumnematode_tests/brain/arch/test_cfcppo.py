"""Unit tests for the CfC (Closed-form Continuous-time) PPO brain architecture.

Covers the spec scenarios at
``openspec/changes/add-cfc-liquid-brain/specs/cfc-liquid-brain/spec.md``:
CfC + AutoNCP topology construction (motor + mlp heads); forward-pass shape +
finite logits/value over the 4-action set; recurrent hidden carry + reset;
non-degenerate logit variance; truncated-BPTT PPO update leaving params finite;
the motor-head learnable logit temperature producing a peaked policy; too-small
units rejection; WeightPersistence Protocol conformance + round-trip; registry +
BrainType + instantiate_brain dispatch; determinism under a fixed seed.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import DEFAULT_ACTIONS, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._registry import (
    Registration,
    get_registration,
    instantiate_brain,
    list_registered_brains,
)
from quantumnematode.brain.arch.cfc_ppo import (
    CfCBrainConfig,
    CfCPPOBrain,
    CfCPPORolloutBuffer,
)
from quantumnematode.brain.arch.dtypes import BrainType, DeviceType
from quantumnematode.brain.modules import ModuleName
from quantumnematode.brain.weights import WeightPersistence
from quantumnematode.env import Direction

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SENSORY_MODULES = [ModuleName.FOOD_CHEMOTAXIS, ModuleName.PROPRIOCEPTION]
INPUT_DIM = 4  # 2 modules * 2 features each
_SEED = 42


def _make_config(**overrides: object) -> CfCBrainConfig:
    """Create a config with sensory modules and small defaults for fast tests."""
    defaults: dict[str, object] = {
        "sensory_modules": SENSORY_MODULES,
        "units": 16,
        "rollout_buffer_size": 32,
        "bptt_chunk_length": 8,
        "critic_hidden_dim": 16,
        "actor_hidden_dim": 16,
        "num_epochs": 2,
        "seed": _SEED,
    }
    defaults.update(overrides)
    return CfCBrainConfig(**defaults)  # type: ignore[arg-type]


def _make_brain(**overrides: object) -> CfCPPOBrain:
    """Create a brain with small defaults for fast tests."""
    config = _make_config(**overrides)
    return CfCPPOBrain(config=config, num_actions=4, device=DeviceType.CPU)


def _make_params(grad_strength: float = 0.5, direction: Direction = Direction.UP) -> BrainParams:
    """Create a BrainParams for testing."""
    return BrainParams(
        food_gradient_strength=grad_strength,
        food_gradient_direction=np.pi / 2,
        agent_direction=direction,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Config Validation
# ──────────────────────────────────────────────────────────────────────────────


class TestCfCBrainConfig:
    """Test cases for CfC PPO brain configuration."""

    def test_default_config(self) -> None:
        """Default configuration values match the spec."""
        config = CfCBrainConfig(sensory_modules=SENSORY_MODULES)
        assert config.units == 32
        assert config.ncp_sparsity == 0.5
        assert config.cfc_mode == "default"
        assert config.actor_head == "motor"
        assert config.motor_logit_scale_init == 1.0
        assert config.actor_hidden_dim == 64
        assert config.actor_num_layers == 2
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
        config = CfCBrainConfig(
            sensory_modules=SENSORY_MODULES,
            units=64,
            ncp_sparsity=0.3,
            cfc_mode="pure",
            actor_head="mlp",
        )
        assert config.units == 64
        assert config.ncp_sparsity == 0.3
        assert config.cfc_mode == "pure"
        assert config.actor_head == "mlp"

    def test_sensory_modules_required(self) -> None:
        """sensory_modules=None is rejected."""
        with pytest.raises(ValueError, match="sensory_modules is required"):
            CfCBrainConfig()

    def test_sensory_modules_non_empty(self) -> None:
        """Empty sensory_modules is rejected."""
        with pytest.raises(ValueError, match="sensory_modules is required"):
            CfCBrainConfig(sensory_modules=[])

    def test_invalid_bptt_chunk_length(self) -> None:
        """bptt_chunk_length < 4 is rejected."""
        with pytest.raises(ValueError, match="bptt_chunk_length must be >= 4"):
            CfCBrainConfig(sensory_modules=SENSORY_MODULES, bptt_chunk_length=2)

    def test_invalid_rollout_buffer_size(self) -> None:
        """rollout_buffer_size < bptt_chunk_length is rejected."""
        with pytest.raises(ValueError, match="rollout_buffer_size"):
            CfCBrainConfig(
                sensory_modules=SENSORY_MODULES,
                rollout_buffer_size=8,
                bptt_chunk_length=16,
            )

    def test_invalid_ncp_sparsity(self) -> None:
        """ncp_sparsity outside [0, 1) is rejected."""
        with pytest.raises(ValueError, match="ncp_sparsity"):
            CfCBrainConfig(sensory_modules=SENSORY_MODULES, ncp_sparsity=1.0)

    def test_invalid_network_and_lr_params(self) -> None:
        """Non-positive actor/critic sizes, layer counts, and learning rates are rejected."""
        with pytest.raises(ValueError, match="actor_hidden_dim"):
            CfCBrainConfig(sensory_modules=SENSORY_MODULES, actor_hidden_dim=0)
        with pytest.raises(ValueError, match="critic_hidden_dim"):
            CfCBrainConfig(sensory_modules=SENSORY_MODULES, critic_hidden_dim=0)
        with pytest.raises(ValueError, match="actor_num_layers"):
            CfCBrainConfig(sensory_modules=SENSORY_MODULES, actor_num_layers=0)
        with pytest.raises(ValueError, match="critic_num_layers"):
            CfCBrainConfig(sensory_modules=SENSORY_MODULES, critic_num_layers=0)
        with pytest.raises(ValueError, match="actor_lr"):
            CfCBrainConfig(sensory_modules=SENSORY_MODULES, actor_lr=0.0)
        with pytest.raises(ValueError, match="critic_lr"):
            CfCBrainConfig(sensory_modules=SENSORY_MODULES, critic_lr=-0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Registry + BrainType + Protocol conformance (task 5.1)
# ──────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    """Registry / Protocol conformance: brain builds via the registry."""

    def test_brain_self_registers_at_import(self) -> None:
        """The @register_brain decorator places cfcppo in the registry."""
        assert "cfcppo" in list_registered_brains()

    def test_registration_metadata(self) -> None:
        """The Registration carries the expected fields."""
        reg = get_registration("cfcppo")
        assert isinstance(reg, Registration)
        assert reg.name == "cfcppo"
        assert reg.config_cls is CfCBrainConfig
        assert reg.brain_cls is CfCPPOBrain
        assert reg.brain_type is BrainType.CFC_PPO
        assert reg.families == ("classical",)

    def test_brain_type_enum_value(self) -> None:
        """BrainType.CFC_PPO value matches the registered name."""
        assert BrainType.CFC_PPO.value == "cfcppo"

    def test_instantiate_via_registry(self) -> None:
        """instantiate_brain() dispatches via the registry without per-arch branches."""
        brain = instantiate_brain("cfcppo", _make_config(), device=DeviceType.CPU)
        assert isinstance(brain, CfCPPOBrain)

    def test_satisfies_classical_brain_protocol(self) -> None:
        """Brain is a runtime ClassicalBrain (has action_set + learn)."""
        from quantumnematode.brain.arch._brain import ClassicalBrain

        brain = _make_brain()
        assert isinstance(brain, ClassicalBrain)


# ──────────────────────────────────────────────────────────────────────────────
# Construction + topology (motor head; spec: builds CfC + AutoNCP)
# ──────────────────────────────────────────────────────────────────────────────


class TestConstructionMotorHead:
    """Brain construction builds the CfC + AutoNCP topology (default motor head)."""

    def test_constructs_cfc_and_critic(self) -> None:
        """CfC core + critic MLP built; input_dim derived from sensory modules."""
        from ncps.torch import CfC

        brain = _make_brain()
        assert brain.input_dim == INPUT_DIM
        assert brain.num_actions == 4
        assert brain.units == 16
        assert isinstance(brain.cfc, CfC)
        assert hasattr(brain, "critic")
        # Hidden state is single (1, units) — no cell state.
        assert brain.h_t.shape == (1, 16)
        assert not hasattr(brain, "c_t")

    def test_motor_head_has_no_actor_mlp(self) -> None:
        """In motor mode the logit_scale exists and no actor MLP is constructed."""
        brain = _make_brain(actor_head="motor")
        assert brain.actor_head == "motor"
        assert brain.actor is None
        assert isinstance(brain.logit_scale, torch.nn.Parameter)
        assert float(brain.logit_scale.detach()) == pytest.approx(1.0)

    def test_logit_scale_in_actor_optimizer(self) -> None:
        """The logit_scale parameter is among the actor optimizer's parameters."""
        brain = _make_brain(actor_head="motor")
        actor_param_ids = {id(p) for p in brain.actor_optimizer.param_groups[0]["params"]}
        assert brain.logit_scale is not None
        assert id(brain.logit_scale) in actor_param_ids

    def test_separate_optimizers_no_overlap(self) -> None:
        """Actor and critic have separate optimizers covering disjoint params."""
        brain = _make_brain()
        actor_params = {id(p) for p in brain.actor_optimizer.param_groups[0]["params"]}
        critic_params = {id(p) for p in brain.critic_optimizer.param_groups[0]["params"]}
        assert brain.actor_optimizer is not brain.critic_optimizer
        assert len(actor_params & critic_params) == 0
        # Actor optimizer covers CfC + feature_norm.
        for p in brain.cfc.parameters():
            assert id(p) in actor_params
        for p in brain.feature_norm.parameters():
            assert id(p) in actor_params

    def test_too_small_units_raises_clear_error(self) -> None:
        """Units <= num_actions + 2 raises a clear AutoNCP-minimum error."""
        # num_actions=4 -> need units > 6; units=6 must raise.
        with pytest.raises(ValueError, match="units > num_actions"):
            _make_brain(units=6)

    def test_action_set_length_must_match_num_actions(self) -> None:
        """Action-set length mismatch with num_actions raises at construction."""
        with pytest.raises(ValueError, match="action_set must have exactly"):
            CfCPPOBrain(
                config=_make_config(),
                num_actions=4,
                device=DeviceType.CPU,
                action_set=DEFAULT_ACTIONS[:2],
            )


# ──────────────────────────────────────────────────────────────────────────────
# Construction with the MLP actor head (task 5.7)
# ──────────────────────────────────────────────────────────────────────────────


class TestConstructionMlpHead:
    """Brain construction with the MLP actor head."""

    def test_mlp_head_builds_actor_mlp(self) -> None:
        """In mlp mode the actor MLP exists (units -> ... -> num_actions), no logit_scale."""
        brain = _make_brain(actor_head="mlp", actor_hidden_dim=16, actor_num_layers=2)
        assert brain.actor_head == "mlp"
        assert brain.actor is not None
        assert brain.logit_scale is None
        # First Linear maps from the hidden-state width (units); last maps to num_actions.
        first_linear = brain.actor[0]
        last_linear = brain.actor[-1]
        assert isinstance(first_linear, torch.nn.Linear)
        assert first_linear.in_features == brain.units
        assert isinstance(last_linear, torch.nn.Linear)
        assert last_linear.out_features == brain.num_actions

    def test_mlp_head_uses_same_cfc_autoncp_core(self) -> None:
        """The mlp head builds the same CfC + AutoNCP recurrent core."""
        from ncps.torch import CfC

        brain = _make_brain(actor_head="mlp")
        assert isinstance(brain.cfc, CfC)
        assert brain.h_t.shape == (1, brain.units)

    def test_mlp_head_actor_in_optimizer(self) -> None:
        """The actor MLP params are in the actor optimizer (and disjoint from critic)."""
        brain = _make_brain(actor_head="mlp")
        actor_params = {id(p) for p in brain.actor_optimizer.param_groups[0]["params"]}
        assert brain.actor is not None
        for p in brain.actor.parameters():
            assert id(p) in actor_params

    def test_mlp_head_forward_and_ppo_step(self) -> None:
        """The mlp head forward-passes (finite num_actions vector) and runs a PPO step."""
        brain = _make_brain(actor_head="mlp", rollout_buffer_size=16, bptt_chunk_length=4)
        brain.prepare_episode()
        params = _make_params()

        actions = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(actions) == 1
        assert isinstance(actions[0], ActionData)

        for step in range(20):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=(step == 19))
        assert brain.history_data.losses, "mlp head should produce losses"
        for loss in brain.history_data.losses:
            assert np.isfinite(loss)


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass: finite logits + value over the action set (task 5.2)
# ──────────────────────────────────────────────────────────────────────────────


class TestForwardPass:
    """Forward pass produces finite logits and a value over the action set."""

    def _logits_value(self, brain: CfCPPOBrain, params: BrainParams) -> tuple[torch.Tensor, float]:
        feat = torch.tensor(brain.preprocess(params), dtype=torch.float32)
        with torch.no_grad():
            normalized = brain.feature_norm(feat)
            motor_out, h_new = brain._cfc_forward(normalized, brain.h_t)
            logits = brain._logits_from_hidden(motor_out, h_new)
            value = brain.critic(h_new.squeeze(0).detach()).item()
        return logits, value

    def test_logits_shape_and_finite(self) -> None:
        """Logits are a finite num_actions-vector; value is a finite scalar."""
        brain = _make_brain()
        brain.prepare_episode()
        logits, value = self._logits_value(brain, _make_params())
        assert logits.shape == (brain.num_actions,)
        assert torch.isfinite(logits).all()
        assert np.isfinite(value)

    def test_run_brain_returns_action_data(self) -> None:
        """run_brain returns a single ActionData drawn from DEFAULT_ACTIONS."""
        brain = _make_brain()
        brain.prepare_episode()
        results = brain.run_brain(_make_params(), top_only=False, top_randomize=False)
        assert len(results) == 1
        assert results[0].action in DEFAULT_ACTIONS
        assert 0.0 <= results[0].probability <= 1.0

    def test_output_dim_equals_num_actions(self) -> None:
        """The CfC motor-output dimensionality equals num_actions."""
        brain = _make_brain()
        feat = torch.tensor(brain.preprocess(_make_params()), dtype=torch.float32)
        with torch.no_grad():
            normalized = brain.feature_norm(feat)
            motor_out, _ = brain._cfc_forward(normalized, brain.h_t)
        assert motor_out.shape == (brain.num_actions,)


# ──────────────────────────────────────────────────────────────────────────────
# Hidden-state carry + reset (task 5.3)
# ──────────────────────────────────────────────────────────────────────────────


class TestHiddenState:
    """Recurrent hidden state carries within an episode and resets at boundaries."""

    def test_hidden_carries_within_episode(self) -> None:
        """Hidden state changes across steps (carried into the recurrence)."""
        brain = _make_brain()
        brain.prepare_episode()
        h_before = brain.h_t.clone()
        brain.run_brain(_make_params(), top_only=False, top_randomize=False)
        assert not torch.equal(h_before, brain.h_t)

    def test_prepare_episode_resets_hidden_to_zero(self) -> None:
        """prepare_episode() resets the hidden state to zeros."""
        brain = _make_brain()
        brain.prepare_episode()
        for _ in range(3):
            brain.run_brain(_make_params(), top_only=False, top_randomize=False)
            brain.learn(_make_params(), reward=0.1, episode_done=False)
        assert not torch.all(brain.h_t == 0)

        brain.prepare_episode()
        assert torch.all(brain.h_t == 0)

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


# ──────────────────────────────────────────────────────────────────────────────
# Non-degenerate variance (spec variance scenario, unit level)
# ──────────────────────────────────────────────────────────────────────────────


class TestVariance:
    """Forward pass produces non-degenerate variance across a sample of passes."""

    def test_logit_variance_strictly_positive(self) -> None:
        """Across >= 100 varied forward passes, per-action logit variance > 0.

        The logits SHALL not collapse to a constant action across the sample
        (interpreted at the categorical-sampling output surface).
        """
        brain = _make_brain()
        brain.prepare_episode()
        rng = np.random.default_rng(0)
        all_logits: list[torch.Tensor] = []
        sampled_actions: list[str] = []
        for _ in range(128):
            params = BrainParams(
                food_gradient_strength=float(rng.uniform(0.0, 1.0)),
                food_gradient_direction=float(rng.uniform(-np.pi, np.pi)),
                agent_direction=Direction.UP,
            )
            feat = torch.tensor(brain.preprocess(params), dtype=torch.float32)
            with torch.no_grad():
                normalized = brain.feature_norm(feat)
                motor_out, h_new = brain._cfc_forward(normalized, brain.h_t)
                logits = brain._logits_from_hidden(motor_out, h_new)
            all_logits.append(logits)
            result = brain.run_brain(params, top_only=False, top_randomize=False)
            action = result[0].action
            assert action is not None  # discrete brain always emits an action
            sampled_actions.append(action)

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
        for p in brain.cfc.parameters():
            assert torch.isfinite(p).all()
        for p in brain.critic.parameters():
            assert torch.isfinite(p).all()
        assert brain.logit_scale is not None
        assert torch.isfinite(brain.logit_scale).all()

    def test_ppo_replays_over_chunks(self) -> None:
        """A buffer larger than the chunk length yields multiple BPTT chunks."""
        buf = CfCPPORolloutBuffer(buffer_size=16, device=torch.device("cpu"))
        for i in range(16):
            buf.add(
                features=np.array([float(i)] * INPUT_DIM, dtype=np.float32),
                action=i % 4,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=False,
                h_state=torch.randn(16),
            )
        returns = torch.ones(16)
        advantages = torch.ones(16)
        chunks = list(buf.get_sequential_chunks(4, returns, advantages))
        assert len(chunks) == 4  # 16 / 4
        for chunk in chunks:
            start = chunk["start"]
            torch.testing.assert_close(chunk["h_init"], buf.h_states[start])
            assert chunk["end"] - start == 4


# ──────────────────────────────────────────────────────────────────────────────
# Motor-head learnable logit temperature (task 5.8 / spec peaked-policy scenario)
# ──────────────────────────────────────────────────────────────────────────────


class TestMotorLogitScale:
    """Motor head applies a learnable logit temperature that can peak the policy."""

    def test_large_logit_scale_peaks_policy(self) -> None:
        """With logit_scale raised large, one action softmax probability > 0.9."""
        brain = _make_brain(actor_head="motor")
        brain.prepare_episode()
        assert brain.logit_scale is not None
        with torch.no_grad():
            brain.logit_scale.copy_(torch.tensor(50.0))

        peaked = False
        for grad in (0.1, 0.4, 0.7, 0.95):
            feat = torch.tensor(brain.preprocess(_make_params(grad)), dtype=torch.float32)
            with torch.no_grad():
                normalized = brain.feature_norm(feat)
                motor_out, h_new = brain._cfc_forward(normalized, brain.h_t)
                logits = brain._logits_from_hidden(motor_out, h_new)
                probs = torch.softmax(logits, dim=-1)
            if float(probs.max()) > 0.9:
                peaked = True
                break
        assert peaked, "Raising logit_scale should be able to peak the policy (>0.9)"

    def test_low_logit_scale_does_not_peak(self) -> None:
        """With a tiny logit_scale the policy stays near-uniform (sanity baseline)."""
        brain = _make_brain(actor_head="motor")
        brain.prepare_episode()
        assert brain.logit_scale is not None
        with torch.no_grad():
            brain.logit_scale.copy_(torch.tensor(0.001))
        feat = torch.tensor(brain.preprocess(_make_params()), dtype=torch.float32)
        with torch.no_grad():
            normalized = brain.feature_norm(feat)
            motor_out, h_new = brain._cfc_forward(normalized, brain.h_t)
            logits = brain._logits_from_hidden(motor_out, h_new)
            probs = torch.softmax(logits, dim=-1)
        # Near-uniform: max prob close to 1/num_actions.
        assert float(probs.max()) < 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Weight persistence round-trip (task 5.5)
# ──────────────────────────────────────────────────────────────────────────────


class TestWeightPersistence:
    """WeightPersistence Protocol conformance + round-trip identical logits."""

    def test_satisfies_weight_persistence_protocol(self) -> None:
        """Brain conforms to the runtime-checkable WeightPersistence Protocol."""
        assert isinstance(_make_brain(), WeightPersistence)

    def test_get_weight_components_exposes_required(self) -> None:
        """At minimum cfc, critic, and feature_norm are exposed."""
        brain = _make_brain()
        comps = brain.get_weight_components()
        for required in ("cfc", "critic", "feature_norm"):
            assert required in comps
            assert comps[required].name == required

    def test_get_weight_components_filter(self) -> None:
        """Filtering by name returns only the requested components."""
        brain = _make_brain()
        filtered = brain.get_weight_components(components={"cfc", "critic"})
        assert set(filtered.keys()) == {"cfc", "critic"}

    def test_get_weight_components_unknown_raises(self) -> None:
        """Requesting an unknown component name raises ValueError."""
        brain = _make_brain()
        with pytest.raises(ValueError, match="Unknown weight components"):
            brain.get_weight_components(components={"nonexistent"})

    def _logits(self, brain: CfCPPOBrain, params: BrainParams) -> torch.Tensor:
        feat = torch.tensor(brain.preprocess(params), dtype=torch.float32)
        with torch.no_grad():
            normalized = brain.feature_norm(feat)
            motor_out, h_new = brain._cfc_forward(normalized, brain.h_t)
            return brain._logits_from_hidden(motor_out, h_new)

    def test_round_trip_identical_logits_motor(self) -> None:
        """Two motor-head brains produce identical logits after a weight round-trip."""
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
        # Same input + same (zeroed) hidden state -> identical logits.
        brain1.prepare_episode()
        brain2.prepare_episode()
        p = _make_params(0.7)
        torch.testing.assert_close(self._logits(brain1, p), self._logits(brain2, p))

    def test_round_trip_identical_logits_mlp(self) -> None:
        """Two mlp-head brains produce identical logits after a weight round-trip."""
        brain1 = _make_brain(actor_head="mlp")
        components = brain1.get_weight_components()
        brain2 = _make_brain(actor_head="mlp")
        brain2.load_weight_components(components)

        brain1.prepare_episode()
        brain2.prepare_episode()
        p = _make_params(0.3)
        torch.testing.assert_close(self._logits(brain1, p), self._logits(brain2, p))


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

        def logits(b: CfCPPOBrain) -> torch.Tensor:
            feat = torch.tensor(b.preprocess(_make_params(0.6)), dtype=torch.float32)
            with torch.no_grad():
                nz = b.feature_norm(feat)
                mo, hn = b._cfc_forward(nz, b.h_t)
                return b._logits_from_hidden(mo, hn)

        torch.testing.assert_close(logits(b1), logits(b2))

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


class TestRolloutBuffer:
    """Rollout buffer stores a single hidden state per step (no cell state)."""

    @pytest.fixture
    def buffer(self) -> CfCPPORolloutBuffer:
        return CfCPPORolloutBuffer(buffer_size=10, device=torch.device("cpu"))

    def test_add_and_full(self, buffer: CfCPPORolloutBuffer) -> None:
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
                h_state=torch.zeros(16),
            )
        assert buffer.is_full()
        assert len(buffer) == 10

    def test_reset(self, buffer: CfCPPORolloutBuffer) -> None:
        """Reset clears the buffer."""
        for _ in range(5):
            buffer.add(
                features=np.zeros(INPUT_DIM, dtype=np.float32),
                action=0,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=False,
                h_state=torch.zeros(16),
            )
        buffer.reset()
        assert len(buffer) == 0
        assert not buffer.is_full()

    def test_gae_computation_positive_returns(self, buffer: CfCPPORolloutBuffer) -> None:
        """GAE returns/advantages have the right shape; positive rewards -> positive returns."""
        for i in range(5):
            buffer.add(
                features=np.zeros(INPUT_DIM, dtype=np.float32),
                action=0,
                log_prob=-1.0,
                value=float(i) * 0.1,
                reward=1.0,
                done=(i == 4),
                h_state=torch.zeros(16),
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
        buf = CfCPPORolloutBuffer(buffer_size=8, device=torch.device("cpu"))
        for i in range(8):
            buf.add(
                features=np.zeros(INPUT_DIM, dtype=np.float32),
                action=0,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=(i == 3),
                h_state=torch.zeros(8),
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
