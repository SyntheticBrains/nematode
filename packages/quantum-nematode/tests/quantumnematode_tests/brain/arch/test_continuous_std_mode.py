# pyright: reportPrivateUsage=false
"""Tests for the state-dependent continuous std mode (roadmap D7).

Covers the `Continuous Action Std Modes` requirement of
``openspec/changes/add-state-dependent-action-std/specs/brain-architecture/spec.md``:
off-mode byte-identity, on-at-init parity as a run property (RNG-free head
allocation), negative-space attribute checks, the load-time mode validator,
cross-mode weight-load errors, the shape pin on recurrent per-step paths,
gradient flow into the head, and the clamp-ceiling monitor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from pydantic import ValidationError
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.cfc_ppo import CfCBrainConfig, CfCPPOBrain
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.arch.lstmppo import LSTMPPOBrain, LSTMPPOBrainConfig
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.arch.transformer_ppo import (
    TransformerPPOBrain,
    TransformerPPOBrainConfig,
)
from quantumnematode.brain.modules import ModuleName

if TYPE_CHECKING:
    from quantumnematode.brain.arch._brain import Brain

_SEED = 2026

_BRAINS: dict[str, tuple[type, type]] = {
    "mlpppo": (MLPPPOBrain, MLPPPOBrainConfig),
    "cfcppo": (CfCPPOBrain, CfCBrainConfig),
    "lstmppo": (LSTMPPOBrain, LSTMPPOBrainConfig),
    "transformerppo": (TransformerPPOBrain, TransformerPPOBrainConfig),
    "connectomeppo": (ConnectomePPOBrain, ConnectomePPOBrainConfig),
}


_REQUIRED: dict[str, dict[str, object]] = {
    "mlpppo": {"sensory_modules": [ModuleName.FOOD_CHEMOTAXIS]},
    "cfcppo": {"sensory_modules": [ModuleName.FOOD_CHEMOTAXIS]},
    "lstmppo": {"sensory_modules": [ModuleName.FOOD_CHEMOTAXIS]},
    "transformerppo": {"sensory_modules": [ModuleName.FOOD_CHEMOTAXIS]},
    "connectomeppo": {},
}


def _make(name: str, **cfg_overrides: object) -> Brain:
    brain_cls, cfg_cls = _BRAINS[name]
    kwargs: dict[str, object] = {**_REQUIRED[name], **cfg_overrides}
    cfg = cfg_cls(seed=_SEED, action_mode="continuous", **kwargs)  # type: ignore[call-arg]
    return brain_cls(config=cfg, device=DeviceType.CPU)


def _params(strength: float = 0.42, angle: float = 0.13) -> BrainParams:
    return BrainParams(
        food_gradient_strength=strength,
        food_gradient_direction=angle,
    )


def _act(brain: Brain) -> tuple[float, float]:
    actions = brain.run_brain(
        _params(),
        reward=None,
        input_data=None,
        top_only=False,
        top_randomize=False,
    )
    continuous = actions[0].continuous
    assert continuous is not None
    return continuous


def _named_params(brain: Brain) -> dict[str, torch.Tensor]:
    modules = {}
    for attr in ("actor", "critic", "rnn", "encoder", "cfc", "topology", "log_std_head"):
        mod = getattr(brain, attr, None)
        if isinstance(mod, torch.nn.Module):
            modules[attr] = mod
    out: dict[str, torch.Tensor] = {}
    for prefix, mod in modules.items():
        for pname, p in mod.named_parameters():
            out[f"{prefix}.{pname}"] = p
    log_std = getattr(brain, "log_std", None)
    if isinstance(log_std, torch.nn.Parameter):
        out["log_std"] = log_std
    return out


class TestModeValidator:
    """`state_dependent` beside `discrete` fails at load (review S5)."""

    def test_state_dependent_with_discrete_raises(self) -> None:
        with pytest.raises(ValidationError, match="requires action_mode"):
            BrainConfig(action_mode="discrete", continuous_std_mode="state_dependent")

    def test_state_dependent_with_continuous_parses(self) -> None:
        cfg = BrainConfig(action_mode="continuous", continuous_std_mode="state_dependent")
        assert cfg.continuous_std_mode == "state_dependent"


@pytest.mark.parametrize("name", list(_BRAINS))
class TestOffModeByteIdentity:
    """`Default mode is byte-identical` scenario."""

    def test_default_equals_explicit_off(self, name: str) -> None:
        default = _make(name)
        explicit = _make(name, continuous_std_mode="state_independent")
        a, b = _named_params(default), _named_params(explicit)
        assert a.keys() == b.keys()
        for key, p_a in a.items():
            assert torch.equal(p_a, b[key]), key

    def test_negative_space(self, name: str) -> None:
        off = _make(name)
        on = _make(name, continuous_std_mode="state_dependent")
        off_owner = off.topology if name == "connectomeppo" else off
        on_owner = on.topology if name == "connectomeppo" else on
        assert not hasattr(off_owner, "log_std_head")
        assert not hasattr(on_owner, "log_std")
        assert hasattr(on_owner, "log_std_head")


@pytest.mark.parametrize("name", list(_BRAINS))
class TestOnAtInitParity:
    """`State-dependent mode matches at initialisation` — a run property."""

    def test_step0_action_byte_identical(self, name: str) -> None:
        off = _make(name)
        action_off = _act(off)
        on = _make(name, continuous_std_mode="state_dependent")
        action_on = _act(on)
        assert action_off == action_on

    def test_head_outputs_zero_log_std_at_init(self, name: str) -> None:
        on = _make(name, continuous_std_mode="state_dependent")
        owner = on.topology if name == "connectomeppo" else on
        head = owner.log_std_head
        assert torch.all(head.weight == 0)
        assert torch.all(head.bias == 0)


class TestShapePins:
    """Recurrent per-step head outputs match the mean's shape (task 3.4)."""

    def test_cfc_per_step_shape(self) -> None:
        brain = _make("cfcppo", continuous_std_mode="state_dependent")
        _act(brain)
        out = brain.log_std_head(brain.h_t.squeeze(0))
        assert out.shape == (2,)

    def test_lstm_per_step_shape(self) -> None:
        brain = _make("lstmppo", continuous_std_mode="state_dependent")
        _act(brain)
        out = brain.log_std_head(brain.h_t.squeeze(0).squeeze(0))
        assert out.shape == (2,)

    def test_connectome_accessor_handles_both_shapes(self) -> None:
        brain = _make("connectomeppo", continuous_std_mode="state_dependent")
        single = brain.topology.state_dependent_log_std(torch.zeros(302))
        batched = brain.topology.state_dependent_log_std(torch.zeros(5, 302))
        assert single.shape == (2,)
        assert batched.shape == (5, 2)


@pytest.mark.parametrize("name", ["mlpppo", "transformerppo", "connectomeppo"])
class TestGradientFlowAndMonitor:
    """A training update moves the head and records the ceiling monitor."""

    def test_update_moves_head_and_logs_monitor(self, name: str) -> None:
        overrides: dict[str, object] = {"continuous_std_mode": "state_dependent"}
        if name != "connectomeppo":
            overrides["rollout_buffer_size"] = 32
        brain = _make(name, **overrides)
        owner = brain.topology if name == "connectomeppo" else brain
        before = {
            "weight": owner.log_std_head.weight.detach().clone(),
            "bias": owner.log_std_head.bias.detach().clone(),
        }
        torch.manual_seed(_SEED + 1)
        for step in range(8):
            brain.run_brain(
                _params(strength=0.3 + 0.05 * step, angle=0.1 * step - 0.3),
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            brain.learn(_params(), reward=0.1 * (step % 3), episode_done=(step == 7))
        moved = not torch.equal(before["weight"], owner.log_std_head.weight) or not torch.equal(
            before["bias"],
            owner.log_std_head.bias,
        )
        assert moved
        assert len(brain.history_data.log_std_clamped_mean) >= 1
        assert len(brain.history_data.log_std_clamped_max) >= 1
        assert brain.history_data.log_std_clamped_max[0] <= 2.0


class TestCrossModePersistence:
    """Same-mode round-trips work; cross-mode loads fail loudly (review S3)."""

    @pytest.mark.parametrize("name", ["mlpppo", "cfcppo", "lstmppo", "transformerppo"])
    def test_same_mode_round_trip(self, name: str) -> None:
        brain = _make(name, continuous_std_mode="state_dependent")
        components = brain.get_weight_components()
        assert "log_std_head" in components
        assert "log_std" not in components
        fresh = _make(name, continuous_std_mode="state_dependent")
        fresh.load_weight_components(components)
        assert torch.equal(fresh.log_std_head.weight, brain.log_std_head.weight)

    @pytest.mark.parametrize("name", ["mlpppo", "cfcppo", "lstmppo", "transformerppo"])
    def test_cross_mode_load_raises(self, name: str) -> None:
        on = _make(name, continuous_std_mode="state_dependent")
        off = _make(name)
        with pytest.raises(ValueError, match="modes must match"):
            off.load_weight_components(on.get_weight_components())
        with pytest.raises(ValueError, match="modes must match"):
            on.load_weight_components(off.get_weight_components())
