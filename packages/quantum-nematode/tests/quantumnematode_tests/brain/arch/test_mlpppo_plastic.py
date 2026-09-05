"""The matched-rule MLP arm: the same rule, honestly matched.

Pins what "matched" has to mean for the yardstick to be one: the same
rule class with the same arithmetic and the same hyperparameter values,
one traced forward per step, only linear weights plastic, and the PPO
machinery genuinely dormant rather than stubbed.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
from typing import Literal

import pytest
import torch
from pydantic import ValidationError
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.learning_rules import ThreeFactorRule
from quantumnematode.learning_rules.three_factor import (
    BASELINE_KEY,
    MEAN_ABS_DELTA_KEY,
    PREDICTION_ERROR_KEY,
    SATURATED_FRACTION_KEY,
    ThreeFactorBatch,
)
from torch import nn

ActionMode = Literal["discrete", "continuous"]

_SEED = 6174
_STEPS = 6
_MODULES = [ModuleName.FOOD_CHEMOTAXIS]


def _make(action_mode: ActionMode = "continuous", **overrides: object) -> MLPPPOBrain:
    cfg = MLPPPOBrainConfig(
        seed=_SEED,
        action_mode=action_mode,
        sensory_modules=_MODULES,
        **overrides,  # type: ignore[arg-type]
    )
    return MLPPPOBrain(config=cfg, device=DeviceType.CPU)


def _plastic(action_mode: ActionMode = "continuous", **overrides: object) -> MLPPPOBrain:
    return _make(
        action_mode,
        learning_rule="three_factor",
        enable_activity_traces=True,
        **overrides,
    )


def _drive(brain: MLPPPOBrain, rewards: list[float] | None = None) -> None:
    rewards = rewards if rewards is not None else [0.4 * (i % 3) - 0.2 for i in range(_STEPS)]
    brain.prepare_episode()
    torch.manual_seed(_SEED + 1)
    for step, reward in enumerate(rewards):
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
        brain.learn(BrainParams(), reward=reward, episode_done=(step == len(rewards) - 1))


def _linears(brain: MLPPPOBrain) -> list[nn.Linear]:
    return [m for m in brain.actor if isinstance(m, nn.Linear)]


class TestSelectionAndValidation:
    def test_default_is_ppo_and_wraps_the_actor(self) -> None:
        brain = _make()
        assert brain._uses_ppo
        assert brain._rule is None
        for topo_layer, actor_layer in zip(brain.topology.layers, _linears(brain), strict=True):
            assert topo_layer is actor_layer

    @pytest.mark.parametrize("rule", ["three_factor", "hebbian"])
    def test_plastic_rule_without_traces_rejected(self, rule: str) -> None:
        with pytest.raises(ValidationError, match="requires enable_activity_traces"):
            MLPPPOBrainConfig(sensory_modules=_MODULES, learning_rule=rule)  # type: ignore[arg-type]

    def test_plastic_defaults_match_the_connectome(self) -> None:
        """One definition: the two arms cannot drift apart silently."""
        mlp = MLPPPOBrainConfig(sensory_modules=_MODULES)
        connectome = ConnectomePPOBrainConfig()
        for field in (
            "learning_rule",
            "plasticity_rate",
            "plasticity_weight_decay",
            "plasticity_weight_bound",
            "plasticity_baseline_rate",
            "enable_activity_traces",
            "trace_decay",
            "freeze_updates",
        ):
            assert getattr(mlp, field) == getattr(connectome, field), field

    def test_plastic_brain_uses_the_generic_rule(self) -> None:
        assert isinstance(_plastic()._rule, ThreeFactorRule)


@pytest.mark.parametrize("action_mode", ["discrete", "continuous"])
class TestPlasticCadenceAndDormantPPO:
    def test_one_update_per_step(self, action_mode: ActionMode) -> None:
        brain = _plastic(action_mode)
        _drive(brain)
        assert len(brain.history_data.plasticity_prediction_error) == _STEPS

    def test_buffer_empty_and_value_unset(self, action_mode: ActionMode) -> None:
        brain = _plastic(action_mode)
        _drive(brain)
        assert len(brain.buffer) == 0
        assert brain.last_value is None

    def test_every_linear_weight_changes(self, action_mode: ActionMode) -> None:
        brain = _plastic(action_mode)
        before = [layer.weight.detach().clone() for layer in _linears(brain)]
        _drive(brain)
        for b, layer in zip(before, _linears(brain), strict=True):
            assert not torch.equal(b, layer.weight)

    def test_only_linear_weights_are_plastic(self, action_mode: ActionMode) -> None:
        brain = _plastic(action_mode)
        biases = [layer.bias.detach().clone() for layer in _linears(brain)]
        critic = [p.detach().clone() for p in brain.critic.parameters()]
        log_std = brain.log_std.detach().clone() if action_mode == "continuous" else None
        _drive(brain)
        for b, layer in zip(biases, _linears(brain), strict=True):
            assert torch.equal(b, layer.bias)
        for c, p in zip(critic, brain.critic.parameters(), strict=True):
            assert torch.equal(c, p)
        if log_std is not None:
            assert torch.equal(log_std, brain.log_std)

    def test_telemetry_recorded(self, action_mode: ActionMode) -> None:
        brain = _plastic(action_mode)
        _drive(brain)
        assert len(brain.history_data.plasticity_baseline) == _STEPS
        assert len(brain.history_data.plasticity_mean_abs_delta) == _STEPS
        assert len(brain.history_data.plasticity_saturated_fraction) == _STEPS
        assert len(brain.history_data.rewards) == _STEPS


class TestOneTracedForwardPerDiscreteStep:
    """The discrete path evaluates the actor twice; only one may accrue."""

    def test_single_discrete_step_accrues_exactly_one_outer_product(self) -> None:
        brain = _plastic("discrete")
        brain.prepare_episode()
        torch.manual_seed(_SEED + 2)
        params = BrainParams(food_gradient_strength=0.5, food_gradient_direction=0.1)
        brain.run_brain(params, reward=None, input_data=None, top_only=False, top_randomize=False)

        # Recompute the single outer product by hand from the plain actor.
        x = torch.tensor(brain.preprocess(params), dtype=torch.float32)
        h = brain._apply_torch_gating(x)
        mods = list(brain.actor)
        expected = []
        i = 0
        while i < len(mods):
            if isinstance(mods[i], nn.Linear):
                pre = h
                h = mods[i](h)
                if i + 1 < len(mods) and not isinstance(mods[i + 1], nn.Linear):
                    h = mods[i + 1](h)
                    i += 1
                expected.append(torch.outer(h.detach(), pre.detach()))
            i += 1
        for trace, one_product in zip(brain.topology.eligibility_traces, expected, strict=True):
            assert torch.allclose(trace, one_product, rtol=0.0, atol=0.0)


class TestFloorsBehaveAsOnTheConnectome:
    def test_frozen_plastic_arm_does_not_learn(self) -> None:
        brain = _plastic(freeze_updates=True)
        before = [p.detach().clone() for p in brain.actor.parameters()]
        _drive(brain)
        for b, p in zip(before, brain.actor.parameters(), strict=True):
            assert torch.equal(b, p)

    def test_frozen_ppo_arm_does_not_learn_either(self) -> None:
        """The flag must mean the same thing under every rule on this brain."""
        brain = _make(freeze_updates=True, rollout_buffer_size=4)
        before = [p.detach().clone() for p in brain.actor.parameters()]
        _drive(brain)
        for b, p in zip(before, brain.actor.parameters(), strict=True):
            assert torch.equal(b, p)

    def test_unfrozen_ppo_arm_does_learn(self) -> None:
        """Guards the freeze test against passing because nothing ever learns."""
        brain = _make(rollout_buffer_size=4)
        before = [p.detach().clone() for p in brain.actor.parameters()]
        _drive(brain)
        assert any(
            not torch.equal(b, p) for b, p in zip(before, brain.actor.parameters(), strict=True)
        )

    def test_hebbian_is_invariant_to_reward(self) -> None:
        a = _make(learning_rule="hebbian", enable_activity_traces=True)
        b = _make(learning_rule="hebbian", enable_activity_traces=True)
        _drive(a, [1.0] * _STEPS)
        _drive(b, [-4.0, 8.0, 0.0, 2.5, -1.0, 6.0])
        for pa, pb in zip(a.actor.parameters(), b.actor.parameters(), strict=True):
            assert torch.equal(pa, pb)

    def test_three_factor_is_not_invariant_to_reward(self) -> None:
        a, b = _plastic(), _plastic()
        _drive(a, [1.0] * _STEPS)
        _drive(b, [-4.0, 8.0, 0.0, 2.5, -1.0, 6.0])
        assert any(
            not torch.equal(pa, pb)
            for pa, pb in zip(a.actor.parameters(), b.actor.parameters(), strict=True)
        )


class TestMatchedUpdateAcrossSubstrates:
    """The same update lands on an MLP layer exactly as it lands on w_chem."""

    def test_same_product_on_both_substrates(self) -> None:
        mlp = _plastic()
        conn = ConnectomePPOBrain(
            config=ConnectomePPOBrainConfig(
                seed=_SEED,
                action_mode="continuous",
                learning_rule="three_factor",
                enable_activity_traces=True,
            ),
            device=DeviceType.CPU,
        )
        hyper = {
            "plasticity_rate": 0.05,
            "weight_decay": 0.0,
            "weight_bound": 10.0,
            "baseline_rate": 0.0,
        }
        rules = [
            ThreeFactorRule(
                t,
                freeze_updates=False,
                modulated=True,
                device=torch.device("cpu"),
                **hyper,
            )
            for t in (mlp.topology, conn.topology)
        ]
        # Scripted traces: 0.5 on every allowed entry of every plastic tensor.
        with torch.no_grad():
            for t in (mlp.topology, conn.topology):
                for trace, mask in zip(t.eligibility_traces, t.plastic_masks, strict=True):
                    trace.copy_(torch.full_like(trace, 0.5) * mask.to(trace.dtype))
        befores = [
            [w.detach().clone() for w in t.plastic_weights] for t in (mlp.topology, conn.topology)
        ]
        reports = [
            rule.step(t, ThreeFactorBatch(reward=2.0))
            for rule, t in zip(rules, (mlp.topology, conn.topology), strict=True)
        ]
        # baseline_rate 0 pins delta == reward == 2.0, so the update is
        # 0.05 * 2.0 * 0.5 on every allowed entry.
        for t, before in zip((mlp.topology, conn.topology), befores, strict=True):
            for w, b, mask in zip(t.plastic_weights, before, t.plastic_masks, strict=True):
                expected = b + 0.1 * 0.5 * mask.to(b.dtype)
                assert torch.allclose(w, expected, rtol=0.0, atol=1e-7)
        for key in (PREDICTION_ERROR_KEY, BASELINE_KEY, MEAN_ABS_DELTA_KEY, SATURATED_FRACTION_KEY):
            assert key in reports[0].extra
            assert key in reports[1].extra
        assert reports[0].extra[PREDICTION_ERROR_KEY] == reports[1].extra[PREDICTION_ERROR_KEY]


class TestInitialisationAndCopying:
    def test_bound_clears_the_mlp_initialisation(self) -> None:
        for seed in (7, 909, 4242):
            brain = MLPPPOBrain(
                config=MLPPPOBrainConfig(
                    seed=seed,
                    action_mode="continuous",
                    sensory_modules=_MODULES,
                    learning_rule="three_factor",
                    enable_activity_traces=True,
                ),
                device=DeviceType.CPU,
            )
            largest = max(layer.weight.detach().abs().max().item() for layer in _linears(brain))
            assert largest < brain.config.plasticity_weight_bound, seed

    def test_deepcopy_keeps_topology_aliased_to_the_copied_actor(self) -> None:
        brain = _plastic()
        copied = copy.deepcopy(brain)
        for topo_layer, actor_layer in zip(copied.topology.layers, _linears(copied), strict=True):
            assert topo_layer is actor_layer
        assert copied.topology.layers[0] is not brain.topology.layers[0]

    def test_ppo_weight_component_keys_are_unchanged(self) -> None:
        keys = set(_make().get_weight_components())
        assert keys == {"policy", "value", "optimizer", "training_state", "log_std"}
        assert not any("trace" in k for k in keys)
