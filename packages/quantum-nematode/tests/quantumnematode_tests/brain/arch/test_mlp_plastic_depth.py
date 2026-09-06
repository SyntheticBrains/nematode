"""Selectable plastic depth on the MLP: ``all`` is today's build, ``hidden`` freezes the readout.

A plastic output layer takes its own output as its post-synaptic factor and rotates into
saturation under the three-factor rule; ``hidden`` keeps that layer off the seam entirely. Pinned
here: the default is byte-identical, the hidden-only seam has one entry fewer and no trace for
the output layer, the forward still runs the whole actor bitwise-equal to the actor's own, the
output layer is bit-identical after rule steps while the hidden weights move, and the gradient
rule ignores the option.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._mlp_topology import MLPTopology
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.learning_rules import ThreeFactorRule
from quantumnematode.learning_rules.three_factor import ThreeFactorBatch
from torch import nn

if TYPE_CHECKING:
    from quantumnematode.brain.arch._topology import BrainTopology

_SEED = 2718
_MODULES = [ModuleName.FOOD_CHEMOTAXIS]
_IN, _HIDDEN, _OUT = 6, 16, 2


def _actor() -> nn.Sequential:
    torch.manual_seed(_SEED)
    return nn.Sequential(
        nn.Linear(_IN, _HIDDEN),
        nn.Tanh(),
        nn.Linear(_HIDDEN, _HIDDEN),
        nn.Tanh(),
        nn.Linear(_HIDDEN, _OUT),
    )


def _topology(plastic_layers: str, actor: nn.Sequential | None = None) -> MLPTopology:
    return MLPTopology(
        actor if actor is not None else _actor(),
        enable_activity_traces=True,
        trace_decay=0.9,
        plastic_layers=plastic_layers,
    )


def _brain(**overrides: object) -> MLPPPOBrain:
    cfg = MLPPPOBrainConfig(
        seed=_SEED,
        action_mode="continuous",
        sensory_modules=_MODULES,
        **overrides,  # type: ignore[arg-type]
    )
    return MLPPPOBrain(config=cfg, device=DeviceType.CPU)


def _plastic_brain(**overrides: object) -> MLPPPOBrain:
    return _brain(
        learning_rule="three_factor",
        enable_activity_traces=True,
        plasticity_rate=0.5,
        **overrides,
    )


def _drive(brain: MLPPPOBrain, steps: int = 12) -> None:
    brain.prepare_episode()
    torch.manual_seed(_SEED + 1)
    for step in range(steps):
        brain.run_brain(
            BrainParams(food_gradient_strength=0.2 + 0.05 * step, food_gradient_direction=0.3),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        brain.learn(
            BrainParams(),
            reward=1.0 if step % 3 else -0.5,
            episode_done=(step == steps - 1),
        )


class TestDefaultIsToday:
    def test_all_exposes_every_linear(self) -> None:
        topo = _topology("all")
        assert len(topo.plastic_weights) == 3
        assert len(topo.eligibility_traces) == 3
        assert len(topo.plastic_masks) == 3
        assert topo.plastic_fan_in_axes == [1, 1, 1]
        assert "trace_2" in dict(topo.named_buffers())

    def test_brain_default_is_all(self) -> None:
        brain = _plastic_brain()
        assert brain.config.plastic_layers == "all"
        linears = [m for m in brain.actor if isinstance(m, nn.Linear)]
        assert len(brain.topology.plastic_weights) == len(linears)

    def test_invalid_setting_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="plastic_layers"):
            _topology("output")


class TestHiddenOnly:
    def test_seam_excludes_the_output_layer(self) -> None:
        topo = _topology("hidden")
        assert len(topo.plastic_weights) == 2
        assert len(topo.eligibility_traces) == 2
        assert len(topo.plastic_masks) == 2
        assert topo.plastic_fan_in_axes == [1, 1]
        buffers = dict(topo.named_buffers())
        assert "trace_1" in buffers
        assert "trace_2" not in buffers
        output = [m for m in topo._actor if isinstance(m, nn.Linear)][-1]
        assert all(w is not output.weight for w in topo.plastic_weights)
        assert topo.layers == [m for m in topo._actor if isinstance(m, nn.Linear)][:-1]

    def test_forward_is_bitwise_equal_to_the_actor_and_credits_hidden_only(self) -> None:
        actor = _actor()
        topo = _topology("hidden", actor)
        torch.manual_seed(_SEED + 2)
        x = torch.randn(_IN)
        out = topo(x)
        assert torch.equal(out, actor(x))
        assert all(float(t.abs().sum()) > 0 for t in topo.eligibility_traces)
        assert "trace_2" not in dict(topo.named_buffers())

    def test_rule_leaves_the_readout_bit_identical_and_moves_hidden_weights(self) -> None:
        brain = _plastic_brain(plastic_layers="hidden")
        linears = [m for m in brain.actor if isinstance(m, nn.Linear)]
        out_w, out_b = linears[-1].weight.detach().clone(), linears[-1].bias.detach().clone()
        hidden_before = [m.weight.detach().clone() for m in linears[:-1]]
        _drive(brain)
        assert torch.equal(linears[-1].weight, out_w)
        assert torch.equal(linears[-1].bias, out_b)
        assert any(
            not torch.equal(m.weight, b) for m, b in zip(linears[:-1], hidden_before, strict=True)
        )

    def test_homeostasis_targets_follow_the_plastic_list(self) -> None:
        brain = _plastic_brain(plastic_layers="hidden", plasticity_homeostasis=True)
        assert isinstance(brain._rule, ThreeFactorRule)
        assert len(brain._rule.norm_targets) == 2

    def test_rule_steps_over_a_hand_built_hidden_topology(self) -> None:
        actor = _actor()
        topo = _topology("hidden", actor)
        torch.manual_seed(_SEED + 3)
        topo(torch.randn(_IN))
        rule = ThreeFactorRule(
            topo,
            plasticity_rate=0.5,
            weight_decay=0.0,
            weight_bound=10.0,
            baseline_rate=0.0,
            freeze_updates=False,
            modulated=True,
            device=torch.device("cpu"),
        )
        output = [m for m in actor if isinstance(m, nn.Linear)][-1]
        before = output.weight.detach().clone()
        rule.step(cast("BrainTopology", topo), ThreeFactorBatch(reward=2.0))
        assert torch.equal(output.weight, before)


class TestGradientRuleIgnoresTheOption:
    def test_learnable_parameters_are_every_actor_parameter(self) -> None:
        topo = _topology("hidden")
        assert len(topo.learnable_parameters) == len(list(topo._actor.parameters()))

    def test_ppo_training_is_identical_under_both_settings(self) -> None:
        """PPO trains through the optimiser, not the seam, so the option changes nothing."""
        runs = []
        for setting in ("all", "hidden"):
            brain = _brain(plastic_layers=setting, rollout_buffer_size=8)
            brain.prepare_episode()
            torch.manual_seed(_SEED + 4)
            for step in range(24):
                brain.run_brain(
                    BrainParams(food_gradient_strength=0.4, food_gradient_direction=-0.2),
                    reward=None,
                    input_data=None,
                    top_only=False,
                    top_randomize=False,
                )
                brain.learn(BrainParams(), reward=0.3, episode_done=(step % 8 == 7))
            runs.append([p.detach().clone() for p in brain.actor.parameters()])
        for a, b in zip(runs[0], runs[1], strict=True):
            assert torch.equal(a, b)
