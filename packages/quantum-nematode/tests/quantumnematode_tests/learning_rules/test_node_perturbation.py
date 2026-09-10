"""An eligibility that carries the perturbation rather than the activity.

The rule failed its positive control with a gradient alignment of +0.009: its updates were not
starved of reward information, they were not aimed. The eligibility was `pre x post` with the
exploration noise applied at the action, so it correlated reward surprise with ordinary activity
rather than with anything the network could have done differently. These pin the repair: the unit
perturbs its own pre-activation, acts on it, and the trace carries that perturbation.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch
from pydantic import ValidationError
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._mlp_topology import MLPTopology
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.learning_rules import ThreeFactorRule
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

_SEED = 9091


def _actor() -> nn.Sequential:
    torch.manual_seed(_SEED)
    return nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 1))


def _mlp(noise: float = 0.0, decay: float = 0.0) -> MLPTopology:
    return MLPTopology(
        _actor(),
        enable_activity_traces=True,
        trace_decay=decay,
        node_noise=noise,
        perturbation_seed=_SEED,
    )


def _brain(**over: object) -> ConnectomePPOBrain:
    config: dict[str, object] = {
        "seed": _SEED,
        "action_mode": "continuous",
        "learning_rule": "three_factor",
        "enable_activity_traces": True,
    }
    config.update(over)
    return ConnectomePPOBrain(
        config=ConnectomePPOBrainConfig(**config),  # type: ignore[arg-type]
        device=DeviceType.CPU,
    )


def _drive(brain: ConnectomePPOBrain, steps: int = 2) -> None:
    brain.prepare_episode()
    for step in range(steps):
        brain.run_brain(
            BrainParams(food_gradient_strength=0.3 + 0.1 * step, food_gradient_direction=0.5),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )


class TestTheUnitActsOnItsPerturbation:
    def test_the_pre_activation_is_displaced_by_exactly_the_exposed_value(self) -> None:
        # The perturbation must pass through the nonlinearity: perturbing the OUTPUT would drop
        # the activation's derivative and give a rescaled, not unbiased, estimator.
        features = torch.randn(4, generator=torch.Generator().manual_seed(3))
        plain, perturbed = _mlp(), _mlp(noise=0.3)
        clean = plain(features)
        noisy = perturbed(features)
        xi = perturbed.plastic_perturbations[0]
        assert float(xi.abs().sum()) > 0.0
        # The hidden layer's activity is tanh(a + xi), not tanh(a) + xi.
        first = cast("nn.Linear", perturbed._layers[0])
        a = features @ first.weight.T + first.bias
        assert torch.allclose(perturbed.plastic_post_activities[0], torch.tanh(a + xi), atol=1e-6)
        assert not torch.allclose(perturbed.plastic_post_activities[0], torch.tanh(a), atol=1e-6)
        assert not torch.equal(clean, noisy)

    def test_the_trace_carries_the_perturbation_not_the_activity(self) -> None:
        features = torch.randn(4, generator=torch.Generator().manual_seed(4))
        topo = _mlp(noise=0.3)
        topo(features)
        xi = topo.plastic_perturbations[0]
        assert torch.allclose(topo.eligibility_traces[0], torch.outer(xi, features), atol=1e-6)
        assert not torch.allclose(
            topo.eligibility_traces[0],
            torch.outer(topo.plastic_post_activities[0], features),
            atol=1e-6,
        )


class TestDisabledIsUnchanged:
    def test_the_mlp_forward_and_trace_are_bit_identical(self) -> None:
        features = torch.randn(4, generator=torch.Generator().manual_seed(5))
        one, other = _mlp(), _mlp()
        assert torch.equal(one(features), other(features))
        assert torch.equal(one.eligibility_traces[0], other.eligibility_traces[0])

    def test_no_perturbations_are_exposed(self) -> None:
        assert _mlp().plastic_perturbations == []
        assert _brain().topology.plastic_perturbations == []

    def test_the_connectome_is_deterministic(self) -> None:
        one, other = _brain(), _brain()
        _drive(one)
        _drive(other)
        assert torch.equal(one.topology.eligibility_traces[0], other.topology.eligibility_traces[0])


class TestTheConnectomePerturbsEverySettlingStep:
    def _perturbed(self, **over: object) -> ConnectomePPOBrain:
        return _brain(
            plasticity_eligibility="node_perturbation",
            plasticity_node_noise=0.05,
            **over,
        )

    def test_the_perturbation_accumulates_over_the_settling(self) -> None:
        # Perturbing only the settled state would leave a synapse's effect through the later
        # steps uncredited; the exposed value is the sum over all of them.
        shallow = self._perturbed(forward_pass_depth=1)
        deep = self._perturbed(forward_pass_depth=8)
        _drive(shallow)
        _drive(deep)
        assert float(deep.topology.node_perturbation.abs().mean()) > float(
            shallow.topology.node_perturbation.abs().mean(),
        )

    def test_the_trace_is_nonzero_once_there_is_pre_synaptic_history(self) -> None:
        brain = self._perturbed()
        _drive(brain, steps=2)
        assert float(brain.topology.eligibility_traces[0].abs().sum()) > 0.0

    def test_perturbations_clear_between_episodes(self) -> None:
        brain = self._perturbed()
        _drive(brain)
        assert float(brain.topology.node_perturbation.abs().sum()) > 0.0
        brain.prepare_episode()
        assert float(brain.topology.node_perturbation.abs().sum()) == 0.0


class TestItDoesNotDisturbTheRestOfTheStream:
    def test_the_action_noise_is_untouched(self) -> None:
        # "Same seed, perturbation on against off" must differ by the perturbation alone, so
        # the perturbation is drawn from its own generator.
        # Sampled AFTER each drive without reseeding: reseeding first would make the
        # comparison vacuous, since it would restore the stream the drive had advanced.
        torch.manual_seed(101)
        plain = _brain()
        _drive(plain)
        after_plain = torch.randn(3)

        torch.manual_seed(101)
        perturbed = _brain(plasticity_eligibility="node_perturbation", plasticity_node_noise=0.05)
        _drive(perturbed)
        after_perturbed = torch.randn(3)

        # The two drives consumed the global stream identically, so what follows them is the
        # same: the perturbation came from its own generator and shifted nothing.
        assert torch.equal(after_plain, after_perturbed)

    def test_the_perturbation_would_shift_the_stream_if_it_shared_one(self) -> None:
        # Guards the test above from passing for the wrong reason: drawing the same shapes
        # from the GLOBAL stream does advance it, so an implementation without a dedicated
        # generator would fail the assertion above.
        torch.manual_seed(101)
        untouched = torch.randn(3)
        torch.manual_seed(101)
        torch.randn(302)  # what a shared-stream perturbation would have drawn
        shifted = torch.randn(3)
        assert not torch.equal(untouched, shifted)


class TestPersistence:
    def test_the_perturbation_is_not_persisted(self, tmp_path: Path) -> None:
        from quantumnematode.brain.weights import load_weights, save_weights

        source = _brain(plasticity_eligibility="node_perturbation", plasticity_node_noise=0.05)
        _drive(source)
        file = tmp_path / "w.pt"
        save_weights(source, file)
        blob = torch.load(file, weights_only=False)
        assert "node_perturbation" not in blob["topology"]
        # And a file written before perturbation existed still loads into a perturbing brain.
        target = _brain(plasticity_eligibility="node_perturbation", plasticity_node_noise=0.05)
        load_weights(target, file)


class TestRefusals:
    def test_a_zero_perturbation_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="plasticity_node_noise"):
            ConnectomePPOBrainConfig(
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_eligibility="node_perturbation",
            )

    def test_the_construction_guard_catches_a_copied_config(self) -> None:
        config = ConnectomePPOBrainConfig(
            seed=_SEED,
            action_mode="continuous",
            learning_rule="three_factor",
            enable_activity_traces=True,
            plasticity_eligibility="node_perturbation",
            plasticity_node_noise=0.05,
        ).model_copy(update={"plasticity_node_noise": 0.0})
        with pytest.raises(ValueError, match="identically zero"):
            ConnectomePPOBrain(config=config, device=DeviceType.CPU)

    def test_the_rule_refuses_a_topology_that_cannot_perturb(self) -> None:
        topo = _mlp()  # no perturbation
        with pytest.raises(ValueError, match="exposes no perturbations"):
            ThreeFactorRule(
                topo,
                plasticity_rate=0.01,
                weight_decay=0.0,
                weight_bound=10.0,
                baseline_rate=0.0,
                freeze_updates=False,
                modulated=True,
                device=torch.device("cpu"),
                eligibility="node_perturbation",
            )

    def test_the_rule_accepts_a_topology_that_does(self) -> None:
        rule = ThreeFactorRule(
            _mlp(noise=0.1),
            plasticity_rate=0.01,
            weight_decay=0.0,
            weight_bound=10.0,
            baseline_rate=0.0,
            freeze_updates=False,
            modulated=True,
            device=torch.device("cpu"),
            eligibility="node_perturbation",
        )
        assert rule.eligibility == "node_perturbation"
