"""e-prop's eligibility, and the four things that would make it something else.

* the trace is the ACTIVATION'S derivative times the pre-synaptic rate. Default that derivative to
  ones and the trace is the Hebbian one under a new name, which would read as a rule behaving
  differently rather than as a mechanism that was never built;
* on one plastic layer, one forward pass and one scalar Gaussian action, symmetric feedback makes
  the update the exact REINFORCE gradient. That is the strongest available check on the derivative,
  the score function and the fold-in together, and it is what licenses reading anything the same
  code returns on the connectome;
* symmetric feedback reaches ONLY the units the readout reads -- 39 of 302 on this substrate --
  because the readout mean-pools four motor classes and e-prop drops the multi-hop paths by which
  any other unit reaches the action. That is the arm's defining property, so a projection leaking
  outside the pool would make it a broad arm under a restricted name;
* an eligibility held across a step that never credited it would be folded in against the wrong
  step's reward, which is silent and reads as a rule that learns slowly.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from quantumnematode.brain.arch._mlp_topology import MLPTopology, activation_derivative
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config
from torch import nn

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "configs").is_dir():
    _root = _root.parent
_CONFIGS = _root / "configs" / "scenarios" / "foraging"
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop"
_SEED = 1
_SIGMA = 0.5


def _buffer(topology: MLPTopology, name: str) -> torch.Tensor:
    """One of the topology's buffers, narrowed: Module.__getattr__ is typed ``Tensor | Module``."""
    value = getattr(topology, name)
    assert isinstance(value, torch.Tensor)
    return value


def _one_step_net(hidden: int = 6, action_dim: int = 2) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, action_dim))


def _topology(routing: str, actor: nn.Sequential | None = None, decay: float = 0.0) -> MLPTopology:
    return MLPTopology(
        actor if actor is not None else _one_step_net(),
        enable_activity_traces=True,
        trace_decay=decay,
        plastic_layers="hidden",
        eligibility="eprop",
        learning_signal=routing,
        learning_signal_seed=_SEED,
    )


class TestTheActivationDerivative:
    @pytest.mark.parametrize("module", [nn.Tanh(), nn.ReLU(), nn.Sigmoid()])
    def test_it_matches_autograd(self, module: nn.Module) -> None:
        pre = torch.linspace(-2.0, 2.0, 9, requires_grad=True)
        module(pre).sum().backward()
        assert pre.grad is not None
        assert torch.allclose(activation_derivative(module, pre.detach()), pre.grad, atol=1e-6)

    def test_no_activation_is_one(self) -> None:
        pre = torch.randn(5)
        assert torch.equal(activation_derivative(None, pre), torch.ones_like(pre))

    def test_an_unknown_activation_raises(self) -> None:
        # Silently returning ones would turn the trace into the Hebbian one under e-prop's name.
        with pytest.raises(TypeError, match="no activation derivative"):
            activation_derivative(nn.Softmax(dim=0), torch.randn(3))


class TestSymmetricFeedbackIsTheGradient:
    """One layer, one pass, one Gaussian action: the update IS REINFORCE's gradient."""

    def test_the_trace_equals_the_score_function_gradient(self) -> None:
        actor = _one_step_net()
        topology = _topology("symmetric", actor)
        features = torch.randn(4)
        mean = topology(features)
        draw = mean.detach() + _SIGMA * torch.randn(2)
        topology.apply_learning_signal((draw - mean.detach()) / _SIGMA**2)

        actor.zero_grad()
        log_prob = (-0.5 * ((draw - actor(features)) / _SIGMA) ** 2).sum()
        log_prob.backward()
        plastic = actor[0]
        assert isinstance(plastic, nn.Linear)
        gradient = plastic.weight.grad
        assert gradient is not None
        assert torch.allclose(_buffer(topology, "trace_0"), gradient, atol=1e-5)

    def test_the_random_routing_is_not_the_gradient(self) -> None:
        # Broadcast alignment is a different direction by construction; a test that passed for both
        # would not be testing the projection at all.
        actor = _one_step_net()
        symmetric = _topology("symmetric", actor)
        random = _topology("random", actor)
        features = torch.randn(4)
        for topology in (symmetric, random):
            topology(features)
            topology.apply_learning_signal(torch.tensor([0.4, -0.9]))
        assert not torch.allclose(_buffer(symmetric, "trace_0"), _buffer(random, "trace_0"))


class TestTheScalarRoutingRemovesTheSignal:
    def test_the_trace_is_the_bare_eligibility(self) -> None:
        # L_j = 1 exactly. A projection of ones would instead give L_j = sum_k score_k, which is
        # still a broadcast of the error and not what this arm ablates.
        topology = _topology("scalar")
        topology(torch.randn(4))
        eps = _buffer(topology, "eprop_0").clone()
        topology.apply_learning_signal(torch.tensor([3.0, -7.0]))
        assert torch.allclose(_buffer(topology, "trace_0"), eps)

    def test_it_registers_no_projection(self) -> None:
        assert not hasattr(_topology("scalar"), "feedback_0")


class TestTheForwardIsUnchanged:
    def test_the_output_is_bitwise_equal_to_the_actors(self) -> None:
        actor = _one_step_net()
        features = torch.randn(4)
        expected = actor(features)
        assert torch.equal(_topology("random", actor)(features), expected)


class TestAnUncreditedEligibilityIsRefused:
    def test_a_second_forward_with_one_pending_raises(self) -> None:
        topology = _topology("random")
        topology(torch.randn(4))
        with pytest.raises(RuntimeError, match="never credited"):
            topology(torch.randn(4))

    def test_the_fold_in_clears_it(self) -> None:
        topology = _topology("random")
        topology(torch.randn(4))
        topology.apply_learning_signal(torch.tensor([0.1, 0.2]))
        topology(torch.randn(4))  # does not raise

    def test_an_episode_boundary_clears_it(self) -> None:
        # The one place a pending eligibility is legitimately dropped: its step has no reward left.
        topology = _topology("random")
        topology(torch.randn(4))
        topology.reset_traces()
        assert not topology._eprop_pending
        assert float(_buffer(topology, "eprop_0").abs().sum()) == 0.0


class TestTheProjectionIsPersisted:
    def test_it_round_trips_through_the_state_dict(self) -> None:
        # A reloaded policy that redrew its projection would learn against a different feedback
        # path than the one it was trained with, and nothing in the run would say so.
        first = _topology("random")
        assert "feedback_0" in first.state_dict()
        second = _topology("random")
        second.load_state_dict(first.state_dict())
        assert torch.equal(_buffer(second, "feedback_0"), _buffer(first, "feedback_0"))

    def test_the_eligibility_is_not_persisted(self) -> None:
        # Per-step state: a checkpoint written under another eligibility must load unchanged.
        assert "eprop_0" not in _topology("random").state_dict()


def _brain(routing: str, seed: int = _SEED) -> ConnectomePPOBrain:
    simulation = load_simulation_config(str(_CONFIGS / f"{_STEM}_{routing}.yml"))
    assert simulation.brain is not None
    config = simulation.brain.config
    assert isinstance(config, ConnectomePPOBrainConfig)
    config.seed = seed
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


def _settle(brain: ConnectomePPOBrain) -> torch.Tensor:
    """One traced forward through the committed klinotaxis head."""
    _logits, hidden = brain.topology.forward_with_hidden(
        torch.tensor([0.3, -0.2, 0.1]),
        None,
        None,
        None,
        None,
    )
    return hidden


class TestSymmetricFeedbackReachesOnlyTheReadoutPool:
    """The readout mean-pools 39 units, and e-prop drops every path by which the rest reach it."""

    @pytest.mark.parametrize("routing", ["symmetric", "random_motor"])
    def test_the_projection_is_zero_outside_the_pool(self, routing: str) -> None:
        topology = _brain(routing).topology
        pool = set(topology._motor_flat_indices.tolist())
        projection = topology.learning_signal_projection()
        assert projection is not None
        reached = {i for i, row in enumerate(projection) if bool(row.abs().any())}
        assert reached == pool

    def test_the_broad_routing_reaches_every_unit(self) -> None:
        topology = _brain("random").topology
        projection = topology.learning_signal_projection()
        assert projection is not None
        reached = [i for i, row in enumerate(projection) if bool(row.abs().any())]
        assert len(reached) == topology.n_neurons

    def test_symmetric_carries_the_pooling_divisor(self) -> None:
        # d mu_k / d h_j is readout[k, class(j)] / |class(j)|: the pool is a MEAN, so a member
        # carries a fraction of its class's influence, not the whole of it.
        topology = _brain("symmetric").topology
        start, stop = topology._motor_class_slices[0]
        member = int(topology._motor_flat_indices[start])
        expected = topology.readout[:, 0].detach() / float(stop - start)
        projection = topology.learning_signal_projection()
        assert projection is not None
        assert torch.allclose(projection[member], expected)

    def test_the_masked_routing_is_the_broad_draw_masked(self) -> None:
        # Same generator draw, so the pair differs by the mask alone at one seed -- which is what
        # makes them comparable at matched breadth.
        broad = _brain("random").topology.learning_signal_projection()
        masked_topology = _brain("random_motor").topology
        masked = masked_topology.learning_signal_projection()
        assert broad is not None
        assert masked is not None
        pool = masked_topology._motor_flat_indices
        assert torch.allclose(masked[pool], broad[pool])


class TestTheConnectomeEligibility:
    def test_it_is_the_summed_settling_derivative(self) -> None:
        # Recomputed independently from the topology's own tensors: sum over settling steps of
        # h_i^(s) * psi_j^(s+1), masked to the chemical edge set. This is the one check that the
        # trace is the DYNAMICS' derivative and not some other outer product of the same vectors.
        brain = _brain("scalar")
        topology = brain.topology
        food = torch.tensor([0.3, -0.2, 0.1])
        topology.forward_with_hidden(food, None, None, None, None)
        recorded = topology.eprop_trace.clone()

        chem = (topology.w_chem.detach() * topology.m_chem).T
        gap = topology.g_gap.detach().T
        h = torch.zeros(topology.n_neurons).index_add(
            0,
            topology._food_neuron_indices,
            food @ topology.food_gains.detach(),
        )
        expected = torch.zeros_like(recorded)
        for _ in range(topology.forward_pass_depth):
            preact = chem @ h + gap @ h
            expected = expected + torch.outer(h, 1.0 - torch.tanh(preact) ** 2)
            h = torch.tanh(preact)
        assert torch.allclose(recorded, expected * topology.m_chem, atol=1e-6)

    def test_it_respects_the_chemical_mask(self) -> None:
        brain = _brain("random")
        _settle(brain)
        outside = ~brain.topology.m_chem.bool()
        assert not bool(brain.topology.eprop_trace[outside].abs().any())

    def test_the_scalar_routing_credits_the_bare_eligibility(self) -> None:
        brain = _brain("scalar")
        _settle(brain)
        eps = brain.topology.eprop_trace.clone()
        brain.topology.apply_learning_signal(torch.tensor([2.0, -5.0]))
        assert torch.allclose(brain.topology.activity_traces, eps)

    def test_a_second_settle_without_the_fold_in_raises(self) -> None:
        brain = _brain("random")
        _settle(brain)
        with pytest.raises(RuntimeError, match="never credited"):
            _settle(brain)

    def test_the_hebbian_trace_is_not_also_accumulated(self) -> None:
        # Two eligibilities in one trace would be neither, and the rule would report itself the
        # dynamics-derived one.
        brain = _brain("random")
        _settle(brain)
        assert float(brain.topology.activity_traces.abs().sum()) == 0.0


class TestTheFrozenFloorIsOneFloor:
    def test_the_forward_is_identical_across_routings(self) -> None:
        # One frozen arm serves all four routings only if the routing cannot reach the forward
        # pass: the projections draw from their own generator and none perturbs anything.
        hiddens = [_settle(_brain(routing)) for routing in ("symmetric", "random_motor", "random")]
        for other in hiddens[1:]:
            assert torch.equal(hiddens[0], other)


class TestTheSymmetricProjectionTracksTheReadout:
    def test_it_is_taken_from_the_readout_the_arm_runs_with(self) -> None:
        # The anatomical contrast overwrites the orthogonal draw AFTER the topology is built, and a
        # checkpoint can substitute the readout again. A projection cached at construction would be
        # the transpose of a readout the arm never ran with.
        topology = _brain("symmetric").topology
        before = topology.learning_signal_projection()
        assert before is not None
        with torch.no_grad():
            topology.readout.mul_(-2.0)
        after = topology.learning_signal_projection()
        assert after is not None
        assert torch.allclose(after, before * -2.0)

    def test_it_matches_the_anatomical_readout_through_the_pool(self) -> None:
        topology = _brain("symmetric").topology
        start, stop = topology._motor_class_slices[1]
        member = int(topology._motor_flat_indices[start])
        projection = topology.learning_signal_projection()
        assert projection is not None
        expected = topology.readout[:, 1].detach() / float(stop - start)
        assert torch.allclose(projection[member], expected)


class TestAPlasticOutputLayerGetsIdentityFeedback:
    """Its post-synaptic units ARE the action dimensions, so its signal is its own error."""

    @staticmethod
    def _all_plastic(routing: str) -> MLPTopology:
        return MLPTopology(
            _one_step_net(),
            enable_activity_traces=True,
            trace_decay=0.0,
            plastic_layers="all",
            eligibility="eprop",
            learning_signal=routing,
            learning_signal_seed=_SEED,
        )

    @pytest.mark.parametrize("routing", ["random", "scalar", "symmetric"])
    def test_the_projection_is_the_identity(self, routing: str) -> None:
        # Exact for every routing, and not a routing choice: a random projection here would scramble
        # the one layer whose gradient is exact.
        topology = self._all_plastic(routing)
        output_index = len(topology.layers) - 1
        projection = topology.learning_signal_projection(output_index)
        assert projection is not None
        assert torch.equal(projection, torch.eye(2))

    def test_the_hidden_layer_still_takes_its_routing(self) -> None:
        random = self._all_plastic("random")
        symmetric = self._all_plastic("symmetric")
        first = random.learning_signal_projection(0)
        second = symmetric.learning_signal_projection(0)
        assert first is not None
        assert second is not None
        assert not torch.allclose(first, second)

    def test_the_output_layers_trace_is_its_own_gradient(self) -> None:
        actor = _one_step_net()
        topology = MLPTopology(
            actor,
            enable_activity_traces=True,
            trace_decay=0.0,
            plastic_layers="all",
            eligibility="eprop",
            learning_signal="random",
            learning_signal_seed=_SEED,
        )
        features = torch.randn(4)
        mean = topology(features)
        draw = mean.detach() + _SIGMA * torch.randn(2)
        topology.apply_learning_signal((draw - mean.detach()) / _SIGMA**2)

        actor.zero_grad()
        (-0.5 * ((draw - actor(features)) / _SIGMA) ** 2).sum().backward()
        output = actor[2]
        assert isinstance(output, nn.Linear)
        gradient = output.weight.grad
        assert gradient is not None
        index = len(topology.layers) - 1
        assert torch.allclose(_buffer(topology, f"trace_{index}"), gradient, atol=1e-5)
