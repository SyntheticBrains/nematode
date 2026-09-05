"""The MLP topology observes the actor without changing it.

Everything the matched-rule arm relies on lives here: the wrapped actor
is the same object, the traced forward is the plain forward bit for bit,
and each layer's trace follows the same-step closed form.
"""

from __future__ import annotations

import copy

import pytest
import torch
from quantumnematode.brain.arch._mlp_topology import MLPTopology
from quantumnematode.brain.arch._topology import BrainTopology, PlasticTopology
from torch import nn

_SEED = 2718
_IN, _HIDDEN, _OUT = 13, 64, 2


def _actor() -> nn.Sequential:
    torch.manual_seed(_SEED)
    return nn.Sequential(
        nn.Linear(_IN, _HIDDEN),
        nn.ReLU(),
        nn.Linear(_HIDDEN, _HIDDEN),
        nn.ReLU(),
        nn.Linear(_HIDDEN, _OUT),
    )


def _inputs(n: int) -> list[torch.Tensor]:
    torch.manual_seed(_SEED + 1)
    return [torch.randn(_IN) for _ in range(n)]


class TestSeamConformance:
    def test_satisfies_both_protocols(self) -> None:
        topo = MLPTopology(_actor(), enable_activity_traces=True, trace_decay=0.9)
        assert isinstance(topo, PlasticTopology)
        assert isinstance(topo, BrainTopology)

    def test_lists_are_aligned_and_shape_matched(self) -> None:
        topo = MLPTopology(_actor(), enable_activity_traces=True, trace_decay=0.9)
        weights, traces, masks = topo.plastic_weights, topo.eligibility_traces, topo.plastic_masks
        assert len(weights) == len(traces) == len(masks) == 3
        for w, t, m in zip(weights, traces, masks, strict=True):
            assert t.shape == w.shape
            assert m.shape == w.shape
            assert bool(m.all())


class TestWrapsRatherThanRebuilds:
    def test_layers_are_the_actors_own_modules(self) -> None:
        actor = _actor()
        topo = MLPTopology(actor, enable_activity_traces=True, trace_decay=0.9)
        linears = [m for m in actor if isinstance(m, nn.Linear)]
        for topo_layer, actor_layer in zip(topo.layers, linears, strict=True):
            assert topo_layer is actor_layer

    def test_actor_state_dict_has_no_traces(self) -> None:
        actor = _actor()
        MLPTopology(actor, enable_activity_traces=True, trace_decay=0.9)
        assert not any("trace" in key for key in actor.state_dict())

    def test_topology_state_dict_holds_only_traces(self) -> None:
        """Layers are referenced, not registered, so no weight is duplicated."""
        topo = MLPTopology(_actor(), enable_activity_traces=True, trace_decay=0.9)
        assert set(topo.state_dict()) == {"trace_0", "trace_1", "trace_2"}

    def test_traces_off_allocates_nothing(self) -> None:
        topo = MLPTopology(_actor(), enable_activity_traces=False, trace_decay=0.9)
        assert topo.state_dict() == {}

    def test_deepcopy_keeps_layers_aliased_to_the_copied_actor(self) -> None:
        """The one place wrap-by-reference could silently split."""
        actor = _actor()
        holder = {
            "actor": actor,
            "topo": MLPTopology(actor, enable_activity_traces=True, trace_decay=0.9),
        }
        copied = copy.deepcopy(holder)
        linears = [m for m in copied["actor"] if isinstance(m, nn.Linear)]
        for topo_layer, actor_layer in zip(copied["topo"].layers, linears, strict=True):
            assert topo_layer is actor_layer
        assert copied["topo"].layers[0] is not holder["topo"].layers[0]


class TestTracedForwardEqualsPlainForward:
    def test_bitwise_equal_with_traces_on(self) -> None:
        actor = _actor()
        topo = MLPTopology(actor, enable_activity_traces=True, trace_decay=0.9)
        for x in _inputs(5):
            assert torch.equal(topo(x), actor(x))

    def test_bitwise_equal_with_traces_off(self) -> None:
        actor = _actor()
        topo = MLPTopology(actor, enable_activity_traces=False, trace_decay=0.9)
        for x in _inputs(3):
            assert torch.equal(topo(x), actor(x))


class TestSameStepEligibility:
    def test_recurrence_matches_the_closed_form(self) -> None:
        actor = _actor()
        decay = 0.8
        topo = MLPTopology(actor, enable_activity_traces=True, trace_decay=decay)
        expected = [torch.zeros_like(layer.weight) for layer in topo.layers]
        for x in _inputs(6):
            # Recompute pre/post by hand from the plain actor.
            h = x
            pres_posts = []
            mods = list(actor)
            i = 0
            while i < len(mods):
                if isinstance(mods[i], nn.Linear):
                    pre = h
                    h = mods[i](h)
                    if i + 1 < len(mods) and not isinstance(mods[i + 1], nn.Linear):
                        h = mods[i + 1](h)
                        i += 1
                    pres_posts.append((pre.detach(), h.detach()))
                i += 1
            topo(x)
            for layer_index, (pre, post) in enumerate(pres_posts):
                expected[layer_index] = decay * expected[layer_index] + torch.outer(post, pre)
                assert torch.allclose(
                    topo.eligibility_traces[layer_index],
                    expected[layer_index],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_one_forward_accrues_one_outer_product(self) -> None:
        """From reset, one forward leaves exactly post (x) pre per layer -- not twice it.

        Determinism across a reset would not catch a double accumulation, so
        the expected single product is computed explicitly from the plain
        actor and compared bit for bit.
        """
        actor = _actor()
        topo = MLPTopology(actor, enable_activity_traces=True, trace_decay=0.9)
        x = _inputs(1)[0]

        expected = []
        h = x
        mods = list(actor)
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

        topo(x)
        for trace, one_product in zip(topo.eligibility_traces, expected, strict=True):
            assert torch.equal(trace, one_product)
            # A doubled accumulation would be exactly 2x and is ruled out explicitly.
            assert not torch.equal(trace, 2 * one_product)

    def test_traced_forward_rejects_batched_input(self) -> None:
        """A batch has no single step to credit; the contract is stated, not silent."""
        topo = MLPTopology(_actor(), enable_activity_traces=True, trace_decay=0.9)
        with pytest.raises(ValueError, match="unbatched"):
            topo(torch.stack(_inputs(2)))

    def test_untraced_forward_still_accepts_batches(self) -> None:
        topo = MLPTopology(_actor(), enable_activity_traces=False, trace_decay=0.9)
        batch = torch.stack(_inputs(2))
        assert torch.equal(topo(batch), topo._actor(batch))

    def test_reset_zeroes_every_trace(self) -> None:
        topo = MLPTopology(_actor(), enable_activity_traces=True, trace_decay=0.9)
        for x in _inputs(3):
            topo(x)
        assert all(t.abs().sum() > 0 for t in topo.eligibility_traces)
        topo.reset_traces()
        assert all(bool((t == 0).all()) for t in topo.eligibility_traces)

    def test_traces_do_not_join_the_autograd_graph(self) -> None:
        topo = MLPTopology(_actor(), enable_activity_traces=True, trace_decay=0.9)
        out = topo(_inputs(1)[0].requires_grad_())
        out.sum().backward()
        assert all(t.grad is None and not t.requires_grad for t in topo.eligibility_traces)
