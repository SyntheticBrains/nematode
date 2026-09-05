"""Plastic-topology seam over an MLP actor's linear layers.

The MLP brain's actor is a plain ``nn.Sequential``. This wraps it for the
plasticity rule without owning it: the topology holds *references* to the
actor's ``Linear`` modules and registers one eligibility trace per layer on
itself. The actor is therefore the same object before and after wrapping,
its state dict is unchanged (no trace buffers appear in it), and its
construction order -- which fixes the torch-RNG stream every PPO result
depends on -- is never re-run. Traces are zero-initialised and consume no
randomness.

**Eligibility for a feedforward layer is the same-step product** of the
layer's output and its input, ``E_l <- decay * E_l + post_l (x) pre_l``,
oriented ``(out, in)`` to match ``nn.Linear.weight``. This is the same
principle as the recurrent connectome's previous-step form, not a
departure from it: the connectome needed ``h_prev`` because its same-step
product ``h (x) h`` is symmetric, giving both directions of a reciprocal
edge identical eligibility. A layer's pre and post are different
populations that the layer itself orders causally within the step, so the
same-step product already says "this input caused this output". Using
the previous step's input would instead credit a layer's synapses for an
output they had no path to.

``post_l`` is the layer's output after its nonlinearity where one follows
it -- the rate-code analogue of the connectome's settled tanh state -- and
the raw output for a final layer with none.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPTopology(nn.Module):
    """``PlasticTopology`` over the ``Linear`` layers of an existing ``nn.Sequential``.

    Parameters
    ----------
    actor
        The brain's actor, held by reference. Every ``nn.Linear`` in it is
        a plastic layer; every other module is treated as the activation
        that follows the preceding layer.
    enable_activity_traces
        Allocate traces. When false nothing is allocated and the traced
        forward degrades to the plain forward.
    trace_decay
        Per-step multiplicative decay of every trace.
    """

    # The wrapped actor. Declared at class level so the type checker knows it
    # even though it is set around ``nn.Module.__setattr__`` (see __init__);
    # without this, attribute access falls through Module's ``__getattr__``
    # and is inferred as a Tensor.
    _actor: nn.Sequential

    # Traces are registered only when enabled (guard every use).
    def __init__(
        self,
        actor: nn.Sequential,
        *,
        enable_activity_traces: bool,
        trace_decay: float,
    ) -> None:
        super().__init__()
        # References, deliberately not registered as submodules: the actor
        # owns these layers and persists them under its own keys. Registering
        # them here too would duplicate every weight in this module's state
        # dict for no benefit.
        # ``nn.Module.__setattr__`` registers ANY Module assigned to an
        # attribute as a submodule, which would put every actor weight in
        # this module's state dict; store the reference around that hook.
        object.__setattr__(self, "_actor", actor)
        self._modules_in_order: list[nn.Module] = list(actor)
        self._layers: list[nn.Linear] = [m for m in actor if isinstance(m, nn.Linear)]
        self.enable_activity_traces = enable_activity_traces
        self.trace_decay = trace_decay

        # Dense substrate: every entry is a synapse. Masks are all-true so the
        # rule's mask-dependent telemetry means the same thing here as on a
        # sparse substrate, rather than being special-cased away.
        self._masks: list[torch.Tensor] = [
            torch.ones_like(layer.weight, dtype=torch.bool) for layer in self._layers
        ]
        if enable_activity_traces:
            for index, layer in enumerate(self._layers):
                self.register_buffer(f"trace_{index}", torch.zeros_like(layer.weight))

    # ── PlasticTopology seam ──────────────────────────────────

    @property
    def layers(self) -> list[nn.Linear]:
        """The plastic layers, in forward order, as the actor's own modules."""
        return self._layers

    @property
    def plastic_weights(self) -> list[torch.Tensor]:
        """Every ``Linear`` weight matrix. Biases are not plastic."""
        return [layer.weight for layer in self._layers]

    @property
    def eligibility_traces(self) -> list[torch.Tensor]:
        """One ``(out, in)`` trace per layer, aligned with ``plastic_weights``."""
        return [getattr(self, f"trace_{index}") for index in range(len(self._layers))]

    @property
    def plastic_masks(self) -> list[torch.Tensor]:
        """All-true masks: on a dense layer every entry is a synapse."""
        return self._masks

    # ── BrainTopology seam ────────────────────────────────────

    @property
    def learnable_parameters(self) -> list[nn.Parameter]:
        """The actor's parameters -- what a gradient rule would optimise."""
        return list(self._actor.parameters())

    def apply_weight_mask(self, weights: torch.Tensor) -> torch.Tensor:
        """Identity: a dense layer has no disallowed edges."""
        return weights

    # ── Lifecycle ─────────────────────────────────────────────

    def reset_traces(self) -> None:
        """Zero every trace at episode start; a documented no-op when off."""
        if self.enable_activity_traces:
            for trace in self.eligibility_traces:
                trace.zero_()

    # ── Forward ───────────────────────────────────────────────

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the actor's forward while recording eligibility.

        Runs the actor's modules in their own order on the same input, so the
        output is bitwise-equal to ``actor(features)``. When traces are
        enabled, records each ``Linear`` layer's input and its output after
        the activation that follows it, and accumulates the eligibility
        outer product under ``torch.no_grad()``. Call exactly once per
        environment step: the rule's alignment semantics credit the trace
        as it stands when that step's reward arrives.
        """
        if not self.enable_activity_traces:
            return self._actor(features)

        modules = self._modules_in_order
        x = features
        layer_index = 0
        i = 0
        while i < len(modules):
            module = modules[i]
            if isinstance(module, nn.Linear):
                pre = x
                x = module(x)
                # The activation following a layer is part of that layer's
                # "post": it is the rate the next population actually sees.
                if i + 1 < len(modules) and not isinstance(modules[i + 1], nn.Linear):
                    x = modules[i + 1](x)
                    i += 1
                with torch.no_grad():
                    trace = getattr(self, f"trace_{layer_index}")
                    trace.mul_(self.trace_decay).add_(torch.outer(x.detach(), pre.detach()))
                layer_index += 1
            else:
                # An activation not preceded by a Linear (never the case for
                # the brain's actor, but the loop stays total).
                x = module(x)
            i += 1
        return x
