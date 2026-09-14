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

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from quantumnematode.brain.arch._node_noise_schedule import NodeNoiseSchedule


def activation_derivative(module: nn.Module | None, pre: torch.Tensor) -> torch.Tensor:
    """Take the elementwise derivative of an activation at its pre-activation.

    Dispatched on the module type rather than taken from autograd: this runs inside the
    rollout's ``no_grad`` region, where building a graph for a vector-sized derivative would
    be the only reason autograd was live. An unrecognised activation RAISES -- a silent
    fallback to ones would turn e-prop's trace into the Hebbian one and read as a rule that
    learns differently rather than as an activation nobody wrote a derivative for.

    ``None`` means no activation follows the layer, whose derivative is one.
    """
    if module is None:
        return torch.ones_like(pre)
    if isinstance(module, nn.Tanh):
        return 1.0 - torch.tanh(pre) ** 2
    if isinstance(module, nn.ReLU):
        return (pre > 0).to(pre.dtype)
    if isinstance(module, nn.Sigmoid):
        s = torch.sigmoid(pre)
        return s * (1.0 - s)
    if isinstance(module, nn.Identity):
        return torch.ones_like(pre)
    msg = (
        f"no activation derivative for {type(module).__name__}: e-prop's eligibility is the "
        "activation's own derivative, and defaulting to ones would silently return the "
        "Hebbian trace."
    )
    raise TypeError(msg)


class MLPTopology(nn.Module):
    """``PlasticTopology`` over the ``Linear`` layers of an existing ``nn.Sequential``.

    Parameters
    ----------
    actor
        The brain's actor, held by reference. The plastic layers are its
        ``nn.Linear`` modules as ``plastic_layers`` selects them -- all of
        them, or all but the output layer; every other module is treated as the activation
        that follows the preceding layer.
    enable_activity_traces
        Allocate traces. When false nothing is allocated and the traced
        forward degrades to the plain forward.
    trace_decay
        Per-step multiplicative decay of every trace.
    plastic_layers
        ``"all"`` (every ``Linear``) or ``"hidden"`` (every ``Linear`` but the
        output layer, which then has no trace and is never written).
    """

    # The wrapped actor. Declared at class level so the type checker knows it
    # even though it is set around ``nn.Module.__setattr__`` (see __init__);
    # without this, attribute access falls through Module's ``__getattr__``
    # and is inferred as a Tensor.
    _actor: nn.Sequential
    # The actor's output layer. Declared at class level for the same reason ``_actor`` is.
    _output_layer: nn.Linear

    # Traces are registered only when enabled (guard every use).
    def __init__(  # noqa: PLR0913 — one parameter per substrate switch
        self,
        actor: nn.Sequential,
        *,
        enable_activity_traces: bool,
        trace_decay: float,
        plastic_layers: str = "all",
        node_noise: float = 0.0,
        node_noise_schedule: NodeNoiseSchedule | None = None,
        perturbation_seed: int | None = None,
        eligibility: str = "hebbian",
        learning_signal: str = "random",
        learning_signal_seed: int | None = None,
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
        linears: list[nn.Linear] = [m for m in actor if isinstance(m, nn.Linear)]
        if plastic_layers not in ("all", "hidden"):
            msg = f"plastic_layers must be 'all' or 'hidden', got {plastic_layers!r}"
            raise ValueError(msg)
        # ``hidden`` keeps the output layer off the seam entirely: no trace, no
        # mask, no fan-in axis, no update. A plastic output layer takes its own
        # output as its post-synaptic factor, and under a Hebbian rule its rows
        # rotate toward the hidden-activity direction that maximises the action
        # mean until the actions saturate; a fixed decoder, as the connectome
        # has, is what a learner behind it needs.
        self.plastic_layers = plastic_layers
        self._layers: list[nn.Linear] = linears if plastic_layers == "all" else linears[:-1]
        self._plastic_ids: set[int] = {id(layer) for layer in self._layers}
        self.enable_activity_traces = enable_activity_traces
        self.trace_decay = trace_decay
        # The actor's output layer, which is what a symmetric learning signal is the transpose of.
        # A reference like ``_actor``: held around Module.__setattr__ so the actor's weights are
        # not registered in this module's state dict a second time.
        object.__setattr__(self, "_output_layer", linears[-1])
        # What the eligibility carries. "eprop" builds it from the forward dynamics -- the
        # activation's own derivative times the pre-synaptic activity -- and is the only mode
        # that needs a learning signal folded in after the action is sampled.
        self.eligibility = eligibility
        self.learning_signal = learning_signal
        # Whether a forward's eligibility is waiting for the signal that gives it a sign. A
        # second forward arriving with one still pending means a call site was missed, which
        # would read as a rule that learns slowly rather than as a rule never credited.
        self._eprop_pending = False
        # Per-unit perturbation. Zero (the default) draws nothing and leaves the forward pass
        # bit-identical. Its own generator, seeded from the run seed, so enabling perturbation
        # shifts nothing else in the random stream -- "same seed, on against off" must differ
        # by the perturbation alone.
        self.node_noise = node_noise
        # ``node_noise`` is the INITIAL scale and answers "does this topology perturb at all",
        # which a schedule never changes: a schedule's floor is required to be positive, so a
        # perturbing topology stays perturbing. Only the magnitude is scheduled, read through
        # ``current_node_noise`` at draw time. A plain integer, not a buffer: it never enters
        # the state dict, so a checkpoint written before schedules existed still loads.
        self._node_noise_schedule = node_noise_schedule
        self._schedule_steps_begun = 0
        self._perturbation_generator = torch.Generator()
        if perturbation_seed is not None:
            self._perturbation_generator.manual_seed(perturbation_seed)

        # Dense substrate: every entry is a synapse. Masks are all-true so the
        # rule's mask-dependent telemetry means the same thing here as on a
        # sparse substrate, rather than being special-cased away.
        self._masks: list[torch.Tensor] = [
            torch.ones_like(layer.weight, dtype=torch.bool) for layer in self._layers
        ]
        if enable_activity_traces:
            for index, layer in enumerate(self._layers):
                self.register_buffer(f"trace_{index}", torch.zeros_like(layer.weight))
                # The post-synaptic activity that trace was built from, kept so a
                # rule term that needs what the units did can read it through the
                # seam. One vector per plastic layer, its rows' worth of units.
                self.register_buffer(
                    f"post_activity_{index}",
                    torch.zeros(layer.weight.shape[0]),
                )
                # The perturbation those units acted on. Transient like the trace: cleared per
                # episode, never persisted, so a checkpoint written before perturbation
                # existed loads unchanged.
                self.register_buffer(
                    f"perturbation_{index}",
                    torch.zeros(layer.weight.shape[0]),
                    # Not persisted: per-step state, redrawn every forward and cleared every
                    # episode. Keeping it out of the state dict is what lets a checkpoint
                    # written before perturbation existed load unchanged.
                    persistent=False,
                )
                if eligibility == "eprop":
                    # The step's UNSIGNED eligibility, psi_j * h_i, held between the forward and
                    # the learning signal that gives it a sign. Transient for the same reason the
                    # perturbation is: per-step state, cleared every episode.
                    self.register_buffer(
                        f"eprop_{index}",
                        torch.zeros_like(layer.weight),
                        persistent=False,
                    )
        if eligibility == "eprop":
            self._build_feedback(linears, learning_signal, learning_signal_seed)

    def _build_feedback(
        self,
        linears: list[nn.Linear],
        routing: str,
        seed: int | None,
    ) -> None:
        """Build one feedback projection per plastic layer, and persist it.

        Persisted rather than transient: a reloaded policy that redrew its projection would be
        learning against a different feedback path than the one it was trained with, and nothing
        in the run would say so.

        ``symmetric`` and ``random_motor`` are defined by which units the readout reads. On this
        substrate the readout is the output ``Linear``, which reads every unit of the layer
        feeding it and none of any earlier layer -- so both are supported only for that layer,
        and an earlier plastic layer raises rather than being credited through a composition of
        downstream weights this eligibility does not compute. ``random_motor`` then coincides
        with ``random`` here, the mask being all-true; it is a distinct arm only on a substrate
        whose readout reads a strict subset.
        """
        action_dim = self._output_layer.weight.shape[0]
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        for index, layer in enumerate(self._layers):
            if layer is self._output_layer:
                # A plastic OUTPUT layer needs no projection under any routing: its post-synaptic
                # units are the action dimensions themselves, so the signal reaching unit k is the
                # score's own k-th component. That is the identity, not a choice, and
                # ``learning_signal_projection`` returns it. A random projection here would
                # scramble the one layer whose gradient is exact.
                continue
            reads_the_output = layer is linears[-2] if len(linears) > 1 else False
            if routing == "symmetric":
                if not reads_the_output:
                    msg = (
                        "learning_signal='symmetric' needs the plastic layer to feed the output "
                        "layer: the readout's transpose reaches no earlier layer, and e-prop "
                        "does not compose the downstream weights that would."
                    )
                    raise ValueError(msg)
                # Derived live in ``learning_signal_projection``, not cached: a readout that is
                # written or loaded after construction would leave a cached transpose describing a
                # readout the arm never ran with.
                continue
            if routing in ("random", "random_motor"):
                if routing == "random_motor" and not reads_the_output:
                    msg = (
                        "learning_signal='random_motor' is the readout-reached subset, which on "
                        "this substrate is the layer feeding the output layer."
                    )
                    raise ValueError(msg)
                feedback = torch.randn(
                    (layer.weight.shape[0], action_dim),
                    generator=generator,
                    dtype=layer.weight.dtype,
                )
            elif routing == "scalar":
                # No projection at all: L_j is exactly 1, so the update's only dependence on the
                # outcome is the scalar modulator at update time. A projection of ones would
                # instead give L_j = sum_k score_k -- still a broadcast of the error, which is
                # not what this arm ablates.
                continue
            else:
                msg = f"unknown learning_signal {routing!r}"
                raise ValueError(msg)
            self.register_buffer(f"feedback_{index}", feedback)

    # ── PlasticTopology seam ──────────────────────────────────

    @property
    def layers(self) -> list[nn.Linear]:
        """The plastic layers, in forward order, as the actor's own modules.

        Every ``Linear`` under ``plastic_layers="all"``; every ``Linear`` but the
        output layer under ``"hidden"``.
        """
        return self._layers

    @property
    def plastic_weights(self) -> list[torch.Tensor]:
        """The plastic ``Linear`` weight matrices. Biases are not plastic."""
        return [layer.weight for layer in self._layers]

    @property
    def eligibility_traces(self) -> list[torch.Tensor]:
        """One ``(out, in)`` trace per layer, aligned with ``plastic_weights``."""
        return [getattr(self, f"trace_{index}") for index in range(len(self._layers))]

    @property
    def plastic_masks(self) -> list[torch.Tensor]:
        """All-true masks: on a dense layer every entry is a synapse."""
        return self._masks

    @property
    def plastic_fan_in_axes(self) -> list[int]:
        """A ``Linear`` weight is ``[out, in]``: a unit's incoming weights are a row."""
        return [1] * len(self._layers)

    @property
    def plastic_post_activities(self) -> list[torch.Tensor]:
        """One ``(out,)`` activity vector per layer: a ``[out, in]`` weight's post axis is ``0``."""
        return [getattr(self, f"post_activity_{index}") for index in range(len(self._layers))]

    @property
    def current_node_noise(self) -> float:
        """The perturbation scale for the current step: the initial one absent a schedule."""
        if self._node_noise_schedule is None:
            return self.node_noise
        # The counter records steps BEGUN, so the one currently running is indexed one
        # lower: the first step must run at the initial scale, not one step into the decay.
        return self._node_noise_schedule.scale_at(max(0, self._schedule_steps_begun - 1))

    def advance_schedule(self) -> None:
        """Advance the perturbation schedule by one step; a no-op without one."""
        if self._node_noise_schedule is not None:
            self._schedule_steps_begun += 1

    def reset_schedule(self) -> None:
        """Return the schedule to its initial scale, as a policy load does."""
        self._schedule_steps_begun = 0

    @property
    def plastic_perturbations(self) -> list[torch.Tensor]:
        """One ``(out,)`` perturbation per layer; empty when this topology is not perturbing."""
        if self.node_noise <= 0.0:
            return []
        return [
            cast("torch.Tensor", getattr(self, f"perturbation_{index}"))
            for index in range(len(self._layers))
        ]

    # ── BrainTopology seam ────────────────────────────────────

    @property
    def learnable_parameters(self) -> list[nn.Parameter]:
        """The actor's parameters -- what a gradient rule would optimise."""
        return list(self._actor.parameters())

    def apply_weight_mask(self, weights: torch.Tensor) -> torch.Tensor:
        """Identity: a dense layer has no disallowed edges."""
        return weights

    # ── Lifecycle ─────────────────────────────────────────────

    def learning_signal_projection(self, index: int) -> torch.Tensor | None:
        """Derive one plastic layer's feedback projection: ``(out, action_dim)``.

        ``None`` under the ``scalar`` routing, whose signal is 1 everywhere. Under ``symmetric``
        it is the output layer's transpose, taken live rather than cached: a readout written or
        loaded after construction would leave a cached transpose describing a readout the arm
        never ran with.

        A plastic output layer is the exception: its post-synaptic units are the action dimensions,
        so its projection is the IDENTITY under every routing -- exact, and not a routing choice.
        """
        if self._layers[index] is self._output_layer:
            # Exact for every routing: this layer's units ARE the action dimensions.
            return torch.eye(
                self._output_layer.weight.shape[0],
                dtype=self._output_layer.weight.dtype,
            )
        if self.learning_signal == "scalar":
            return None
        if self.learning_signal == "symmetric":
            return self._output_layer.weight.detach().T
        return cast("torch.Tensor", getattr(self, f"feedback_{index}"))

    def apply_learning_signal(self, score: torch.Tensor) -> None:
        """Fold the step's learning signal into every plastic layer's trace.

        ``score`` is the derivative of the action log-probability with respect to the action
        mean -- for a Gaussian head, ``(u - mu) / sigma ** 2`` at the PRE-SQUASH draw. The
        per-unit signal is ``L = B @ score``, and the trace takes ``eps * L`` broadcast down
        each unit's row. A no-op unless the eligibility is e-prop's, so every other substrate
        and mode pays one attribute read.
        """
        if not self.enable_activity_traces or self.eligibility != "eprop":
            return
        with torch.no_grad():
            flat = score.detach().reshape(-1)
            for index in range(len(self._layers)):
                eps = cast("torch.Tensor", getattr(self, f"eprop_{index}"))
                projection = self.learning_signal_projection(index)
                if projection is None:
                    signal = torch.ones(eps.shape[0], dtype=eps.dtype)
                else:
                    signal = projection.to(eps.dtype) @ flat.to(eps.dtype)

                trace = cast("torch.Tensor", getattr(self, f"trace_{index}"))
                # ``signal`` is one scalar per post-synaptic unit and a ``[out, in]`` weight's
                # post axis is 0, so it multiplies each row.
                trace.mul_(self.trace_decay).add_(eps * signal.unsqueeze(1))
        self._eprop_pending = False

    def reset_traces(self) -> None:
        """Zero every trace at episode start; a documented no-op when off."""
        if self.enable_activity_traces:
            for trace in self.eligibility_traces:
                trace.zero_()
            # Perturbation state is per-step and belongs to the episode that drew it.
            for perturbation in self.plastic_perturbations:
                perturbation.zero_()
            if self.eligibility == "eprop":
                for index in range(len(self._layers)):
                    cast("torch.Tensor", getattr(self, f"eprop_{index}")).zero_()
                # An episode boundary is the one place a pending eligibility is legitimately
                # dropped: the step it belonged to has no further reward coming.
                self._eprop_pending = False

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

        With traces enabled ``features`` must be a single unbatched vector:
        eligibility is one outer product per step, and a batch would have
        no single "this step" to credit. Batched evaluation belongs to the
        untraced actor (the PPO replay path), which this does not replace.
        """
        if not self.enable_activity_traces:
            return self._actor(features)
        if features.dim() != 1:
            msg = (
                "MLPTopology.forward with traces enabled expects one unbatched feature "
                f"vector (got shape {tuple(features.shape)}). Eligibility is one outer "
                "product per environment step; batched evaluation belongs to the "
                "untraced actor."
            )
            raise ValueError(msg)

        if self._eprop_pending:
            msg = (
                "the previous step's e-prop eligibility was never credited: "
                "apply_learning_signal must be called once per environment step, immediately "
                "after the action is sampled. Crediting it now would attribute this step's "
                "reward to the previous step's dynamics."
            )
            raise RuntimeError(msg)
        modules = self._modules_in_order
        x = features
        layer_index = 0
        i = 0
        perturbing = self.node_noise > 0.0
        scale = self.current_node_noise
        while i < len(modules):
            module = modules[i]
            if isinstance(module, nn.Linear):
                pre = x
                x = module(x)
                pre_activation = x
                if perturbing and id(module) in self._plastic_ids:
                    # Into the PRE-activation, so the unit acts on its perturbation through
                    # its own nonlinearity. Perturbing the output instead would drop the
                    # activation's derivative and give a rescaled, not unbiased, estimator.
                    with torch.no_grad():
                        # Drawn on the generator's own device (CPU) and moved to the
                        # activation's: a CPU generator cannot fill a non-CPU tensor, so a
                        # dedicated generator and a GPU substrate would otherwise collide.
                        noise = (
                            torch.randn(
                                x.shape,
                                generator=self._perturbation_generator,
                                dtype=x.dtype,
                            )
                            * scale
                        ).to(x.device)
                        getattr(self, f"perturbation_{layer_index}").copy_(noise)
                    x = x + noise
                # The activation following a layer is part of that layer's
                # "post": it is the rate the next population actually sees.
                activation: nn.Module | None = None
                if i + 1 < len(modules) and not isinstance(modules[i + 1], nn.Linear):
                    activation = modules[i + 1]
                    pre_activation = x
                    x = activation(x)
                    i += 1
                # A non-plastic layer (the output layer under ``hidden``) still
                # runs, so the output stays bitwise-equal to the actor's; it
                # simply accrues no eligibility.
                if id(module) in self._plastic_ids:
                    with torch.no_grad():
                        if self.eligibility == "eprop":
                            # The step's UNSIGNED eligibility: psi_j * h_i, the local part of
                            # d h_j / d w_ij. Held rather than added to the trace, because the
                            # sign lives in a learning signal that does not exist until the
                            # action is sampled -- see ``apply_learning_signal``.
                            psi = activation_derivative(activation, pre_activation.detach())
                            getattr(self, f"eprop_{layer_index}").copy_(
                                torch.outer(psi, pre.detach()),
                            )
                            getattr(self, f"post_activity_{layer_index}").copy_(x.detach())
                            self._eprop_pending = True
                            layer_index += 1
                            i += 1
                            continue
                        trace = getattr(self, f"trace_{layer_index}")
                        # Perturbing, the eligibility carries what the unit VARIED rather than
                        # what it did: `pre (x) xi` is the part of its output the network could
                        # have done otherwise, which is what a reward surprise can be
                        # correlated with. `pre (x) post` correlates reward with ordinary
                        # activity, which is reinforced correlation and not a gradient estimate.
                        post = (
                            getattr(self, f"perturbation_{layer_index}")
                            if perturbing
                            else x.detach()
                        )
                        trace.mul_(self.trace_decay).add_(torch.outer(post, pre.detach()))
                        # The same post-synaptic factor the trace just took, so a
                        # term reading it and the Hebbian term agree on the step.
                        getattr(self, f"post_activity_{layer_index}").copy_(x.detach())
                    layer_index += 1
            else:
                # An activation not preceded by a Linear (never the case for
                # the brain's actor, but the loop stays total).
                x = module(x)
            i += 1
        return x
