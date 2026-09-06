# Design: freeze the matched-rule yardstick's readout

## Context

`MLPTopology` wraps the actor's `Linear` layers by reference, keeps one eligibility trace per
layer, exposes them all on the plastic-topology seam, and in its forward credits each `Linear`
with `post ⊗ pre` where `post` is that layer's output after the activation that follows it. For
the output layer there is no activation: `post` is the action mean itself. Under the three-factor
rule that makes the output rows rotate toward the hidden-activity direction that maximises the
mean's magnitude, and homeostasis, holding the norm, turns the rotation into saturation. The
connectome's readout is frozen and anatomical, so the two arms are not structurally matched
today: one learns behind a fixed decoder, the other learns its decoder with a self-amplifying
rule.

## Goals / Non-Goals

**Goals**

- Let the MLP arm learn its hidden weights under a frozen readout, matching the connectome.
- Keep the default byte-identical and the PPO path untouched.
- Keep the seam contract: aligned lists, one entry per plastic tensor, nothing else.

**Non-Goals**

- Changing what `post` means for a plastic output layer (for example, the sampled action rather
  than the mean, which would be a policy-gradient-flavoured rule on one substrate and not the
  other). The matched rule stays the same rule; the arm's structure changes instead.
- Freezing anything on the connectome; its readout is already frozen.

## Decisions

### D1. `plastic_layers: "all" | "hidden"` on the MLP config

`all` is the historical build. `hidden` marks every `Linear` but the last as plastic. The
setting reaches the topology at construction; the topology builds its plastic list, masks,
traces and fan-in axes from that list only, so the seam shrinks by one entry and the rule needs
no knowledge of the option. Ratified with Chris over leaving the arm as is (a yardstick that
saturates by construction) and over redefining the output layer's post factor (a different rule
on one substrate).

The option has no effect under the gradient rule: PPO trains the actor's parameters through the
optimiser, not through the seam, and `learnable_parameters` keeps returning every actor
parameter. The forward with traces on still runs the whole actor in order and is bitwise-equal
to `actor(features)`; only the crediting loop skips the non-plastic layer.

### D2. What "frozen readout" makes the two arms

Both arms then learn a recurrent or feedforward body under a fixed linear decoder that was set
at construction: anatomical on the connectome, a random orthogonal draw on the MLP. The
comparison is "same rule, same hyperparameters, same structure of what is plastic", which is what
the ranking test needs. The MLP's decoder being a random full-rank draw rather than an anatomical
pooling is the residual asymmetry. Its direction is not established — a full-rank decoder can
read any hidden pattern, an anatomical one only its motor pools — so it is stated and left
untested rather than claimed to be neutral.

### D3. Byte-identity and the seam

With `all`, the plastic list, masks, traces, axes and forward crediting are exactly today's.
The existing MLP tests (seam conformance, refactor equivalence against the frozen PPO
reference, matched-rule invariance) keep proving it. With `hidden`, tests pin that the output
weight is bit-identical after rule steps, that the seam lists have one entry fewer, that no
trace buffer exists for the output layer, that the forward output is bitwise-equal to the actor's,
and that homeostatic targets are captured for the hidden layers only.

## Risks / Trade-offs

- **Reverses a ratified choice.** The all-layers choice was made before any run under the rule
  existed; the evidence that overturns it is recorded and specific.
- **The variant test's key set grows by one** for the MLP arm when the panel change sets the
  option; that test already carries an MLP-specific set.

## Open Questions

None.
