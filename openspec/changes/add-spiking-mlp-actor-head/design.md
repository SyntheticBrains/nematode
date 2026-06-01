# Design: Configurable MLP actor head for `SpikingPPOBrain`

## Context

- `SpikingPPOBrain` ([`brain/arch/spiking_ppo.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/spiking_ppo.py))
  computes logits from `LeakyIntegratorReadout` — a non-spiking membrane that integrates the hidden
  layer's spikes. The recurrent layer's hidden membrane `v` is read only by the critic (detached).
- `CfCPPOBrain` is the precedent: a `actor_head: "motor" | "mlp"` field where the `"mlp"` head reads the
  recurrent hidden state through a small actor MLP. This change mirrors that pattern for the spiking arm.
- Empirical basis (cross-architecture comparison, spiking C3 evaluation): the default spike head reaches
  ~17% on the foraging+predator cell and fails the full integrated cell; the MLP-head prototype reaches
  ~70% (→91%) and a ~92% from-scratch post-convergence plateau on the integrated cell. The MLP head is
  load-bearing.

## Decision 1 — `actor_head` config field; the MLP head reads the hidden membrane `v`

Add `actor_head: "spike" | "mlp"`. The recurrent adaptive-LIF core, the direct-current encoder, and the
detached-membrane critic are **identical** in both modes — only the logit source changes:

- **`"spike"` (default)**: the existing `LeakyIntegratorReadout` produces the logits (linear projection of
  the hidden spikes into a carried output membrane). Byte-equivalent to current behaviour.
- **`"mlp"`**: a small actor MLP (`Linear(hidden_size → actor_hidden_dim) → ReLU → … → Linear(→ num_actions)`,
  `actor_num_layers` deep) maps the recurrent layer's **hidden membrane potential `v`** (the top layer's
  membrane, `_hidden_membrane(state)`) to the logits. The leaky-integrator readout is still advanced (it is
  part of the carried neuron state) but its output is not used for the logits in this mode.

**Why read the membrane `v`, not the spikes?** The spikes are binary and sparse; the membrane is the
continuous recurrent state that already drives the spikes. Reading `v` (non-detached) gives the policy a
smooth, high-information signal and lets policy gradients flow into the spiking core — exactly the
information the critic already uses and the constrained spike readout discards. This is why it is the lever
that unblocks learning (and mirrors CfC's mlp head reading the recurrent hidden state).

**Legitimacy.** The deployed policy is still driven by the spiking recurrent core (the dynamics, the
recurrence via spikes); the MLP is a readout on that core's membrane — the same standing CfC's mlp head
has as a legitimate CfC arm.

## Decision 2 — Default `"spike"` (backward-compatible)

The default is `"spike"` so existing configs, the merged foraging smoke, and the archived spec's behaviour
are unchanged. Hard cells opt into `"mlp"` explicitly (as the downstream C3 comparison config does), exactly
as CfC defaults to `"motor"` and opts into `"mlp"`.

## Decision 3 — Optimizer + WeightPersistence + validation

The actor MLP's parameters join `_actor_parameters()` (so they are in the actor optimizer and the
`max_grad_norm` clip) and are exposed/loaded via `WeightPersistence` under an `"actor_mlp"` component (only
when present). `actor_head` is validated to `{"spike", "mlp"}` in the existing `_validate_config`. The
per-step neuron state is unchanged (the MLP is stateless).

## Risks

| Risk | Mitigation |
|---|---|
| MLP head changes the default behaviour | Default is `"spike"`; the MLP path is constructed only when `actor_head == "mlp"`. A test asserts the spike-head default builds no actor MLP. |
| Weight round-trip misses the new component | `WeightPersistence` includes `"actor_mlp"` when present; a test round-trips logits in `mlp` mode. |
| Unknown `actor_head` silently mis-builds | `_validate_config` rejects values outside `{"spike", "mlp"}` with a clear error; a test asserts it. |
