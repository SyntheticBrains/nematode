# Proposal: Configurable MLP actor head for `SpikingPPOBrain`

## Why

`SpikingPPOBrain` currently produces its action logits from a single non-spiking leaky-integrator
readout — a linear projection of the hidden spikes. That is a **constrained, low-capacity policy head**:
the rich continuous hidden membrane `v` (the recurrent state) feeds only the critic (detached), never the
policy. On hard multi-objective cells this caps the policy's expressiveness and, empirically, prevents
the arm from learning at all.

This is the same failure CfC hit — its NCP motor-direct head failed the integrated cell while an **MLP
actor head on the recurrent hidden state** rescued it. The cross-architecture comparison's spiking
evaluation reproduced it decisively: on the foraging + predator cell the default spike head reached only
**17%** while an MLP-head prototype reached **70%** (climbing to 91%), and on the full integrated cell
the spike head fails entirely whereas the MLP head converges to a **top-of-field ~92% post-convergence**
plateau. The MLP head is therefore not a tuning nicety — it is the difference between a useless arm and a
competitive one.

This change formalizes the prototyped MLP actor head as a first-class, configurable option, so the
spiking arm can be run with the readout that actually learns hard cells. It is a small, backward-compatible
enhancement to the existing (already-merged) `SpikingPPOBrain`.

## What Changes

- **New config field `actor_head: "spike" | "mlp"`** on `SpikingPPOBrainConfig` (default `"spike"`,
  byte-equivalent to current behaviour), plus `actor_hidden_dim: int = 64` and `actor_num_layers: int = 2`.
- **MLP actor head**: with `actor_head: "mlp"`, the action logits are produced by a small actor MLP
  (`Linear(hidden_size → actor_hidden_dim) → ReLU → … → Linear(→ num_actions)`) that reads the recurrent
  layer's **hidden membrane potential `v`** (non-detached, so policy gradients flow into the spiking core),
  instead of the leaky-integrator readout. The recurrent spiking core and the detached-membrane critic are
  unchanged in both modes.
- **Registry / config validation**: `actor_head` validated to `{"spike", "mlp"}`; the actor MLP joins the
  actor optimizer and is covered by `WeightPersistence`.
- **Tests** extending `test_spikingppo.py` (construction + forward + PPO update + weight round-trip in
  `mlp` mode; validation rejects unknown `actor_head`).

## Impact

- **Backward-compatible**: `actor_head` defaults to `"spike"` — existing configs and the foraging smoke are
  unchanged.
- **No new dependency**: reuses the existing `SpikingPPOBrain` networks; the actor MLP is a standard
  `torch.nn` block on the already-computed hidden membrane.
- **Reuses**: the `WeightPersistence` protocol, the paired-field validator pattern, the `_actor_parameters`
  grad-clip/optimizer wiring.
- **Out of scope**: the cross-architecture comparison's C3 config + ranked n=8 evaluation (downstream on
  the comparison branch); a spike-rate readout variant; population-coded input.
