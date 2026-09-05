# Add the minimal rate-based three-factor learning rule

## Why

Phase 7's flagship is a pre-registered 2×2 — learning rule × wiring — asking whether the wild-type connectome becomes load-bearing *under a rule family the real animal could plausibly use*. Logbook 034 established that under PPO it does not: a degree-preserving rewired null matches wild-type, so the specific wiring is inert to gradient descent. The 2×2's plastic row cannot be run until that rule exists.

Two prerequisites shipped. `ConnectomePPORule` gave the `LearningRule` seam a real consumer, and `ConnectomeTopology` gained a per-synapse eligibility buffer `E`, currently allocated, decayed, masked — and read by nothing. This change supplies the third factor and makes `E` load-bearing.

Nothing here runs the panel. The arms (A.4–A.6) and the panel itself (A.7) follow; this delivers the mechanism they hold constant.

## What Changes

### 1. `ConnectomeThreeFactorRule` in `learning_rules/`

A second `LearningRule` implementation beside the PPO one. The update is the minimal reward-modulated Hebbian form:

```text
Δw_chem = η · δ_t · E_t ,   δ_t = r_t − b_t
```

where `E_t` is the eligibility trace, `δ_t` is a reward prediction error against a running baseline `b_t`, and `η` is the plasticity rate. Three factors, named: **pre-synaptic** and **post-synaptic** activity (jointly, via `E`) and a **global neuromodulatory** signal (`δ`). No gradients, no optimiser, no critic, no backward pass — `step` runs entirely under `torch.no_grad()`.

The baseline is an exponential moving average of reward. It is not a refinement: without it, any task whose rewards are predominantly one-signed drives monotonic weight growth regardless of behaviour, and the rule measures reward *magnitude* rather than reward *surprise*.

**Cadence is per environment step**, not per rollout buffer. The neuromodulator arrives with the reward and gates the standing eligibility; batching that would discard the temporal alignment the rule exists to exploit.

### 2. The eligibility trace becomes temporally causal (dated amendment)

The shipped v1 formula `E ← λE + M∘(h hᵀ)` is a symmetric outer product of one settled state. The trace-substrate requirement pre-authorised exactly this amendment, and the rule needs it:

```text
E ← λ·E + M_chem ∘ (h_{t−1} ⊗ h_t)
```

The pre-synaptic term is the **previous** step's settled hidden state; the post-synaptic term is the current one. This makes the trace causal (pre precedes post) and directional (reciprocal edge pairs, which the symmetric form gave identical eligibility, are now distinguished). The adjacency mask already separated non-reciprocal edges, so this is a sharpening rather than a repair — but "this synapse participated in causing that activity" is the claim a three-factor rule rests on, and the symmetric form did not support it.

The first step of an episode has no previous state; eligibility accrues from the second step onward, with `h_0` seeded at reset.

### 3. Rule selection on the existing brain

`ConnectomePPOBrainConfig` gains `learning_rule: "ppo" | "three_factor"`, defaulting to `"ppo"`. Selecting `three_factor` requires `enable_activity_traces` — a rule with no trace to read would silently learn nothing — and that pairing is validated at load time rather than discovered after a run.

Following the wiring-control and std-mode precedent, this is a config option on the existing brain, not a new brain class: the D10 panel's arms must differ in exactly one dimension, which is impossible across a forked class. The default path stays **byte-identical**, verified against a frozen reference.

### 4. Plastic scope: chemical synapses only

The rule updates `w_chem` and nothing else. Sensory gains and the motor readout stay at initialisation, identical across arms through the shared seed.

This is the clean statement of the hypothesis — the *connectome* learns — and it keeps the wild-type vs rewired-null contrast about wiring alone. A wild-type advantage arising from an adapting readout would be precisely the confound the 2×2 exists to exclude. The cost is accepted and detectable: if a frozen readout makes the task unlearnable, the arm fails the D2 frozen-weights sanity floor, which is what that floor is for. A floor failure is diagnostic, not silent.

### 5. Stability

Unbounded Hebbian growth is the classic failure mode, so two bounded, configurable stabilisers ship with the rule: a weight-decay term (`Δw −= η·λ_w·w`) and a magnitude clamp. Both default to values that constrain without dominating, and both are reported in telemetry so a run that saturates is visible rather than inferred.

**Dale's law is preserved**: the update SHALL NOT flip the sign of an existing synapse. A chemical synapse's sign is a property of its neurotransmitter, not of experience, and a rule that silently converts excitatory synapses to inhibitory would be modelling something the animal cannot do.

### 6. Telemetry and a smoke config

Per-update `δ`, baseline, mean |Δw|, saturated-synapse fraction, and sign-clamp hits, so the rule's health is observable during the panel rather than reconstructed after it. One plastic wild-type config lands as a smoke arm; the panel's full arm set is A.4–A.6.

## Impact

- **Affected specs**: new `learning-rules` capability; `connectome-ppo-brain` gains rule selection and a dated amendment to the trace formula.
- **Affected code**: new `learning_rules/three_factor.py`; the trace update and reset in `ConnectomeTopology`; rule construction and per-step dispatch in `ConnectomePPOBrain`; config fields and their validator.
- **Default behaviour unchanged**: `learning_rule` defaults to `"ppo"` and traces default off, so every existing config, weight file, and Logbook 029 comparison is untouched — the substrate frozen under Amendment A stays frozen.
- **Not in scope**: the panel arms (A.4–A.6), the 2×2 itself (A.7), neuromodulator grounding (7a-ii), and any performance claim. This change ships a mechanism and the evidence that it is correctly wired, not a result.
