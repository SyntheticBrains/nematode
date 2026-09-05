# Make the three-factor rule's scaling substrate-invariant

## Why

The L4 panel's registered pilot pinned nothing, and the reason is scale, not the rule's
ability to learn. Two facts from that pilot fix the problem this change solves:

- **The modulator is dominated by one step.** The reward prediction error at an episode's
  terminal step is about −10 (a death penalty against a baseline near −0.3), while ordinary
  steps sit at ±0.05. One update at the default rate therefore moves weights by order 1
  against an initialisation scale near 0.3, and the connectome arms random-walk onto the
  ±3 bound; the Hebbian floor, whose update never changes sign for a co-active pair, ends
  with 96% of its synapses clamped.
- **A matched rate is not a matched rule.** The MLP's per-weight eligibility trace is roughly
  a thousand times smaller than the connectome's, so the same `plasticity_rate` is lethal on
  the dense substrate (dead ReLU units after one −10 kick in episode one) and negligible on
  it at any cooler rate. No single rate serves both substrates, which empties the matched-rule
  yardstick the panel's second success test depends on.

Diagnostic probes show the connectome learning cleanly at a rate two orders below the
registered grid. The defect is therefore in what `η` means: today it is an absolute step in
units that differ by orders of magnitude between substrates and between ordinary and terminal
steps. The pre-registration for the panel anticipated this branch and routed it here rather
than to a wider grid: a rule change must be pre-registered on its own.

Both design choices below were settled with Chris before this proposal was written:

- **The modulator becomes bounded and scale-free**: `δ̃ = tanh(δ / σ)`, with `σ` a running
  root-mean-square of the prediction error. Sign-preserving, monotone, in `[−1, 1]`, one
  fewer arbitrary constant than a clip; linear near zero and saturating for large surprise,
  bounded, as firing rates are.
- **The eligibility trace is normalised per plastic tensor by a running scale**:
  `Δw = η · δ̃ · E / ρ`, with `ρ` a running root-mean-square of the trace over the tensor's
  edge set. `η` then means the root-mean-square step per unit modulator on every substrate,
  while step-to-step variation of the trace is kept: a step with more co-activity still
  moves more.

## What Changes

- Two opt-in switches and two shared scale hyperparameters on the plasticity config mixin
  (so every plastic brain gets the same definition): `plasticity_normalise_modulator`,
  `plasticity_normalise_trace`, `plasticity_scale_rate`, `plasticity_scale_floor`. Both
  switches default off, and with both off the rule is **byte-identical** to today's, proven
  against the frozen reference.
- The rule maintains the two running scales as its own state beside the baseline, keeps them bias-corrected so the first observation counts fully, updates them under a freeze and in unmodulated mode (so
  telemetry stays comparable across arms), and applies them only to the Hebbian term — the
  decay and the clamp are unchanged.
- Three new telemetry keys recorded beside the existing four: the effective modulator, the
  modulator scale, and the trace scale.
- Tests: the product with normalisation on; invariance of the Hebbian step to a constant
  rescaling of the trace; the modulator's bound; the bias correction; byte-identity with both off;
  freeze and unmodulated interplay; the keys; and the matched-rule invariance across the two
  substrates under normalisation.
- Docs: the architectures table's plasticity row, the CHANGELOG.

Out of scope: any change to the trace substrate, the panel's registration (its grid is
re-registered in the panel change once this lands), the MLP's activation function (the
substrate is frozen), and the existing arm configs (the panel change turns the switches on
for every arm at its pin step).

## Capabilities

**Modified**: `learning-rules` — gains a substrate-invariant scaling mode for the
three-factor rule; the existing requirements are untouched and their default behaviour is
preserved bit for bit.

## Impact

- Edited: `learning_rules/three_factor.py`, `brain/arch/_plasticity_config.py`,
  `brain/arch/_brain.py` (three history fields), tests under `learning_rules/`,
  `docs/architectures.md`, `CHANGELOG.md`.
- No config changes; no change to the PPO path; trace-off and switch-off builds are
  byte-identical.
