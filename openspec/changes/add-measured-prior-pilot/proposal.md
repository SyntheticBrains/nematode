## Why

Phase 8's **B.1b**, the pilot between B.1a's data and B.1c's 2×3.

B.1a vendored the Creamer–Leifer–Pillow fitted weights and added `weight_prior` with the
`measured_weight_scale` multiplier. Three things stand between that and a registrable 2×3:

- **The multiplier is a pin nobody has swept.** It maps coefficients of a 2 Hz linear dynamical
  system on calcium signals onto this rate model, and there is no natural mapping. Decision **D16**
  forbids a Phase 8 contrast at a pin unswept for the learner it uses, and B.1c runs two learners.
- **The pathway may be unlearnable.** Lee 2026 fitted c302's global conductances to the Randi atlas
  and found no functional sensory-to-command step. D17 names the pilot's sign-only-versus-magnitude
  arms, with the reading learner's positive control, as the place this failure would show. If it
  shows, B.1 closes *unmet-with-reason* with the pathway named.
- **B.1c's PPO arm cannot be built.** D17 requires a measured PPO arm to run "under D15's shared-init
  protocol", because under PPO a measured prior is an initialisation. B.1a refuses every measured
  prior with a non-default weight draw.

## What Changes

- **The pairing with `per_neuron_fanin` is defined and allowed.** `dense_mask` stays refused.
  - **This is more than removing a refusal.** The fan-in draw shares each neuron's incoming multiset
    across wirings. If covered edges are overwritten naively, the wild type discards its draws at the
    covered positions and the null discards its first *k*, so the two keep different subsets of the
    same block and the multiset is no longer shared.
  - **The definition:** on the wild type, every edge keeps its fan-in draw unless the prior covers it.
    On the null, each neuron's first *k* incoming edges receive the wild type's covered values and
    the rest receive the wild type's **uncovered** draws, each in wild-type pre-synaptic order. Every
    neuron then carries its wild type's exact multiset under every prior, and the wild type's
    uncovered edges stay bit-identical to its random build.
- **`measured_shuffled` draws its permutation from a generator of its own.** It currently uses a
  second generator at the same seed as the draw, so once fan-in draws coexist with the shuffle, both
  would be read from the same bits. No run has used this prior, so the change costs nothing.
- **The pilot's configs, a committed generator and an analysis script.**
  - **Arms:** the wild type and its rewired null, each learning and frozen, under the random prior,
    `measured_signs`, and `measured` at multipliers 0.25, 0.5, 1, 2 and 4.
  - **Both learners, both on hard350:** PPO under the fan-in draw, and the reading learner
    (`readout_only`) at A.2's centre.
  - **Seeds:** 8 per learner, on disjoint bands never used before.
  - **Size:** 448 runs, about 9–10 hours at 16 workers.
- **A rule for choosing a pin**, added to `architecture-comparison-protocol`: the value is chosen on
  the learner's own gate, never on the contrast the value will carry.
- **Registered before any seed runs:** each level is gated against its own frozen floor, and the
  multiplier is selected by a rule fixed in advance. The wiring gap is recorded descriptively, and
  the branches (magnitude is the obstacle, the pathway is unlearnable, the sign moves across the
  multiplier) are named with their consequences for B.1c.

## Capabilities

### Modified Capabilities

- `connectome-ppo-brain`: the measured-prior requirement defines the fan-in pairing and gives the
  shuffle its own generator.
- `architecture-comparison-protocol`: adds the pin-selection rule.

## Impact

- **Code:** `brain/arch/connectome_ppo.py`, covering the refusal, the rewired placement under the
  fan-in draw, and the shuffle's generator. No change under the default prior or under `edge_order`,
  so every committed arm is unaffected.
- **Scripts and configs:** `scripts/campaigns/generate_measured_prior_configs.py`,
  `scripts/analysis/measured_prior_pilot.py`, 48 configs under `configs/scenarios/foraging/`, and a
  minimal refactor of `scripts/analysis/operating_point_surface.py` so the learning gate can be
  shared rather than copied.
- **Records:** Logbook 072 with its supporting files; tracker B.1b and B.1c; roadmap D17 and § B.1.
