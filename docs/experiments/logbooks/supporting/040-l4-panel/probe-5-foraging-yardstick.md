# Probe 5 (foraging-only cell): is the yardstick's failure ours?

Run 2026-09-06 from commit `50397337` (the panel's pinned state plus the launch record), before
the panel was launched, at Chris's request: does the tanh MLP under the pinned recipe learn
*anything* on an easier cell? Connectome and MLP, plastic and frozen, on the Fick-adaptive
foraging cell (no predator, no thermotaxis), seeds 101–102, 600 episodes, 8 runs, 6 min.
Diagnostic, not part of the registration.

## Full-clear successes per 100-episode block and plateau tail (last 150)

| run | blocks 1–6 | tail % |
|---|---|---|
| connectome frozen, s101 / s102 | 0 0 0 0 0 0 / 7 9 12 12 13 9 | 0.0 / 10.7 |
| connectome plastic, s101 / s102 | 3 0 0 0 0 0 / 35 37 17 0 11 24 | 0.0 / 22.0 |
| **MLP frozen, s101** | **93 90 87 92 96 96** | **96.0** |
| MLP frozen, s102 | 0 0 0 0 0 0 | 0.0 |
| **MLP plastic, s101** (same initial weights as the frozen run) | **10 2 0 0 2 0** | **0.0** |
| MLP plastic, s102 | 0 0 0 0 0 0 | 0.0 |

## Reading

- **The plastic MLP destroys a working policy.** At seed 101 the frozen MLP forages at 96% from its
  random initialisation; the plastic MLP starts from the same weights and is at 10% in its first
  hundred episodes and zero for the remaining five hundred. That is not a failure to find a
  solution; it is a rule that removes one within a few episodes.
- **The actions saturate.** The tracked per-step probability of the plastic run's actions reaches
  `1e14` by its third episode and `1e17` by the end, against at most `18` for the frozen run. Under
  the tanh-squashed Gaussian the density carries the squash Jacobian `1 / (1 − a²)`, so a density
  of `1e17` means the pre-squash action mean sits near 20 and the actions are pinned at the squash
  limits: the worm does one thing, forever.
- **The weights are bounded, so this is direction, not magnitude.** An in-process reproduction with
  synthetic inputs holds every row norm at its target and every activation below 1.2. The output
  mean cannot exceed about 13 in magnitude with these norms; it reaches the limit by *alignment*:
  the output layer is plastic and its post-synaptic factor is its own output, so the update
  `Δw ∝ m · (u ⊗ h)` pushes each output row toward the hidden-activity direction that maximises
  `|u|`, and homeostasis, holding the norm, turns that into pure rotation into saturation. The
  connectome cannot do this: its readout is frozen and anatomical, and its plastic weights sit
  behind bounded recurrent units.
- **The connectome on this cell** learns weakly within 600 episodes on one seed (10.7 → 22.0) and
  not at all on the other; it is a different cell from the panel's (Fick fields, adaptive
  sensor) and is not read further here.

## Consequence

The yardstick as configured is broken by design, not by the task: the all-layers-plastic choice
made for the matched-rule MLP arm (ratified as "conservative" when the arm was added) gives it a
plastic readout that self-amplifies into saturation under this rule. Ratified with Chris: the MLP
arm learns its hidden weights under a **frozen readout**, like the connectome, through a
`plastic_layers` option on the MLP (default `all`, byte-identical), in its own pre-registered
change; this probe is re-run to confirm the arm is functional before the panel launches.
