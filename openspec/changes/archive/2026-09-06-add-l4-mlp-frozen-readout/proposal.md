# Freeze the matched-rule yardstick's readout

## Why

The panel's matched-rule MLP arm exists so that "recovery" has a meaning: the same three-factor
rule, the same hyperparameters, on a dense substrate. When it was added, every `Linear` layer
including the output layer was made plastic, ratified as the conservative choice on the reasoning
that a random output layer has nothing to preserve. Two pilots and a targeted probe show that
choice breaks the arm under the rule it was built to carry
(`docs/experiments/logbooks/supporting/040-l4-panel/probe-5-foraging-yardstick.md`):

- On the foraging-only cell, the **frozen** tanh MLP forages at 96% from its random initialisation
  at seed 101; the **plastic** MLP, starting from the same weights, is at 10% in its first hundred
  episodes and zero for the remaining five hundred. The rule does not fail to find a solution; it
  removes one within a few episodes.
- Its tracked action density under the tanh-squashed Gaussian climbs from 4 to 386 by the third
  episode and to `1e17` by the end (the frozen run stays under 18). The density carries the squash
  Jacobian, so `1e17` means the pre-squash action mean sits near 20: the actions are pinned at the
  squash limits and the worm does one thing forever.
- The weights are bounded throughout (homeostasis holds every row norm; an in-process
  reproduction keeps every activation under 1.2). The failure is *direction*: the output layer's
  post-synaptic factor is its own output, so `Δw ∝ m · (u ⊗ h)` rotates each output row toward
  the hidden-activity direction that maximises `|u|`, and with the norm held that rotation ends in
  saturation. The connectome cannot do this — its readout is frozen and anatomical, and its
  plastic weights sit behind bounded recurrent units.

A yardstick that saturates by construction says nothing about dense substrates and makes D2
test (ii) vacuous. Ratified with Chris: the MLP arm learns its hidden weights under a **frozen
readout**, the structure the connectome arm already has, through a default-off option so every
existing build is byte-identical.

## What Changes

- `plastic_layers` on the MLP brain configuration: `all` (default, byte-identical) or `hidden`.
  Under `hidden` the output `Linear` is not on the plastic-topology seam: no trace, no mask, no
  fan-in axis, no homeostatic target, and the rule never writes it.
- The MLP topology's forward still runs the whole actor bitwise-identically and credits
  eligibility only to the plastic layers.
- Tests on both settings; docs.

Out of scope: the panel's registration (its MLP config sets `hidden` by a further dated amendment,
the pin having already been made, and the foraging probe is re-run there), any change to the rule, the connectome,
or the PPO path (which trains every layer regardless of this option).

## Capabilities

**Modified**: `brain-architecture` — the requirement that fixed the MLP's plastic set as every
linear weight is modified to select it by `plastic_layers`, every existing scenario keeping its
name; the all-layers asymmetry statement is retracted with its reason.

## Impact

- Edited: `brain/arch/mlpppo.py`, `brain/arch/_mlp_topology.py`, tests, `docs/architectures.md`,
  `CHANGELOG.md`.
- No config changes here; default builds are byte-identical.
