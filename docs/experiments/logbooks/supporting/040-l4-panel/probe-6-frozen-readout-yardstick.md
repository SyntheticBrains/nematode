# Probe 6 (frozen readout): the yardstick with `plastic_layers: hidden`

Run 2026-09-06 from commit `e9380fe4` (the MLP arm on `plastic_layers: hidden`, all other pinned
values as before) on the rule as merged in PR #317. The hidden-readout MLP, plastic and frozen,
on the panel's C3 cell and on the Fick-adaptive foraging cell, seeds 101–102, 600 episodes.
Diagnostic, not part of the registration.

## Full-clear successes per 100-episode block, plateau tail (last 150), peak tracked action density

| cell | run | blocks 1–6 | tail % | peak density |
|---|---|---|---|---|
| foraging | MLP frozen, s101 | 93 90 87 92 96 96 | 96.0 | 18 |
| foraging | MLP frozen, s102 | 0 0 0 0 0 0 | 0.0 | 34 |
| foraging | **MLP plastic (hidden), s101** | **1 0 0 0 0 0** | **0.0** | **1002** |
| foraging | MLP plastic (hidden), s102 | 0 0 0 0 0 0 | 0.0 | 4084 |
| C3 | MLP frozen, s101 / s102 | all 0 | 0.0 / 0.0 | 19 / 32 |
| C3 | MLP plastic (hidden), s101 / s102 | all 0 | 0.0 / 0.0 | 5673 / 785 |

For comparison, probe 5's all-layers plastic MLP reached a density of `1e17` on the foraging
cell; the frozen readout brings that to `1e3` (a pre-squash mean near 3.5 instead of 20), but the
policy that foraged at 96% is still gone within the first hundred episodes.

## Reading

- **Freezing the readout was necessary and is not sufficient.** The output layer's
  self-amplification is gone (the density falls by fourteen orders of magnitude), and the hidden
  layers still destroy the policy on their own.
- **Why the hidden layers collapse.** The actor's input is a mostly-zero feature vector, so for
  every hidden unit the eligibility `post ⊗ pre` points at the same few active inputs. Under a
  Hebbian update every row of a dense layer rotates toward the same direction; homeostasis holds
  each row's norm but not its independence, so the sixty-four units converge on one pattern, the
  representation collapses toward rank one, and a fixed readout of a rank-one representation is a
  constant, large action mean. This is the textbook failure of Hebbian learning without lateral
  decorrelation (the reason Oja's rule alone yields the first principal component for every unit,
  and the reason cortical circuits carry lateral inhibition). The connectome does not suffer it:
  its wiring is sparse (about twelve chemical inputs per neuron, not sixty-four), its units are
  recurrent and bounded, and its eligibility is a *temporal* correlation across a step rather
  than an instantaneous one within a layer.
- **This is a property of the matched rule on a dense substrate, not a defect of the arm.** The
  readout, the activation, the noise, the runaway control and the modulator have each been
  examined and fixed where they were ours; what remains is the rule itself meeting a substrate
  whose structure gives it nothing to be legible to. The minimal three-factor rule, as
  registered, cannot hold a policy on a dense feedforward stack, let alone find one.

## Consequence for the panel

The MLP arm enters the panel as registered, with the frozen readout (the strictly better and
structurally matched configuration). Its plateau is expected at chance. D2 test (ii) will pass
by construction, and the logbook reads it as the design already prescribes: the yardstick
result is a finding about local rules on dense stacks, the ranking half of "recovery" is empty,
and any structure-function claim rests on T1 and T4 alone. A rule with lateral decorrelation
would be a different rule and a new pre-registration.
