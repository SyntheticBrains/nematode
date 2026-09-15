# Design: the wiring as fixed features, and what that contrast can carry

## What is being contrasted, precisely

Under `readout_only` the connectome's chemical matrix is **frozen** and the 2×4 motor readout learns
by its own exact gradient. The learner therefore consumes the wiring as a **fixed feature map**: the
settling dynamics turn a 3-feature klinotaxis input into 302 activations, those are mean-pooled into
four motor-class means, and eight parameters decode them into `(speed, turn)`.

Rewiring changes the feature map and nothing else. `rewire_degree_preserving` preserves the neuron
set and ordering, and per-post fan-in — so the strict mask's shape, the weight-init scale
`1/√(chemical in-degree)`, the gap-junction normalisation, and the motor pool are all preserved. What
differs is **which pre-synaptic neuron reaches which post-synaptic neuron**.

So the question is sharp: *do the wild-type edges compute better four-dimensional features for this
task than a degree-matched shuffle of the same edges?*

## Why this cell

`hard350` is the only cell where the comparator, the learner and the metric already meet:

| | on `hard350` | source |
|---|---|---|
| the same contrast under PPO | wild type 892 episodes vs null 1165, **+23.5%**, 32 paired seeds | V.3 |
| this learner's level | `readout_only` at 17.570 foods, **52.61% full clear**, 16/16 seeds | R.2 |
| the metric's viability | `episodes_to_30pct_success` well-posed — far above the 30% threshold, far below ceiling | R.2 + V.3 |

The last row is not a formality. V.3's calibration found the band **one step wide**: at
`max_steps: 250` learning was plainly happening while **no arm crossed the 30% threshold the primary
metric needs**, so the contrast would have been censored for every seed and read as a clean null. At
350, with a learner at 52.61% mean full clear, that failure mode is off the table — which is the
reason this cell is used rather than V.1's thermal one, whose operating point was never calibrated
for this learner.

## The matched projection, by construction and by test

`random` routing draws the feedback projection `B` from a `torch.Generator()` seeded with the run
seed; `rewire_degree_preserving` draws from a separate numpy generator; and rewiring preserves
`n_neurons`. So at seed *N* the wild-type and rewired arms receive the **same** `B`, and the pair
differs by the wiring alone.

That is a reading of the code, so the change **asserts it by test** rather than relying on it. It
matters because the alternative — each wiring exploring through a different random projection — would
confound the wiring with the feedback path, and the confound would be invisible in the results.

## The reading

Block V's instrument, unchanged: `episodes_to_30pct_success` through the committed
`connectome_structure_efficiency.py`, four efficiency metrics, paired by seed, BH-FDR, against the
registered **≥ 20%** minimum on time-to-competence.

Three checks travel with the primary, as they did in V.1 and V.3:

- **Two learning gates** — each wiring against its **own** frozen floor. A contrast between two arms
  that did not learn is not a wiring result, which is what the gates exist to exclude.
- **The untrained prior** — wild-type frozen against rewired frozen. V.1 measured this at −0.17
  (q = 0.735) and V.3 at −0.01 (q = 0.841): the prior is indistinguishable between wirings, and if
  that changed here it would mean the rewiring is doing something to the substrate before any
  learning, which would make the primary uninterpretable.
- **Credited drift**, which under this learner should read **0.00** on `w_chem` for both wirings —
  the check that the substrate really was frozen, as it was in R.2.

## The power arithmetic, registered because a null closes the phase

| | k needed for p ≤ 0.05 | power at a 66% win rate | at 75% | at 81% |
|---|---|---|---|---|
| 16 pairs | 12/16 (75%) | 32% | 63% | 83% |
| **32 pairs** | 22/32 (69%) | 45% | **85%** | 97% |

V.3's observed per-seed win rate on this contrast was **21–26 of 32 (66–81%)**. Sixteen pairs would
have given 63% power at the midpoint of that range — missing a real effect of the comparator's size
about **37%** of the time — which is not an acceptable basis for a null that closes a phase and
stands as "the wiring is inert even as fixed features". Thirty-two pairs match the comparator's own
power.

These are **sign-test** figures. The registered test is a paired rank test under BH-FDR, which uses
magnitudes and so has somewhat more power; the sign test is quoted because it is the conservative
floor and needs no assumption about the effect's distribution.

## Outcomes

| verdict | test | what follows |
|---|---|---|
| `wiring_is_legible` | wild type beats the null on time-to-competence by ≥ 20%, significant under BH-FDR, both learning gates pass, prior indistinguishable | The first wiring result in this project under a **plausible** learner. Phase 7 closes with three citable results. L.1 asks "how much more" and **L.4/L.5 open** — which part of the wiring carries it |
| `wiring_is_inert_as_features` | no significant advantage at the registered bar, gates and prior clean | 034's degree-statistics verdict extends to a **second learning regime**: the wiring is endpoint-inert under gradient learning, harmful under local rules that write it, and indistinguishable from a degree-matched shuffle as fixed features. **L.1 is promoted to MUST** — the width question a null raises |
| `below_bar` | a significant advantage under the registered bar | Reported as suggestive with the bar unmet, as V.1 and V.3 would have been. L.1 becomes the follow-up either way |
| `void` | a learning gate fails, or the untrained prior separates | The contrast is uninterpretable: either the arms did not learn, or the rewiring changed the substrate before learning did |

## What this may not be cited as

- **Evidence for D2's primary.** That requires *plastic* wild-type to beat *plastic* rewired-null.
  Nothing here makes the wiring plastic, and a positive cannot convert Phase 7's SPLIT into a GO.
- **A dynamics claim.** It is a performance claim: the wild-type edges compute better features for
  this task. Bars (a) and (b) of the claim discipline are not attempted.
- **A result about the wiring in general.** One cell, one readout width, one feature dimensionality,
  one learner. The readout is also standing in for the entire motor periphery, which the substrate
  ladder's body rung is what would replace.
- **A comparison with V.3's +23.5% as a quantitative delta.** The two run under different learning
  regimes, and the project's own commensurability rule forbids treating cross-regime deltas
  quantitatively. Same cell, same metric, same bar — read as two answers to one question, not as a
  difference.
