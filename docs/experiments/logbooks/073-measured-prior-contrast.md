# 073: Measured Weights Do Not Change What the Wiring Is Worth Under PPO, and Leave the Reading Learner Unresolved (Phase 8 B.1c)

**Status**: completed — **under PPO the measured weights do not move the wiring effect, and under the
reading learner the panel cannot tell.**

- **PPO** (fan-in draw, pooled readout): the wild type's lead over its rewired null is present at all
  three priors, **+0.078**, **+0.091** and **+0.070** `auc_success`, each interval excluding zero. The
  measured prior moves it by +0.013 [−0.001, +0.027], inside the registered minimum on both sides:
  **no move**. Whether the fitted *placement* matters is **unresolved**, because the upper bound,
  +0.040, just crosses the 0.037 minimum. Verdict: `null_placement_unresolved`.
- **The reading learner**: the null's lead shrinks from −0.109 under random weights to −0.026 under
  measured ones. That interaction, +0.082 [−0.002, +0.157], spans the 0.140 minimum, so the verdict is
  **unresolved** at this panel's sensitivity. Placement does not move the gap by the minimum.
- **Registered branches that did not fire:** no learner reaches `legible` or `hides`, and no Lee
  failure occurs.

B.1's measured-weight rung is therefore resolved for shipment 8a without a positive.

**Date**: 2026-09-24 to 2026-09-25.

**OpenSpec change**: `add-measured-prior-contrast`. It extends `architecture-comparison-protocol`
with one requirement: a measured-weight positive is read against its placement-shuffled control.

**Pre-registration**: [supporting/073-measured-prior-contrast/launch.md](supporting/073-measured-prior-contrast/launch.md),
written before any seed ran.

## Objective

D17 asks whether the animal's own synaptic weights make its wiring legible where random ones did
not. Every connectome brain here has been anatomically constrained in topology and randomly
initialised in weight, and every wiring result so far, block V's included, was read on random
weights.

[Logbook 072](072-measured-prior-pilot.md) showed that the Creamer–Leifer–Pillow fitted weights
leave the pathway learnable. It fixed the multiplier at 1.0, where the covered edges carry the random
draw's magnitude, so this contrast compares structure rather than size.

## Method

Wiring {wild type, rewired null} × prior {random, measured, measured-shuffled}, each arm learning and
frozen, on hard350. Every level is gated against its own frozen floor.

| learner | draw, readout | seeds | runs | wall clock |
|---|---|---|---|---|
| PPO | `per_neuron_fanin`, pooled | 225–256 (32) | 384 | 5.9 h |
| reading (`readout_only`) | `edge_order`, A.2's centre | 257–304 (48) | 576 | 16.2 h |

**Two interactions per learner**, each paired by seed on the wiring gap (wild type minus null,
positive when the wild type is better):

1. **measured × wiring**, `gap(measured) − gap(random)`;
2. **placement × wiring**, `gap(measured) − gap(shuffled)`.

The shuffled arm keeps the fitted values and destroys their assignment, with a different permutation
at every seed. That makes the second interaction the only one that can say "the animal's weights"
rather than "a distribution of values".

**The metric.** `auc_success` is the primary on both learners. This is a departure from A.2's
censoring rule, registered before launch: on episodes, PPO's detectable effect was 1.14× its
committed wiring effect, too coarse to see a sign move. Episodes are reported beside. The censoring
rule would have chosen episodes for PPO and `auc_success` for the reading learner.

**The minimum** is 2/3 of each learner's committed wiring effect on the draw it runs: 0.037 for PPO
(A.1's fan-in effect, +0.055) and 0.140 for the reading learner (A.2's centre, −0.2105). The q values
come from BH-FDR across the four primary interactions, using a two-sided Wilcoxon, with 80% bootstrap
intervals. Every statistic and instrument is A.2's, unmodified.

## Results

### The gates: every level readable on both learners

Plateau success against each level's own frozen floor. Every floor is 0.0 and no level saturates, so
all six interactions are readable and the Lee branch did not fire at panel scale.

| level | PPO wild type | PPO null | reading wild type | reading null |
|---|---|---|---|---|
| random | 76.9 | 70.9 | 51.6 | 64.0 |
| measured | 76.7 | 69.3 | 49.8 | 53.4 |
| shuffled | 76.7 | 71.4 | 47.4 | 58.1 |

**The reading learner's substrate stayed frozen.** Its `w_chem` drift is **0.0** on every seed at
every level, so that half is not void. Under PPO, which writes the matrix, the same check reads up to
1.44 relative drift, which is its positive control.

### The primary

`auc_success`, with 80% intervals and BH-FDR q.

| learner | interaction | effect | interval | q | minimum | state |
|---|---|---|---|---|---|---|
| PPO | measured × wiring | +0.0128 | [−0.0009, +0.0269] | 0.317 | 0.037 | **no_move** |
| PPO | placement × wiring | +0.0204 | [−0.0008, +0.0403] | 0.317 | 0.037 | **unresolved** |
| reading | measured × wiring | +0.0821 | [−0.0024, +0.1568] | 0.317 | 0.140 | **unresolved** |
| reading | placement × wiring | +0.0610 | [−0.0123, +0.1309] | 0.317 | 0.140 | **no_move** |

**Verdicts, read off the registered map:**

- **PPO: `null_placement_unresolved`.** The measured prior does not move the wiring effect by
  two-thirds of its committed size in either direction: its interval sits inside ±0.037. Whether the
  fitted placement matters cannot be said. The placement interval, reaching +0.040, crosses the
  minimum, but only just.
- **Reading: `unresolved`.** The measured-versus-random interval spans the minimum. The map reads that
  row first, so the placement contrast's `no_move` does not rescue a verdict.

**Sensitivity achieved against sensitivity registered.** From each interaction's own per-seed spread,
the MDEs are 0.50× and 0.72× of the reference effect on PPO (measured, placement), and 0.71× and 0.66×
on the reading learner. Registration expected about 0.75×. The PPO measured contrast was sharper than
planned, and the rest landed near plan. That is why PPO could reach `no_move` and the reading learner
could not.

### The gaps, reported beside

The wiring gap per level, on `auc_success` and on episodes, with 80% intervals:

| learner | level | `auc_success` | episodes to 30% |
|---|---|---|---|
| PPO | random | +0.078 [+0.063, +0.092] | +470 [+327, +616] |
| PPO | measured | +0.091 [+0.076, +0.104] | +547 [+369, +712] |
| PPO | shuffled | +0.070 [+0.056, +0.084] | +492 [+328, +659] |
| reading | random | −0.109 [−0.169, −0.036] | −456 [−688, −184] |
| reading | measured | −0.026 [−0.094, +0.036] | −132 [−399, +130] |
| reading | shuffled | −0.087 [−0.137, −0.033] | −81 [−260, +120] |

**Block V's wiring advantage is present under measured weights.** Under PPO with the fan-in draw the
wild type leads at every prior, each interval excluding zero on both metrics. The random level's
+0.078 is 1.4 times A.1's +0.055 for the same draw and cell on fresh seeds. This reading is beside the
primary and not a registered test. It is the strongest statement this panel makes: on the edges the
fitted table covers, the effect block V found on random weights does not depend on those weights
being random.

**The reading learner's null lead narrows under measured weights**, from −0.109 to −0.026, with the
measured level's interval spanning zero. At this panel's sensitivity it is unresolved. It points the
same way as Logbook 072's descriptive observation, where the null's lead closed at larger multipliers.
The shuffled level sits between random and measured on `auc_success`, and on episodes it is nearer
the measured level than the random one. Both are descriptions, not registered readings.

**Episodes, reported beside:** no interaction is significant after correction on either learner (q =
0.875 throughout). The reading learner's measured interaction on episodes is +324.5 [−29.5, +650.4],
the same direction as on `auc_success`.

## Corrections made in the open

- **PPO's reference effect was wrong in the first draft** and was corrected at spec review, before
  launch. The draft used A.1's `edge_order` effect (+0.061 / +580.4) for a panel that runs PPO under
  `per_neuron_fanin`, where A.1's effect is +0.055 / +392.4. On the draft's registered primary,
  episodes, the panel could not have seen a sign move. `auc_success` became the primary on both
  learners, registered with its reason.
- **The draft's verdict map was asymmetric** and was corrected at spec review. "Hides" was granted on
  the measured-versus-random contrast alone, while "legible" needed the placement control. Both now
  need it.
- **None of the figures above changed after scoring.**

## What this establishes, and what it does not

**Establishes.**

- Under PPO on hard350, with the fan-in draw and a pooled readout, replacing the random draw with the
  fitted weights on the 1,049 edges they cover does not change the wiring effect by two-thirds of its
  committed size in either direction.
- The wild type's advantage is present at every prior, on both metrics, on fresh seeds.
- Every arm on both learners learns under every prior.
- The reading learner's substrate stayed frozen throughout.

**Leaves unresolved.**

- Whether the fitted placement matters under PPO: an interval edging past a small minimum.
- Whether measured weights move the reading learner's wiring effect. The point estimate narrows the
  null's lead by about 40% of the reference effect, and the interval spans the 2/3 minimum.

**Does not establish.**

- That measured weights make the wiring legible, or hide it: neither verdict was reached on either
  learner.
- That the measured prior has no effect smaller than the minimum; `no_move` is about the registered
  minimum and nothing finer.
- Anything beyond head scope. The fitted table reaches 1,049 of 3,709 chemical edges (77.0% of the
  1,363 coverable at head scope) and none onto a body motor neuron, so the command-to-motor and motor
  layers carry the random draw in every arm.
- That the conclusion stands alone. The weights come from a preprint fitted on another connectome
  (White 1986 + Witvliet 2020).

## Registered consequences

- **B.1c: met.** The 2×3 ran under both learners at swept points, with coverage reported at head and
  full scope, and each learner has a registered verdict: PPO `null_placement_unresolved`, reading
  `unresolved`.

- **B.1: met, without a positive.** The data was vendored, the pilot was run, and the 2×3 was read.
  The payoff D17 named "either way" arrives as its null side under PPO, and as unresolved under the
  reading learner.

- **The Lee risk: did not materialise** at panel scale on either learner, and is closed for B.1.

- **Logbook 034 and block V: no citation-site change owed.** The registration tied a Logbook 034
  update to a `null` verdict and a block V update to `hides`, and neither was reached.

  The block V observation above, its advantage present under measured weights, is recorded here and
  in the tracker as a reported-beside reading, not as a registered result.

  Every PPO statement carries its conditions in the same sentence. The fan-in draw and pooled readout
  sit at a point whose depth and initial noise come from A.2's `edge_order` surface, with readout
  width unresolved there.

- **The reading learner's unresolved interaction** is *deferred-with-destination* to the 8a
  synthesis (S8a). S8a decides whether it earns a larger panel or stands as unresolved; it gates
  nothing in 8b.

- **A.5, the publication decision**, now has B.1's result to weigh.

## Artefacts

Under [supporting/073-measured-prior-contrast/](supporting/073-measured-prior-contrast/):

- `launch.md`: the registration.
- `contrast.json`: every level's gates (with each seed's plateau and floor), gaps, both interactions
  on both metrics with their q, the states, the verdicts, the censoring rule's own choice, the drift
  evidence and the coverage counts.
- `per-seed.csv`: 80 rows, one per learner and seed, holding every arm's plateau and floor, every
  gap, and both interactions on both metrics.

Drivers:

- `scripts/analysis/measured_prior_contrast.py`, whose stems come from
  `scripts/analysis/measured_prior_pilot.py` and whose statistics come from
  `scripts/analysis/operating_point_surface.py`;
- configs generated by `scripts/campaigns/generate_measured_prior_configs.py`.

Raw campaign logs, 960 runs across two campaign directories, are archived off-repo under A.0.
