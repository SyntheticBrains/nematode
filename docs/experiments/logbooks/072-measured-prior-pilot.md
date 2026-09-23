# 072: Measured Weights Are Learnable at Every Scale Tried, and Both Learners Take the Matched One (Phase 8 B.1b)

**Status**: completed — **the measured weights do not break the pathway, and B.1c runs at the
magnitude-matched multiplier, 1.0, on both learners.** Under PPO and under the reading learner, every
level — the sign-only prior and the measured prior at 0.25, 0.5, 1, 2 and 4 times the random draw's
magnitude — beats its own frozen floor on both wirings, and none saturates. The Lee 2026 failure
mode, a measured grounding that leaves the sensory-to-motor pathway unlearnable, does not appear
here at pilot scale. No level's wiring gap moved to the other side of zero, so **B.1c carries no
registered condition from this pilot.** One pattern in the reading learner's gap is reported below
as descriptive only.

**Date**: 2026-09-23 to 2026-09-24.

**OpenSpec change**: `add-measured-prior-pilot` (extends `connectome-ppo-brain`: a measured prior is
defined under the per-neuron fan-in draw, and the shuffle gets its own random stream; extends
`architecture-comparison-protocol`: a pin is chosen on the learner's own gate, never on the contrast
it will carry).

**Pre-registration**: [supporting/072-measured-prior-pilot/launch.md](supporting/072-measured-prior-pilot/launch.md),
written before any seed ran.

## Objective

B.1a vendored the Creamer–Leifer–Pillow fitted weights (bioRxiv 2024.09.22.614271, a preprint) and
added `weight_prior` with a multiplier, `measured_weight_scale`, that maps the fitted values onto this
rate model. Two things stood between that and B.1c's registered 2×3:

1. **The multiplier was a pin nobody had swept.** The fitted values are coefficients of a 2 Hz linear
   dynamical system on calcium signals, and nothing maps them onto this model. D16 forbids a Phase 8
   contrast at a pin unswept for the learner it uses, and B.1c runs two learners.
2. **The pathway might be unlearnable.** Lee 2026 (bioRxiv 2026.09.06.749731) fitted c302's global
   conductances to the Randi atlas and found no functional sensory-to-command step. D17 names this
   pilot's sign-only-versus-magnitude arms as the place the same failure would show here.

The pilot does **not** ask whether measured weights make the wiring legible. That is B.1c's question.

## Method

One cell, **hard350**, the cell A.2 swept both learners on. Seven levels per learner, each on both
wirings, learning and frozen, with **every level gated against its own frozen floor**, because the
prior and the multiplier change the substrate before any learning.

| | PPO half | reading half |
|---|---|---|
| learner | PPO | `readout_only`, the three-factor rule with e-prop eligibility |
| weight draw | `per_neuron_fanin` (D15's shared initialisation, which B.1c's PPO arm runs under) | `edge_order` (where A.2 swept it) |
| seeds | 113–120 | 121–128 |
| runs | 224 | 224 |
| wall clock | 3.15 h | 6.02 h |

| level | `weight_prior` | multiplier |
|---|---|---|
| `random` | `random` | — |
| `sign` | `measured_signs` | — |
| `m025` … `m4` | `measured` | 0.25, 0.5, 1, 2, 4 |

**The PPO half needed a definition this change adds.** Under the fan-in draw a rewired neuron holds
the same block of drawn values as its wild type. Overwriting covered edges naively would have the two
wirings keep different subsets of that block. On the null, each neuron's block is therefore reordered
so that the wild type's covered values come first and its uncovered draws after, which keeps every
neuron's wild-type multiset of incoming values under every prior. Tests assert the multiset, and 6
of the 8 placement tests fail with the reorder removed.

**The gate reads twice, with one job each.** Both come from A.2's own `learning_gates`: the
instrument's paired Wilcoxon on per-seed plateau against the floor, with an 80% bootstrap interval.

- *The wild type learns* (its lower bound above zero) decides the Lee branch.
- *The level passes* (both arms learn, and the level is not saturated at the 90% bar) decides the
  multiplier.

**The selection rule was fixed in advance:** 1.0 if it passes, otherwise the passing level nearest it
on the log scale. **The wiring gap never enters it.** `select` takes the gate records and nothing else,
and a test pins that signature. The gap is recorded at every level, on the metric A.2's per-level
censoring rule chooses, as description only.

Every instrument ran unmodified. The gate and the drift check gained a floor-level argument so the
pilot could call them rather than copy them, and A.2's 190 tests pass unchanged.

## Results

### The Lee branch: not triggered on either learner

Plateau success (percent of episodes clearing every food, over the final quarter) against each level's own
frozen floor. Every learning arm beat its floor on **8 of 8 seeds** at every level.

| level | PPO wild type | PPO null | reading wild type | reading null |
|---|---|---|---|---|
| `random` | 77.2 | 70.8 | 53.6 | 64.2 |
| `sign` | 76.4 | 71.7 | 49.2 | 54.3 |
| `m025` | 77.0 | 70.2 | 36.6 | 55.0 |
| `m05` | 76.3 | 71.7 | 37.6 | 59.6 |
| `m1` | 74.0 | 74.7 | 46.4 | 60.6 |
| `m2` | 74.6 | 73.2 | 58.9 | 57.5 |
| `m4` | 75.5 | 72.2 | 55.9 | 50.6 |

Frozen floors are 0.0 everywhere under PPO and at most 0.4 under the reading learner. The smallest
lower bound against a floor is +25.3 points (the reading wild type at `m025`), so no gate is marginal.

**Neither the sign-only prior nor any multiplier leaves the pathway unlearnable, under either
learner.** Under PPO, which rewrites the weights it starts from, the prior barely moves the plateau:
74.0–77.2 on the wild type at every level. The reading learner, which can only read the fixed matrix,
is where a bad prior would show, and it shows **reduced** learning rather than none: the wild type
reaches 36.6 at 0.25 and 37.6 at 0.5, against 53.6 under the random prior.

### The selection: 1.0 on both learners

Every level passes on both learners, so the rule chooses the default. **B.1c's multiplier is 1.0 for
PPO and for the reading learner.** This is the point where the covered edges carry the random draw's
expected magnitude, so B.1c's measured-versus-random contrast compares structure, not size.

### The wiring gap: descriptive, and no movement

Wild type minus null (positive means the wild type is better), on each level's primary metric, with
80% intervals. **At eight seeds none of these is an estimate**, and none entered the selection.

| level | PPO, episodes to 30% | reading, `auc_success` |
|---|---|---|
| `random` | +270 [−104, +635] | −0.094 [−0.227, +0.036] |
| `sign` | +287 [+10, +574] | −0.026 [−0.204, +0.162] |
| `m025` | −77 [−380, +221] | −0.185 [−0.270, −0.089] |
| `m05` | +216 [−146, +591] | −0.199 [−0.289, −0.091] |
| `m1` | +171 [−105, +444] | −0.155 [−0.270, −0.041] |
| `m2` | −31 [−276, +215] | +0.010 [−0.098, +0.113] |
| `m4` | +55 [−289, +376] | +0.006 [−0.188, +0.202] |

**Registered sign movement: none.** The `random` level's intervals include zero on both learners, so
its side is taken from its mean, as registered: positive under PPO, negative under the reading
learner. No level's interval excludes zero on the opposite side. Under PPO the only interval that
excludes zero (`sign`) is on the same side as `random`. Under the reading learner the three that do
(`m025`, `m05`, `m1`) are also on the same side.

**The reference points reproduce their committed readings in direction.** PPO's `random` level under
the fan-in draw carries the wild type ahead on `auc_success` (+0.059 [+0.024, +0.098], reported beside
the primary), where A.1 established `auc_success` survival under the same draw. The reading learner's
`random` level has the null ahead, as A.2's reading centre and Logbooks 064 and 066 did.

**One pattern in the reading gap, described and not carried.** At multipliers from 0.25 to 1 the null
leads with intervals excluding zero. At 2 and 4 the gap sits at zero, because the wild type's plateau
rises to 58.9 and 55.9 while the null's does not. This is the same side as `random`, so under the
registered rule it is not a movement. At eight seeds it is a shape, not a finding. It is set down so
that B.1c's registration can decide, before its runs, whether a multiplier-sensitivity arm is worth
its cost.

### Frozen-substrate evidence

The reading learner never wrote its chemical matrix: drift against each level's own floor is **0.0**
on every seed at every level, with evidence complete, so the reading half is not void. Under PPO the
same check reads 0.54–1.46 relative drift, which is what it should read on a learner that rewrites
the matrix. It is run there as the check's own positive control.

## Corrections made in the open

- **A misreported wall clock.** Before this logbook, the PPO campaign was reported in conversation as
  taking 8.5 hours against a 3.1-hour estimate. It took **3.15 hours**. The progress reader measured a
  finished campaign's elapsed time from launch to the moment of reading, and it was read five hours
  after the campaign ended, while the reading half was running. The reader now stops a finished
  campaign's clock at its last completion marker, with a test. Both campaigns landed inside their
  registered estimates: 3.15 h against 3.1, and 6.02 h against 6.4.

- **Two defects fixed in the specification before any code, at spec review.**

  - The first draft let B.1c run a value arm at a multiplier where the wild type had failed its gate,
    contradicting the gate requirement.
  - It also chose the multiplier on the wild type alone, which could have picked a level where the
    null is broken or both arms saturate.

  Both were fixed before launch, and neither case arose.

- **None of the figures above changed after scoring.**

## What this establishes, and what it does not

**Establishes.** On hard350, under PPO with the fan-in draw and under the reading learner at A.2's
centre, the Creamer–Leifer–Pillow weights leave the pathway learnable at every scale from 0.25 to 4
times the random draw's magnitude, as signs alone or with their magnitudes. B.1c's multiplier is 1.0
on both learners, chosen by a rule that never saw the contrast. The fan-in pairing keeps every
neuron's wild-type multiset on the null, which is what B.1c's PPO arm needs to run under D15.

**Leaves unresolved.** Whether measured weights change the wiring contrast. Eight seeds per level
cannot say, and the pilot does not ask it. Whether the reading learner's lower plateau at small
multipliers is a real cost of the measured structure or seed noise; the gate is coarse at this size.
`measured_shuffled`, which B.1c runs as its control and this pilot did not.

**Does not establish.** That the measured weights help. Every comparison here is a level against its
own floor, never measured against random, and no such claim is made. Nor that Lee's failure mode is
absent in general: the pathway here is this model's, on this cell, at a settling depth of 4, and the
fitted table reaches no body motor neuron (Logbook 071's hop measurement is the relevant geometry).

## Registered consequences

- **B.1b**: *met*. Both learners have a swept multiplier, chosen by the registered rule: 1.0.

- **The Lee risk on B.1**: the pilot's branch did not fire on either learner, so B.1 does not close
  here. B.1c keeps the risk registered, since its own gate is the finer one.

- **B.1c carries**:

  - multiplier 1.0 on both learners;
  - PPO under `per_neuron_fanin`, and with it the standing condition that its depth and initial-noise
    settings come from A.2's surface, measured under `edge_order`, stated in the same sentence as any
    PPO claim;
  - readout width under PPO still unresolved (tracker B.1c, from A.2).

  No condition arises from the sign-movement rule.

- **The reading gap's multiplier shape** is offered to B.1c's registration as a candidate
  sensitivity arm. It is not carried as a condition, because the registered rule did not produce it.

## Artefacts

Under [supporting/072-measured-prior-pilot/](supporting/072-measured-prior-pilot/):

- `launch.md` — the registration.
- `pilot-ppo.json`, `pilot-reading.json` — every level's gates (with each seed's plateau and floor),
  both metrics' gaps with intervals, the metric choice, the drift evidence, the selection and the
  sign-movement read.
- `per-seed-ppo.csv`, `per-seed-reading.csv` — 56 rows each: per level and seed, both arms' plateau
  and floor and the gap on both metrics.

Drivers: `scripts/analysis/measured_prior_pilot.py`; configs generated by
`scripts/campaigns/generate_measured_prior_configs.py`. Raw campaign logs, 448 runs across two
campaign directories, are archived off-repo under A.0.
