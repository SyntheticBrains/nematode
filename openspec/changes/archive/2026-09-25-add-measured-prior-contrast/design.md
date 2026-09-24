## Context

This change is B.1c, D17's 2×3. It builds on:

- **B.1a**, the vendored table, `weight_prior`, and the coverage report;
- **B.1b** (Logbook 072), which established that the pathway is learnable at every scale, chose
  multiplier 1.0 on both learners, and defined the fan-in pairing;
- **A.2** (Logbook 071), the swept operating point for each learner;
- **A.1** (Logbook 070), D15's shared initialisation under PPO.

Decisions already taken:

- readout width under PPO stays pooled, stated beside any PPO claim;
- no multiplier-2 sensitivity arm;
- 32 seeds for PPO and 48 for the reading learner, sized from the power analysis below;
- A.3 may run in a separate worktree while this campaign runs;
- A.5 waits for this readout.

## Goals / Non-Goals

**Goals**

- Read whether measured weights move the wiring gap, and whether their **placement** is what moves
  it, on each learner.
- Register the sensitivity, the minimum, the verdict map and the gates before launch.

**Non-Goals**

- Thermal (A.1 showed it is underpowered against its own effect).
- Other multipliers.
- Readout width under PPO.
- The sign-only prior as a B.1c arm. B.1b did not reach its magnitude branch, and a sign-only arm
  would need a sign-shuffled control that does not exist.

## Decisions

### Decision A: Arms, seeds, and which configs are new

| learner | draw, readout | levels | arms | seeds | runs |
|---|---|---|---|---|---|
| PPO | `per_neuron_fanin`, pooled | random, measured, shuffled | 12 | 225–256 | 384 |
| reading | `edge_order`, A.2's centre | random, measured, shuffled | 12 | 257–304 | 576 |

**Arms.** Each level has four arms: the wild type and the null, each learning and frozen. Every level
is gated against its own floor, because the prior changes the substrate before any learning.

**Configs.**

- The random level is B.1b's parents: A.1's `_fanin` arms for PPO and A.2's reading centre.
- The measured level is B.1b's `_measured_m1` configs.
- All of those are committed and reused unchanged.
- Only the eight `_measured_shuffled` configs are new. Each is its parent plus
  `weight_prior: measured_shuffled` at the default multiplier.

**Seeds.** B.1b fixed 225 as B.1c's first seed. The two learners take disjoint bands, as every Phase
8 campaign has. A test checks both bands against every band spent before.

**Cost.** At B.1b's measured per-run times and 15.9× parallelism, PPO takes about 5.4 h and the
reading learner about 15.5 h.

### Decision B: Two interactions per learner, both paired by seed

The gap is wild type minus null, oriented so that positive means the wild type is better. That is
the instrument's own orientation.

1. **Measured × wiring**: `gap(measured) − gap(random)`.
2. **Placement × wiring**: `gap(measured) − gap(shuffled)`.

Both are computed with A.2's `interaction`, which pairs each seed's gap at one level with the same
seed's gap at the other.

**Why the second contrast carries the claim.** A measured prior changes two things at once: the
distribution of values on covered edges, and which synapse holds which value. The shuffled arm keeps
the distribution and destroys the assignment, so only the second contrast isolates what "the
animal's weights" means.

**The metric: `auc_success` is the primary on both learners, a departure registered with its
reason.** A.2's censoring rule would give PPO `episodes_to_30pct_success`, because B.1b's crossing
rates are all 1.0. That metric cannot carry this panel. On PPO's own draw its detectable effect is
1.14× the committed wiring effect (Decision C), so the panel could not see even a sign move on it.
`auc_success` detects 0.75× of the same effect. This is the pattern A.1 recorded under the fan-in
draw, where `auc_success` established survival and episodes stayed inconclusive.

The choice is made **before launch, from committed data**, and it is **the same metric for every
interaction of both learners**, so the verdict map never combines states read on different scales.
`episodes_to_30pct_success` is reported beside every interaction, with the censoring rule's own
choice recorded next to it, as the metric requirement asks of a departure.

**The family.** The four primary-metric interactions (2 contrasts × 2 learners) are BH-FDR corrected
together, with the two-sided folded p (`two_sided`). The reported-beside metric's four are corrected
as their own family and never decide a verdict.

### Decision C: Sensitivity, from frozen committed data

The interaction requirement forbids sizing a panel from its own results. The spread instead comes
from **B.1b's committed per-seed CSVs**: the sd of each seed's `m1 − random` interaction. The
minimum detectable effect is `2.487 × sd / √n`, as A.2 used.

| learner | metric | sd (8 seeds) | n | MDE | reference effect | MDE ÷ reference |
|---|---|---|---|---|---|---|
| PPO | `auc_success` (primary) | 0.092 | 32 | 0.041 | +0.055 (A.1, hard350, fan-in) | **0.75** |
| PPO | `episodes_to_30pct_success` (beside) | 1013.6 | 32 | 445.6 | +392.4 (A.1, hard350, fan-in) | 1.14 |
| reading | `auc_success` (primary) | 0.444 | 48 | 0.159 | −0.210 (A.2 reading centre) | **0.76** |

**PPO's reference is the effect under its own draw.** A.1 measured hard350 at +580.4 episodes and
+0.061 `auc_success` under `edge_order`, and its committed interaction for `per_neuron_fanin` is
−188.0 and −0.006. The fan-in effect is therefore +392.4 and +0.055, and that is the effect this
panel runs on. A first draft used the `edge_order` figures and overstated PPO's sensitivity.

**Stated now, not after the fact:**

- The spreads come from 8 seeds.
- The placement contrast's spread is assumed equal to the measured contrast's, because the pilot had
  no shuffled arm.
- The registered minimum (2/3) sits **below** the MDE on both primary rows. An effect at the
  minimum is therefore not reliably detected, and a reading between the two is reported as
  **unresolved**, never as "no move".

### Decision D: The registered minimum, in both directions

The reference effect is each learner's **committed** wiring effect on hard350, on the draw it runs:
A.1's under `per_neuron_fanin` for PPO (+0.055 `auc_success`), and A.2's reading centre under
`edge_order` for the reading learner (−0.210). It is not the in-campaign random-level gap, which is
itself a reading of this panel and would let the result size its own bar.

The minimum is **2/3 of |reference|** on the primary:

- PPO: 0.037 `auc_success`;
- reading: 0.140 `auc_success`.

Each interaction is classified from its mean, its 80% bootstrap interval, and its BH-FDR q, with
significance at q < 0.05:

| state | condition |
|---|---|
| `move_wt` | q < 0.05, interval above zero, mean ≥ minimum: moves the gap toward the wild type |
| `move_null` | q < 0.05, interval below zero, mean ≤ −minimum: moves it toward the null |
| `below` | q < 0.05, interval excludes zero, absolute mean < minimum |
| `no_move` | interval inside (−minimum, +minimum) and including zero |
| `unresolved` | anything else |

`unresolved` covers three cases, named so the classifier and its tests cover each:

- the interval spans the minimum;
- q ≥ 0.05 while the interval excludes zero;
- q < 0.05 while the interval spans zero, where the Wilcoxon and the bootstrap disagree.

### Decision E: The verdict map, per learner

Rows are the measured × wiring state. Columns are the placement × wiring state.

| measured × wiring | placement × wiring | verdict |
|---|---|---|
| `move_wt` | `move_wt` | **legible**: the animal's weights make the wiring legible |
| `move_wt` | anything else | **value_distribution_wt**: a distribution effect toward the wild type, not placement; never reported as legibility |
| `move_null` | `move_null` | **hides**: the animal's weights hide the wiring, the symmetric finding |
| `move_null` | anything else | **value_distribution_null**: a distribution effect toward the null, not placement; never reported as hiding |
| `no_move` | `no_move` | **null**: extends Logbook 034's degree-statistics verdict to measured weights |
| `no_move` | `move_wt` or `move_null` | **placement_only**: the shuffle moves the gap and the fitted placement does not. A finding about the shuffle, carried to the synthesis, not legibility |
| `no_move` | `below` or `unresolved` | **null_placement_unresolved** |
| `below` | any | **below_minimum**: licenses nothing on its own |
| `unresolved` | any | **unresolved** at this panel's sensitivity, with the MDE beside it |

### Decision F: Gates, read before any interaction

- **Floors.** Every learning arm is gated against its own level's floor with A.2's `learning_gates`.
  - An interaction involving a level where either wiring fails its floor is **unreadable**, never a
    verdict.
  - If the wild type fails at the measured level, B.1 closes for that learner *unmet-with-reason*
    (the Lee branch at panel scale), with the pathway named.
- **Saturation.** A level where both arms saturate makes its interactions unreadable. That is two arms
  tied at the ceiling.
- **Drift.** The reading learner's `w_chem` drift must be 0 against its floor on every seed, or that
  half is void. On PPO, where the matrix is written, the same check runs as its positive control.

### Decision G: Standing conditions, in the same sentence as any claim

- **PPO:** the fan-in draw with the pooled readout. Its depth and initial-noise settings come from
  A.2's surface measured under `edge_order`, and readout width is unresolved there.
- **Coverage, at both scopes, as D17 requires:**
  - 1,049 of 3,709 chemical edges carry a fitted value at full scope, which is 77.0% of the 1,363
    coverable at head scope;
  - none reaches a body motor neuron, so the command-to-motor and motor layers stay on the draw.
- **The source** is a preprint fitted on another connectome (White 1986 + Witvliet 2020), so the
  result never stands alone.

### Decision H: Reuse, not copies

`scripts/analysis/measured_prior_contrast.py` holds this panel's stems, levels, seeds, interactions,
classification and verdict map.

- **Imported from B.1b:** `PARENTS`, `stem_for`, `level_keys` and the manifest format.
- **Imported from A.2:** `score_level`, `learning_gates`, `interaction`, `wiring_gap`,
  `censoring_rates`, `choose_metric`, `apply_family_correction`, `two_sided`, `substrate_drift`.

**The vocabulary is shared; the panels are not.** `measured_prior_pilot` separates the two:

- A table of every level any panel may use, which `level_keys` and `stem_for` read. It gains
  `shuffled` (`weight_prior: measured_shuffled` at the default multiplier), so both panels spell
  every stem one way.
- The pilot's own panel: `LEVELS`, `ALL_LEVELS`, its 56-stem map, and its completeness check. This
  is unchanged. Adding `shuffled` there would make the committed B.1b campaign read as incomplete if
  it were ever re-scored.

`build_manifest` and `require_complete` gain the one argument needed to take another panel's levels
and stem map, defaulting to the pilot's own. B.1b's tests pin the default through the change.

The contrast module declares its own three levels (`random`, `m1`, `shuffled`) and builds its own
stem map from the shared vocabulary.

**The generator.** `generate_measured_prior_configs.py` is extended to write the contrast's new
arms from the contrast's panel definition, leaving every existing file alone.

**One property of the shuffled arm, stated.** The permutation is keyed on the run seed, so each
seed gets a different shuffle. The control therefore averages over permutations rather than resting
on one that happened to be lucky or unlucky. The launch record says so.

## Risks / Trade-offs

- **The reading half is still coarse.** Its MDE is 0.76 of its reference even at 48 seeds, so
  unresolved readings there are a likely outcome and are registered as one.
- **PPO's primary departs from the censoring rule.** Registered before launch with its reason
  (Decision B). The censoring rule's own choice is reported beside it on every interaction.
- **The reference effects come from other campaigns.** That is deliberate (Decision D). It does mean
  a learner whose in-campaign random gap differs greatly from its reference is read against a
  minimum sized for a different effect, and the record reports both.
- **A 21-hour campaign on one working tree.** No branch switches until it completes. A.3, if run
  meanwhile, runs in its own git worktree.
