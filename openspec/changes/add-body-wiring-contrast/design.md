## Overview

The wild type against three nulls, through the body, under two learners. A pilot sets the minimum and
the panel's size; the panel reads the contrasts.

## Decisions

### Decision A: The cell

hard350's food layout at **500 steps**, the length C.1d's fallback chose, through the frozen kinematic
body (steering gain 2, one-step reversals, wave floor 0.25), on Emmons 2024 with reversal on, settling
dynamics at depth 4, 3,000 episodes, **`entropy_coef` 0.004** for every learning arm. It is a new
reference frame: nothing here is a delta against 029, block V or any point-worm result.

### Decision B: The wirings

| wiring | role | differs from the wild type in |
|---|---|---|
| wild type | | |
| chemical-only null (`rewired_chemical_only`) | **primary** (D21) | chemical placement, autapses held |
| boundary-preserving null (`rewired_boundary_held`) | **second registered reading**, the only one that carries an interior-wiring claim | chemical placement in the interior only |
| current null (`rewired_degree_preserving`) | beside | chemical and gap placement |

**The boundary under the body.** The boundary null holds every chemical edge out of a sensory neuron
and into a motor-side cell. Under the body drive the motor side is every cell the drive reads: the 162
cells with junctions onto the body wall muscles. They include all 39 neurons of the four motor classes
the direction channel reads.

**What that leaves to rewire.** The old boundary, the 39 motor-class neurons, held 544 of the 3,709
chemical edges (15%). The body's boundary holds **2,134 (58%)**, so the boundary null rewires only the
**1,575 interior edges**. That follows D21's amendment, and it narrows the second reading: an interior
claim covers 42% of the chemical wiring, and a `no_move` there is weaker evidence than it was under the
point worm's boundary. Both are stated in the panel's registration.

### Decision C: The learners

- **PPO** writes the chemical weights, the sensor gains, the 25 body-drive gains and the noise.
- **Frozen-wiring PPO** (`freeze_wiring: true`) keeps the chemical weights at their initial draw and
  trains the rest. It asks whether the wiring helps when it is only read. The e-prop reading learner
  cannot run through the body: its rule reads and writes a two-number readout the body replaces.

`freeze_wiring` is refused under any learner but PPO, and alongside `freeze_updates`, which already
trains nothing. Each wiring has one **frozen floor** (`freeze_updates`), shared by both learners. **MLP-PPO** runs beside
at the same cell and entropy, as the non-connectome reference.

### Decision D: The metrics

**Primary**: `auc_success`. **Beside**: episodes to 30% success. The probes showed bimodal seeds (the
chemical-only null's split between about 20% and 80%), so the frequency of competent seeds (plateau ≥
30%) is read beside the level, paired by seed (protocol principle 10), by an **exact McNemar test** on
the seeds where the two wirings disagree. It is reported as description, never as a verdict.

### Decision E: The pilot, which sets the minimum

**Before it, a frozen-wiring probe.** No run had yet trained only the gains over a fixed wiring, so the
pilot's frozen-wiring half had no evidence that it learns. The wild type and the chemical-only null
under `freeze_wiring`, seeds 1601–1604, 8 runs, read before the pilot registers. If neither learns, the
frozen-wiring learner is dropped before the pilot, with its reason recorded.

**Read 2026-10-09: the frozen-wiring learner leaves.** The wild type plateaued at 0% on all four seeds
and the chemical-only null at 0, 0, 25.3 and 0, against 0% floors; foods per episode rose only from
2–4 to 4–5 on most seeds. The gate preflight on the probe runs, mapped to the registered stems, reads
the level `fails_floor` (wild type 0.0% against its floor), and PPO `readable` (66.7% and 52.3%). By the
rule below, the learner leaves: through the body, the drive gains and sensor gains over a fixed wiring
do not carry foraging, within 3,000 episodes at these settings. **The reading-learner half of C.1e
closes unreachable-with-reason**, and the
pilot runs PPO only: 2 wirings × (learning, frozen) × 16 = **64 runs**.

The wild type and the chemical-only null, learning under both learners, with both frozen floors, on
**seeds 1701–1716**: 2 learners × 2 wirings × 16 + 2 frozen × 16 = **96 runs**.

**What it fixes, per learner, before the panel registers:**

- **The reference effect**: the pilot's paired wild-type-minus-null `auc_success` mean.
- **The minimum**: 2/3 of |reference|, floored at **0.0367** for both learners. The floor is a
  judgement, not a measurement on this cell: it is B.1c's committed PPO minimum, 2/3 of the point
  worm's wiring effect, carried over only to keep a near-zero pilot from making a trivial difference
  count as a move.
- **The spread** and so the panel's seed count: the smallest n at which the MDE, `2.487 × sd / √n`, is
  at most the minimum, **never fewer than the pilot's 16**, capped at 64. A paired rank test on a
  handful of seeds fires on the consistency of the sign rather than the size, and a bimodal pilot's
  spread is unstable: resampling the probes moved n between 5 and 14.
- **The floor's size on this cell**: the 0.0367 floor as a share of the wild type's mean
  `auc_success`, reported beside the minimum, never used to re-read it.
- **The gates**, read by the gate preflight on the pilot runs: each learning arm beats its floor, and
  no level has both learning arms at or above 90%. **A learner whose learning arms do not beat their
  floors leaves the panel**, recorded with its reason; a saturated learner likewise.

The pilot is registered and committed as Logbook 084 before the panel registers.

### Decision F: The panel *(revised 2026-10-09, after the pilot, before the panel registered)*

**What the pilot fixed** (Logbook 084): the minimum is its floor, **0.0367**, since 2/3 of the pilot's
+0.0229 falls below it; the per-seed spread is 0.161, so the panel takes the cap, **64 seeds**, where its
MDE is 0.050. An effect between the minimum and 0.050 will read `unresolved`.

**The panel is cut to what can be read.** It runs the **primary contrast alone**: the wild type against
the chemical-only null under PPO, learning and frozen, with MLP-PPO beside, on **seeds 1801–1864**, 320
runs, about 20 hours. Its one registered reading, wild type minus null on `auc_success`, is classified
at 0.0367 by `mc.classify` (two-sided Wilcoxon q, 80% bootstrap interval). The verdict map:

| state | verdict |
|---|---|
| `move_wt` | **wild type ahead**: the boundary stage runs |
| `move_null` | null ahead |
| `below` | a difference below the minimum |
| `no_move` | no wiring effect at the minimum |
| `unresolved` | unresolved at this sensitivity, with the achieved interval as the bound |

Gates first: if either learning arm fails its floor, or both reach 90%, the panel is unreadable.

**The boundary stage, gated.** The boundary-preserving null runs only if the primary reads `move_wt`,
on the same seeds, PPO and frozen, against the panel's wild-type runs, at the same minimum. An
interior-wiring claim needs a wiring effect to locate; without one its 128 runs would answer a question
that did not arise. The four-wiring panel this replaces would have cost about 29 hours, and the current
null, read only beside, is not run.

## Risks

- **The wild type may trail.** Probe 2, before the entropy fix, had two null seeds learn and no wild
  type; probe 3 had the wild type ahead on average. Four seeds settle nothing; the pilot is sized to.
- **Cost.** A connectome learning run took about 51 minutes at 12 workers; the pilot's 96 runs are about
  six rounds, about 5.5 hours at 16 workers, measured again in the pilot.
- **Saturation.** No probe seed reached 90% (best 83.5%); the pilot's gate reading confirms it per level.
