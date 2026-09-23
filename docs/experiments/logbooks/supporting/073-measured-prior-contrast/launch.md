# B.1c — the wiring × weight-prior 2×3: registration and launch

**Registered 2026-09-24, before any seed ran.** Change: `add-measured-prior-contrast`.

Phase 8 **B.1c**, roadmap decision **D17**, and the last required item before shipment 8a. This
record fixes the design, the metric, the sensitivity, the minimum, the state classification, the
verdict map and the gates before either campaign launches. **No seed in either band has been
touched.**

## The question

Do the animal's own synaptic weights make its wiring legible where random ones did not?

Every connectome brain in this project has been constrained in topology and randomly initialised in
weight. Block V's wiring effect was read on random weights, and so was every null before it,
including Logbook 034's degree-statistics verdict. The Creamer–Leifer–Pillow fitted weights
(bioRxiv 2024.09.22.614271, a preprint) are the first measured weights on this substrate.
[Logbook 072](../../072-measured-prior-pilot.md) showed they leave the pathway learnable, and
selected the magnitude-matched multiplier, 1.0, on both learners.

## The design

Wiring {wild type, rewired null} × prior {random, measured, measured-shuffled}, each arm learning and
frozen, on **hard350**.

| learner | draw, readout | seeds | arms | runs |
|---|---|---|---|---|
| PPO | `per_neuron_fanin`, pooled | 225–256 (32) | 12 | 384 |
| reading (`readout_only`) | `edge_order`, A.2's centre | 257–304 (48) | 12 | 576 |

| level | `weight_prior` | configs |
|---|---|---|
| `random` | `random` | B.1b's parents, unchanged |
| `m1` | `measured` at 1.0 | B.1b's `_measured_m1` arms, unchanged |
| `shuffled` | `measured_shuffled` at 1.0 | the 8 new `_measured_shuffled` arms |

**Every level has its own frozen floor**, because the prior changes the substrate before any
learning.

**The shuffled arm's permutation is keyed on the run seed, so each seed gets a different shuffle.**
The control averages over permutations rather than resting on one that happened to be lucky or
unlucky.

**Why the learners are on different draws.**

- Under PPO a measured prior is an initialisation, so D17 requires D15's shared initialisation. That
  is the fan-in draw, under which B.1b made every neuron keep its wild-type multiset of incoming
  values on the null.
- The reading learner freezes the chemical matrix, so its prior is the substrate it reads. It stays
  where A.2 swept it.

## Two interactions per learner

The wiring gap is wild type minus null, positive when the wild type is better. Both interactions are
paired by seed with A.2's `interaction`:

1. **measured × wiring**: `gap(m1) − gap(random)`. Does the measured prior move the gap?
2. **placement × wiring**: `gap(m1) − gap(shuffled)`. Does it matter which synapse carries which
   fitted value?

The second is what can make a positive mean "the animal's weights". A measured prior changes the
distribution of values on the covered edges and their assignment at once. The shuffled arm keeps the
first and destroys the second.

## The metric: `auc_success` on both learners, a registered departure

A.2's censoring rule would give PPO `episodes_to_30pct_success`, because B.1b's crossing rates are
all 1.0. That metric cannot carry this panel. On PPO's own draw its detectable effect is **1.14×**
the committed wiring effect (below), so the panel could not see even a sign move on it.
`auc_success` detects 0.75×. This repeats A.1, where under the fan-in draw `auc_success` established
survival and episodes stayed inconclusive.

The departure is made here, before launch, from committed data, and it is **one metric for every
interaction of both learners**. `episodes_to_30pct_success` is reported beside every interaction,
with the censoring rule's own choice recorded next to it.

## Sensitivity, from frozen committed data

The spread is the sd of each seed's `m1 − random` interaction in
[B.1b's committed per-seed CSVs](../072-measured-prior-pilot/). The minimum detectable effect is
`2.487 × sd / √n`.

| learner | metric | sd (8 seeds) | n | MDE | reference effect | MDE ÷ reference |
|---|---|---|---|---|---|---|
| PPO | `auc_success` (primary) | 0.092 | 32 | 0.041 | +0.0550 | **0.75** |
| PPO | `episodes_to_30pct_success` (beside) | 1013.6 | 32 | 445.6 | +392.4 | 1.14 |
| reading | `auc_success` (primary) | 0.444 | 48 | 0.159 | −0.2105 | **0.76** |

**Where the reference effects come from.**

- **PPO:** A.1's hard350 wiring gap under `per_neuron_fanin` (Logbook 070's JSON, `shared_gap_mean`).
  This is the `edge_order` effect of +0.061 and +580.4 plus A.1's fan-in interaction of −0.006 and
  −188.0. The first draft of this design used the `edge_order` figures and overstated PPO's
  sensitivity; the spec review caught it.
- **Reading:** A.2's reading centre (Logbook 071's reading JSON).

**Stated now:**

- The spreads come from 8 seeds.
- The placement contrast's spread is assumed equal to the measured contrast's, because the pilot had
  no shuffled arm.
- The minimum below sits **below** both primary MDEs, so an effect at the minimum is not reliably
  detected, and a reading between the two is reported as unresolved.

## The registered minimum, in both directions

The minimum is **2/3 of |reference|**: **0.0367** `auc_success` for PPO and **0.1403** for the reading
learner. It is never a fraction of the in-campaign random gap, which would let the result size its
own bar.

Each interaction is classified from its mean, its 80% bootstrap interval, and its q. The q comes from
the two-sided folded Wilcoxon, BH-FDR corrected across the **four primary interactions** (2
contrasts × 2 learners). The four episodes interactions form a separate family and never decide a
verdict.

| state | condition |
|---|---|
| `move_wt` | q < 0.05, interval above zero, mean ≥ minimum |
| `move_null` | q < 0.05, interval below zero, mean ≤ −minimum |
| `below` | q < 0.05, interval excludes zero, absolute mean < minimum |
| `no_move` | interval inside (−minimum, +minimum) and including zero |
| `unresolved` | anything else |

`unresolved` covers three cases:

- the interval spans the minimum;
- q ≥ 0.05 while the interval excludes zero;
- q < 0.05 while the interval spans zero, where the Wilcoxon and the bootstrap disagree.

## The verdict map, per learner

| measured × wiring | placement × wiring | verdict |
|---|---|---|
| `move_wt` | `move_wt` | **legible**: the animal's weights make the wiring legible |
| `move_wt` | anything else | **value_distribution_wt**: a distribution effect, never reported as legibility |
| `move_null` | `move_null` | **hides**: the animal's weights hide the wiring |
| `move_null` | anything else | **value_distribution_null**: a distribution effect, never reported as hiding |
| `no_move` | `no_move` | **null**: extends Logbook 034's degree-statistics verdict to measured weights |
| `no_move` | `move_wt` or `move_null` | **placement_only**: the shuffle moved the gap and the fitted placement did not; a finding about the control |
| `no_move` | `below` or `unresolved` | **null_placement_unresolved** |
| `below` | any | **below_minimum**: licenses nothing on its own |
| `unresolved` | any | **unresolved** at this panel's sensitivity, with the MDE beside it |

## Gates, read before any interaction

- **Floors.** Every learning arm is gated against its own level's floor, with A.2's `learning_gates`
  (the paired test against the floor, and the 90% saturation bar).
- **Lee.** If the wild type fails its floor at `m1`, the learner's verdict is `lee_unlearnable`. B.1
  then closes *unmet-with-reason* for that learner, with the pathway named.
- **Unreadable.** A level where either wiring fails its floor, or where both arms saturate, makes
  every interaction it enters unreadable. The learner's verdict is then `unreadable`, whatever the
  gaps say.
- **Drift.** The reading learner's `w_chem` must not move against its floor on any scored seed, or
  that half is void. On PPO, which writes the matrix, the same check must read large, as its own
  positive control.

## Standing conditions, carried in the same sentence as any claim

- **PPO:** the fan-in draw with the pooled readout. Its depth and initial-noise settings come from
  A.2's surface measured under `edge_order`, and readout width is unresolved there (tracker B.1c,
  from A.2).
- **Coverage, at both scopes, as D17 requires:**
  - 1,049 of Cook 2019's 3,709 chemical edges carry a fitted value at full scope, which is 77.0% of
    the 1,363 coverable at head scope;
  - none reaches a body motor neuron, so the command-to-motor and motor layers stay on the draw.
  - The analysis JSON records these counts from `coverage`.
- **The source** is a preprint fitted on another connectome (White 1986 + Witvliet 2020), so the
  result never stands alone.

## Launch

The runner takes one seed range per campaign, so each learner is its own campaign, run one after the
other. The family spans both, so they are scored together.

```bash
arms() { uv run python -c "import sys;sys.path.insert(0,'scripts/analysis');import measured_prior_contrast as m, measured_prior_pilot as p;print(' '.join(p.stem_for('$1',a,l) for l in m.LEVELS for a in m.ARMS))"; }
for half in ppo reading; do
  if [ $half = ppo ]; then seeds=225-256; else seeds=257-304; fi
  cfgs=(); for s in $(arms $half); do cfgs+=(--config configs/scenarios/foraging/$s.yml); done
  uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds $seeds --runs 3000 --workers 16 \
    --output-dir campaigns/b1c-$half \
    -- --theme headless --track-experiment --no-detailed-export --no-file-log
done

# Score, both learners together
uv run python scripts/analysis/measured_prior_contrast.py \
  --ppo-campaign campaigns/b1c-ppo --reading-campaign campaigns/b1c-reading \
  --out-dir build/b1c --out build/b1c/contrast.json --csv build/b1c/per-seed.csv
```

## Artefact retention (A.0)

- **Committed:** the parsed per-seed CSV, the analysis JSON and this launch record, under this
  directory.
- **Archived off-repo:** the raw campaign logs.
- **Deletion:** a campaign directory is removed only after its CSV is committed. Any field that
  cannot be compared because its source is gone is named **uncompared**.

## Cost

B.1b measured the per-run times: PPO at 16.4 min learning and 10.5 frozen, the reading learner at
19.4 and 31.9. Both campaigns ran at 15.9× on 16 workers. That puts PPO at about **5.4 hours** and
the reading learner at about **15.5**, **about 21 in all**.
