# B.1b — the measured-prior pilot: registration and launch

**Registered 2026-09-23, before any seed ran.** Change: `add-measured-prior-pilot`.

Phase 8 **B.1b**, roadmap decisions **D16** and **D17**. This record fixes the pilot before either
campaign launches: the arms, the seeds, the gate, the metric rule, the selection rule and every
branch the outcome can take. **No seed in either band has been touched.**

## The questions

1. **Can the measured weights be learned at all?** Lee 2026 (bioRxiv 2026.09.06.749731) fitted c302's
   global conductances to the Randi atlas and found no functional sensory-to-command step. D17 names
   this pilot's sign-only-versus-magnitude arms as the place the same failure would show here.
2. **Which multiplier does B.1c run at, per learner?** `measured_weight_scale` maps coefficients of a
   2 Hz linear dynamical system on calcium signals onto this rate model, and there is no natural
   mapping. D16 forbids a contrast at a pin unswept for the learner it uses.

The pilot does **not** ask whether the measured weights make the wiring legible. That is B.1c's
question, and a pilot that answered it would have spent B.1c's seeds on its own operating point.

## The design

One cell, **hard350**, the cell A.2 swept both learners on.

| | PPO half | reading half |
|---|---|---|
| learner | PPO | `readout_only`, the three-factor rule with e-prop eligibility |
| weight draw | `per_neuron_fanin`, D15's shared initialisation, which B.1c's PPO arm runs under | `edge_order`, the draw A.2 swept it at |
| parents | A.1's committed `…_hard350_fanin` arms | A.2's reading centre |
| seeds | 113–120 | 121–128 |
| arms | 28 | 28 |
| runs | 224 | 224 |

**Seven levels per learner**, each on both wirings, learning and frozen:

| level | `weight_prior` | `measured_weight_scale` |
|---|---|---|
| `random` | `random` (the parents, unchanged) | — |
| `sign` | `measured_signs` | — |
| `m025` | `measured` | 0.25 |
| `m05` | `measured` | 0.5 |
| `m1` | `measured` | 1.0 |
| `m2` | `measured` | 2.0 |
| `m4` | `measured` | 4.0 |

At 1.0 the covered edges carry the random draw's expected magnitude, so a measured arm and a random
arm differ in structure, not in size. The grid spans 16× around that point on a log scale.

**Every level has its own frozen floor.** The prior and the multiplier change the substrate before
any learning, so a floor from another level would gate one substrate against another. A test asserts
that each floor is built from the same chemical weights as its level's learning arm.

**Why the PPO half moves draw and the reading half does not.** Under PPO a measured prior is an
initialisation, and D17 requires that arm to run under D15's shared initialisation; this change
defines the pairing so every neuron keeps its wild-type multiset of incoming values on the null. The
reading learner freezes the chemical matrix, so its prior is the substrate it reads rather than an
initialisation, and A.2 swept it under `edge_order`. **The PPO half is therefore at a draw A.2 did
not sweep.** Its committed reference is A.1's fan-in reading on hard350: `auc_success` survival
established, and the primary inconclusive at −32%. The pilot's `random` level re-reads that point on
fresh seeds, and every PPO figure states the draw.

**What is not in the pilot.** `measured_shuffled`, which has `measured`'s value distribution at every
multiplier and so inherits the chosen level; it is B.1c's control. Thermal. Readout width under PPO,
which tracker B.1c already carries as unresolved.

## The gate: two readings, each with one job

Both come from A.2's own `learning_gates`, called with each level as its own floor. The test is the
instrument's paired Wilcoxon with an 80% bootstrap interval on the per-seed plateau against the
floor.

- **The wild type learns**: the wild-type arm's lower bound against its floor is above zero. This asks
  whether the substrate can be learned at all, so it concerns the wild type alone.
- **The level passes**: both learning arms beat their floors **and** the level is not saturated (both
  plateaus under the instrument's 90% bar). This asks whether a wiring contrast could be read there.
  It asks whether each arm learns, never which learns more.

## The metric rule

A.2's, unchanged: censoring counted per level against the `random` level, with
`episodes_to_30pct_success` primary where the arms' crossing rates are within 0.10, and
`auc_success` primary otherwise. Both metrics are recorded at every level.

## The branches, in the order they are read

1. **Broken arm.** A learning arm that does not beat its floor is reported as a gate failure at that
   level, never as a wiring reading.
2. **Uninformative.** If the wild type does not learn at the `random` level, no failure of a measured
   level can be attributed to the measured weights, and the record says so for that learner.
3. **Pathway unlearnable (Lee).** The wild type learns at `random`, and at **no** `measured` level
   nor at `sign`. B.1 closes *unmet-with-reason* for that learner, with the pathway named, and B.1c
   does not run that learner.
4. **Magnitude is the obstacle.** `sign` passes and no `measured` level does. B.1c's `measured` and
   `measured_shuffled` arms close *unmet-with-reason* for that learner, with the fitted magnitudes
   named. Whether B.1c runs the sign-only prior as its measured arm is left to B.1c's own change; it
   would need a sign-shuffled control, which does not exist, and this pilot builds nothing for it.
5. **Selected.** Among the `measured` levels that pass:
   - choose **1.0** if it passes;
   - otherwise choose the passing level nearest 1.0 on the log scale, ties to the smaller multiplier.
6. **No level passes, and neither branch 3 nor 4 applies.** No multiplier is chosen, B.1c's value
   arms close *unmet-with-reason* for that learner, and the record names which gate failed at each
   level.

**The wiring gap never enters the selection.** `measured_prior_pilot.select` takes the gate records
and nothing else, and a test pins that signature. A multiplier picked because it shows the contrast
best would build B.1c's answer into its operating point, and B.1c's fresh seeds would then
re-measure the contrast at a point chosen for its size.

## The gap is descriptive, and one pattern in it is carried

The wiring gap is recorded at every level with its interval, on both metrics. At eight seeds it
estimates nothing and is reported as such.

**Sign movement** is recorded where a level's gap interval excludes zero on the side opposite to the
`random` level's. That side is the `random` level's interval where it excludes zero, and its mean
otherwise, and the record says which. A movement becomes a **registered condition B.1c carries**. It
is not a trigger for a crossing, and not a finding.

## The reading half owes drift evidence, or it is void

As in A.2: the reading learner leaves `w_chem` fixed, so each learning arm's chemical matrix is
compared against its own level's frozen floor on every scored seed, and any non-zero drift or
missing evidence voids that half. `--track-experiment` is therefore not optional. The PPO half runs
the same check, where it must read large.

## Launch

The runner takes one seed range per campaign, so each learner is its own campaign, run one after the
other.

```bash
arms() { uv run python -c "import sys;sys.path.insert(0,'scripts/analysis');import measured_prior_pilot as m;print(' '.join(m.stem_for('$1',a,l) for l in m.ALL_LEVELS for a in m.ARMS))"; }
for half in ppo reading; do
  seeds=$([ $half = ppo ] && echo 113-120 || echo 121-128)
  uv run python scripts/run_campaign.py \
    $(for s in $(arms $half); do printf -- "--config configs/scenarios/foraging/%s.yml " "$s"; done) \
    --seeds $seeds --runs 3000 --workers 16 \
    --output-dir campaigns/b1b-$half \
    -- --theme headless --track-experiment --no-detailed-export --no-file-log
done

# Score
uv run python scripts/analysis/measured_prior_pilot.py \
  --campaign campaigns/b1b-ppo --half ppo --out-dir build/b1b-ppo \
  --out build/b1b-ppo/pilot.json --csv build/b1b-ppo/per-seed.csv
```

The reading half is scored the same way from `campaigns/b1b-reading` with `--half reading`.

## Artefact retention (A.0)

Committed: the parsed per-seed CSV, the analysis JSON and this launch record, under this directory.
Archived off-repo: the raw campaign logs. A campaign directory is removed only after its CSV is
committed, and any field that cannot be compared because its source is gone is named **uncompared**
rather than counted as matching.

## Cost

At A.2's measured per-run times (PPO: 14.9 min learning, 10.9 frozen; reading: 20.0 learning, 33.4
frozen) and its 15.5× parallel efficiency at 16 workers: **about 3.1 hours for PPO and 6.4 for the
reading half, 9.5 in all.** A multiplier that makes an arm fail runs every episode to its full
budget, so the reading half's frozen-arm cost is the likelier bound for failing levels. Detailed
export is off.
