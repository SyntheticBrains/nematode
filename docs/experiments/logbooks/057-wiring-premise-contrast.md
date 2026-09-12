# 057: The Wiring Premise — the Wild-Type Connectome Learns Faster on a Behaviour It Is Wired For (7a-ii V.1 / Phase 7)

**Status**: completed — **`specific_wiring_efficiency`**, the first positive wiring result in the
project, replicated on an independent panel and narrower than it first looked. Every wiring contrast
in this project had run on one cell: the integrated C3 demand of food chemotaxis **plus predator
evasion plus thermotaxis** at 2400 steps. [Logbook 034](034-connectome-structure-controls.md)
recorded the limitation in its own words — "**Single task**… not a universal statement" — and both
architecture rankings place the connectome competitive on foraging and behind on predator evasion,
the least worm-like of the three demands. This ran the same contrast under PPO on cells matched to
behaviours the animal performs. On a **foraging cell under lethal thermal pressure**, the wild-type
wiring reaches competence **~35% sooner** than its degree-preserving rewired null — 396 episodes
against 614, pooled over **64 paired seeds**, all four efficiency metrics significant (q = 0.000 to
0.001, 43–48 of 64 seeds). The controls that make it mean something both hold: **both wirings learn**
(gates 64/64, q = 0.000) and **the untrained prior is indistinguishable** (−0.17, q = 0.735), so the
advantage is created by learning rather than inherited from the graph. Two things bound the claim.
It is **speed, not performance** — the endpoint is saturated and the null is nominally *ahead* there
(−0.47 foods of 20, q = 0.004), consistent with 034, 043 and this panel's own food-only cell. And it
is **PPO, not a local rule**, so 7b's gate as written is still unmet while its premise is now
supported. The estimate shrank across panels — **+46.4% → +32.6% → +31.8%** — a textbook winner's
curse in which the first panel drew high and the two independent panels agree.

**Branch**: `feat/wiring-premise-contrast`.

**Date**: 2026-09-12.

**OpenSpec change**: `add-wiring-premise-contrast` (extends `architecture-comparison-protocol`: a
wiring contrast is run on a cell matched to the behaviour under claim, and gated on that cell showing
learning).

## Objective

Test the premise [Logbook 056](056-l4-ladder-reread.md) found undemonstrated — that learning finds a
wild-type advantage — where it has the best chance of holding: on a single behaviour the animal
performs, under an optimiser known to learn on this substrate, on a cell whose difficulty this
repository has calibrated.

This also gates shipment 7b, whose MUST is a comparative cross-connectome sweep on klinotaxis and
thermotaxis. If the contrast were null on both *within one species under a working optimiser*, that
sweep would have no within-species signal to find.

## Background

Eight logbooks had asked whether the wild-type wiring is load-bearing, all on the C3 cell, and
Logbook 056 classified five of their registered contrasts as uninformative about both the wiring and
the instrument because no tested optimiser had shown the effect they sought. Two PPO regimes had
looked and found nothing: 034's registered contrast on C3 (−3.28, q = 0.770, the null nominally
higher) and 043's low-noise PPO arms (the null ahead by 12.6 on 0 of 8).

What none of them varied was the **cell**.

## Method

### The cells, and what they demand

Both are **klinotaxis-sensed foraging cells** — head-sweep chemosensing, `sensing_mode: klinotaxis`
— and they differ in whether survival pressure is present:

| cell | `max_steps` | demand | thermotaxis projection |
|---|---|---|---|
| food-only (C1) | 800 | collect the target | — |
| food-under-thermal-pressure | 500 | collect the target while avoiding lethal zones | onto AFDL/AFDR |

The scored quantities are **food metrics on both cells**. Nothing here scores temperature avoidance
directly, so the result is about **foraging under thermal pressure**, not about thermotaxis.

### Arms and the family

Four connectome arms per cell on paired seeds, 3000 episodes: `wt_ppo`, `rn_ppo` and a frozen-weights
floor for each wiring. Within a wiring pair `wiring` is the only key that differs; within a learning
pair `freeze_updates` is; `rewire_seed` is unset, so each seed's rewiring derives from its run seed
and the arms pair. Eight registered tests, both cells corrected together under BH-FDR — the contrast,
two learning gates and the untrained prior per cell — with 80% bootstrap CIs, through
`scripts/analysis/wiring_premise.py`, which calls the same metric and statistics layers as the 034
control.

### Two things the pilot changed, both before a registered seed was spent

The pilot ran on disjoint seeds 101–104 and **fired the registered saturation clause**: on the
food-only cell both wirings cleared **100.00% on every seed**, a contrast of exactly zero, with
frozen random weights already at 57.60%. The registered remedy — `target_foods_to_collect` 10 → 20,
recipe untouched — was applied once as registered and **did not unsaturate either cell**.

Re-reading the same runs on the committed efficiency harness
(`connectome_structure_efficiency.py`, 034's own follow-up) separated them: the food-only cell is
learned by both wirings inside forty episodes of three thousand, while the thermal cell showed
301 episodes against 580. **The primary therefore moved to the efficiency axis on the thermal cell**,
under a dated amendment, with the learning gates left on the peak axis and a **minimum effect of 20%
off time-to-competence** registered beside significance.

Changing a scored axis after seeing data is the thing a registration exists to prevent. It is
defensible here on four counts, and the record states them so a reader can weigh them: the pilot ran
on **disjoint seeds** and its registered job was to fix the protocol; the new axis is **not invented
for this result** but 034's committed follow-up applied unchanged; the **minimum effect was fixed in
advance**; and the same axis on the same seeds reads **flat and nominally negative** on the food-only
cell, so it is not an instrument that manufactures positives.

## Results

### The primary: three panels

| | panel 1 (1–16) | panel 2 (17–32) | panel 3 (33–64) | pooled (64) |
|---|---|---|---|---|
| `auc_success` | +0.14, q = 0.001 | +0.10, q = 0.115 | +0.07, q = 0.012 | **+0.10, q = 0.000** |
| `auc_foods` | +1.21, q = 0.009 | +0.86, q = 0.274 | +0.07, q = 0.027 | **+0.55, q = 0.001** |
| `episodes_to_30pct_success` | +264.8, q = 0.009 | +187.8, q = 0.302 | +207.9, q = 0.012 | **+217.1, q = 0.001** |
| `episodes_to_90pct_foods_plateau` | +308.4, q = 0.014 | +155.8, q = 0.334 | +294.8, q = 0.012 | **+263.4, q = 0.001** |
| time-to-competence gain | +46.4% | +32.6% | +31.8% | **+35.4%** |
| wild-better seeds | 12–14/16 | 7–10/16 | 21–24/32 | 43–48/64 |

Pooled: the wild type reaches a 30% full-clear rate in **396 episodes** against the null's **613**.

### The controls

| test | pooled (64) | reading |
|---|---|---|
| V6 `wt_ppo − wt_frozen` | +16.64, q = 0.000, **64/64** | the wild type learns this cell |
| V7 `rn_ppo − rn_frozen` | +16.94, q = 0.000, **64/64** | so does the null — the contrast is not one arm failing |
| V8 `wt_frozen − rn_frozen` | −0.17, q = 0.735, 28/64 | **the untrained prior is indistinguishable** |

### The endpoint, which moves the other way

The peak axis is `saturated` on both cells and is not used for a verdict, but its registered test is
reported: pooled over 64 seeds the null is nominally **ahead** on mean foods, −0.47 of 20 (q = 0.004,
44/64 positive), with full clears 91.87% against 93.09%. Its direction flipped between panels
(+0.13, +0.30, −1.15) — on a metric at ceiling, a minority of collapsed wild-type seeds dominates the
mean while a majority of seeds stay ahead.

### The secondary cell

Food-only, 16 seeds: peak `SATURATED` (100.00% on both wirings, contrast exactly 0.00 on 0/16),
efficiency `DEGREE-STATISTICS` with all four metrics nominally negative (−25.7% on time-to-competence,
wild-better 6–7/16). Its prior is also indistinguishable (−0.07, q = 0.847).

## Analysis

1. **The premise holds, on a cell that can measure it.** Under an optimiser known to learn, on a
   behaviour the animal performs, the specific wiring is worth about a third off the time to reach
   competence. The two controls are what license the reading: both wirings do learn, and their
   untrained priors are indistinguishable — the third independent confirmation of that, after 041's
   +3.3 over 64 seeds and 044's +0.77. **Whatever the wild type contributes, it contributes during
   learning.**
2. **It is speed, not performance, and the two must not be conflated.** Both wirings finish at the
   same place; if anything the null finishes marginally higher. That is consistent with every prior
   measurement — 034, 043 and this panel's own food-only cell all put the null level or nominally
   ahead on endpoints — and it says those results were not wrong, but were measuring the axis on
   which this substrate has no advantage.
3. **The shrinkage is the estimate converging, not the effect dissolving.** +46.4%, +32.6%, +31.8%:
   panel 1 drew high, and the two independent panels agree. Panel 2's failure to replicate was
   underpower at the true effect size, not a contradiction — its 80% intervals were entirely above
   zero and each contained panel 1's estimate, and a Mann-Whitney between the two panels' per-seed
   deltas gives **p = 0.318**. The panels are one population.
4. **Why the paired test and the interval disagreed at n = 16.** The effect is heterogeneous: large
   on most seeds, slightly negative on a few. A paired rank test reads only signs, so it needs
   consistency; the bootstrap interval reads magnitude. At 16 seeds the sign count was the binding
   constraint, which is why panel 3 was registered at 32.
5. **What the C3 results were, in this light.** Logbook 056 classified five contrasts as
   uninformative because no optimiser had shown a wild-type advantage. That classification stands —
   those are C3 contrasts, and 056 scoped its claim to "this task, this substrate" — but its general
   reading no longer holds and is superseded here. A wild-type advantage exists; it is on an axis and
   a cell none of those contrasts measured.

## Conclusions

- **`specific_wiring_efficiency`** on the food-under-thermal-pressure cell: +35.4% off
  time-to-competence over 64 paired seeds, all four metrics at q ≤ 0.001, replicated on an
  independent 32-seed panel.
- **No endpoint advantage**, and a small endpoint deficit pooled. The claim is about learning speed.
- **PPO, not a local rule.** 7b's gate as written is unmet; its premise is now supported.
- **The food-only cell cannot answer this question**: both wirings solve it perfectly and reach
  competence in forty episodes.
- **No committed verdict changes.** Every C3 result stands in its own units, on its own cell.

## Next Steps

Neither is run here; both are registered as the follow-ups this result earns.

- [ ] **Why the seeds differ.** Each seed's rewiring is deterministic given `rewire_seed`, so the 64
  rewired graphs can be regenerated offline and tested for a graph property that predicts that
  seed's time-to-competence — path length from AFDL/AFDR to the motor readout being the obvious
  candidate. No compute. Exploratory, and anything it finds needs its own registered test.
- [ ] **Pathway or difficulty?** The two cells differ in difficulty, not only in modality. A
  food-only cell made hard by some route other than temperature separates "the evolved AFD
  pathway helps" from "the wiring helps once the task stops saturating". The second is the
  broader claim and the one worth registering against.
- [ ] B.8 inherits this: a phase with a positive, bounded, replicated result.

## Data References

- Records: `supporting/057-wiring-premise-contrast/` — `launch.md` (the protocol, the amendment and
  both replications registered before they ran), `pilot/`, `panel1/`, `panel2/`, `panel3/`,
  `per-seed-primary.csv` (all 64 seeds).
- Configs: `configs/scenarios/{foraging,thermal_foraging}/connectomeppo_small_continuous2d_*_t20.yml`
  and their `_rewired_null`, `_frozen` and `_rewired_null_frozen` variants.
- Harness: `scripts/analysis/wiring_premise.py`, reusing `t7_continuous_ranking.plateau_tail`,
  `weight_search_architecture_ranking.{paired_seed_wilcoxon_bootstrap,bh_fdr}` and
  `connectome_structure_efficiency.analyse`.
- 320 runs in total (52 pilot, 128 panel 1, 64 panel 2, 128 panel 3); all succeeded.
