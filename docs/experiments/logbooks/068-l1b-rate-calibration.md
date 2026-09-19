# 068: The Pool-Hiding Effect Is a Property of the Learning Rate (7a-ii L.1b / Phase 7)

**Status**: completed — **`width_favours_the_shuffle_at_this_rate`**, and the robust finding is the
one the reading's name does not carry: **[L.1](066-l4-readout-width.md)'s interaction does not survive
a one-decade change in the learning rate**. 192 of 192 runs succeeded; the panel is 96 paired seeds
across two rates.

**The registered primary.** At `plasticity_rate` **0.0001** the width × wiring interaction on
`auc_success` is **−0.0657** (CI [−0.0883, −0.0428], q = 0.0005, wild-type-gains-more on **31 of 96**
seeds). At 0.001 it is **+0.2818**. The **three-way** — the interaction at 0.001 minus the interaction
at 0.0001, per seed — is **+0.3475** (CI [+0.2957, +0.3966], q = 0.000, **81 of 96** seeds). The rate
dependence is *larger than L.1's whole effect*, and the sign of the interaction reverses.

**What is established, and it is the point of the panel.** L.1's positive is a fact about
(width × wiring × **rate**), not about width × wiring. The rate was never swept at the per-neuron
width: R.2 pinned 0.001 on the 8-parameter readout and waived its registered check,
[L.0](064-l4-frozen-features.md) checked it at the pooled width on the wild type alone, and L.1
inherited it. **The phase's only positive for the wild-type wiring under a biologically plausible
learner may not be cited without its rate.**

**What is *not* established: the reverse direction at 0.0001.** The registered reading fires on the
primary alone, and three things say the panel does not support reading it as a wiring result:

1. The magnitude, **0.0657**, is **less than half of the 0.141 minimum** the *positive* branch would
   have needed. The registration put **no minimum on the reverse branch** — an asymmetry in my own
   registered rules, named here rather than exploited.
2. The registered secondary is **voided at 0.0001 by censoring**: 58 of 96 wild-type and 44 of 96
   null pooled seeds never reach 30% competence, against **0 of 96 in both wide cells**. On it the
   interaction is **+20.4 episodes, p = 0.658** — nothing.
3. An **unregistered** graded axis, `auc_foods`, gives **+0.2828, p = 0.637** — nothing, and the
   opposite sign.

**The dominant effect at 0.0001 is capacity, and nothing in this programme has seen one this large.**
The width main effect is **+0.6178** (CI [+0.6047, +0.6309], q = 0.000) on **96 of 96** seeds, where
L.1 at 0.001 **detected no width main effect at all** (−0.0112, q = 0.591). At the lower rate the
8-parameter readout learns slowly and the 78-parameter one thrives. **The wild type leads at neither
width at 0.0001**: the wiring effect is −0.0320 pooled and −0.0977 wide.

**My honest prior was `pool_effect_is_rate_specific`** — right that the effect would not survive,
wrong in expecting a failure to detect rather than a significant reversal. It stands as written.

**Branch**: `feat/l1b-rate-calibration`.

**Date**: 2026-09-19.

**OpenSpec change**: `add-l1b-rate-calibration` (extends `plasticity-evaluation`: a positive carrying
an inherited learner setting is re-read at a calibrated operating point before a synthesis cites it).

## Objective

L.1 read `pooling_hid_structure` at 0.001: widening the readout from the four-class mean-pool
(8 parameters) to one weight per motor neuron (78) helped the wild type and hurt its
degree-preserving null, interaction +0.2818, the wiring effect's sign flipping between widths. It is
the phase's first and only positive for the wild-type wiring under a plausible learner.

[L.4/L.5](067-l4-feature-ablations.md) then measured the two **wide** learning arms at 0.0001 as its
rate-matched baseline and found the wiring effect there **−0.0977, null ahead**. L.1's sign-flip form
was already known not to hold at the lower rate. What was unknown — and what a synthesis citing L.1
has to know — is whether L.1's **registered primary, the interaction**, holds there.

> *At 0.0001, does widening the readout still help the wild type more than the shuffle — or was L.1's
> interaction a property of 0.001?*

This ran **before the synthesis** because the protocol says re-read before shipping, and the one
re-read that could change what the synthesis says about its own headline cost 192 runs.

## Method — two arms complete a 2×2 that was three-quarters measured

The 2×2 at 0.0001 was missing exactly its two **pooled** learning cells. Everything else existed:

| cell | rate | source |
|---|---|---|
| `wt_pooled`, `rn_pooled` | 0.0001 | **this campaign**, 192 runs |
| `wt_wide`, `rn_wide` | 0.0001 | L.4's rate-matched baseline, `campaigns/feature-ablations` |
| the four frozen floors | inert under `freeze_updates` | L.1, `campaigns/readout-width` |
| L.1's four learning arms | 0.001 | L.1, for the three-way and the reference 2×2 |

Two configs, each one key (`plasticity_rate`) from its committed pooled parent **and** one key
(`readout_width`) from the wide arm at the same rate, both asserted by exact-key test. The wild-type
config is L.0's rate-check config promoted to a registered arm with its header rewritten; the null's
is new.

**Every reused cell is licensed by evidence, not argument.** At seed 1, all eight reused arms were
re-run under the current path and reproduced their committed logs on **all nine fields** the parser
reads. Before comparing, each committed run's export was checked to still exist, so no field was
"identical" by being absent on both sides.

**Eight tests in one BH-FDR family**: the interaction at 0.0001, both main effects there, the
three-way, and four learning gates. The priors are L.1's committed tests on the same floors
(q = 0.462) and were not re-run. The full protocol, both dated stop-clause passes and a dated
correction to a mis-cited statistic are in
[`supporting/068-l1b-rate-calibration/launch.md`](supporting/068-l1b-rate-calibration/launch.md).

## Results

### Gates, read before anything else

| arm | Δ over its own-width floor (foods) | q | seeds |
|---|---|---|---|
| `wt_pooled` @ 0.0001 | **+11.569** | 0.000 | 96/96 |
| `rn_pooled` @ 0.0001 | **+12.292** | 0.000 | 95/96 |
| `wt_wide` @ 0.0001 | **+16.159** | 0.000 | 96/96 |
| `rn_wide` @ 0.0001 | **+15.959** | 0.000 | 96/96 |

All four arms plainly learn, so the interaction is interpretable. The priors on these floors are
L.1's and detect no pre-update difference between the wirings at either width.

### The 2×2 at each rate — `auc_success`, the registered primary (n = 96 paired)

| rate | | pooled (8 params) | per-neuron (78) | widening buys |
|---|---|---|---|---|
| **0.0001** | wild type | 0.1163 | **0.7013** | **+0.5850** |
| | rewired null | 0.1484 | **0.7990** | **+0.6506** |
| | wiring effect | **−0.0320** (null ahead) | **−0.0977** (null ahead) | |
| **0.001** (L.1) | wild type | 0.4140 | **0.5437** | +0.1297 |
| | rewired null | 0.5106 | 0.3585 | −0.1521 |
| | wiring effect | **−0.0966** (null ahead) | **+0.1852** (wild ahead) | |

L.1's four cells, its interaction of **+0.2818** and its censoring counts are reproduced **exactly**
by this harness through its own efficiency call, which is the check that the instrument still
reproduces the record one half of the three-way comes from.

| contrast (at 0.0001 unless stated) | Δ | CI | q | seeds |
|---|---|---|---|---|
| **interaction (primary)** | **−0.0657** | [−0.0883, −0.0428] | **0.000** | 31/96 |
| **three-way**, 0.001 minus 0.0001 | **+0.3475** | [+0.2957, +0.3966] | **0.000** | **81/96** |
| width main effect | **+0.6178** | [+0.6047, +0.6309] | 0.000 | **96/96** |
| wiring main effect | −0.0649 | [−0.0811, −0.0482] | 0.000 | 32/96 |
| interaction at 0.001 (reference, outside the family) | +0.2818 | [+0.2281, +0.3340] | — | 74/96 |

**Read the rows in order.** The three-way is the result: the interaction moves by +0.3475 between the
two rates, on 81 of 96 seeds, which is more than L.1's whole effect. The width main effect is the
mechanism of that move: at 0.0001 capacity dominates everything, where at 0.001 it was undetectable.
The interaction at 0.0001 is significantly negative, and the next section is why that is reported and
not explained.

### Why the reverse direction is reported and not read as a wiring result

**The magnitude against the registered minimum.** `pool_effect_survives_the_rate` required a positive
interaction **and** `abs(Δ) ≥ 0.141` — half of L.1's +0.2818 — on the reasoning that a claim which
has lost more than half its size to a one-decade rate change is a claim about the rate. The observed
interaction is **−0.0657: 23% of L.1's, and negative.** The registration put **no minimum on the
reverse branch**, so the reading fires at a size that would not have been credited in the positive
direction. That asymmetry is a **defect in my own registered rules**, recorded here as L.1's missing
minimum was recorded rather than retro-fitted. Applying the minimum symmetrically, this panel reads
*the effect does not survive the rate, and no direction is credited at 0.0001*.

**The registered secondary is voided at this rate.** `episodes_to_30pct_success` with censoring
counted per cell:

| cell | mean episodes | censored |
|---|---|---|
| `wt_pooled` @ 0.0001 | 2550.9 | **58/96 (60%)** |
| `rn_pooled` @ 0.0001 | 2449.6 | **44/96 (46%)** |
| `wt_wide` @ 0.0001 | 306.1 | 0/96 |
| `rn_wide` @ 0.0001 | 184.4 | 0/96 |
| the four cells at 0.001 (L.1) | 1051.4 / 729.4 / 239.9 / 370.4 | 17 / 8 / 2 / 6 |

A censoring spread of **0.604** across the cells of a difference of differences is exactly what the
primary-metric choice was registered to avoid, and it is why `auc_success` is the primary. On the
censored axis the interaction at 0.0001 is **+20.4 episodes, p = 0.658** — not significant. Its point
estimate agrees in direction with the primary, which at p = 0.658 is agreement of signs and not
corroboration, and it is reported as that.

**An unregistered graded axis finds nothing.** On `auc_foods` the interaction at 0.0001 is **+0.2828
foods, p = 0.637** (48/96) — not significant, opposite in sign to the primary. At 0.001 the same axis
gives **+4.7669, p = 1.9e-8**, agreeing with L.1. This axis is **not** the registered secondary and
carries no verdict; it is reported because a reader deciding how much weight the −0.0657 can bear
should know that the only axis it appears on is the primary.

**Not a floor artefact, and the committed floor requirement does not apply.** The pooled cells at
0.0001 are *slow*, not floored: at plateau they reach **21.52%** and **28.33%** full clear against
frozen floors of **0.01%** and **0.04%**, on mean foods of 14.370 and 15.603 of 20. The low
`auc_success` is area under a curve that rises late. The requirement covering a floored primary
applies where no seed is competent, which is not this case, and it is named here because it would
have been the natural place to look.

### Sensitivity

| | registered | realised |
|---|---|---|
| interaction sd | 0.339 to 0.445 (assumed) | **0.1754** |
| standard error at n = 96 | 0.0346 to 0.0454 | **0.0179** |
| detectable at 80% | 0.097 to 0.127 | **0.0501** |
| power at the 0.141 minimum | 0.87 to 0.98 | **1.00** |

The realised spread is half what was assumed, because the pooled cells at 0.0001 vary far less than
they do at 0.001. So the panel could have detected an interaction as small as 0.050, and the 0.141 it
was sized for at **full power**. A null here would have been a strong null; what it found instead is
a reversal three times smaller than the minimum.

## Integrity

- **192 of 192 runs succeeded**, in 22,652 s of wall clock at 16-way parallelism, and all 192 parse.
- **No code changed during the campaign.** No commit touching `packages/`, the runner, the campaign
  driver or `configs/` since the launch commit `8bcad837`; a clean tree at launch and at scoring.
- **Every reused cell was identity-checked** at seed 1 on all nine parsed fields, with the committed
  exports confirmed present first. L.1's cell means, interaction and censoring counts come back
  exactly.
- **A mis-cited statistic was corrected before the campaign, not after.** This change's artefacts had
  quoted L.0's rate-check figures (17.41 / 12.08 / 12.81 foods) without naming the statistic; those
  are **whole-run** means, while every contrast in this programme reads the **plateau-tail** mean,
  which on the same runs is 18.085 / 14.488 / 12.842. L.0's ordering and its spread reading are
  unchanged; the citable margin at the pooled width is 3.60 foods rather than 5.33.
- **One field could not be compared in the pilot's extra identity point**, and is named rather than
  counted as a pass: `peak_action_density` reads a file inside a run's export, and L.0's 2026-09-15
  exports have since been deleted, so it is absent on the committed side. The registered identity
  check is untouched — all eight of its committed runs still hold their exports.

## What this establishes, and what it does not

1. **L.1's interaction does not survive a one-decade change in the learning rate.** Three-way
   +0.3475, q = 0.000, 81 of 96 seeds, against an original effect of +0.2818. This is the result, and
   it is robust: it is significant on the primary and on the censored axis (−472.8 episodes,
   p = 0.018), and the realised sensitivity was high enough to have detected a fifth of it.
2. **L.1's `pooling_hid_structure` stands as read at 0.001 and may not be cited without that rate.**
   The verdict is not rewritten; a dated condition is carried beside it wherever it is cited. The
   "sign flips" form belongs to (width × wiring × rate).
3. **At 0.0001 the wild type leads at neither width**, and the dominant effect is capacity:
   +0.6178 on 96 of 96 seeds, where L.1 detected no width main effect. **How much readout capacity
   matters is set by the learning rate**, which no rung in this programme had measured.
4. **The reverse direction at 0.0001 is not credited.** It fires the registered reading, at 23% of
   the registered minimum, on the primary axis alone, with the secondary voided by censoring and an
   unregistered graded axis finding nothing. **May not be cited as** the shuffle being favoured by
   width, nor as evidence about the wiring at 0.0001 in either direction.
5. **Not a mechanism, not an endpoint claim, not a read-across to block V, and not a choice of
   rate.** Both wirings learn better at 0.0001 at the wide width and worse at the pooled width, so
   neither rate is "correct"; which rate a later rung runs at is that rung's calibration to make.
6. **No committed verdict changes.** D2's primary remains unmet: `w_chem` is frozen throughout.

## Consequences

**For the synthesis (Z.1).** Phase 7's one positive for the wild-type wiring under a plausible
learner is carried **with its operating point in the same sentence as the claim**: at the per-neuron
readout width and `plasticity_rate` 0.001, the wild-type connectome's features are legible to a
learner that reads them where a degree-matched shuffle's are not; one decade lower the interaction
reverses, and the wild type leads at neither width. That is a narrower claim than L.1's headline and
it is the defensible one.

**For the rung that follows.** The phase-after-7 rung named in
[067](067-l4-feature-ablations.md) — gap-junction coupling as a dynamical term — keeps its
precondition and gains a second: **a rate × width calibration before any contrast**, since this panel
shows capacity's importance is rate-set, and a rung that pins a rate inherited from another capacity
is measuring its own pin. The cheapest form is the one run here: the missing cells of a crossed panel
at a second rate, ~192 runs.

**For the protocol.** Principle 7's note gains this panel's arithmetic: the inherited rate was not
merely suboptimal, it **set the sign** of the registered primary. And a new asymmetry is recorded
under principle 10: a minimum effect registered for one direction of a two-sided reading must be
registered for both, or the unguarded direction fires on a size the guarded one would refuse.

## Artefacts

- [`supporting/068-l1b-rate-calibration/launch.md`](supporting/068-l1b-rate-calibration/launch.md) — the protocol, both stop-clause passes, the dated statistic correction
- [`supporting/068-l1b-rate-calibration/rate_calibration.json`](supporting/068-l1b-rate-calibration/rate_calibration.json) — the full reading, the eight-test family, both rates
- [`supporting/068-l1b-rate-calibration/per-seed.csv`](supporting/068-l1b-rate-calibration/per-seed.csv) — one row per rate, width, wiring and seed, with the censoring column
- `scripts/analysis/l4_rate_calibration.py` — the harness
