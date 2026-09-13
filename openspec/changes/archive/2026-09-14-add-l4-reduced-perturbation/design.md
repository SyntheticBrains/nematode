# R.1c — a declarable perturbation set for the connectome: design

## What the substrate does today, and why 44% of it cannot help

The eligibility is `E ← decay·E + M_chem ∘ (h_prev ⊗ perturbation)`, so a unit's perturbation writes
eligibility on **every chemical synapse onto it**, regardless of whether that unit can influence the
action. The action comes from a mean-pool over the **39 VB/DB/VA/DA motor neurons** only. With
`forward_pass_depth: 4`, a perturbation injected at step `s` reaches the pool only if its unit is within
`4 − s` hops.

| settling step | hop budget | units that can reach the readout |
|---|---|---|
| 1 | 3 | 277 of 302 |
| 2 | 2 | 247 of 302 |
| 3 | 1 | 109 of 302 |
| 4 | 0 | **39 of 302** |

**672 of 1208 draws per decision are causally connected; 536 are not.** The largest budget is 3, at step
1, so the **25** units at four hops or more can never contribute at any step: 1 at four hops, 7 at five,
7 at six, 3 at seven or more, and 7 unreachable in the directed chemical graph. 277 + 25 = 302.

**A masked synapse stays put only because homeostasis cancels an unconditional decay, and the
cancellation is close but not exact.** The rule applies `− rate · weight_decay · weight` to **every**
plastic weight, trace or no trace, so a masked unit's incoming weights shrink on every update. The
homeostatic rescale returns each unit's incoming norm to its construction target, and both operations
are purely radial, so they cancel — measured on the recipe's own values (rate 1e-3, decay 1e-3) with
every trace held at zero:

| updates | incoming-norm change | largest single weight's excursion | 1 − cosine of the weight vector |
|---|---|---|---|
| 1 000 | 0.0000% | 1.8e-05 | 0 |
| 10 000 | 0.0000% | 1.9e-04 | 6e-08 |
| 100 000 | 0.0000% | 1.9e-03 | 2.6e-06 |

So the norm is held **exactly** and the direction to a cosine of **1 − 3e-05** extrapolated across a
full run's ~1.05M updates. What remains is float32 round-off, accumulating linearly in the largest
single-weight excursion — about 1.9e-02 over a run, roughly 9% of these weights' rms — while leaving
the vector the unit actually computes with unchanged. That is jitter, not decay and not learning.

**Without homeostasis it is decay.** The same bench run loses **2.0% of the norm per 20 000 updates**,
which compounds to a collapse over a run: a `motor` arm would shrink its 3,386 uncredited synapses and
be a covert global weight-decay experiment reported as a dimension result. So masked arms are **only
supported with homeostasis on**; a config validator refuses the combination, and the test asserts both
directions — norm conserved with it, norm lost without it — so the dependency is measured rather than
assumed.

**And it is measured in the campaign, not only on the bench.** Drift is reported separately for
**excluded** and **credited** synapses per arm, so the claim that excluded weights only jitter is
checked on the real substrate at the real scale rather than argued from this table. Note the residual is
a property of the rule that **every** plastic result already carries; what a restricted set changes is
that it becomes the *only* update those synapses receive.

Two distinct things follow, and the design keeps them apart:

- **A correctness fix.** Masking the 536 disconnected draws removes variance from the estimator and
  **no signal**, because those draws provably cannot move the outcome they are credited against.
- **A dimension knob.** Restricting the perturbed set further trades adaptable synapses for a smaller
  dimension, which is a real cost and not a free improvement.

## The arms

One cell: the **calibrated hard-food connectome cell** (`max_steps: 350`,
`target_foods_to_collect: 20`) — block V's cell, so a result here is directly comparable both to R.1's
MLP sweep and to the PPO reference already on the record.

| arm | perturbed set | units | adaptable synapses | draws/decision |
|---|---|---|---|---|
| `full` | every unit, every step — **today's behaviour** | 302 | 3709 | 1208 |
| `causal` | per-step reach mask | 277 | 3538 | **672** |
| `hop1` | within 1 hop of the readout pool | 109 | 1476 | 436 |
| `motor` | the readout pool itself | 39 | 323 | 156 |
| `motor_last` | the readout pool, **last settling step only** | 39 | 323 | **39** |

Realised causally-connected draws, measured from the built masks: `full` 672 of 1208, `causal` 672
of 672, `hop1` **366 of 436** — a hop-1 unit cannot reach the pool from the last step — `motor` 156
of 156, `motor_last` 39 of 39.

Each carries **its own frozen control** at the same σ and the **same mask**, freezing only the update —
R.1's lesson, so the perturbation's cost to the policy is matched across each pair and the contrast
measures the update alone. Eight seeds, 3000 episodes. **Ten arms, 80 runs.**

**Correction, 2026-09-13, before any arm ran.** The table above first gave `causal` as 302 units and
3709 adaptable synapses, on the reasoning that it removes only draws that cannot matter. The
implementation shows that is wrong in one respect: the **25** units at four hops or more are never
perturbed at *any* step, so their **171** incoming synapses are never credited either. `causal` is
therefore 277 units and 3538 synapses, coinciding with the units of a "within 3 hops" set.

What it gives up is credit those synapses could never have earned informatively — their units cannot
influence the action from any settling step — so the claim "no signal lost" still holds, while the
claim "every synapse stays adaptable" does not. Both are now stated.

`hop2` (247 units, 988 draws) is deliberately omitted: it is within 1.2× of `full` on the axis under
test and would spend 16 runs to interpolate a curve the other arms already bracket.

### The capability reference is already on the record

Block V ran PPO on this exact cell and substrate over **32 seeds**
([Logbook 058](../../../docs/experiments/logbooks/058-wiring-premise-difficulty.md)), so **no PPO arm is
re-run here**:

| arm (058, 32 seeds) | mean foods of 20 | full clear |
|---|---|---|
| wild type under PPO | **19.31** | — |
| wild type, frozen weights | **3.82** | **0.0%** |

So the reachable gap on this cell is **15.49 foods**, which fixes the relative minimum at **1.55 foods**
rather than leaving it to be computed later. Note 058's frozen arm does **not** perturb, so it is a
reference and not the comparator: every arm here carries its own frozen control at its own mask and σ,
which is the only matched null.

For scale, the MLP's frozen arms on the same cell ran 1.69–3.89 foods in R.1, beside this substrate's
3.82 — a small consistency check, not a comparison.

## The reading

**Plateau-tail mean foods** through I.2's graded family, learning against its own frozen control, paired
by seed, one-sided, BH-FDR across the five masks. Full-clear success recorded alongside; where it sits
at the floor the record says so rather than reporting a null.

**A shift counts only if significant *and* both**: at least **1.0 foods** of the cell's 20, and at least
**10%** of the reachable gap. Against 058's committed reference the gap is 19.31 − 3.82 = **15.49
foods**, so the relative minimum is **1.55 foods**; where an arm's own frozen mean differs from 3.82 the
harness recomputes it per arm and reports both.

**Drift** per arm — relative weight distance from its own frozen control — so starved-and-still is
distinguishable from writing-a-lot-in-a-worsening-direction, which is what separated the two regimes in
R.1.

**Every run records its perturbation set**: the declared mask, the unit count, the adaptable-synapse
count and the draws per decision. A dimension claim that does not carry those numbers is the thing R.1
had to reconstruct after the fact.

## Outcomes, fixed before the run

| verdict | test | what follows |
|---|---|---|
| `causal_mask_sufficient` | the `causal` arm beats its own frozen control by both minima | the strongest available result: the failure was **uninformative noise**, not scale, and nothing had to be given up to fix it. R.1b runs at the causal mask with all 3709 synapses adaptable |
| `dimension_reducible` | some reduced arm (`hop1`, `motor`, `motor_last`) beats its control by both minima, and `causal` does not | the dimension binds on the connectome as it does on the MLP. **R.1b is unblocked at the winning mask**, which becomes its registered dimension, and the record states which synapses that mask gives up |
| `not_reducible` | no arm beats its own frozen control | the connectome's failure is **not** the perturbation dimension. R.1b stays blocked, R.2 (e-prop) becomes the live path, and R.1's result stays bounded to the MLP |

A partial reading — `motor_last` working where `motor` does not, or the ordering non-monotone — is
recorded as **partial with the ordering stated**, not resolved toward the nearest verdict.

**A deliverable that does not depend on the learning arms.** The causal-reach table is a property of the
substrate at a given depth, computed from the loaded connectome, and it stands whatever the arms do. A
`not_reducible` verdict still ships the measurement that 44.4% of this substrate's perturbation is
credited against an outcome it cannot influence.

### Stop clauses — void until found

- **`full` does not reproduce the known failure.** Every plastic connectome result to date has it at or
  below its floor; if it learns here, something changed since 055 and nothing in the sweep is
  interpretable until that is found.
- **A mask does not mask.** Each arm asserts its realised draws-per-decision against the declared value,
  and that a masked unit's incoming synapses do not move across an episode. A mask that silently does
  nothing would make its arm a duplicate of `full` reported under another name.
- **The frozen floor is at the ceiling.** If the frozen arms already clear the cell, there is nothing to
  add and the finding is about the cell, not the rule.

## Honest prior

**`not_reducible` for `causal` alone, and better than even for `motor`.** The arithmetic is the reason to
say so in advance:

- the causal mask takes draws per decision from 1208 to 672 — a **1.8×** reduction;
- R.1's MLP curve needed **16×** (128 → 8 units) to move from 3.0% to 90.7% full clear, and 128 → 64 (a
  2× reduction) only reached 12.9%.

So a 1.8× reduction is very unlikely to be sufficient on its own, and the causal mask should be argued
for as **correctness** rather than as a knob. `motor` at 156 draws is a **7.7×** reduction and 39 units,
which sits between the MLP's 32 (53.6% full clear) and 64 (12.9%) — the informative place to be.

Against it: only **323 synapses** are adaptable at `motor`, against 3709 unmasked. There may simply be
too little left to express — though the frozen prior is a long way from competent on this cell (3.82 of
20 foods and **0.0% full clear** over 32 seeds), so there is a large gap to move into rather than a small
one. That is the risk the `hop1` arm exists to bracket, at 1476 synapses.

## Risks

- **Reduced masks buy dimension with adaptable synapses.** Unlike R.1's MLP sweep, where narrowing the
  network narrowed everything together, here the substrate is fixed and the mask only chooses who gets
  credit. A win at `motor` is a win for a **restricted learner**, and the record says so.
- **Nested-by-proximity is not the only ordering.** It is chosen because a random subset of 302 units
  would mostly pick units with little influence on a 39-unit readout, confounding "how many" with
  "which". The cost is that the arms differ in identity as well as count, which the record states rather
  than claims to have separated.
- **The masked arms are coupled to homeostasis** by the decay cancellation above. That is pinned and
  tested, but it means these arms cannot be re-run at `plasticity_homeostasis: false` without the mask
  becoming a decay manipulation, and the configs say so.
- **Eight seeds**, the same power limit every block-I and R result carries; hence both effect minima.
- **One cell.** A reduced dimension that works here says nothing about the 2400-step C3 cell, where the
  horizon is seven times longer and R.1 found the horizon interacting with the dimension.
- **The hop distances are computed on the directed chemical graph** and ignore gap junctions, which are
  frozen and bidirectional. A gap junction could carry influence the hop count misses, so the causal mask
  is **conservative in the wrong direction** — it may mask a draw that could have mattered through a gap
  path. The arm is registered with that stated, and the gap-inclusive variant is named as the follow-up
  rather than silently folded in.

______________________________________________________________________

## Amendment, 2026-09-13 — calibrate σ on this substrate before the campaign

The pilot passed both registered gates: the platform has room (frozen 1.9 of 20 foods against PPO's
19.31) and `full` has not learned (2.55 of 20, **0.00% full clear**, all 3000 episodes FAILED). The
direction is as predicted — `motor` +1.11 foods on 4/4 seeds against `full` +0.61 on 4/4 — but at 7% of
the 15.49-foods reachable gap it is **below the registered 1.55-foods minimum**, so the campaign as
registered would most likely record `not_reducible`.

**It also surfaced an uncalibrated knob that may matter more than the mask.** These frozen controls
perturb; 058's did not. σ 0.2 takes the frozen prior from **3.82 foods to 1.9 — it halves it** before any
learning happens. σ 0.2 was selected on the **one-step control** and carried to the MLP; **no connectome
arm has ever had it calibrated**, and this is the tension 052 named: the σ that makes the rule learn is
the σ that makes a competent policy unrunnable.

**Added, before the campaign**: the `motor` arm at **σ ∈ {0.05, 0.1}**, learning and frozen, on the
pilot's disjoint seeds 101–104 — **16 runs**, about 30 minutes. σ 0.2 is not re-run; the pilot's eight
`motor` runs are that point of the grid.

**The decision rule, fixed before these run.** The campaign takes the σ that **maximises the learning
arm's plateau-tail mean foods**, with the learning-minus-frozen gap and the frozen arm's retention of the
unperturbed 3.82-food prior reported beside every point. Ties go to the **larger** σ, which carries more
signal. The absolute level is primary rather than the gap, because the question is whether the rule
**learns the cell** — not whether it beats a floor its own noise damaged.

**And it can stop the campaign.** If no σ lifts the learning arm above the **unperturbed frozen prior of
3.82 foods**, then at 39 perturbed units — inside the band where the MLP reached 15.7 to 18.8 foods — this
substrate is not learning this cell at any tested scale. That answers the campaign's question at pilot
cost, and `not_reducible` is recorded from 24 runs instead of 104.

**A measurement the pilot already delivers, independent of what follows.** The connectome badly
underperforms the MLP at matched dimension: R.1 reached 18.83 foods at 16 units, 15.72 at 32 and 8.34 at
64, while this substrate's 39-unit arm reaches **2.99**. Perturbation dimension does not transfer across
substrates — registered as a caveat, now a number.

### Calibration outcome, 2026-09-13 — σ 0.1, and a correction to this amendment's own reasoning

16/16 runs, 3000 episodes each. The σ 0.2 point is the pilot's eight `motor` runs.

| σ | learning foods | frozen foods | gap | seeds favouring | frozen as % of 3.82 | full clear |
|---|---|---|---|---|---|---|
| 0.05 | 4.321 | 1.732 | +2.589 | 4/4 | 45.3% | 0.17% |
| **0.1** | **4.361** | 1.731 | **+2.630** | 4/4 | 45.3% | 0.03% |
| 0.2 | 2.989 | 1.883 | +1.106 | 4/4 | 49.3% | 0.00% |

**The registered rule selects σ 0.1** — it maximises the learning arm's plateau-tail foods, 4.361 — and
**the stop clause does not fire**: 4.361 is above the 3.82 prior, so the campaign proceeds. σ 0.2 was
indeed too large for this substrate: dropping it lifts the learning arm by **46%** and the gap from
+1.106 to +2.630.

**But this amendment's stated reason for running the calibration was wrong, and that is recorded rather
than quietly dropped.** It said σ 0.2 "takes the frozen prior from 3.82 foods to 1.9 — it halves it". The
frozen arm sits at **1.73 at σ 0.05 too**, and at 45–49% of 3.82 at *every* σ, so the perturbation scale
is **not** what costs the difference. The likely cause is the **action-noise setting**: these arms pin
`initial_log_std: -1.0` as part of I.1's plastic recipe — an action std of **0.368** — while 058's frozen
arm takes the default **1.0**. A wider action distribution collects more food by accident on this cell.

So the calibration was **worth running and its result stands**, but for a different reason than the one
registered: not that σ 0.2 damages the prior, but that it costs the *learning* arm 46% of its level.

**The relative minimum's reference is restated accordingly.** The registered 1.55 foods is 10% of
19.31 − 3.82, and that 3.82 comes from an arm at a different action noise. Against these arms' own frozen
level the gap is 19.31 − 1.73 = **17.58**, so the minimum is **1.76 foods**. The campaign is held to the
**more demanding** figure; +2.630 clears both, so nothing about the reading turns on the choice. No fully
matched PPO reference exists at `initial_log_std: -1.0` on this cell, and none is run — the comparator
that decides every arm is its own frozen control, not either reference.

### A gap in this change's own registration, surfaced before the campaign

`dimension_reducible` is defined as "some reduced arm beats its control by both minima", and its stated
consequence is "**R.1b is unblocked at the winning mask**". At σ 0.1 the `motor` arm would satisfy that
definition: +2.630 foods clears both minima and 4/4 seeds favour it, which at eight seeds reaches
p = 0.004.

**It would not deliver what R.1b needs.** R.1b is block V's contrast on **time to competence**, and this
arm reaches **4.36 of 20 foods at 0.03% full clear** — it never becomes competent, so a time to
competence is undefined for it. The verdict's condition is therefore **weaker than its consequence**:
beating a frozen control by the registered minima is not learning the cell, and only the latter makes the
wiring contrast measurable.

This is recorded now, before the campaign, rather than discovered after it. The campaign's reading will
state the two separately: whether a mask **beats its floor** by the registered bar, and whether any mask
**reaches competence**, which is what R.1b's gate actually requires.

### Amendment, 2026-09-13 — calibrate the action noise too, before the campaign

Inspecting what the rule may write turned up a structural fact worth recording whatever the arms do.
The connectome's `plastic_weights` is **`w_chem` alone**; `food_gains` and `readout` live in the
optimiser's parameter list, which is used **only under PPO**. So under the rule **both ends of the
network are frozen**, and after just 300 PPO episodes on this cell:

| tensor | relative change | cosine to init | writable by the rule |
|---|---|---|---|
| `readout` (2×4, motor classes → action) | **0.783** | +0.865 | **no** |
| `w_chem` (3709 chemical synapses) | 0.486 | +0.899 | yes |
| `food_gains` (sensory projection) | **0.177** | **+0.985** | no |

**PPO's largest single adaptation is the motor readout, and the rule cannot make it** — while the frozen
sensory projection costs little, since PPO barely rotates it. The readout starts from an anatomical prior
(speed as the B-vs-A contrast, turn as D-vs-V, unit-normed) and PPO still moves it 78% in a tenth of a
run. Relative change across differently-sized tensors is not a clean importance measure — eight readout
entries against 3709 synapses — so this is a hypothesis, not a conclusion, and it is registered as
**R.1d** rather than folded into this change, whose question is the perturbation dimension.

**What belongs here is the action noise**, for the same reason σ did: `initial_log_std: -1.0` comes from
I.1's recipe, an action std of 0.368, and it has never been calibrated on this substrate — while 058's
frozen arm at the default std **1.0** collects **3.82 foods** against these arms' **1.73**. Running the
mask comparison at an uncalibrated operating point would hand it a handicap nobody chose.

**Added**: the `motor` arm at σ 0.1 with `initial_log_std ∈ {0.0, −0.5}`, learning and frozen, seeds
101–104 — **16 runs**. The −1.0 point is the σ-calibration's eight σ 0.1 runs and is not re-run.

**The decision rule, fixed before these run.** As for σ: the setting **maximising the learning arm's
plateau-tail mean foods**, with the learning-minus-frozen gap and the **full-clear rate** reported at
every point. Ties on foods go to the higher full-clear rate; ties on both go to the recipe's pinned
−1.0, so the registered recipe is displaced only when it is actually beaten.

**One reading the record must not make.** A wider action distribution collects more food by wandering, so
if the learning arm rises and **the gap does not**, the gain is the task being easier to stumble through
rather than the rule learning better. Where that happens the record says so, and the setting is chosen on
the absolute level only if the gap survives.

### Action-noise outcome, 2026-09-13 — the pinned −1.0 stands, and action noise is off the list

16/16 runs. The −1.0 point is the σ-calibration's σ 0.1 runs.

| `initial_log_std` | action std | learning foods | frozen foods | gap | seeds favouring | full clear |
|---|---|---|---|---|---|---|
| **−1.0** (pinned) | 0.368 | **4.361** | 1.731 | **+2.630** | 4/4 | 0.03% |
| −0.5 | 0.607 | 4.056 | 2.452 | +1.604 | 4/4 | 0.00% |
| 0.0 | 1.000 | 4.154 | 3.254 | +0.900 | 4/4 | 0.00% |

**The registered rule keeps −1.0**, which maximises the learning arm's foods. Wider action noise lifts
the **frozen** arm — 1.73 → 2.45 → 3.25 — while leaving the **learning** arm flat at 4.06–4.36, so the gap
collapses from +2.630 to +0.900. The registered guard reads *matched* only because the learning arm did
not rise; what a wider action distribution buys is floor food by wandering, which is what the guard was
written to catch.

**Two things this settles.** The 1.73-against-3.82 discrepancy is confirmed as the action-noise setting:
at std 1.0 the frozen arm reaches 3.25, and the remainder to 058's 3.82 is that its arm carries no
perturbation at all. And **action noise is not the blocker** — the useful part of a null.

**The observation that matters most is the invariance.** The learning arm sits at **4.1–4.4 foods under
every knob tried**: σ across a fourfold range (4.321, 4.361, 2.989) and action noise across 2.7× (4.361,
4.056, 4.154), with full clear never leaving ~0%. Two independent sweeps hitting the same ceiling is the
signature of a **structural** limit rather than a hyperparameter one — which is what the frozen-readout
finding predicts and which is why R.1d is registered separately rather than pursued as more tuning.

**The campaign runs at σ 0.1 and `initial_log_std: -1.0`**, the latter unchanged from the recipe.

### Correction, 2026-09-13 — the campaign was launched at the rejected σ and restarted

The ten arm configs were generated **before** the σ calibration and carried `plasticity_node_noise: 0.2`.
The calibration then chose **0.1**, this record said the campaign would run at 0.1, and the campaign was
launched with the original configs — **at 0.2, the value the calibration rejected**. Caught at 32 of 80
runs and stopped rather than left to finish: σ 0.2 costs the learning arm 46% of its level at the `motor`
mask, so the whole grid would have been measured at an operating point already known to be wrong.

All ten configs now pin **σ 0.1** with the calibration's reason in the file, the exact-key test pins it so
a config drifting back cannot pass as registered, and the campaign is relaunched.

**The 32 completed runs are kept**, under `campaigns/reduced-perturbation-sigma02-aborted/`. They are the
σ 0.2 points for `full` and `causal` on the registered seeds 1–8, and they are informative:

| set (σ 0.2, seeds 1–8) | learning foods | frozen foods | shift | seeds favouring | p | full clear |
|---|---|---|---|---|---|---|
| `full` | 2.238 | 2.350 | **−0.112** | 4/8 | 0.63 | 0.00% |
| `causal` | 2.185 | 2.378 | **−0.192** | 3/8 | 0.73 | 0.00% |

Two readings follow, and both stand independently of the relaunch.

**`full` reproduces the known failure on the registered seeds** — the stop clause's letter this time, not
only its rationale: the learning arm sits *below* its own frozen control. The pilot's +0.61 at 4/4 seeds
was small-sample noise, and it becomes −0.112 at 4/8 on eight registered seeds. That is the clearest
available argument for why the registered campaign uses eight seeds and the pilot decides nothing on its
own.

**The causal mask alone does not rescue anything at σ 0.2**, exactly as the honest prior said it would
not: a 1.8× reduction in draws where the yardstick needed 16×. Removing 536 provably uninformative draws
per decision leaves the arm at 2.185 foods against its floor's 2.378.
