# 060: The Perturbation Dimension — the Rule Solves a Multi-Step Cell at Eight Units and Fails at 128 (7a-ii R.1 / Phase 7)

**Status**: completed — **`mixed`**, and the mixture is the finding. The three-factor rule under node
perturbation **solves a multi-step foraging task at eight perturbed units** — 19.64 foods of 20 and
**90.7% full clear** against PPO's 19.69 and 87.6%, on 8 registered seeds, from a frozen control at
1.69 foods — and collapses **monotonically** as the dimension grows, to **3.0% full clear at the 128
units every failing MLP yardstick arm ran** (rho −1.000, drift rising 0.91 → 1.39). On the one-step
positive control the same dimension costs almost nothing: every width from 8 to 128 passes, the
yardstick's exact two-layer arrangement passes and reaches criterion *fastest*, and time-to-criterion
**falls** with the dimension (slope −0.149, CI [−0.239, −0.061], against a prediction of +1.0). So the
constraint is neither the dimension nor the horizon alone but their **product**: the per-unit credit is
diluted by the number of perturbed units times the number of decisions the reward is shared over.
Neither pre-registered row fits, and the registration fixed the treatment in advance — a mixed reading
is recorded as mixed with both halves stated. **No committed verdict is changed**; [Logbook
059](059-7a-shipment.md)'s re-registration condition is **met**, and 7b's gate **stands as written**
because the wiring contrast under a local rule has still not been run.

**Branch**: `feat/l4-perturbation-scale`.

**Date**: 2026-09-13.

**OpenSpec change**: `add-l4-perturbation-scale` (extends `plasticity-evaluation`: a
stochastic-gradient result records the perturbation dimension it was measured at, and a
scale-dependent mechanism is tested where the rule works before a failure is attributed to it).

## Objective

Ask whether the three-factor rule under node perturbation was ever run at a **perturbation
dimension** it could work at, before any further conclusion is drawn from its failures.

Node perturbation forms its eligibility from what each unit's own noise did to a scalar outcome. With
N perturbed units the per-trial gradient estimate has signal-to-noise ~1/√N, and the trials needed for
a given amount of progress grow roughly as **N** (Werfel, Xie & Seung 2005). That dimension differs
across every platform this rule has been measured on, and **no experiment had varied it**:

| platform | plastic layers | perturbed units | draws per decision | committed result |
|---|---|---|---|---|
| the one-step control | 1 (`Linear(K, 8)`) | **8** | 8 | **passes** — 89.0% of the floor-to-optimum gap ([048](048-l4-rule-positive-control.md)) |
| the MLP yardstick | 2 (`64 → 64`) | **128** | 128 | fails, and below its own frozen control ([055](supporting/055-l4-horizon-multistep/details.md)) |
| the connectome | 1 (302 neurons) | **302** | **1208** — every unit at each of 4 settling steps | fails on every multi-step task (040–047) |

The pattern this phase has read as "one-step works, multi-step fails" was therefore equally consistent
with "**8 units works, 128 and 302 do not**", and nothing in the record separated the two.

## What ran

**S1, the arithmetic, where the rule works.** The committed one-step contextual-association control at
`HIDDEN ∈ {8, 16, 32, 64, 128}`, seeds 1–8, 20 000 trials, at I.1's passing configuration with the
shape as the only axis. This platform has **no capacity confound** — the task is solvable by 8 units
and every further unit adds only noise — so the dimension is isolated in a way it cannot be on a
behavioural cell, where width is capacity too. It also yields a **rate**, trials-to-criterion, which
is the quantity 1/N makes a claim about and which a pass/fail reading discards.

The width is a **parameter** on the control with the pinned 8 as its default, so every value recorded
by I.0–I.3b still reproduces. Verified before the sweep: **77 common leaves against the committed 048
record, 0 differing, none missing.**

## The rule passes at every width, and the dependence runs the other way

| shape | perturbed units | gap fraction | above floor | passes | median trials to criterion | censored |
|---|---|---|---|---|---|---|
| `8x1` | 8 | 0.890 | 8/8 | **yes** | 4450 | 0 |
| `16x1` | 16 | 0.887 | 8/8 | **yes** | 3200 | 0 |
| `32x1` | 32 | 0.861 | 8/8 | **yes** | 3400 | 0 |
| `64x1` | 64 | 0.860 | 8/8 | **yes** | 3650 | 0 |
| `128x1` | 128 | 0.847 | 8/8 | **yes** | 2650 | 0 |

Four readings, in the order they matter:

1. **Every cell passes the control's own registered rule.** The 8-unit cell reproduces I.1's pass at
   **0.890**, which is the registered stop clause's condition and is what makes the rest
   interpretable.
2. **Nothing is void and nothing is censored.** The analytic reference reaches **0.9986** of the gap at
   every width, so what a frozen random readout can reach never limits a cell and the
   reachability-normalised column tracks the raw one to three decimals. Every seed crosses the
   criterion at every width — **0 censored of 40** — so the fit is over the whole sample rather than
   over its survivors.
3. **Time-to-criterion does not grow with N.** Slope **−0.149** on `log2(trials)` against `log2(N)`,
   bootstrap CI over seeds **[−0.239, −0.061]**, against a prediction of **+1.0** and a registered bar
   of **+0.5**. The interval excludes zero **from below**: the dependence is real and runs *opposite*
   to the prediction. The spread narrows with N too — 3000–11000 trials at 8 units against 2300–4400
   at 128.
4. **What 1/N does cost is in the level, and it is small.** The gap fraction declines monotonically
   — 0.890, 0.887, 0.861, 0.860, 0.847 — a perfect rank correlation (**rho −1.000**) spanning
   **0.043 of the gap across a 16-fold change in N**. Real, and two orders short of what would be
   needed to take a learner below its own frozen floor.

Every derived budget is reported with `extrapolation: true` **and** `is_a_budget_constraint: false`,
because the flag reads the fitted interval rather than the point estimate and this interval does not
exclude zero from above. A negative slope predicts a *smaller* requirement at larger N: 2 944 trials
for the yardstick's 128 units, 2 590 for 302, 2 106 for 1208. Those are what the fit says and they are
**not budgets**; reporting them as budgets would invert the result.

## The depth control: the yardstick's own arrangement passes too

The width grid's flat result left **exactly one** shape difference between the platform the rule
passes and the platform it fails: 128 units as **one** layer of 128 on the control, against **two** of
64 in every failing yardstick arm. Registered as a **dated amendment**, chosen after seeing the slope
and reported with that provenance.

| shape | perturbed units | gap fraction | above floor | median trials |
|---|---|---|---|---|
| `128x1` | 128 | 0.847 | 8/8 | 2650 |
| **`64x2`** — the yardstick's exact arrangement | **128** | **0.853** | **8/8** | **1250** |

The two shapes are **indistinguishable in level**, and the two-layer arrangement reaches criterion
**fastest of every cell in the sweep** — a median of 1250 trials (range 700–1700) against 2650 at one
layer and 4450 at 8 units. Depth does not cost this rule anything. It helps.

So at a **matched perturbation dimension and a matched architecture**, this estimator learns a
one-step task to 0.853 of the floor-to-optimum gap on 8 of 8 seeds, while the same architecture on a
multi-step cell ends **below its own frozen control** (0.393 foods against 2.233, [055](supporting/055-l4-horizon-multistep/details.md)).

## The registered divergence, recorded rather than read away

The pre-registered `not_scale_limited` clause reads "CI contains 0 **and** 128 units passes".

- **Its letter is unmet.** The CI does not contain 0; it excludes 0 on the **opposite** side.
- **Its rationale — "the arithmetic is not the binding constraint" — is satisfied more strongly** than
  a flat result would have satisfied it: not merely no cost in time, but a measured benefit.

Reading a clause loosely toward the verdict it nearly fits is the move this phase keeps catching
itself making, so the harness carries **`opposite_direction`** as its own band rather than folding the
result into either neighbour, and the divergence is recorded as a decision.

## S2's pilot: the rule solves the cell at eight units

The registered pilot ran the two extreme widths, all three arms, on **disjoint seeds 101–104**. Its
job was to establish whether the calibrated hard-food cell — whose 350-step budget was calibrated on
the *connectome* — leaves an MLP room at both ends of the width grid.

| width | perturbed units | PPO foods | PPO clear | frozen foods | **learning foods** | **learning clear** | effect | drift |
|---|---|---|---|---|---|---|---|---|
| 4 | **8** | 19.76 | 90.4% | 0.68 | **19.63** | **92.8%** | **+18.94** | 0.958 |
| 64 | **128** | 19.79 | 94.5% | 3.27 | 4.78 | 4.1% | +1.51 | 1.400 |

**The registered purpose is satisfied and the declared remedy is not applied.** PPO sits at ~19.8 of
20 foods and 90–94% full clear while the frozen controls sit at 0.68 and 3.27 foods and 0% clear, so
the platform has room at both extremes and the cell stands as calibrated.

**And the pilot inverted the registered prior.** At eight perturbed units the rule reaches **19.63
foods and 92.8% full clear — level with PPO** — from a frozen floor of 0.68. At the yardstick's own
128 units it reaches 4.78 against a frozen 3.27, while PPO on that same width reaches 19.79. The two
configurations differ in **one key**, `freeze_updates`, so the update is what produces 19.63 against
0.68; each arm's rule, eligibility, σ, width, depth and activation were re-read from the loaded config
rather than assumed.

**Drift separates the two regimes mechanistically**: **0.958** of the weight's own norm where the rule
solves the cell, **1.400** where it barely leaves its floor — against the **1.28–1.31** I.3b measured
at 128 units on the C3 cell. At high N the rule writes *more* and achieves *less*.

### What the pilot does not license

**No significance claim, and none is available.** At four pairs the smallest p an exact one-sided
paired test can return is **0.0625**, above the 0.05 level, so the floor half of each capability gate
**cannot fire whatever the data does**. Both widths read `capability_undecided` rather than failed, and
every effect size above is **descriptive**. The registered campaign runs eight seeds, where that floor
is 0.0039.

### Why this does not contradict S1

A one-step task has a **single credited decision**, so the estimator's dimension barely matters there —
which is what S1 measured, and it found almost no cost. Over 350 steps the per-unit credit is diluted
across units **and** time. The two sweeps together say the dimension bites **only when the horizon
does**, which is also why I.3's delay result and I.3b's drift measurement pointed at the horizon while
this one points at the dimension: they are the same constraint seen along two axes.

The registered prior was wrong, and specifically so. It read: "*A rule starved of signal per trial
moves little and slowly; this one writes 1.28–1.31× its own weight norm in a direction that makes the
policy worse … no budget repairs a sign.*" On a multi-step cell the drift at low N is **0.958 and the
policy is right**, so the 128-unit drift was not a sign error — it was what an over-dimensioned
estimator's noise looks like.

### Two instrument defects the pilot found, fixed before the campaign

Both are the shape that turns a sample size into a finding:

- **Drift took its seed set from a module constant.** Reusing I.3b's measurement meant reading seeds
  1–8, so on the pilot's disjoint 101–104 it found no pair and reported nothing available — the right
  answer for the wrong reason, and a silent mismatch for any campaign not on seeds 1–8. The seed set
  is now a parameter, and an unreadable pair reports **unavailable** rather than a drift of 0.0, since
  0.0 is the claim *the policy did not move*.
- **The capability gate reported `fails` when its test could not fire.** It now returns
  `underpowered` with the smallest reachable p, and a width whose gate is undecided is recorded as
  neither a null nor uninterpretable.

## The campaign: a monotone collapse from 90.7% to 3.0% full clear across the dimension

120 runs, five widths × three arms × seeds 1–8, 3000 episodes, 5022 s wall clock. All 120 succeeded.

| perturbed units | learning foods | learning full clear | frozen foods | PPO foods | PPO clear | shift | q | seeds favouring | drift | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **8** | **19.64** | **90.7%** | 1.69 | 19.69 | 87.6% | **+17.94** | 0.007 | **8/8** | 0.91 | `beats_control` |
| 16 | 18.83 | 79.2% | 1.31 | 19.94 | 96.8% | +17.53 | 0.007 | 8/8 | 1.07 | `beats_control` |
| 32 | 15.72 | 53.6% | 3.89 | 19.93 | 96.8% | +11.83 | 0.007 | 8/8 | 1.27 | `beats_control` |
| 64 | 8.34 | 12.9% | 3.54 | 19.92 | 97.0% | +4.80 | 0.055 | 5/8 | 1.38 | `no_improvement` |
| **128** | **4.23** | **3.0%** | 2.00 | 19.72 | 94.6% | +2.22 | 0.024 | 7/8 | 1.39 | `beats_control` |

**The rule matches PPO at eight perturbed units.** 19.64 foods of 20 and **90.7% full clear** against
PPO's 19.69 and 87.6%, from a frozen control at 1.69 foods. The same rule at the yardstick's 128 units
reaches 4.23 foods and **3.0%**.

**The collapse is monotone across the whole grid** — foods 19.64, 18.83, 15.72, 8.34, 4.23; full clear
90.7%, 79.2%, 53.6%, 12.9%, 3.0% — a perfect rank correlation of the shift against the dimension
(**rho −1.000**).

**No width is capacity-limited and none is uninterpretable.** The capability arm reaches 19.69–19.94
foods and 87.6–97.0% full clear at *every* width, the narrow end included, so the small-N end of the
grid was never in doubt and the declared one-layer alternative was not needed. Four hidden units per
layer, below the input dimension, is enough for both optimisers on this cell.

**Drift rises monotonically with the dimension: 0.91, 1.07, 1.27, 1.38, 1.39.** At eight units the rule
moves about its own weight norm and the resulting policy is *right*; at 128 it moves more and the
policy is wrong. I.3b measured 1.28–1.31 at 128 units and read it as a signal pointing the wrong way.
It is better read as **an over-dimensioned estimator's noise**: the same magnitude of writing is
productive at low N and destructive at high N.

### Two things in this table that are not clean, stated as they are

**The 64-unit cell's label is ragged.** Its shift (+4.80) is more than twice the 128-unit cell's
(+2.22), yet its q is worse — 0.055 against 0.024 — because only 5 of 8 seeds favour learning there
against 7 of 8 at 128. A paired rank test responds to the **consistency of the sign, not the size of
the shift**, which is the property I.3b's protocol warned about in advance. The effect sizes are
monotone; the verdict labels are not, and the raggedness is the test's, not the data's.

**At 128 units on this cell the rule is slightly above its frozen control**, where on the 2400-step C3
cell [I.3b](supporting/055-l4-horizon-multistep/details.md) found it **below** (0.393 against 2.233).
That is consistent rather than contradictory: the per-unit credit is diluted by units **and** steps, so
128 units is merely useless over 350 steps and actively harmful over 2400.

## Verdict: `mixed`, and that is the accurate label

| half | reading |
|---|---|
| **S1** | `opposite_direction` — the rule passes at every width to 128, and time-to-criterion *falls* with the dimension (slope −0.149, CI [−0.239, −0.061]) |
| **S2** | `rescued` — the rule solves the cell at 8 units and collapses monotonically to 3.0% full clear at 128 |

Neither registered row fits. `scale_limited` required a **positive** S1 slope *and* 128 units failing
the control; both are false. `not_scale_limited` required a flat S1 *and* no rescue; the second is
false. The registration anticipated exactly this and fixed the treatment in advance: a mixed reading is
**recorded as mixed with both halves stated**, not resolved toward whichever verdict is nearer.

Stated as one sentence: **the perturbation dimension does not bind on a one-step task and binds
decisively on a multi-step one.** That is a better result than either registered row, because it names
the interaction rather than the axis — the per-unit credit is diluted by the number of perturbed units
*times* the number of decisions the reward has to be shared over, and only the product matters.

It also explains the two halves of the phase's record in one mechanism, which neither I.3's horizon
finding nor this dimension finding does alone. They are the same constraint measured along two axes.

## What this licenses, and what it does not

**Licensed.** [Logbook 059](059-7a-shipment.md) fixed a condition in advance: "*If the programme
produces a rule that learns the hard-food cell, B.5, B.1, B.4 and B.4b become askable and are
re-registered fresh on the block-V cells with the block-V bar.*" **That condition is met.** The rule
learns the block-V hard-food cell at 8 perturbed units, level with PPO, on 8 registered seeds.

The obvious next registration is the one 7b's gate actually asks for and which **has not been run**:
the **wild-type-versus-rewired-null wiring contrast under this rule at a working dimension**, held to
block V's registered ≥ 20% bar. Until that runs, 7b's gate stands as written — its letter still
requires a local rule beating the null, and no such contrast exists.

**Not licensed: anything about the connectome.** It perturbs **302 neurons at each of four settling
steps — 1208 draws per decision**, far beyond the failing end of this grid, where 128 units already
sits at 3.0% full clear. This result does **not** predict that the connectome works; read plainly it
predicts the opposite at its present dimension. What it makes worth building is a way to *reduce* the
connectome's perturbation dimension — perturbing a subset of units rather than all 302 — which the
substrate does not currently support, since `node_noise` is applied to every pre-activation.

**No committed verdict is changed.** B.8's shipped negative recorded what had been measured, and every
arm it summarised ran at 128 or 302 perturbed units. What this result shows is that the *scope* was
narrower than the phrasing implied and the cause was a dimension nobody varied. Any re-read of those
results is a **new registration**, as this change registered in advance, and not a relabelling.

### What this may not be cited as

- **A result about the connectome**, or about any substrate at a dimension above 32 units.
- **A result about any other cell.** One cell: food-only klinotaxis, 350 steps, target 20, on the
  calibrated continuous-2D substrate. The 2400-step C3 cell still fails at 128 units.
- **A claim that the rule is a good gradient estimator.** Its measured alignment is +0.263 and nothing
  here changes it; what changes is the reading of what that alignment is sufficient for.
- **A measurement of the 1/N exponent.** Five widths, eight seeds, and S1 shows the law does not even
  hold in the predicted direction on a one-step task.
- **A wiring result.** No rewired null was run under this rule at any dimension.

## Conclusions

- **The rule solves a multi-step foraging task at eight perturbed units, level with PPO** — 19.64 foods
  of 20, 90.7% full clear, 8/8 seeds, from a frozen floor of 1.69.
- **Performance collapses monotonically with the perturbation dimension**, to 3.0% full clear at the
  128 units every failing yardstick arm ran, with drift rising 0.91 → 1.39 as it does.
- **On the one-step control the dimension costs almost nothing** — every width passes, and
  time-to-criterion falls rather than rises — so the constraint is the **product** of dimension and
  horizon, not either alone.
- **Verdict `mixed`**, with both halves stated, under the clause registered for exactly this outcome.
- **059's re-registration condition is met**, and the wiring contrast under a working local rule is the
  next registration. **7b's gate still stands as written.**
- **No committed verdict changed.**

## Next Steps

- [ ] The wiring contrast — wild type against its degree-preserving rewired null — **under this rule at
  8 perturbed units**, on the block-V cell, against the registered 20% bar. This is what 7b's gate asks
  for.
- [ ] A reduced-perturbation-dimension mechanism for the connectome, without which no connectome arm
  follows from this result.
- [ ] Re-registration of B.5, B.1, B.4 and B.4b on the block-V cells at a working dimension, per 059.
- [ ] R.2 (e-prop) is **not** retired: it remains the named fallback if the wiring contrast fails.

## Data References

- S1, the depth control and S2: [`supporting/060-l4-perturbation-scale/`](supporting/060-l4-perturbation-scale/) —
  [`launch.md`](supporting/060-l4-perturbation-scale/launch.md) (protocol, the pilot outcome and the
  dated amendment), [`details.md`](supporting/060-l4-perturbation-scale/details.md),
  [`scale.json`](supporting/060-l4-perturbation-scale/scale.json),
  [`s1-per-seed.csv`](supporting/060-l4-perturbation-scale/s1-per-seed.csv),
  [`s2-per-seed.csv`](supporting/060-l4-perturbation-scale/s2-per-seed.csv).
- Campaign directories `campaigns/perturbation-scale-pilot/` and `campaigns/perturbation-scale/` are
  gitignored; every figure above is in the committed records.
