# 060: The Perturbation Dimension — the Rule Solves a Multi-Step Cell at Eight Units and Fails at 128 (7a-ii R.1 / Phase 7)

**Status**: PENDING — S1 and the S2 pilot are complete; the registered 120-run campaign is running.
This record is written up to that point and its verdict is not yet assigned.

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
