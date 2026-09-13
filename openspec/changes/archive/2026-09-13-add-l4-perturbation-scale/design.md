# R.1 — the perturbation dimension: design

## The confound, in the committed record

Node perturbation forms its eligibility from what each unit's own noise did to a scalar outcome. With
N perturbed units the per-trial estimate of the reward gradient has signal-to-noise ~1/√N, and the
trials needed for a given amount of progress grows roughly as **N** (Werfel, Xie & Seung 2005). The
dimension therefore belongs beside every result the rule produced, and it has never been reported:

| platform | plastic layers | perturbed units | draws per decision | result |
|---|---|---|---|---|
| the one-step control | 1 (`Linear(K, 8)`) | **8** | 8 | **passes** — 89.0% of the floor-to-optimum gap (I.1) |
| the MLP yardstick | 2 (`64 → 64`) | **128** | 128 | fails, and below its own frozen control (I.3b) |
| the connectome | 1 (302 neurons) | **302** | **1208** — every unit at each of 4 settling steps | fails on every multi-step task (040–047) |

The pattern the phase has read as "one-step works, multi-step fails" is equally consistent with "8
units works, 128 and 302 do not". Nothing in the record separates them, because no experiment ever
varied the width.

## S1 — the arithmetic, where the rule works

The committed one-step contextual-association control, unchanged in every other respect, at
`HIDDEN ∈ {8, 16, 32, 64, 128}`. Seeds 1–8, 20 000 trials, `plasticity_node_noise 0.2`,
`plasticity_rate 1e-3`, `trace_decay 0.9`, homeostasis on, action noise `exp(-1)` — I.1's passing
configuration, with width as the only axis.

**Why this platform carries the arithmetic.** The task is solvable by 8 units, so no width in the grid
lacks capacity and every further unit adds only noise. N is isolated in a way it can never be on a
behavioural cell, where width is capacity too.

### What is measured

1. **The registered pass rule**, per width: mean score at least halfway from the closed-form cue-blind
   floor to the closed-form optimum, on at least 7 of 8 seeds. This is the control's own bar, not a
   new one.
2. **Trials-to-criterion**, per seed: the first trial at which a trailing 100-trial mean crosses that
   halfway threshold. This is the quantity 1/N makes a claim about; a pass/fail reading discards it.
   The trailing mean is computed at **every trial**, so the reported time is the trial the crossing
   happened on. Non-overlapping block means would be a different statistic — they can only report the
   end of the block a crossing fell inside, quantising every criterion time to the block length.
3. **The analytic reference at every width.** The floor and the optimum are closed-form properties of
   the *task*, but what a network with a **frozen random readout** can reach is a property of the
   *width*. The analytic arm measures it. A width where the reference itself misses the pass bar is
   **void at that width** — the existing control's own void clause, applied per cell.
4. **Reachability-normalised score**: the rule's gap fraction divided by the analytic arm's gap
   fraction at the same width, reported beside the raw fraction. The raw fraction is what the earlier
   results quote; the normalised one is what is comparable across widths.

### The 1/N test

Ordinary least squares of `log2(trials_to_criterion)` on `log2(N)` over the per-seed values, with a
bootstrap CI over seeds. The prediction is a slope of **+1**; the registered bar for "depends on N in
the predicted direction at all" is a slope of **≥ 0.5 with a CI excluding 0**. A Spearman of the
per-width medians against N is reported as description only: at five widths it has almost no power,
and the per-width gates carry the verdict.

**Censoring is reported, not absorbed.** A seed that never crosses the threshold inside 20 000 trials
has no trials-to-criterion. Those seeds are **excluded from the fit and counted in the record**, per
width. V.3 came within one review comment of a false null from exactly this: gates reading positive
while 0% of seeds crossed the threshold being fitted.

### The derived budget, and what it is not

If the fit holds, it says what budget 128 and 1208 draws per decision would need against what the
panels spent. That number is an **extrapolation from a five-point fit read outside its range**, is
labelled as one wherever it appears, and is never reported as a measurement. It is given for both
readings of the connectome's dimension — 302 units and 1208 draws per decision — because the two
differ by 4× and nothing in the record establishes which the arithmetic tracks.

## S2 — the rescue, where the rule fails

The MLP yardstick on the calibrated hard-food cell — block V's cell, so a rule that learns it is
directly comparable to the **+23.5%** PPO achieved there — at
`actor_hidden_dim ∈ {4, 8, 16, 32, 64}` with `num_hidden_layers: 2` and `plastic_layers: hidden`:
**8, 16, 32, 64 and 128 perturbed units**, the same grid as S1. Seeds 1–8, 3000 episodes.

### Three arms per width

| arm | `learning_rule` | `freeze_updates` | σ | what it is for |
|---|---|---|---|---|
| learning | `three_factor`, `node_perturbation` | false | **0.2** | the measurement |
| frozen | `three_factor`, `node_perturbation` | **true** | **0.2** | the do-nothing floor **at that width** |
| capability | `ppo` | false | — | whether the width can hold a competent policy at all |

Each width needs **its own** frozen control, because what the perturbation costs a policy is not
constant in the width being varied, and a control from another width would not be the right null —
I.3b's reason, applied to a different axis.

**The frozen arm perturbs.** It carries the same σ 0.2 as the learning arm and freezes only the
*update*, exactly as I.3b's controls did. This matters for what the contrast means: the cost the
perturbation imposes on the policy is **present in both arms and cancels**, so S2 measures the
benefit of the update alone and not the net effect of switching perturbation on. That is the right
null here — 1/N is a claim about the quality of the gradient estimate, which is precisely what a
matched-σ pair isolates — and it is stated because the opposite reading is the natural one.

**The capability arm shares the architecture it certifies.** `activation: tanh`, the width, the layer
count and `initial_log_std: -1.0` are all identical to the plastic arms; `entropy_coef` stays at the
committed base's 0.05 on all three, so entropy is not a moving part. A relu arm, or one at another
width, would establish capacity for a different network. It is **not** the committed calibrated MLP
arm for this cell — that one is relu at width 64 — and may not be cited as one.

The capability arm exists because **the prediction runs toward small N**, which is exactly where
capacity runs out. Without it, a width-4 null is indistinguishable from a width-4 refutation. A width
whose PPO arm misses the registered floor is reported **uninterpretable**, not as a null. PPO is a
capability floor and **not a ceiling for the rule**: the rule is never scored against it.

### The reading

**Plateau-tail mean foods**, I.2's graded family, learning against its own frozen control, paired by
seed, one-sided, BH-FDR across the five widths. Full-clear success is recorded and is expected to be
undefined or at the floor on every plastic arm; where it is, the record says so rather than reporting
a null.

**A shift counts only if it is significant *and* both:**

- at least **1.0 foods** of the cell's 20 — I.3b's 0.5-of-10 bar in proportion; and
- at least **10%** of that width's own PPO-minus-frozen gap, so a width whose reachable gap is tiny
  cannot clear the bar on a shift that means nothing.

**Drift**, per width: relative weight distance from the frozen control. This separates *starved of
signal and sitting still* — which is what 1/N predicts at large N — from *writing a great deal in a
worsening direction*, which is what I.3b measured at 1.28–1.31× the weight's own norm and which no
budget fixes.

### The pilot

The 350-step budget and the 20-food target were calibrated on the **connectome**. Nothing establishes
that they leave an MLP room at either extreme of the width grid. Before the campaign: widths 4 and 64,
all three arms, **seeds 101–104**, disjoint from the registered 1–8.

**Declared alternative at the small-N end.** Width 4 sits **below the input dimension** —
`food_chemotaxis` + `proprioception` is six to seven features — and it is the only grid point
producing **8 perturbed units**, the count at which the rule passes its control. If its capability arm
fails the floor, the sweep would lose exactly the comparison it exists for. The declared alternative
is `num_hidden_layers: 1` **at width 8**: also 8 perturbed units, with no sub-input bottleneck. It is
used **only** if width 4 fails its capability gate, its one-layer architecture is stated wherever its
number appears, and it is reported as an added point rather than as a replacement for width 4, whose
failure is recorded.

**Declared remedy, in order, if the platform has no room** — the capability arm at the ceiling, or the
frozen arm already at it, or both extremes at the floor: fall back to the committed C1 food-only cell
(`max_steps: 800`, `target_foods_to_collect: 10`), where the calibrated MLP arm exists and the
yardstick's own committed value sits at 0.35 of 10. The fallback is a **platform** change and is
recorded as an amendment with the pilot table that forced it; the block-V comparability is given up
and the record says so.

## Outcomes, fixed before the run

| verdict | S1 | S2 | what follows |
|---|---|---|---|
| `scale_limited` | slope ≥ 0.5, CI excluding 0, **and 128 units fails the control's own pass rule** | at least one width beats its frozen control by both minima **and** the trend across widths is in the predicted direction — an isolated win against the trend is not a scale story and is recorded as mixed | the phase's failures are **located**: a scale property, not a mechanism one. A reduced-perturbation connectome variant becomes the obvious registration, and every panel negative is re-read as under-budgeted rather than refuted. The re-read is a **new registration**, not a re-labelling of committed verdicts |
| `arithmetic_only` | slope ≥ 0.5, CI excluding 0 | no width beats its frozen control | the arithmetic is real **and does not rescue the task**. The multi-step failure is then a **second, independent** defect, and I.3b's worsening-direction drift is the standing candidate. Node perturbation closes as a family member; **R.2 (e-prop) proceeds**, carrying a perturbation-dimension note. This is the expected outcome |
| `not_scale_limited` | CI contains 0 **and** 128 units passes the control | reported, and cannot be read as being about scale | the 1/N arithmetic is not the binding constraint on this implementation. The record's 8-vs-128 confound closes the uninteresting way and **every existing negative keeps its reading** |

Mixed readings — a slope below the bar with 128 failing, a positive S2 with a flat S1, or a rescue
whose trend runs against the prediction — are recorded as **mixed with both halves stated**, not
resolved toward whichever verdict is nearer.

**Both S2 conditions are evaluated in one place.** The trend requirement in the `scale_limited` row is
enforced by the combined verdict rather than folded into S2's own `rescued` label, which asks the
narrower question "did any width beat its control". So an isolated success with the trend running the
other way produces the same non-`scale_limited` verdict whether it is read from the table, from the
harness or from the tests.

**An S2 that returns `void` is an absence, not a negative.** Where no width is interpretable — every
capability gate failed or undecided — S2 has tested nothing and cannot supply the "no width beats its
frozen control" half of `arithmetic_only`. The registered treatment is the one already fixed for a
platform-limited S2: **S1 carries the result alone.**

### Stop clauses — void until found, not results

- **`HIDDEN = 8` does not reproduce I.1's pass.** The platform has drifted since 2026-09-10; nothing
  in the sweep is interpretable until that is found.
- **The analytic reference fails at a width.** That width is void; if it fails at `HIDDEN = 8`, the
  whole sweep is.
- **The pilot finds no room after the declared remedy.** S2 is reported as platform-limited and S1
  carries the change alone. S1 does not depend on S2 and is not weakened by its absence.

## Honest prior

**I expect `arithmetic_only`, and S1's 128-unit cell is the genuinely open question.**

The reason to doubt a rescue is I.3b's drift number. A rule starved of signal per trial moves little
and slowly; this one writes **1.28–1.31× its own weight norm** in a direction that makes the policy
worse, at every eligibility horizon. That is not the signature of too little signal — it is the
signature of a signal pointing the wrong way, and no budget repairs a sign.

The reason to run it anyway is that the record genuinely cannot say whether the rule was ever given a
workable scale, S1 costs minutes, and a confirmed slope changes what the phase's negatives mean even
if no width rescues the cell: it converts "the rule fails on multi-step tasks" into "the rule fails on
multi-step tasks *and* was run at 16–151× the dimension its one success was measured at", which is a
materially different sentence to publish.

## Risks

- **The sweep cannot separate units from weights.** In a fully-connected layer the weight count is
  proportional to the width, so a slope in N is equally consistent with a per-**weight** law — which
  is what weight perturbation obeys — rather than the per-**unit** law node perturbation predicts.
  Separating them needs a substrate where the two counts move independently, which this platform is
  not. What the sweep does separate is **scale from task**, which is the question asked; no claim is
  made that the exponent is attributable to the unit count specifically. On S2 the weight count grows
  faster than the width (roughly `in·W + W²`), which is a second reason its trend is read as coarse.
- **Five widths is a weak trend.** Accepted: the per-width gates carry the verdict and the Spearman is
  labelled descriptive.
- **Eight seeds.** The same power limitation every block-I result carries. The paired test fires on the
  consistency of the sign, which is why both effect-size minima are registered.
- **Width is capacity in S2.** Controlled by the PPO arm, with uninterpretable as an available answer.
- **The connectome's dimension is ambiguous** — 302 units, 1208 draws per decision. Both are reported
  and the extrapolation is given for both.
- **The grid tops out at 128**, the yardstick's own width, and does not reach the connectome's. That is
  deliberate: 302 units on the MLP would be a different architecture, and the connectome itself is out
  of scope until S1 and S2 both move.

______________________________________________________________________

## Amendment, 2026-09-13 — the depth control, added after S1's width grid returned flat

S1's registered grid ran and **the rule passes the control at every width, 8 through 128**, with
time-to-criterion showing no growth in N (slope **−0.149**, CI [−0.239, −0.061] — the dependence
exists and runs the *other way*). The registered sweep therefore closes its own question, and in doing
so it leaves **exactly one shape difference** between the platform the rule passes and the platform it
fails: the control at 128 units is **one** plastic layer of 128; the yardstick is **two** of 64.

Perturbed units match at 128. Depth does not. So a matched-unit comparison of the two shapes is the
cheapest remaining way to separate *shape* from *task*, and the registered sweep cannot make it.

**Added**: one cell, the yardstick's exact arrangement — `hidden 64, layers 2`, hidden-only plasticity,
**128 perturbed units** — on the same one-step control, same seeds, same 20 000 trials, same σ, scored
by the same pass rule and reported beside the matched one-layer cell.

**What it can settle.** If it passes, then at a matched unit count *and* a matched architecture the
estimator learns a one-step task while failing every multi-step one, and the failure localises to
multi-step credit assignment — which is what I.3's delay result predicted in advance (the pinned
`trace_decay 0.9` falls below the cue-blind floor by twenty steps of delay, on episodes of 350 to
2400). If it fails, depth is implicated and the width sweep was asking the wrong question about the
right platform.

**What it cannot.** The control's task is one-step by construction. A pass says the arrangement is not
the problem; it says nothing about how the rule behaves over a long episode, which is S2's question
and 055's finding.

Recorded as an amendment rather than as registered in advance: it was chosen **after** seeing S1's
slope, and its result is reported with that provenance attached.
