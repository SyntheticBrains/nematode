# 061: The Connectome's Perturbation Dimension Is Not What Stops the Rule (7a-ii R.1c / Phase 7)

**Status**: completed — **`not_reducible`**. The mechanism R.1b needed is built: the per-unit
perturbation is now restrictable to a declared set, and because the eligibility is
`pre ⊗ perturbation`, restricting the draw restricts what can be credited. Recon then found that the
readout mean-pools **only the 39 motor neurons**, so at settling step `s` a unit can influence the
action only from within `depth − s` hops — and on this graph at depth 4, **672 of 1208 draws per
decision are causally connected while 536 are not, and every one of them is credited**. Masking exactly
those, and then restricting further to 109 and 39 units, **does not make the rule learn**: no set beats
its own frozen control by the registered minima, and **none reaches competence** — full clear never
exceeds 0.08% against PPO's 19.31 foods of 20 on the same cell. **R.1b stays blocked.** What the
campaign does establish is a mechanism: **credited synapses drift 1.37–1.38× their own norm at every
set from 302 units down to 39**, matching the 1.28–1.31 measured on the MLP, while **excluded synapses
drift 0.013–0.016**. So the rule writes more than its own weight norm on whatever it credits, and the
dimension changes that not at all. Two knobs calibrated on this substrate for the first time — σ and
the action noise — are both nulls, leaving the learning arm pinned at **4.1–4.4 foods** throughout. The
invariance across three independent axes is what points at a **structural** limit, and one is measured
here: the tensor PPO changes most by relative norm on this cell is the **motor readout**, which the rule
is forbidden from touching. That becomes **R.1d**.

**Branch**: `feat/l4-reduced-perturbation`.

**Date**: 2026-09-14.

**OpenSpec change**: `add-l4-reduced-perturbation` (extends `connectome-substrate`: the perturbed unit
set is declarable and travels with the run; and `plasticity-evaluation`: a perturbation that cannot
reach the scored outcome is not counted as exploration).

## Objective

Build the mechanism R.1b needs and ask whether the connectome's **perturbation dimension** is what stops
the rule learning there.

[R.1](060-l4-perturbation-scale.md) found the rule solving a multi-step foraging cell at **8 perturbed
units** and collapsing to **3.0% full clear at 128**. The connectome perturbs **302 neurons at each of
four settling steps — 1208 draws per scored decision** — and there was no way to lower it: `node_noise`
applied to every pre-activation. So no connectome arm followed from R.1, and **R.1b, the wiring contrast
7b's gate asks for, was blocked on this.**

## A substrate measurement that stands whatever the arms do

The motor readout mean-pools **only the 39 VB/DB/VA/DA motor neurons**, and the eligibility is
`E ← decay·E + M_chem ∘ (h_prev ⊗ perturbation)` — so a unit's perturbation writes eligibility on every
synapse onto it whether or not it can reach the readout. At settling step `s` it can reach the readout
only from within `depth − s` hops. On the Cook 2019 graph at `forward_pass_depth: 4`:

| settling step | hop budget | units that can still reach the readout |
|---|---|---|
| 1 | 3 | 277 of 302 (91.7%) |
| 2 | 2 | 247 of 302 (81.8%) |
| 3 | 1 | 109 of 302 (36.1%) |
| 4 | 0 | **39 of 302 (12.9%)** |

**672 of 1208 draws per decision are causally connected. The other 536 — 44.4% — cannot change the action
at all, and every one of them is credited.** The largest budget is 3, so the **25** units at four hops or
more can never contribute at any step: 1 at four hops, 7 at five, 7 at six, 3 at seven or more, and 7
unreachable in the directed chemical graph.

This is a property of the substrate at a given depth, not a result about the rule, and it holds
regardless of what the learning arms do.

## The mechanism

A declarable perturbation set, five names, each a boolean row per settling step. `full` is the default and
is built **without consulting the graph**, so every recorded plastic connectome result reproduces
unchanged. The noise is **drawn and then masked**, so the random stream does not depend on the declared
set: two sets at one seed differ only in which draws are *used*, which is what makes them comparable.

| set | units | adaptable synapses | draws/decision | causally connected |
|---|---|---|---|---|
| `full` — today's behaviour | 302 | 3709 | 1208 | 672 |
| `causal` — per-step reach mask | 277 | 3538 | 672 | 672 |
| `hop1` | 109 | 1476 | 436 | 366 |
| `motor` | 39 | 323 | 156 | 156 |
| `motor_last` | 39 | 323 | 39 | 39 |

Every figure is asserted by test against the built masks. The design's first table gave `causal` as 302
units and 3709 synapses; that was **corrected before any arm ran**, because the 25 never-reaching units
are perturbed at no step and so their 171 incoming synapses are never credited either. "No signal lost"
survives; "every synapse stays adaptable" does not.

**A masked synapse stays put only because homeostasis cancels an unconditional decay.** The rule writes
`− rate · weight_decay · weight` to every plastic weight, trace or no trace. Measured at the recipe's own
rates with every trace held at zero, the homeostatic rescale holds each unit's incoming **norm exactly**
and its direction to a cosine of **1 − 3e-05** across a run's ~1.05M updates, leaving float32 round-off —
about 1.9e-02 on the largest single weight. **Without homeostasis it is decay**: 2.0% of the norm per
20 000 updates. A config validator refuses the combination, and the test asserts the norm in both
directions.

## Two knobs calibrated on this substrate for the first time, and both are nulls

Neither σ nor the action noise had ever been calibrated on the connectome: both came from the one-step
control by way of the MLP.

**σ (`plasticity_node_noise`), at the `motor` mask:**

| σ | learning foods | frozen | gap | full clear |
|---|---|---|---|---|
| 0.05 | 4.321 | 1.732 | +2.589 | 0.17% |
| **0.1** | **4.361** | 1.731 | **+2.630** | 0.03% |
| 0.2 — the carried value | 2.989 | 1.883 | +1.106 | 0.00% |

At σ 0.2 the learning arm is **31.5% lower than at σ 0.1** (2.989 against 4.361 — equivalently, σ 0.1
is 45.9% higher), so the campaign runs at **0.1**.

**Action noise (`initial_log_std`), at σ 0.1:**

| `initial_log_std` | action std | learning | frozen | gap | full clear |
|---|---|---|---|---|---|
| **−1.0** — the recipe's value | 0.368 | **4.361** | 1.731 | **+2.630** | 0.03% |
| −0.5 | 0.607 | 4.056 | 2.452 | +1.604 | 0.00% |
| 0.0 | 1.000 | 4.154 | 3.254 | +0.900 | 0.00% |

The pinned value stands. Wider action noise lifts the **frozen** arm while leaving the learning arm flat,
so the gap collapses — floor food bought by wandering, which is what the registered guard was written to
catch. It also resolves a discrepancy: 058's frozen arm collects 3.82 foods against these arms' 1.73
because of **action noise**, not perturbation, since at std 1.0 the frozen arm reaches 3.25.

### The invariance is the part that matters

**The learning arm sits between 2.4 and 4.4 foods under everything tried** — 4.06–4.36 across the two
knobs at the `motor` mask, 2.99 at the σ the calibration rejected, and 2.41–3.84 across the campaign's
five sets — σ across a fourfold range, action
noise across 2.7× — with full clear never leaving ~0% against PPO's 19.31 foods on the same cell. Two
independent sweeps hitting one ceiling is the signature of a **structural** limit rather than a
hyperparameter one.

## What the rule is structurally forbidden from doing

The connectome's `plastic_weights` is **`w_chem` alone**. `food_gains` and `readout` live in the
optimiser's parameter list, which is used **only under PPO** — so under the rule **both ends of the
network are frozen**. After just 300 PPO episodes on this cell:

| tensor | relative change | cosine to init | writable by the rule |
|---|---|---|---|
| `readout` (2×4, motor classes → action) | **0.783** | +0.865 | **no** |
| `w_chem` (3709 chemical synapses) | 0.486 | +0.899 | yes |
| `food_gains` (sensory projection) | **0.177** | **+0.985** | no |

**Of the three measured tensors the motor readout shows the largest relative norm change, and it is
the one the rule cannot touch.** The frozen sensory
projection — the first candidate — costs little, since PPO barely rotates it. The readout starts from an
anatomical prior (speed as the B-vs-A contrast, turn as D-vs-V, unit-normed) and PPO still moves it 78% in
a tenth of a run.

Relative change across an eight-entry matrix and 3709 synapses is **not** a clean importance measure, so
this is a hypothesis rather than a conclusion. It is registered as **R.1d** and not pursued here, whose
question is the dimension. The safe form of the test is seeding the readout from PPO and freezing it
there, **not** making it plastic: Logbook 040 recorded a 96% forager collapsing to zero within three
episodes once its readout learned.

## The campaign

Ten arms — five declared sets × (learning, frozen) — seeds 1–8, 3000 episodes, at the calibrated
σ 0.1 and the recipe's `initial_log_std: -1.0`. 80 runs, all succeeded, 9389 s wall clock at 15.8×.

| set | units | synapses | draws | learning | frozen | shift | q | favouring | full clear | drift credited | drift excluded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full` | 302 | 3709 | 1208 | 2.413 | 2.796 | −0.384 | 0.727 | 3/8 | 0.00% | **1.37** | n/a |
| `causal` | 277 | 3538 | 672 | 2.891 | 2.797 | +0.094 | 0.723 | 3/8 | 0.00% | **1.37** | 0.013 |
| `hop1` | 109 | 1476 | 436 | 2.756 | 2.882 | −0.126 | 0.723 | 4/8 | 0.02% | **1.38** | 0.016 |
| `motor` | 39 | 323 | 156 | 3.751 | 3.150 | +0.601 | 0.312 | **7/8** | 0.02% | **1.38** | 0.016 |
| `motor_last` | 39 | 323 | 39 | 3.843 | 3.409 | +0.435 | 0.312 | 6/8 | 0.08% | **1.38** | 0.016 |

**No set beats its own frozen control** by the registered minima — the binding bar is ~1.6 foods and the
largest shift is +0.601 at q = 0.312. **No set reaches competence**: full clear never exceeds 0.08%
against PPO's **19.31 foods** on this exact cell and substrate over 32 seeds. Verdict
**`not_reducible`**, and **R.1b stays blocked**.

### The mechanism the drift split establishes

This is the campaign's durable finding, and it exists because a column of `nan` had to be explained.

| set | credited-synapse drift | excluded-synapse drift |
|---|---|---|
| `full` (302 units) | 1.37 | — no unit is excluded |
| `causal` (277) | 1.37 | 0.013 |
| `hop1` (109) | 1.38 | 0.016 |
| `motor` (39) | 1.38 | 0.016 |
| `motor_last` (39) | 1.38 | 0.016 |

**The rule writes 1.37–1.38× the weight's own norm on whatever it credits, at every dimension from 302
units to 39.** That is the same signature I.3b measured on the MLP yardstick (1.28–1.31), reproduced on
a different substrate across a 7.7-fold change in perturbation dimension. **The dimension changes the
drift not at all.**

And **excluded synapses drift 0.013–0.016** — one to two per cent. The design claimed, from a bench
measurement, that the homeostatic rescale leaves them as jitter rather than decay; that claim now holds
on the real substrate at the real scale, which is why it was registered as a measurement instead of an
argument.

### What is suggestive and does not reach the bar

The two reduced sets carry the highest learning levels (3.751, 3.843) and the only meaningful positive
shifts (+0.601 on **7 of 8** seeds, +0.435 on 6 of 8), and the trend across draws per decision runs in
the predicted direction — **rho −0.800, p = 0.104**, descriptive at five points. So the dimension may do
something. But at **q = 0.312**, roughly a third of the required effect, and with full clear at 0.02%,
it is nowhere near what R.1b needs, and the registered reading is the one that stands.

## Three axes, one ceiling

| axis | range tried | learning arm |
|---|---|---|
| σ (`plasticity_node_noise`) | 0.05 → 0.2, fourfold | 4.321, 4.361, 2.989 |
| action noise (`initial_log_std`) | −1.0 → 0.0, 2.7× in std | 4.361, 4.056, 4.154 |
| perturbation dimension | 1208 → 39 draws, 31-fold | 2.413 → 3.843 |

**The learning arm never leaves 2.4–4.4 foods of 20, and full clear never leaves ~0%.** Three
independent axes, one ceiling — which is the signature of a structural limit rather than a
hyperparameter one. The credited-synapse drift being invariant at ~1.4× across all three says the same
thing from the mechanism side: the rule is not starved of signal at any setting, it is writing a great
deal in a direction that does not help.

## What this licenses, and what it does not

**Licensed.** The mechanism exists and is reusable: any future connectome arm can declare its perturbed
set, and every run records the set, its units, its adaptable synapses, its draws per decision and how
many of those can reach the readout. The causal-reach measurement stands on its own: **44.4% of this
substrate's perturbation at depth 4 is credited against an outcome it cannot influence.**

**Not licensed: R.1b.** Block V's contrast is on **time to competence**, and no arm here becomes
competent, so that time is undefined for every one of them. This is the gap between the registered
verdict's condition and its stated consequence, surfaced before the campaign rather than after it:
beating a frozen control by the registered minima is not learning the cell, and only the latter makes
the wiring contrast measurable. **7b's gate is untouched.**

**The live hypothesis is R.1d**, registered separately: under the rule the connectome's
`plastic_weights` is `w_chem` alone, so `food_gains` and `readout` are frozen — and PPO moves the
readout **0.783** relative in 300 episodes against the sensory projection's 0.177 and the chemical
synapses' 0.486 — the largest of the three by that measure, which is not the same as the most important. The safe form of the
test is seeding the readout from PPO and freezing it there, **not** making it plastic: Logbook 040
recorded a 96% forager collapsing to zero within three episodes once its readout learned.

### Correction, 2026-09-14 — the causal mask ignored gap junctions

Raised in review after archiving, and quantified. The reachability above was computed over the **directed
chemical graph alone**, while the forward pass propagates `chem_mat.T @ h + gap_mat.T @ h` and these arms
ran `enable_gap_junctions: true`. Over **chemical + gap**, treating gap junctions as bidirectional as the
forward pass does: cumulative units within 0/1/2/3 hops become 39/123/272/**283**, the units that can never
reach the readout at depth 4 fall from **25 to 19**, and the causally connected draws rise from **672 to
717**.

So the mask **withheld 45 draws per decision** — 6.3% of the 717 that can reach the readout — and 6 of the
25 units it excluded at every step are reachable through a gap path. **The claim "removes no causally
usable signal" is withdrawn as stated**: it removes none usable over chemical edges, and is otherwise
slightly over-tight. **No verdict changes** — `causal` read +0.094 at q = 0.723 against `full`'s −0.384, so
a mask between them is between two flat arms, and the verdict rested on no set beating its own frozen
control. For any future arm the correct construction is the **gap-inclusive** one at 717 draws.

This is also the first time the requirement this change added fired on its own record: a mask restricted by
causal reach must name the connection types its distance measure ignores and the direction of the error.

### What this may not be cited as

- **A result about the perturbation dimension in general.** One substrate, one cell, one rule family, at
  a 31-fold range of draws per decision. R.1 found the dimension decisive on the MLP; this finds it not
  decisive here, and the two together say it does not transfer, not that it never matters.
- **Evidence the causal mask is worthless.** It removes 536 draws per decision that provably cannot
  change the action. That it does not rescue learning was the registered prior — a 1.8× reduction where
  the yardstick needed 16× — and the mask remains the correct default for any future arm.
- **A claim about any other cell**, in particular the 2400-step C3 cell where the horizon is seven times
  longer.
- **A statement that the connectome cannot learn.** PPO reaches 19.31 foods on this cell. What is
  established is that this rule does not, at any perturbation dimension tried.

## Corrections made on the record

Four, all recorded where they occurred rather than quietly folded in:

1. **`causal`'s registered numbers were wrong** — 302 units and 3709 synapses in the design's first
   table, against the implementation's 277 and 3538, because the 25 never-reaching units are perturbed
   at no step. Corrected before any arm ran. "No signal lost" survives; "every synapse stays adaptable"
   does not.
2. **The σ calibration's stated reason was wrong.** It claimed σ 0.2 halves the frozen prior; the frozen
   arm sits at 1.73 at σ 0.05 too. The cause is the action-noise setting, and the calibration's result
   stands for a different reason than the one registered — σ 0.2 costs the *learning* arm 46%.
3. **The campaign was launched at the rejected σ.** The ten configs predated the calibration and carried
   σ 0.2. Caught at 32 of 80 runs and restarted; the partial data is kept and reported, and it is where
   the cleanest evidence for eight seeds over four comes from — `full` went from +0.61 on 4/4 pilot
   seeds to −0.112 on 4/8 registered ones.
4. **Drift read nothing for all 80 runs**, because the harness reused a reader written for the MLP
   checkpoint layout (`state["policy"]`) while connectome checkpoints keep tensors under
   `state["topology"]`. Fixed, and fixing it is what made the split measurement above possible.

## Conclusions

- **`not_reducible`.** No declared set beats its own frozen control by the registered minima; none
  reaches competence. The connectome's failure is **not** the perturbation dimension.
- **Credited-synapse drift is 1.37–1.38× at every dimension from 302 units to 39**, matching the MLP's
  1.28–1.31 — the rule writes more than its own norm on whatever it credits, and narrowing what it
  credits does not change that.
- **Excluded synapses drift 0.013–0.016**, confirming the homeostasis cancellation at real scale.
- **44.4% of this substrate's perturbation cannot reach the action**, a measurement that stands
  independently of every learning arm.
- **Three axes, one ceiling**: σ, action noise and dimension all leave the learning arm at 2.4–4.4 foods
  of 20.
- **R.1b stays blocked** and 7b's gate is untouched. **No committed verdict changed.**

## Next Steps

- [ ] **R.1d** — seed the readout from PPO and freeze it there, then let the rule learn `w_chem` alone.
  The one structural asymmetry measured rather than guessed.
- [ ] **R.2 (e-prop)** remains the registered fallback, and this result strengthens the case for it: the
  failure is not a scale or a noise setting on this substrate.
- [ ] **R.2b**, the matched dimension × horizon sweep, is unaffected and still cheap.

## Data References

- [`supporting/061-l4-reduced-perturbation/`](supporting/061-l4-reduced-perturbation/) —
  [`launch.md`](supporting/061-l4-reduced-perturbation/launch.md) (protocol, both calibrations, and the
  four corrections as they happened), [`reduced_perturbation.json`](supporting/061-l4-reduced-perturbation/reduced_perturbation.json),
  [`per-seed.csv`](supporting/061-l4-reduced-perturbation/per-seed.csv).
- Campaign directories are gitignored. `campaigns/reduced-perturbation` holds the registered 80 runs;
  `campaigns/reduced-perturbation-sigma02-aborted` holds the 32 runs at the rejected σ, reported in the
  launch record.
