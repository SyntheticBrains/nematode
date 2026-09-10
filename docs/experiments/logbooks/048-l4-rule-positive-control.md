# 048: The Rule Has No Positive Control — and Fails One (7a-ii I.0 / Phase 7)

**Status**: completed — **`fail`, with the control valid**. Seven registered results had asked
whether the wild-type connectome's wiring is legible to the minimal three-factor rule. None of them
had established that the rule can learn **anything**, and Logbook 040 had already recorded the two
facts that should have stopped the sequence: the matched-rule MLP yardstick sits at chance, and the
rule destroys a 96% policy on a dense feedforward network within three episodes. "The wiring is not
legible to this rule" and "this rule does not learn" predict the same null. This is the control that
separates them: a one-step contextual association with both bounds in closed form, driven through
the committed rule over the committed seam, with no environment, runner, connectome or action head
— everything that could explain a null removed rather than controlled for. The analytic reference
closes **99.9%** of the available gap on 8 of 8 seeds and the unmodulated floor arm does not pass,
so both validity conditions hold. **The rule fails at every rate across two orders of magnitude and
ends below the cue-blind floor**: a policy that ignored the cue entirely would score better than
what the rule produces after 20,000 trials. The registered diagnosis says why — a live trace, a
well-behaved modulator, and a gradient alignment of **+0.031 mean, +0.009 median**. The updates are
not starved of reward information; they are not *aimed*. **The three-factor rule as implemented is
not a policy-gradient estimator**, and the seven negative results are reframed as characterising a
non-learner.

**Branch**: `feat/l4-rule-positive-control` (PR #336).

**Date**: 2026-09-10.

**OpenSpec change**: `add-l4-rule-positive-control` (archived; the task, the harness and the
registered pass rule; extends capabilities `learning-rules` and `plasticity-evaluation`).

## Objective

Establish whether the three-factor rule, exactly as the panels ran it, learns a task on which
reward-modulated Hebbian learning is supposed to work — and if it does not, say so before another
substrate rung is built on the assumption that it does.

## Background

[Logbooks 040](040-l4-panel.md)–[047](047-l4-structured-instruction.md) built and tested four
substrate or rule interventions after the first panel: sign grounding, three consolidation
mechanisms, two decorrelating terms and a routed third factor. None moved the wild-type connectome
off its floors. The roadmap's [reframing of 2026-09-09](../../roadmap.md) named what every one of them
had assumed — a working instrument — and made block I the response. This is its first item.

There was a specific theoretical reason to expect the rule to fail. The eligibility trace is
`pre × post` with exploration noise applied only at the action output; the three-factor rules that
are policy-gradient estimators put the *noise inside the eligibility* — the deviation of a unit's
activity from its mean, times pre (Williams 1992; node perturbation, Fiete & Seung 2006; the
Frémaux & Gerstner 2016 review). With deterministic units and output-only noise, an internal
synapse's update carries no information about which way to move to make the sampled action more
likely. That predicts a drift toward whatever correlation structure is present, indifferent to
reward, on every substrate — which is what every panel measured, and what the clone-destruction
diagnostic measured directly.

## The control

A one-step continuous contextual association, the smallest thing reward-modulated Hebbian learning
should solve:

- a cue drawn uniformly from **4** one-hot alternatives, targets `t(c)` spread over `[-1, 1]`;
- an **unsquashed** action `a = μ(c) + ε`, `ε ~ N(0, σ²)` at the arms' frozen `σ = e⁻¹ = 0.368`;
- reward `−(a − t(c))²`, with the targets **absent from the observation**, so the association is
  reachable only through reward.

Both bounds are computed, not measured. **Cue-blind floor** `−Var[t] − σ² = −0.6909`: the best a
policy ignoring the cue can do, since the exploration cost falls on every policy alike.
**Optimum** `−σ² = −0.1353`. The gap a learner can win is therefore exactly `Var[t] = 0.5556`.

**What is deliberately absent.** No environment, no episode runner, no reward shaping, no plateau
metric, no 2400-step horizon, no connectome, and not the brain's tanh-squashed action head. The
instrument under test is the rule, its eligibility and its modulator; a failure here has nothing
left to blame. Traces are reset before every trial, so each one-step trial's eligibility is its own.

**The instrument is pinned to the panels'**: `plasticity_rate 1e-3`, both scaling switches on,
homeostasis on, the panels' decay and bound, over `Linear(4, 8) → tanh → Linear(8, 1)` with the
hidden layer plastic behind a frozen readout — their arrangement, and the one most favourable to
the rule. One declared grid on the rate, `{1e-4, 1e-3, 1e-2}`, under which **any rate passing counts
as a pass**, so a failure cannot be a rate artefact.

## Hypothesis

Pre-registered before the run (`supporting/048-l4-rule-positive-control/launch.md` committed
first). Over 8 seeds × 20,000 trials:

- **Pass** — the three-factor arm beats the cue-blind floor on ≥ 7 of 8 seeds **and** its mean is at
  least halfway from that floor to the optimum, at any rate in the grid.
- **Fail** — it does not.
- **Void** — the analytic reference does not itself pass, or the unmodulated arm does. **A void
  control is not a negative result about the rule**, and the record must say which.

The bar is deliberately weak. The claim under test is not that this rule is efficient; it is that it
moves policies toward reward at all.

## Results

| arm | mean | seeds above floor | result |
|---|---|---|---|
| `analytic` (reference) | **−0.1361** | 8/8 | **passes** |
| `hebbian` (floor) | −0.8753 | 3/8 | does not pass |
| `three_factor` @ 1e-4 | −0.7895 | 1/8 | does not pass |
| `three_factor` @ 1e-3 (pinned) | −0.7482 | 1/8 | does not pass |
| `three_factor` @ 1e-2 | −0.7504 | 2/8 | does not pass |

Floor −0.6909, optimum −0.1353, halfway threshold −0.4131.

**The control is valid.** The reference reaches −0.1361 against an optimum of −0.1353 — it closes
99.9% of the gap, on every seed — so the task is learnable, the topology expresses the answer and
the optimiser works. The unmodulated arm does not pass, so the task does not leak its answer
without reward. Neither void condition fires.

**The rule does not merely fail to learn.** At every rate its mean sits near −0.75 against a floor
of −0.69: after 20,000 trials it is *worse than ignoring the cue*. One or two seeds of eight finish
above the floor at any rate.

### The diagnosis

| quantity | three-factor arm |
|---|---|
| mean modulator | +0.0020 |
| mean absolute weight change per step | 2.62 × 10⁻⁴ |
| **gradient alignment** | **+0.031 mean, +0.009 median** (min −0.035, max +0.260) |
| alignment by rate (mean / median) | 1e-4: +0.077 / +0.074 · 1e-3: +0.020 / +0.025 · 1e-2: −0.002 / +0.001 |

The trace is live, the weights move every step, and the modulator is a well-behaved centred
prediction error. What the update is not is aimed: its cosine against the gradient-descent direction
of the same trials is indistinguishable from orthogonal, at every rate. The unmodulated arm's is
−0.010, statistically the same thing.

**On the reference arm's alignment.** Measured while it is still learning — 300 or 1,000 trials —
it is exactly **+1.0**, which is the end-to-end check that the sign convention and the accumulation
are right. Pooled over the full 20,000 it reads +0.45, because it converges after roughly 5,000
trials and its per-block gradient then becomes numerically negligible, so the cosine decays into
rounding noise. **No such dilution applies to the three-factor arm**: it never converges — it ends
below the cue-blind floor — so every one of its blocks is one in which a learning rule would have
had a gradient to follow.

## Analysis

- **This is the measurement the theory predicted.** A Hebbian eligibility with output-only noise
  gives an internal synapse no information about which way to move to make the sampled action more
  likely; the rule reinforces whatever correlation structure is present, which is a drift with no
  particular relationship to reward. The alignment measures exactly that, and finds it.
- **The failure is not about difficulty, horizon, reward sparsity or metric.** Those are the
  explanations a foraging null leaves open, and the control removes all of them. One step, dense
  reward, four cues, a closed-form optimum, and a reference that solves it to 99.9% in the same
  topology.
- **It is not a rate artefact.** Two orders of magnitude, and the alignment is near zero at all
  three — falling, if anything, as the rate rises.
- **What the seven results still are.** They remain valid records of what this rule does on those
  substrates: the drift, the fixed points, the destruction of cloned policies, the sign-grounding
  collapse, the routing degradation. What they cannot support is the inference that the wild-type
  wiring carries no learnable signal, because the instrument that produced them does not learn where
  learning is easiest.

## Conclusions

- **The three-factor rule as implemented is not a policy-gradient estimator**, and the direct
  evidence is a gradient alignment of +0.009 with a healthy modulator and a live trace.
- **A positive control should have come first.** Logbook 040 recorded the yardstick at chance and
  the sequence continued for seven more registered results. The cost of the omission is that every
  one of them now needs re-reading; the cost of the control itself was a day.
- **The registered `void` outcome earned its place even though it did not fire.** Without the
  reference and floor arms, a failed control and a failed rule would be indistinguishable — and the
  reference passing to 99.9% is what makes this a fact about the rule.
- **I.1 is the critical path**: an eligibility with the exploration noise inside it, cleared on the
  MLP yardstick before any connectome arm.

## Limitations

- One task. A rule could fail here and work on a task with different structure — though a rule that
  cannot learn a one-step association with dense reward is not a promising candidate for a
  2400-step foraging task with sparse reward.
- The action is unsquashed and the brain's head is not under test. The head is shared with the PPO
  arms, which do learn, so it is not a likely culprit — but this control would not see it if it were.
- The topology is the panels' favourable arrangement at their pinned recipe; an all-plastic variant
  was not run and is a named follow-up.
- The alignment is measured against the gradient of the *immediate* loss. A rule estimating a
  longer-horizon return would legitimately differ; on a one-step task there is no such difference to
  hide behind.

## Next Steps

**I.1** — a three-factor variant whose eligibility carries the exploration noise (node perturbation
on the units, or the action noise propagated into the trace), so the rule is a policy-gradient
estimator rather than reinforced correlation. It clears on the **MLP yardstick first**: if the
yardstick does not learn, the variant is not an instrument either. Then **I.2**'s statistic and
graded metric, **I.3**'s unexamined knobs, and **I.4**'s re-read of Logbooks 040–047 stating which
results survive as findings about the wiring. B.3's receptor layer and B.5's panel stay queued
behind block I, and the 7a shipment decision is taken after it.

## Data References

- Registration and design: `openspec/changes/archive/2026-09-10-add-l4-rule-positive-control/`;
  capabilities `openspec/specs/learning-rules/spec.md`,
  `openspec/specs/plasticity-evaluation/spec.md`.
- Everything the control produced:
  [supporting/048-l4-rule-positive-control/](supporting/048-l4-rule-positive-control/details.md) —
  `launch.md` (the task, the pins, the pass rule, the void conditions and each outcome's reading,
  all before the run), `control.json`, `per-seed.csv`, `details.md`.
- The reframing this answers: `docs/roadmap.md` § Phase 7, second reframing paragraph.
- Tooling: `plasticity/positive_control.py`, `scripts/analysis/l4_rule_positive_control.py`.
