# A positive control for the three-factor rule (7a-ii I.0)

## Why

Seven registered results have asked whether the wild-type connectome's wiring is legible to the
minimal three-factor rule. None of them established that the rule can learn **anything**. Logbook
040 recorded the two facts that matter here and neither was followed up: the matched-rule MLP
yardstick **sits at chance**, and the rule **destroys a 96% policy on a dense feedforward network
within three episodes**. A rule that cannot learn on the substrate where learning is easiest
cannot support the inference every panel since has drawn, because "the wiring is not legible to
this rule" and "this rule does not learn" predict the same null.

This change builds the missing control: a task on which reward-modulated Hebbian learning is
**known to work**, with a provable chance floor, run through the rule exactly as implemented. The
project already uses this pattern — `bit-memory-positive-control` is a deliberately-artificial
task with a provable floor, built to show the architecture comparison can separate working memory
at all. This is its analogue for the learning rule.

The control is deliberately not an environment. It drives the committed `ThreeFactorRule` over the
committed `MLPTopology` seam with synthetic observations and rewards, so what is under test is the
rule, its eligibility and its modulator — not the foraging task, the runner, the reward shaping or
the metric. If the rule passes here, the gap to foraging is horizon, reward structure and metric,
which block I's later items examine. If it fails here, the seven negative results are
characterising a non-learner and no further substrate rung is worth running until the rule is
fixed.

Ratified with Chris 2026-09-09 as the first item of block I, ahead of any further substrate work.

## What Changes

- **A minimal contextual task** with an analytically known optimum and a provable chance floor: a
  cue is drawn from a small set, the policy emits a continuous action, and reward is highest when
  the action matches that cue's target. A cue-blind policy is pinned at a computable expected
  reward; a cue-sensitive one can reach the optimum. Pure functions, no environment, no runner.
- **A harness** that drives the committed rule over the committed seam on that task, with three
  arms fixed in advance: the **three-factor rule** (the instrument under test), the **unmodulated
  Hebbian rule** (the floor — it sees no reward, so it must not solve a task whose answer only
  reward reveals), and an **analytic reference** that follows the task's exact gradient (the
  ceiling — it proves the task is learnable in this setup and the topology can express the answer).
- **A registered pass rule**: the three-factor arm must beat the cue-blind floor by a stated margin
  on a stated number of seeds, and the reference arm must clear it, or the control is void rather
  than negative — a task the reference cannot solve says nothing about the rule.
- **Diagnostic telemetry** kept whatever the outcome: the rule's modulator, its eligibility
  magnitude, and the alignment between the update it applies and the analytic gradient, which is
  the quantity that distinguishes "learns slowly" from "moves in an unrelated direction".
- Records under `supporting/048-l4-rule-positive-control/`, tests, docs.

Out of scope: fixing the rule. If the control fails, the eligibility variant (I.1) is the response
and is its own change; this one establishes the fact, not the remedy.

## Capabilities

**Modified**: `learning-rules` (the control and its pass rule), `plasticity-evaluation` (the
protocol and what a void result means).

## Impact

- New: the task module, the harness, the supporting directory. Edited: `docs/architectures.md`,
  `CHANGELOG.md`. No change to any rule, substrate or config on a run-time path — this change adds
  a measurement and alters nothing it measures.
