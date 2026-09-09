# Rule positive-control launch record

Written and committed before the registered run, per the registration.

- **Date**: 2026-09-09
- **Pinned state**: `feat/l4-rule-positive-control` at the commit implementing the task, the
  harness and their tests.
- **The question**: seven registered results asked whether a connectome's wiring is legible to the
  minimal three-factor rule. None established that the rule learns anything — Logbook 040 recorded
  the matched-rule MLP yardstick at chance and the rule destroying a competent policy on a dense
  network within three episodes. Does the rule, exactly as the panels ran it, learn a task where
  reward-modulated Hebbian learning is supposed to work?

## The task

One-step continuous contextual association, both bounds in closed form:

- cue `c` drawn uniformly from **4** one-hot alternatives; targets `t(c)` spread over `[-1, 1]`;
- action `a = μ(c) + ε`, `ε ~ N(0, σ²)`, **unsquashed** — the brain's action head is not under
  test, and leaving it out is what keeps the bounds analytic;
- reward `−(a − t(c))²`; the targets never appear in the observation.

**Cue-blind floor** `−Var[t] − σ² = −0.6909`. **Optimum** `−σ² = −0.1353`. **Gap** `Var[t] = 0.5556` — the noise sits on both sides, so what a learner can win is the target variance and
nothing else. `σ = e^(−1) = 0.3679`, the arms' frozen `initial_log_std: −1.0`.

## The instrument, pinned

The panels' rule at the panels' pins: `plasticity_rate 1e-3`, both scaling switches on,
homeostasis on, `weight_decay 0.001`, `weight_bound 3.0`, `baseline_rate 0.01`, `trace_decay 0.9`.
Topology `Linear(4, 8) → tanh → Linear(8, 1)` with **the hidden layer plastic and the readout
frozen** — the panels' arrangement, and the one most favourable to the rule. Traces are reset
before every trial, so each one-step trial's eligibility is its own.

One declared sensitivity grid, on the rate alone: `{1e-4, 1e-3, 1e-2}`. **Any rate passing counts
as a pass** — the claim under test is that the rule learns at all, and a fail must not be a rate
artefact.

## Arms

| arm | what it establishes |
|---|---|
| `three_factor` | the instrument under test |
| `hebbian` | the floor: it never sees reward, so it must **not** solve a task only reward answers |
| `analytic` | the ceiling: gradient descent on the task's own loss, which must pass or the control is **void** |

## Pass rule, fixed before the run

8 seeds, 20,000 trials, scored on mean reward over the final 1,000 trials.

- **Pass** — the three-factor arm beats the cue-blind floor on ≥ 7 of 8 seeds **and** its mean is
  at least halfway from the floor to the optimum, at any rate in the grid.
- **Fail** — it does not.
- **Void** — the analytic arm does not pass, or the unmodulated arm does. A void control is not a
  negative result about the rule, and the record says which.

The bar is deliberately weak: the claim is not that the rule is efficient, only that it moves
policies toward reward at all.

## Diagnosis, kept whatever the outcome

The modulator, the eligibility magnitude, and the **cosine between the rule's update accumulated
over 100-trial blocks and the analytic gradient summed over the same blocks**. An alignment near
zero alongside a healthy modulator and a live trace is the diagnosis, and it points at I.1.

## What each outcome licenses

- **Pass** — the rule learns when the task is trivial; the seven negatives stand as findings about
  the substrate or about the gap to foraging, and block I's remaining items close that gap.
- **Fail** — the rule does not learn where learning is easiest; the seven negatives are reframed as
  characterising a non-learner, I.1 becomes the critical path, and no substrate rung runs until it
  lands. **This is the outcome the reframing expects, and naming it here is what stops it being
  rationalised afterwards.**
- **Void** — nothing is concluded and the control is rebuilt.

## Command

```bash
uv run python scripts/analysis/l4_rule_positive_control.py \
  --out docs/experiments/logbooks/supporting/048-l4-rule-positive-control/control.json \
  --csv docs/experiments/logbooks/supporting/048-l4-rule-positive-control/per-seed.csv
```

## Results

*(written here after the run)*
