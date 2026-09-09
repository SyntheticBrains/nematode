# Tasks: a positive control for the three-factor rule

## 1. The task

- [ ] 1.1 A pure task module: uniform cue sampling over `K = 4` one-hot cues, fixed targets spread
  across the action range, reward `−(a − t(c))²`, and closed-form `cue_blind_floor()` and
  `optimum(noise)` derived from the task's own parameters.
- [ ] 1.2 Tests: the floor equals the negative target variance and no cue-blind action beats it
  (checked numerically over a grid); the optimum equals the negative exploration variance; the
  observation carries the cue and not the target; sampling is uniform under a fixed seed.

## 2. The harness

- [ ] 2.1 Drive the committed `ThreeFactorRule` over a small `MLPTopology` on the task: forward,
  sample an action with the panels' noise, reward, `rule.step`, repeat. No environment, no runner,
  no connectome.
- [ ] 2.2 The unmodulated floor arm (the same rule with the third factor off) and the analytic
  reference arm (gradient descent on the task's own loss through the same topology).
- [ ] 2.3 Report per run: final mean reward against the floor and the optimum, the modulator and
  its scale, the eligibility magnitude, and the cosine between the rule's applied update and the
  analytic gradient of the same step.
- [ ] 2.4 Tests: the reference arm reaches the optimum on a fixed seed; the unmodulated arm does
  not beat the floor; the alignment is ~1 for the reference arm by construction (it is the
  gradient) and is computed the same way for the rule.

## 3. The registered result

- [ ] 3.1 Fix the pass rule in code before any run: 8 seeds, 20,000 trials, pass = beats the
  cue-blind floor on ≥ 7 seeds **and** the mean is at least halfway from the floor to the optimum;
  `void` when the reference does not pass or the floor arm does.
- [ ] 3.2 Tests for the harness: each outcome branch including `void` on a failed reference and
  `void` on a solving floor arm; a missing arm is reported and never imputed.
- [ ] 3.3 Commit the launch record — the task, the arms, the pass rule, the void conditions and
  what each outcome licenses — then run it.
- [ ] 3.4 Records under `docs/experiments/logbooks/supporting/048-l4-rule-positive-control/`:
  `launch.md`, `control.json`, `per-seed.csv`, `details.md`.

## 4. Documentation

- [ ] 4.1 `docs/architectures.md`: the control, what it isolates, and that a substrate null is read
  against it.
- [ ] 4.2 `CHANGELOG.md`.
- [ ] 4.3 Tracker and roadmap updated with the outcome at close-out; if the control fails, the
  re-read of 040–046 (I.4) is what carries the reframing, not this record.
