# Tasks: a positive control for the three-factor rule

## 1. The task

- [x] 1.1 A pure task module: uniform cue sampling over `K = 4` one-hot cues, fixed targets spread across the action range, reward `−(a − t(c))²` on an unsquashed action, and closed-form `cue_blind_floor(noise) = −Var[t] − σ²` and `optimum(noise) = −σ²` derived from the task's own parameters.
- [x] 1.2 Tests: the floor equals the negative target variance less the noise variance and no cue-blind mean beats it (checked numerically over a grid); the optimum equals the negative exploration variance; the
  observation carries the cue and not the target; sampling is uniform under a fixed seed.

## 2. The harness

- [x] 2.1 Drive the committed `ThreeFactorRule` over `Linear(4, 8) → tanh → Linear(8, 1)` with `plastic_layers: hidden` (frozen readout), at the panels' pinned recipe (`plasticity_rate 1e-3`, both normalisations, homeostasis, the panels' decay and bound, noise at `log_std −1.0`): reset traces, forward, sample an unsquashed action, reward, `rule.step`, repeat per trial. A declared rate grid `{1e-4, 1e-3, 1e-2}` as sub-arms. No environment, no runner, no connectome, no action head.
- [x] 2.2 The unmodulated floor arm (the same rule with the third factor off) and the analytic
  reference arm (gradient descent on the task's own loss through the same topology).
- [x] 2.3 Report per run: final mean reward against the floor and the optimum, the modulator and
  its scale, the eligibility magnitude, and the cosine between the rule's update accumulated over 100-trial blocks and the analytic gradient summed over the same blocks.
- [x] 2.4 Tests: the reference arm reaches the optimum on a fixed seed; the unmodulated arm does not beat the floor; traces are reset before every trial; the block alignment is ~1 for the reference arm by construction and is computed the same way for the rule.

## 3. The registered result

- [x] 3.1 Fix the pass rule in code before any run: 8 seeds, 20,000 trials, pass = beats the cue-blind floor on ≥ 7 seeds **and** the mean is at least halfway from the floor to the optimum, on **any** rate of the declared grid; `void` when the reference does not pass or the floor arm does.
- [x] 3.2 Tests for the harness: each outcome branch including `void` on a failed reference and
  `void` on a solving floor arm; a missing arm is reported and never imputed.
- [x] 3.3 Commit the launch record — the task, the arms, the pass rule, the void conditions and
  what each outcome licenses — then run it.
- [x] 3.4 Records under `docs/experiments/logbooks/supporting/048-l4-rule-positive-control/`:
  `launch.md`, `control.json`, `per-seed.csv`, `details.md`.

## 4. Documentation

- [ ] 4.1 `docs/architectures.md`: the control, what it isolates, and that a substrate null is read against it.
- [ ] 4.1b `docs/experiments/README.md`: the convention that a registered substrate result is read against the rule's positive control, and that while the control has not passed a null is recorded as consistent with the rule not learning.
- [ ] 4.2 `CHANGELOG.md`.
- [ ] 4.3 Tracker and roadmap updated with the outcome at close-out; if the control fails, the
  re-read of 040–046 (I.4) is what carries the reframing, not this record.
