# Tasks

## 1. The schedule on the configuration

- [x] 1.1 `plasticity_node_noise_final` and `plasticity_node_noise_anneal_episodes` on the
  plasticity configuration mixin, both defaulting to unset, with the existing field's docstring
  restated as the *initial* scale.
- [x] 1.2 Validators: half-specified schedules, a final scale above the initial one and a zero
  final scale are refused, and the existing zero-initial-scale refusal stands. Repeat every refusal
  at brain construction, since `model_copy` skips validators.
- [x] 1.3 Tests for each refusal and each permitted combination, including the copied-configuration
  path.

## 2. The schedule on the substrates

- [x] 2.1 A single scheduling helper the two topologies share, so the shape exists once: steps
  begun in, current scale out, with the geometric path and the floor after `E`.
- [x] 2.2 A plain-integer counter on each topology and an `advance_schedule()` seam method, added
  to the `PlasticTopology` protocol; `_mlp_topology.py` and `connectome_ppo.py` draw at the current
  scale rather than a construction-time constant, and the connectome's per-settling-step draws
  within one episode all use that episode's scale.
- [x] 2.3 Both brains' `prepare_episode` call `advance_schedule()` beside `reset_traces()`; the
  rule's load-time reset sets the counter to zero and does not advance it. The counter is not a
  buffer and never enters `state_dict`.
- [x] 2.4 Tests: the scale falls monotonically and hits the floor exactly at `E`; resetting the
  traces alone leaves it unchanged; a load restarts it; the same draw reaches both the
  pre-activation and the eligibility; with the fields unset the weight trajectory is bit-identical
  to the constant-scale arm.

## 3. Clearance on the positive control

- [x] 3.1 An annealed arm in `scripts/analysis/l4_rule_positive_control.py` at the registered
  values (0.2 → 0.02 over the first 10,000 of 20,000 trials), the harness calling
  `advance_schedule()` once per trial, with the bounds and length recorded in the result.
- [x] 3.2 The harness reports gradient alignment over the decay and over the floor separately,
  from the blocks it already computes, beside the floor-phase score.
- [x] 3.3 The result records whether trace normalisation was enabled, so the rate regime travels
  with the number.
- [x] 3.4 Run it. Records under `supporting/051-l4-sigma-annealing/`. **A failure stops here** and
  is written up as a property of this schedule. **Ran 2026-09-10: FAILS.** Mean −0.4795 against a
  −0.4131 threshold, 38.0% of the gap where the constant σ = 0.2 arm closes 89.0%; 7/8 seeds above
  floor, so the seed clause is met and the mean clause is not. The arm lands on the constant
  σ = 0.05 arm (38.4%), because a geometric decay puts σ below 0.05 at trial 6,021 — **70% of the
  budget runs below a scale already shown not to clear the bar.** Alignment +0.1756 over the decay
  against +0.0510 at the floor, so the estimator was aimed while the scale was large: the failure
  is in how the budget was spent, not in the schedule breaking the estimator.

## 4. The clone assay, only if 3 passes — NOT RUN

**Gate 3 failed, so this section did not run**, as registered. The arms, their configs and the
harness's binned trajectory are implemented and tested, so a later schedule can be screened
without rebuilding them; no campaign was launched and no assay result exists.

- [~] 4.1 (built, not run) Two arm configs at the registered values (0.2 → 0.02 over the first 1,000 of 2,000
  episodes): the annealed screening arm, and its frozen control carrying the identical schedule with
  `freeze_updates: true` — one `freeze_updates` key per file, checked before the run.
- [~] 4.2 (built, not run) The arms join the existing screen registry; the assay's pass rule, comparator, budget and
  metric are untouched.
- [~] 4.3 (built, not run) The harness bins both arms' curves per 250 episodes with σ(e) stated per bin, so the
  learning arm is read against the frozen control bin by bin rather than against a single endpoint.
- [ ] 4.4 Run both arms. **Not run**: gate 3 failed.

## 5. Documentation

- [x] 5.1 `docs/architectures.md`: the schedule, its shape, and the rate coupling with the two
  regimes named.
- [x] 5.2 `CHANGELOG.md`.
- [x] 5.3 Tracker and roadmap updated with the outcome at close-out, as I.1b.
- [x] 5.4 The logbook: this closes into I.1's record or takes its own, decided at close-out by
  whether the outcome changes I.1's reading.
