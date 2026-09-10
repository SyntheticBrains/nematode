# Tasks

## 1. The schedule on the configuration

- [ ] 1.1 `plasticity_node_noise_final` and `plasticity_node_noise_anneal_episodes` on the
  plasticity configuration mixin, both defaulting to unset, with the existing field's docstring
  restated as the *initial* scale.
- [ ] 1.2 Validators: half-specified schedules and a final scale above the initial one are refused,
  a zero final scale is permitted, and the existing zero-scale refusal applies to the initial scale
  only. Repeat both at brain construction, since `model_copy` skips validators.
- [ ] 1.3 Tests for each refusal and each permitted combination, including the copied-configuration
  path.

## 2. The schedule on the substrates

- [ ] 2.1 A single scheduling helper the two topologies share, so the shape exists once: episodes
  begun in, current scale out, with the geometric path and the floor after `E`.
- [ ] 2.2 `_mlp_topology.py` and `connectome_ppo.py` read the current scale at draw time rather than
  a construction-time constant. The connectome's per-settling-step draws within one episode all use
  that episode's scale.
- [ ] 2.3 The counter advances in `prepare_episode`, is cleared on a policy load, and is not
  persisted — added to the transient-buffer set rather than left to be discovered by a checkpoint
  refusal.
- [ ] 2.4 Tests: the scale falls monotonically and hits the floor exactly at `E`; a warm start
  restarts the schedule; the same draw reaches both the pre-activation and the eligibility; with the
  fields unset the weight trajectory is bit-identical to the constant-scale arm.

## 3. Clearance on the positive control

- [ ] 3.1 An annealed arm in `scripts/analysis/l4_rule_positive_control.py`, with the schedule
  compressed to the control's budget and the declared bounds recorded in the result.
- [ ] 3.2 The harness reports gradient alignment at the schedule's initial and final scales, not
  only at the end of the run.
- [ ] 3.3 The result records whether trace normalisation was enabled, so the rate regime travels
  with the number.
- [ ] 3.4 Run it. Records under `supporting/051-l4-sigma-annealing/`. **A failure stops here** and
  is written up as a property of this schedule.

## 4. The clone assay, only if 3 passes

- [ ] 4.1 Two arm configs: the annealed screening arm, and its frozen control carrying the identical
  schedule with `freeze_updates: true` — one `freeze_updates` key per file, checked before the run.
- [ ] 4.2 The arms join the existing screen registry; the assay's pass rule, comparator, budget and
  metric are untouched.
- [ ] 4.3 The harness reports the frozen control's score as a trajectory over the schedule, so the
  learning arm can be read against it rather than against a single endpoint.
- [ ] 4.4 Run both arms. Records under the same supporting directory.

## 5. Documentation

- [ ] 5.1 `docs/architectures.md`: the schedule, its shape, and the rate coupling with the two
  regimes named.
- [ ] 5.2 `CHANGELOG.md`.
- [ ] 5.3 Tracker and roadmap updated with the outcome at close-out, as I.1b.
- [ ] 5.4 The logbook: this closes into I.1's record or takes its own, decided at close-out by
  whether the outcome changes I.1's reading.
