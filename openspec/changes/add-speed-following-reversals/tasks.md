# Tasks: an on-food reversal speed that follows the run speed

Phase 10, amending E.2's primitives before C.2's registered fit. The maintainer's order of
2026-11-10: Logbook 131, then this change, then the fit.

## 1. The record

- [ ] 1.1 `scripts/analysis/reversal_speed_coupling.py`:

  - the held-out cohorts' line on every on-lawn start, with a bootstrap over recording dates;
  - the inside stratum and the band cohorts, beside;
  - the deciles' speeds and durations;
  - the still-worm library's paused-worm displacement, reported beside the slowest decile's speed
    before.

  Write `supporting/132-speed-following-reversals/coupling.json`.

## 2. The data

- [ ] 2.1 `data/locomotion/primitives.json`: add `on_food_reversal_speed_coupling`, with the
  record's intercept, slope, low, high and window (3 s), condition `on food` and status
  `measured`.
  - Its source names Scheer & Bargmann 2023 (Zenodo 8310289), the held-out OD 2 cohorts, every
    on-lawn start, the rule, the script and its record.
  - Update the file's description and `PROVENANCE.md`.
- [ ] 2.2 `env/primitives.py`: a `SpeedCoupling` in `PrimitiveKinematics`, loaded and checked:
  intercept > 0, slope ≥ 0, 0 < low < high, window > 0.

## 3. The primitives

- [ ] 3.1 `PrimitiveState` gains the run-speed window.
  - `step_primitives` takes the option and pushes each run step's commanded speed.
  - `_start_bout` takes the window's mean as an argument (None until the window fills; design
    Decision 10). For a reversal on food it gives the line on v̄, clipped, or the pooled speed while
    the mean is None.
  - The window's length is N = round(window ÷ step).
- [ ] 3.2 `continuous.on_food_reversal_speed: pooled | follows_run_speed` in `config_loader.py`,
  passed to the environment's config.
  - Refuse it without primitives, and refuse `follows_run_speed` with
    `reversal_kinematics: off_food`.
  - Classify it as not step-coupled in `test_step_duration.py`'s `Continuous2DConfig` set.
- [ ] 3.3 `Continuous2DEnvironment.move_agent_primitives` passes the option.
- [ ] 3.4 `BatchedEvaluator`: the same window as an `(animals, N)` float64 array, pushed for each
  idle animal that ran, read in `_start_bouts` for a reversal on food. The arithmetic follows design
  Decision 9.

## 4. Tests

- [ ] 4.1 Pin the new entry. A coupling with a negative slope, an inverted range or a missing
  record is refused.
- [ ] 4.2 Unit tests of the reading, each spec scenario:
  - after steady runs;
  - clipped at both ends;
  - before the window fills;
  - an omega between the runs and the reversal;
  - 6 run steps at a 0.5-s step.
- [ ] 4.3 Per-step: forced reversals on a lawn after runs at known speeds back at the line's speed.
  Off every lawn and in a no-food phase they keep the off-food values. A depleted cell counts as a
  lawn.
- [ ] 4.4 Unset, and `pooled`, are byte-identical to before the option on a lawn assay: positions
  and RNG state.
- [ ] 4.5 The refusals.
- [ ] 4.6 The batched parity suite passes with the option on: bouts' kinds, start steps,
  durations and reversal speeds match over 300 steps. A batched evaluator that ignores the window
  fails it.
- [ ] 4.7 `speedup_identity_check.py check` passes with the option unset, against a baseline taken
  before the change. It covers C.1's batched simulation path.

## 5. The confirmation

- [ ] 5.1 Grade Logbook 131's refit answer (`hinge4.json`) on 10 fresh populations of 200
  (880,000,000 + 200·g + i), every statistic and mutant recorded, twice, paired: under the package's
  option, and under Logbook 131's patch (`reversal_coupling_probe.patch`).
  - The rule, fixed before the grade: the option carries the probe's result if the wild type passes
    in at least 8 of 10. Otherwise the difference (the line, the speed's definition, the clip) is
    diagnosed before C.2's fit relies on the option.
  - This is description. Commit the record under `supporting/132-speed-following-reversals/`.

## 6. Documentation and tracker

- [ ] 6.1 Logbook 132:
  - the measurement and its strata;
  - why no tracker floor;
  - the delta-turn mismatch;
  - the decisions;
  - the parity;
  - the confirmation.
- [ ] 6.2 The roadmap's D36 row amended and dated. The tracker gets E.2's amendment note and the
  seed rows (9601–9632, 880 M).
- [ ] 6.3 AGENTS.md's primitives entry, `docs/usage.md`, `configs/README.md` and the CHANGELOG.

## 7. Close-out

- [ ] 7.1 The full suite and `pre-commit run -a`.
- [ ] 7.2 `openspec validate add-speed-following-reversals --strict`.
- [ ] 7.3 Archive, with the spec's Purpose checked.
- [ ] 7.4 PR.
