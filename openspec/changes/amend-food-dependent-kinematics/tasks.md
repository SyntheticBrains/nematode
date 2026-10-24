# Tasks: the D36 amendment — food-dependent reversal kinematics

Phase 10, block E, amending E.2 before B.3 registers. The maintainer's decision of 2026-10-24:
food-dependent kinematics, as its own change landing before B.4.

## 1. The record

- [ ] 1.1 `scripts/analysis/primitive_kinematics_check.py`: write each mean quantity's SD per stratum
  beside its value. Regenerate `supporting/105-reference-extraction/kinematics.json`, and check that
  every existing value is unchanged.

## 2. The data

- [ ] 2.1 `data/locomotion/primitives.json`: add `on_food_reversal_duration_s` (gamma, 2.047, SD 0.899)
  and `on_food_reversal_speed_mm_per_s` (0.069), with condition `on food` and status `measured`. Each
  source names Scheer & Bargmann 2023 (Zenodo 8310289), the held-out OD 2 cohorts more than 0.5 mm
  inside the lawn, the script and its record. Update the file's description and `PROVENANCE.md`.
- [ ] 2.2 `env/primitives.py`: the loader accepts `measured` only with a script and record in its
  source. `PrimitiveKinematics` gains the on-food reversal speed and duration.

## 3. The primitives

- [ ] 3.1 `_start_bout` and `step_primitives` take whether the worm is on a lawn. A reversal draws its
  condition's entries, with one gamma draw either way.
- [ ] 3.2 `continuous.reversal_kinematics: food_dependent | off_food` in `config_loader.py`, passed to
  the environment's config. Refuse it when explicitly set with free locomotion. Classify it as not
  step-coupled in `test_step_duration.py`'s `Continuous2DConfig` set.
- [ ] 3.3 `Continuous2DEnvironment.move_agent_primitives`: on a lawn by `lawn_field.cell_at`, before the
  move.
- [ ] 3.4 `BatchedEvaluator._start_bouts`: on a lawn by `_cells_under(pos) >= 0`, before the move.

## 4. Tests

- [ ] 4.1 Pin both new entries. A `measured` entry without a record is refused.
- [ ] 4.2 Reversals forced on a lawn and off every lawn each match their condition, by duration and
  distance (KS, p > 0.01). The existing forced-bout test runs in a 1-km world without lawns, so it
  covers off food. Add a variant from a lawn's centre, the lawn too large to leave in 2,000 bouts. A depleted cell draws on-food. A bout crossing the edge keeps its speed.
  An assay's no-food phase draws off-food.
- [ ] 4.3 `off_food` is byte-identical to before the option on a lawn assay, positions and RNG state.
- [ ] 4.4 The free-locomotion refusal.
- [ ] 4.5 The batched parity suite passes on the new default. A batched evaluator that ignores the lawn
  fails the trajectory check.
- [ ] 4.6 E.2's positive control runs in the full suite as usual. It reads the first step on the lawn,
  so it cannot move.

## 5. The pilot, on the vendored values

- [ ] 5.1 Re-run the D36 pilot on seeds 9001–9064 through the option: constant-hazard controllers at
  0.01–0.15 mm/s on Flavell's and Scheer's lawns, roaming fraction under each setting. Commit the
  record under `docs/experiments/logbooks/supporting/108-food-dependent-kinematics/`.

## 6. Documentation and tracker

- [ ] 6.1 Logbook 108:
  - the disagreement and the form chosen;
  - the gamma's fit;
  - the speed's whole-second measurement and tracking noise;
  - the pilot, both its first run (SD 1.33 s) and the re-run;
  - the records that stay under `off_food`;
  - the stream shift after an on-lawn reversal.
- [ ] 6.2 The roadmap's D36 row amended and dated. The tracker: E.2's and B.1's amendment notes closed,
  and 9001–9064 added to the seed table as used.
- [ ] 6.3 AGENTS.md's primitives entry, `docs/usage.md`, `configs/README.md`, and the CHANGELOG.

## 7. Close-out

- [ ] 7.1 Full suite and `pre-commit run -a`.
- [ ] 7.2 `openspec validate amend-food-dependent-kinematics --strict`.
- [ ] 7.3 Archive, with the spec's Purpose checked.
- [ ] 7.4 PR.
