# Tasks: E.2 — locomotor primitives

Phase 10, block E + T, tracker item E.2 (roadmap D36). The maintainer's decisions of 2026-10-23:

- a hazard interface;
- kinematics from published values;
- a food-contact sense;
- pause as a run at zero speed.

## 1. Kinematics data

- [x] 1.1 `data/locomotion/primitives.json` and `PROVENANCE.md` (design Decision B's table): each value
  with its source, figure, condition and whether it was verified or read from a figure.
- [x] 1.2 A loader with schema checks, and a test pinning every value.

## 2. The primitive layer

- [x] 2.1 `continuous.locomotion` in the config and `Continuous2DParams`.
- [x] 2.2 `env/primitives.py`:
  - bout state;
  - competing-hazard starts, drawn only at a step with no bout running;
  - run, reversal, omega and delta execution;
  - the per-episode ventral side.
- [x] 2.3 `move_agent_primitives` and the runner's dispatch.
- [x] 2.4 Tests:
  - each spec scenario;
  - byte identity under `free`, RNG included.

## 3. The action space

- [x] 3.1 `action_space: primitives` in `dtypes.py`, with its validator, and its dimension and bounds
  in `_policy.py`.
- [x] 3.2 MLP-PPO and the connectome brain size their heads for it.
- [x] 3.3 Config-loader cross-validation (Decision D).
- [x] 3.4 Tests:
  - the action shape and bounds;
  - every refusal;
  - both brains act and update on it.

## 4. The senses

- [x] 4.1 `food_contact` and `locomotor_state` modules, the `BrainParams` fields, and the agent's
  population of them.
- [x] 4.2 Tests:
  - the edge-crossing and reversal scenarios;
  - the refusals;
  - an unlisted module leaves a brain unchanged.

## 5. Capture

- [x] 5.1 `BehaviourStep.primitive`, written only under primitives, and `speed_signed` recorded under
  primitives.
- [x] 5.2 A test for the labelled bout and the byte-identical default capture.

## 6. Non-learning controllers and the positive control

- [x] 6.1 `env/hazard_controllers.py`: `ConstantHazards` and `PirouetteHazards` on the Brain
  protocol (Decision F), run through the standard runner.

- [x] 6.2 The kinematics check (design § Positive control 1). Durations are compared at whole steps;
  distances and angles are compared exactly.

- [x] 6.3 The expressiveness check (§ 2), as a slow-tier test on fixture seeds 5994–5999. It compares
  the pirouette rule against constant hazards, with the inverted rule as the must-fail arm.

  If the pirouette arm fails, stop for a dated decision.

## 7. Configs

- [x] 7.1 `generate_lawn_configs.py` adds `_s1_primitives` arms:
  - the connectome with modulators;
  - the MLP reading internal state, food contact and locomotor state.
- [x] 7.2 Loader tests, and a smoke-tier run of each, with capture labelled.

## 8. Documentation

- [x] 8.1 AGENTS.md, `docs/usage.md` and the CHANGELOG.
- [x] 8.2 The dated D36 amendment (Decision G) in the roadmap and the tracker's design, with the
  fixture-seed note in its seed table.
- [x] 8.3 The tracker:
  - E.2 closed;
  - the lawn-edge question answered (a food-contact sense);
  - the dated note on delta turns and off-food sources;
  - B.1's on-food kinematics check added.

## 9. Close-out

- [ ] 9.1 Full suite, `.test_durations`, and `pre-commit run -a`.
- [ ] 9.2 `openspec validate add-locomotor-primitives --strict`.
- [ ] 9.3 Archive.
- [ ] 9.4 PR.
