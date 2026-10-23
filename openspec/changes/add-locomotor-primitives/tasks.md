# Tasks: E.2 — locomotor primitives

Phase 10, block E + T, tracker item E.2 (roadmap D36). The maintainer's decisions of 2026-10-23:

- a hazard interface;
- kinematics from published values;
- a food-contact sense;
- pause as a run at zero speed.

## 1. Kinematics data

- [ ] 1.1 `data/locomotion/primitives.json` and `PROVENANCE.md` (design Decision B's table): each value
  with its source, figure, condition and whether it was verified or read from a figure.
- [ ] 1.2 A loader with schema checks, and a test pinning every value.

## 2. The primitive layer

- [ ] 2.1 `continuous.locomotion` in the config and `Continuous2DParams`.
- [ ] 2.2 `env/primitives.py`:
  - bout state;
  - competing-hazard starts;
  - run, reversal, omega and delta execution;
  - the per-episode ventral side.
- [ ] 2.3 `move_agent_primitives` and the runner's dispatch.
- [ ] 2.4 Tests:
  - each spec scenario;
  - byte identity under `free`, RNG included.

## 3. The action space

- [ ] 3.1 `action_space: primitives` in `dtypes.py`, with its validator, and its dimension and bounds
  in `_policy.py`.
- [ ] 3.2 MLP-PPO and the connectome brain size their heads for it.
- [ ] 3.3 Config-loader cross-validation (Decision D).
- [ ] 3.4 Tests:
  - the action shape and bounds;
  - every refusal;
  - both brains act and update on it.

## 4. The senses

- [ ] 4.1 `food_contact` and `locomotor_state` modules, the `BrainParams` fields, and the agent's
  population of them.
- [ ] 4.2 Tests:
  - the edge-crossing and reversal scenarios;
  - the refusals;
  - an unlisted module leaves a brain unchanged.

## 5. Capture

- [ ] 5.1 `BehaviourStep.primitive`, written only under primitives.
- [ ] 5.2 A test for the labelled bout and the byte-identical default capture.

## 6. Non-learning controllers and the positive control

- [ ] 6.1 `env/hazard_controllers.py`: `ConstantHazards`, `PirouetteHazards` and the driver loop.

- [ ] 6.2 The kinematics check (design § Positive control 1).

- [ ] 6.3 The expressiveness check with its must-fail arm (§ 2), as a slow-tier test on seeds
  5994–5999.

  If the pirouette arm fails, stop for a dated decision.

## 7. Configs

- [ ] 7.1 `generate_lawn_configs.py` adds `_s1_primitives` arms:
  - the connectome with modulators;
  - the MLP reading internal state, food contact and locomotor state.
- [ ] 7.2 Loader tests, and a smoke-tier run of each, with capture labelled.

## 8. Documentation

- [ ] 8.1 AGENTS.md, `docs/usage.md` and the CHANGELOG.
- [ ] 8.2 The tracker:
  - E.2 closed;
  - the lawn-edge question answered (a food-contact sense);
  - the dated note on delta turns and off-food sources;
  - B.1's on-food kinematics check added.

## 9. Close-out

- [ ] 9.1 Full suite, `.test_durations`, and `pre-commit run -a`.
- [ ] 9.2 `openspec validate add-locomotor-primitives --strict`.
- [ ] 9.3 Archive.
- [ ] 9.4 PR.
