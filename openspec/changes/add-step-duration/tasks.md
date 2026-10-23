# Tasks: E.1 — one step duration, and a 1-second point worm

Phase 10, block E + T, tracker item E.1 (roadmap D35). The maintainer's decisions of 2026-10-23:

- E.1 lands alone, with E.2 next;
- configs are authored at the reference step and derived;
- a non-default step is refused outside lawn foraging with `connectomeppo` or `mlpppo`.

## 1. The key and the derivation

- [ ] 1.1 `SimulationConfig.step_seconds`, with the ratio validation (design Decision A).
- [ ] 1.2 `derive_step_coupled` and its call at the end of `load_simulation_config`. It covers the rule
  table, the INFO record per derived constant, and idempotence (Decisions B, C).
- [ ] 1.3 The refusals (Decision D), each with its message.
- [ ] 1.4 Tests:
  - each rule's derived value at `r = 5`;
  - the 1-s lawn worm's values (the spec scenario);
  - idempotence;
  - `r = 1` and unset derive nothing;
  - each refusal;
  - the field-classification walk over every model the lawn point worm reads;
  - all committed scenario configs still load.

## 2. The environment's step

- [ ] 2.1 `Continuous2DParams.step_seconds`, mapped by `create_env_from_config`.
  `ContinuousEnvironment` sets `BodyParams.step_seconds` from it, with the fallback unchanged
  (Decision E).
- [ ] 2.2 Tests:
  - the 5-s scenario still holds;
  - the configured step is the environment's step;
  - derived `max_step_mm` divided by the crawl speed equals it.

## 3. The positive control

- [ ] 3.1 The scripted-worm tests (design § Positive control):
  - straight at full speed for 30 s;
  - turning in place;
  - the grazed cell over 60 s;
  - the STAM weights by age in seconds;
  - the adaptive baseline's step response;
  - the must-fail arm.

## 4. Captures and the classifier

- [ ] 4.1 `write_behaviour_capture` writes `step_seconds` only when it is not 5 (Decision F). Readers
  take an absent key as 5.
- [ ] 4.2 `roaming_dwelling.downsample_to_reference`, with its refusal.
  - `lawn_states.py` and `patch_leaving_reference.py` take the step from the capture or config and
    downsample before measuring.
  - Intake is summed per 5-s block.
- [ ] 4.3 `behavioural_chemotaxis_validation.py` refuses non-5-s captures.
- [ ] 4.4 Tests:
  - the default capture is byte-identical;
  - a 1-s capture is labelled;
  - a downsampled 1-s track classifies as its 5-s subsample;
  - the 2-s refusal;
  - the harness refusal.

## 5. The 1-second configs

- [ ] 5.1 `generate_lawn_configs.py` adds the `_s1` arms of `lawns_modulated` (connectome) and
  `lawns_internal` (MLP) (Decision G).
  - `test_lawn_configs.py` loads both and asserts the derived values.
- [ ] 5.2 Smoke: three episodes each on seeds 5991–5993, with capture on. Each must:
  - complete;
  - classify;
  - record wall-clock per worm-hour beside the 5-s arms, as a note for T.1.

## 6. Documentation

- [ ] 6.1 Correct the STAM module docstring's "~1–2 seconds" per step.
- [ ] 6.2 `step_seconds` in the configuration docs and AGENTS.md's commands section.
- [ ] 6.3 CHANGELOG under *Unreleased*.
- [ ] 6.4 Update the tracker (`phase10-tracking/tasks.md`):
  - H.2 closed (on #470's merge, 2026-10-23);
  - E.1 closed, with its dated amendment. The kinematic body is refused at a non-default step, so
    "body step" moves to Phase 11 (D41).

## 7. Close-out

- [ ] 7.1 Full suite: `uv run pytest -q -m "not nightly"`.
- [ ] 7.2 Regenerate `.test_durations`, then `uv run pre-commit run -a`.
- [ ] 7.3 `openspec validate add-step-duration --strict`.
- [ ] 7.4 Archive.
- [ ] 7.5 PR.
