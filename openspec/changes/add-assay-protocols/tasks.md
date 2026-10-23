# Tasks: E.3 — assay protocols

Phase 10, block E + T, tracker item E.3. The maintainer's decisions of 2026-10-24:

- satiety is one slow reserve;
- each assay uses its reference's arena;
- the food history is simulated;
- lawn density is uniform.

## 1. Config

- [ ] 1.1 `AssayConfig` and its phases, `continuous.arena_shape`, `lawns.placement` and
  `satiety.starvation_ends_episode` (design Decisions A–C).
- [ ] 1.2 Load-time validation:
  - phase names are unique and the assay phase exists;
  - the duration sum matches `max_steps` × step, before and after derivation;
  - the assay's scope;
  - centred placement with one lawn, and no random placement on a circle.
- [ ] 1.3 Classify the new fields in E.1's field walk.

## 2. Environment

- [ ] 2.1 The circular wall, in moves and sensing positions.
- [ ] 2.2 Centred lawn placement.
- [ ] 2.3 Phase transitions: lawns on, freshly placed, or off; placement and heading.
- [ ] 2.4 The runner's phase clock, and optional starvation termination in `SatietyManager` and the
  runner.

## 3. Capture

- [ ] 3.1 `BehaviourStep.phase`, the file's `assay` record, and satiety and on-lawn status on bare
  phases.

## 4. Checks

- [ ] 4.1 The design § Checks items 1–6 as tests. Each spec scenario has a test.

## 5. Configs

- [ ] 5.1 `scripts/campaigns/generate_assay_configs.py` and its eight configs, with loader tests.
- [ ] 5.2 Smoke-tier runs of the eight, with capture, on seeds 5991–5993.

## 6. Documentation and tracker

- [ ] 6.1 AGENTS.md, `docs/usage.md` and the CHANGELOG.
- [ ] 6.2 The tracker:
  - close E.3;
  - date the protocol corrections: Scheer's off-lawn start, López-Cruz's fit, the two clocks;
  - add to B's items: López-Cruz's time zero and the fit's start bin, with γ registered from the
    deposited data.

## 7. Close-out

- [ ] 7.1 Full suite, CI-timed `.test_durations` (AGENTS.md § CI sharding) and `pre-commit run -a`.
- [ ] 7.2 `openspec validate add-assay-protocols --strict`.
- [ ] 7.3 Archive.
- [ ] 7.4 PR.
