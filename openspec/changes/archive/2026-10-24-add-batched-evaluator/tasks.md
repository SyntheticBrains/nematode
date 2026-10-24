# Tasks: T.1 — the batched evaluator

Phase 10, block E + T, tracker item T.1. The maintainer's decisions of 2026-10-24:

- frozen evaluation only;
- layered parity;
- sampled noise from per-animal streams;
- at least 20× per core.

## 1. Groundwork

- [x] 1.1 The batch-safe connectome `modulator_release`, with a test that pins the single-sample
  result.
- [x] 1.2 The per-step reference harness: one episode per seed, logging features, action means,
  modulators, the STAM derivative and the trajectory (design Decisions C and D).

## 2. The evaluator

- [x] 2.1 `quantumnematode/evaluation/batched.py`:
  - the state arrays;
  - the declared surface and its refusals;
  - the per-step order;
  - vectorised sensing: odour at three points, STAM ring buffers, adaptive fold change, food contact,
    internal and locomotor state, proprioception;
  - primitives: vectorised mapping, per-animal start tests and bout starts;
  - vectorised moves with the arena walls;
  - intake and satiety;
  - assay phases.
- [x] 2.2 `quantumnematode/evaluation/controllers.py`: `BatchedMLP`, `BatchedConnectome`,
  `BatchedConstantHazards` and `BatchedPirouetteHazards`, with per-animal noise and a mean-action mode.
- [x] 2.3 `BatchedTrajectories`: the compact arrays, the final state and the protocol record, and
  `to_behaviour_steps(i)`.

## 3. Parity

- [x] 3.1 Layer 1 (state) and layer 2 (trajectory) tests in the default tier, with the batch-size
  invariance test:
  - 4 seeds, 300 steps;
  - tolerances 1e-5 for features, means and modulators;
  - exact for bouts;
  - 1e-3 for positions and headings and 1e-6 for satiety and density, over the first 150 steps.
- [x] 3.2 Layer 3 (statistics) as four slow-tier tests, one per assay × controller.
- [x] 3.3 The must-fail faults for each layer.

## 4. The cost pilot

- [x] 4.1 `scripts/analysis/batched_cost_pilot.py`, with one-thread pinning and a default-thread
  figure beside it.
- [x] 4.2 Run it, then commit `cost_pilot.json` and Logbook 104: parity, speed, and W's projected
  budget.

## 5. Documentation and tracker

- [x] 5.1 AGENTS.md, `docs/usage.md` and the CHANGELOG.
- [x] 5.2 Close T.1 in the tracker with the pilot's numbers, and note W's sizing inputs.

## 6. Close-out

- [ ] 6.1 Full suite and `pre-commit run -a`. Durations are left for the next standalone refresh.
- [ ] 6.2 `openspec validate add-batched-evaluator --strict`.
- [ ] 6.3 Archive.
- [ ] 6.4 PR.
