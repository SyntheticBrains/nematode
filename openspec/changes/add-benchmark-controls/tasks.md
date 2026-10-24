# Tasks: B.4 — the benchmark's controls

Phase 10, block B, tracker item B.4. The maintainer's decisions of 2026-10-25:

- a registered allowance across controls;
- negatives matched to band data;
- the untrained policy as both arms, 8 networks × 25 animals;
- random walks matched to the wild type's speed.

## 1. Matched parameters (band data only)

- [ ] 1.1 `scripts/analysis/benchmark_negatives_params.py`:
  - per-state and pooled speed, curvature SD, reversal and omega hazards, and heading persistence,
    from the OD 2 band cohorts;
  - curvature and persistence read only on seconds above `FLOOR_MM`, with the count left out;
  - hazards as onsets per second outside every bout.

  A test that it reads no held-out row.
- [ ] 1.2 Run it. Write the parameters to `supporting/110-benchmark-controls/negatives_params.json`.

## 2. The negatives' controllers

- [ ] 2.1 `BatchedStateHazards` and `BatchedCurvatureWalk` in `evaluation/controllers.py`. Each
  draws from the per-animal `noise` generators the evaluator passes to `act`, and refuses
  `noise=None`.
- [ ] 2.2 Tests:
  - the chain's stationary roaming fraction matches the transitions';
  - the HMM worm is the same on and off food;
  - the random walk's curvature is uniform, and the correlated walk's lag-1 autocorrelation matches;
  - an animal is the same alone and in a batch;
  - one untrained seed gives one network across the three assay configs.

## 3. Registration

- [ ] 3.1 A pilot on seeds 9201–9264 and band data only: every negative runs and is graded at n = 64,
  and the cost is recorded. Each negative's pilot pass or fail is written into the launch record as
  description before it is committed. A negative passing here is brought to the maintainer before the
  launch.
- [ ] 3.2 `supporting/110-benchmark-controls/launch.md`, after the pilot (3.1):
  - the units, their failure rates and the allowance;
  - the negatives and their parameters;
  - the seeds;
  - W's rule;
  - the verdict map.

  Committed before 4.2, with the pilot's readings in it.

## 4. The reading

- [ ] 4.1 `scripts/analysis/benchmark_controls.py`: refuses to run without the committed launch
  record. It reads the held-out units, runs the negatives, reads W's minimum, and writes the verdict.
- [ ] 4.2 Run it once. Commit the record.

## 5. Documentation and tracker

- [ ] 5.1 Logbook 110:
  - the verdict;
  - every unit and negative;
  - W's minimum;
  - each statistic's discrimination, as description.
- [ ] 5.2 The tracker: close B.4, or record VOID with the pivot. Add the seed bands used.
- [ ] 5.3 AGENTS.md, the CHANGELOG.

## 6. Close-out

- [ ] 6.1 Full suite and `pre-commit run -a`.
- [ ] 6.2 `openspec validate add-benchmark-controls --strict`.
- [ ] 6.3 Archive, with the new spec's Purpose written.
- [ ] 6.4 PR.
