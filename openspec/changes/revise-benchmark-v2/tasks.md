# Tasks: B v2 — kinematics, a measurement model, and an instrument that resists its known exploit

Phase 10, block B, reopened once. The maintainer's decisions of 2026-10-27:

- four statistics: per-state speed, reversal rate, bout shape, and the rise;
- jitter and missingness;
- fresh N2 data;
- a separate kinematics family;
- adversaries as conditions of validity;
- three changes in sequence.

## 1. Probes and the measurement model

- [ ] 1.1 `scripts/analysis/benchmark_v2_controls.py probe`: re-read the planning probes into `supporting/115-benchmark-v2/probes.json`:
  - the per-state speeds;
  - the angular-speed artefact;
  - the detected reversal rates;
  - jitter and missingness, with missing-run lengths;
  - the blind worm on v1.
- [ ] 1.2 `benchmark/measurement.py`: noise and missingness (Bernoulli or runs, per 1.1), per-animal streams, simulated sources only.

## 2. The v2 statistics and bands

- [ ] 2.1 `grading`:
  - per-state window-speed histograms;
  - the on-lawn reversal tally;
  - roaming-bout lists and Kaplan–Meier at 5 min;
  - the rise;
  - four families;
  - D, D_fit and z\* for v2;
  - v1 unchanged.
- [ ] 2.2 `benchmark_bands.py --version v2`: bands from band cohorts, generators after v1's, κ calibration and the rule check; `bands_v2.json` and its checksum in `PROVENANCE.md`.
- [ ] 2.3 `benchmark_grade.py --bands v1|v2`; the measurement model on `batched` and `captures`.
- [ ] 2.4 Tests:
  - v1 reproduced exactly;
  - the measurement model's noise, missingness and per-animal streams, and real tracks untouched;
  - each new statistic on toy tracks;
  - the KM survival with censoring;
  - four-family weights;
  - the held-out refusal.

## 3. Registration

- [ ] 3.1 `supporting/115-benchmark-v2/launch.md` Part A.
- [ ] 3.2 Probe, bands, rule check and the oracle (stop if it fails); Part B with `bands_v2.json`.

## 4. The controls

- [ ] 4.1 The N2 positive control (read once, guarded).
- [ ] 4.2 B.4's six negatives on v2.
- [ ] 4.3 The blind worm fitted against v2 (2 restarts, the σ guard), graded 3 of 5.
- [ ] 4.4 C.1's v1 answer graded on v2.
- [ ] 4.5 W's minimum v2.
- [ ] 4.6 The verdict into `controls_v2.json`.

## 5. Documentation and tracker

- [ ] 5.1 The track-import round trip on v2; `IMPORT.md`.
- [ ] 5.2 Logbook 115.
- [ ] 5.3 The tracker:
  - B v2;
  - the declared non-goals;
  - W.1 and W.2's reconstruction and weight-noise checks (Yad & Nabila 2026);
  - H.1 closed;
  - the F.0b-i re-run and C.1b as the next changes;
  - the seed table.
- [ ] 5.4 AGENTS.md and the CHANGELOG.

## 6. Close-out

- [ ] 6.1 Full suite and `pre-commit run --all-files` after `git add -A`.
- [ ] 6.2 `openspec validate revise-benchmark-v2 --strict`.
- [ ] 6.3 Archive.
- [ ] 6.4 PR.
