# Tasks: B v2 — kinematics, a measurement model, and an instrument that resists its known exploit

Phase 10, block B, reopened once. The maintainer's decisions of 2026-10-27:

- three statistics: per-state speed, reversal rate and bout shape (the rise dropped);
- chemotaxis as a fifth family, from Chen 2025 (6 recordings, leave-one-out);
- jitter and missingness;
- fresh N2 data;
- a separate kinematics family;
- adversaries as conditions of validity;
- three changes in sequence.

## 1. Probes and the measurement model

- [x] 1.1 `scripts/analysis/benchmark_v2_controls.py probe`: re-read the planning probes into `supporting/115-benchmark-v2/probes.json`:
  - the per-state speeds;
  - the angular-speed artefact;
  - the detected reversal rates;
  - jitter and missingness, with missing-run lengths;
  - the blind worm on v1.
- [x] 1.2 `benchmark/measurement.py`: noise and missingness (Bernoulli or runs, per 1.1), per-animal streams, simulated sources only.

## 1b. Chemotaxis data and assay

- [x] 1b.1 Chen 2025's deposit (downloaded by hand into the off-repo archive): `extract_benchmark_references.py chen`, naive N2 1-s midbody tracks, plates, dates and the gradient profile; `PROVENANCE.md`.
- [x] 1b.2 The leave-one-recording-out validation and its allowance, registered in Part A before any chemotaxis statistic is computed.
- [x] 1b.3 The `chemotaxis_chen` assay on the 1-s primitives worm; its tracker parameters in the measurement model.
- [x] 1b.4 The chemotaxis statistics (drift, quintile ratio, weathervane slope) on real and simulated tracks; the off-protocol recordings reported; the literature-watch topic for chemotaxis datasets.

## 2. The v2 statistics and bands

- [x] 2.1 `grading`:
  - per-state window-speed histograms;
  - the on-lawn reversal tally;
  - roaming-bout lists and Kaplan–Meier at 5 min;
  - five families;
  - D, D_fit and z\* for v2;
  - v1 unchanged.
- [x] 2.2 `benchmark_bands.py --version v2`: bands from band cohorts, generators after v1's, κ calibration and the rule check; the chemotaxis bands from Chen's 6 recordings; `bands_v2.json` and its checksum in `PROVENANCE.md`.
- [x] 2.3 `benchmark_grade.py --bands v1|v2`; the measurement model on `batched` and `captures`.
- [x] 2.4 Tests:
  - v1 reproduced exactly;
  - the measurement model's noise, missingness and per-animal streams, and real tracks untouched;
  - each new statistic on toy tracks;
  - the KM survival with censoring;
  - four-family weights;
  - the held-out refusal;
  - the landscape's 180° registration, the pixel scale and the fragmentation;
  - the three chemotaxis statistics on toy tracks;
  - the chemotaxis presence test (a gradient-blind population is absent; an undefined resample counts as absent);
  - the measurement-model round trip on toy tracks.

## 3. Registration

- [x] 3.1 `supporting/115-benchmark-v2/launch.md` Part A.
- [x] 3.2 Probe (with Chen's facts), bands, rule check, the measurement-model round trip and the reachability oracle (stop if either fails); Part B with `bands_v2.json`.
- [x] 3.2a Amendment 1, after the Scheer round trip failed on its denoiser: the losses on gap-filled tracks and the jitter on paused stretches replace it, registered before the jitter check is read.

## 4. The controls

- [x] 4.1 The N2 positive control: 4 units, allowance 1 (read once, guarded).
- [x] 4.1b The chemotaxis leave-one-recording-out validation.
- [x] 4.2 B.4's six negatives on v2; the four open-loop negatives each fail the chemotaxis family.
- [x] 4.3 The blind worm fitted against v2 (2 restarts, the σ guard), graded 3 of 5.
- [x] 4.4 C.1's v1 answer graded on v2.
- [x] 4.5 W's minimum v2, on both family sets.
- [x] 4.6 The verdict into `controls_v2.json`.

## 5. Documentation and tracker

- [x] 5.1 The track-import round trip on v2; `IMPORT.md`.
- [x] 5.2 Logbook 115.
- [x] 5.3 The tracker:
  - B v2;
  - the declared non-goals;
  - W.1 and W.2's reconstruction and weight-noise checks (Yad & Nabila 2026);
  - H.1 closed;
  - the F.0b-i re-run and C.1b as the next changes;
  - the seed table.
- [x] 5.4 AGENTS.md and the CHANGELOG.

## 6. Close-out

- [x] 6.1 Full suite and `pre-commit run --all-files` after `git add -A`.
- [x] 6.2 `openspec validate revise-benchmark-v2 --strict`.
- [x] 6.3 Archive.
- [x] 6.4 PR.
