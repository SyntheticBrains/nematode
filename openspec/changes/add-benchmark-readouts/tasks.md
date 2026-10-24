# Tasks: B.2 — the benchmark's readouts

Phase 10, block B, tracker item B.2. The maintainer's decisions of 2026-10-24:

- reversal onsets, tuned on Roberts;
- López-Cruz's bins are recording time, 4 min after removal;
- the fit starts at the wild type's peak bin;
- the quintile ratio's reference is a digitised Pierce-Shimomura 1999 figure.

## 1. The readouts

- [ ] 1.1 `quantumnematode/benchmark/readouts.py`: `reversal_onsets`, the reference-onset rule,
  matching, `local_search_curve`, `fit_local_search` and `quintile_ratio`.
- [ ] 1.2 Unit tests on synthetic tracks:
  - a straight run with one reversal;
  - pixels and mm reading alike;
  - unit cancellation in the quintile ratio;
  - the fit recovering known parameters.

## 2. The detector's calibration and gates

- [ ] 2.1 `scripts/analysis/reorientation_detector_calibration.py`: the gates as constants, the cohort
  split, the grid search on the calibration cohorts, then every gate once.
- [ ] 2.2 The simulated validation: ConstantHazards on López-Cruz's assay, seeds 7601–7632, with the
  bout labels read only here.
- [ ] 2.3 Run it, and commit the record under
  `docs/experiments/logbooks/supporting/107-benchmark-readouts/`.

## 3. López-Cruz

- [ ] 3.1 `scripts/analysis/local_search_reference.py`: the wild type's fit over the registered window,
  the adjacent starts, and pdfr-1 against its control, written to
  `data/benchmark/lopezcruz2019_reference.json`.

## 4. The quintile ratio's reference

- [ ] 4.1 Digitise Pierce-Shimomura 1999's figure, or record that it could not be obtained.
- [ ] 4.2 Commit it with its calibration points and method.

## 5. The 1-s classifier path

- [ ] 5.1 A test that a simulated 1-s capture is classified through the 5-s path.

## 6. Documentation and tracker

- [ ] 6.1 Logbook 107, AGENTS.md, `docs/usage.md` and the CHANGELOG.
- [ ] 6.2 Close B.2 in the tracker, and add the 7601–7632 band to its seed table.

## 7. Close-out

- [ ] 7.1 Full suite and `pre-commit run -a`.
- [ ] 7.2 `openspec validate add-benchmark-readouts --strict`.
- [ ] 7.3 Archive.
- [ ] 7.4 PR.
