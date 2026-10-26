# Tasks: B.5 — track import

Phase 10, block B, tracker item B.5. The maintainer's decisions of 2026-10-26:

- CSV and WCON's core;
- lawn membership by a column or a boundary;
- local search included;
- a package module and a CLI.

## 1. Code

- [ ] 1.1 `quantumnematode/benchmark/track_import.py`: `read_csv`, `read_wcon`, `resample_1s`, `lawn_membership`, `to_steps` and `import_population`.
- [ ] 1.2 The `tracks` subcommand of `benchmark_grade.py`.
- [ ] 1.3 Tests:
  - a skeleton reduced to its midpoint;
  - gaps become NaN;
  - pixels refused without a scale;
  - unknown WCON fields refused;
  - a circular and a polar lawn;
  - the column taking precedence;
  - a 1-s track passing through unchanged;
  - grading with the simulator disabled;
  - WCON's `@nematode_benchmark` block, origin offsets and ignored `@` keys.

## 2. The round-trip control

- [ ] 2.1 `scripts/analysis/track_import_check.py`: band-cohort Scheer (small lawn and uniform well) and Roberts' calibration cohorts, as CSV and as WCON, against the native path. It refuses held-out animals.
- [ ] 2.2 Run it, and write `supporting/114-track-import/roundtrip.json`.

## 3. Documentation and tracker

- [ ] 3.1 `data/benchmark/IMPORT.md`: the contract, and the positioning against Tierpsy and OWMD.
- [ ] 3.2 Logbook 114; the tracker's B.5; AGENTS.md and the CHANGELOG.

## 4. Close-out

- [ ] 4.1 Full suite and `pre-commit run --all-files` after `git add -A`.
- [ ] 4.2 `openspec validate add-track-import --strict`.
- [ ] 4.3 Archive, with the new spec's Purpose written.
- [ ] 4.4 PR.
