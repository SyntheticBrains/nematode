# Tasks: the benchmark's imports, decoupled from the simulator

## 1. Code

- [x] 1.1 `report/behaviour.py` with `BehaviourStep`, moved verbatim, and re-exported from `report.dtypes`.
- [x] 1.2 `validation/__init__.py`: `adaptive_sensor`'s four names resolve lazily, with `__all__` unchanged.
- [x] 1.3 `benchmark/track_import.py` imports `BehaviourStep` from `report.behaviour`.
- [x] 1.4 The fresh-interpreter test, and the lazy names still resolving.

## 2. Docs

- [x] 2.1 `data/benchmark/IMPORT.md`, Logbook 114's limit (a dated note), the tracker's B.5 note, and the CHANGELOG.

## 3. Close-out

- [x] 3.1 Full suite and `pre-commit run --all-files` after `git add -A`.
- [x] 3.2 `openspec validate refactor-benchmark-light-imports --strict`.
- [x] 3.3 Archive.
- [x] 3.4 PR.
