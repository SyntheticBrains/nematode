# Tasks: the `wormbench` package

Not a phase task. The maintainer decided on 2026-11-23 and 2026-11-24 to make the benchmark a
standalone package:

- named `wormbench`, a new package in this repository;
- the small reference files in the wheel, the tracks on Zenodo;
- a measured route for simulated tracks;
- chemotaxis import as a follow-up.

The spec review's findings of 2026-11-24 are written into the design. No campaign runs in this
change, no experiment's input changes, and no PD1074 held-out row is read.

**It merges only after PR #541's registered fit, grade and read are done, or after PR #541 records
C.2 as banked.**

Commands that run tests or checks during a fit use one worker (`-n 0`), or wait.

## 1. Review

- [x] 1.1 `/nematode-review-spec`, by an independent reviewer: 4 blocking and 12 should-fix
  findings, all written into the design, and the maintainer's answers recorded.
- [ ] 1.2 A second review of the revised change before code starts.

## 2. Baselines, before any code moves

- [ ] 2.1 `scripts/analysis/wormbench_identity_check.py baseline|check` (Decision K). Take the
  baseline in a separate worktree at `origin/main`, with one worker.
- [ ] 2.2 `speedup_identity_check.py baseline` there, with `--workers 1` and `--out` in the scratch
  directory.

## 3. The package

- [ ] 3.1 `packages/wormbench/` with its `pyproject.toml` (Decision A), README, NOTICE,
  `CITATION.cff` and Ruff configuration.
- [ ] 3.2 Move the modules of Decision B. Make the path handling resolve when used (Decision D): no
  import-time path, `None` defaults, a lazy module `__getattr__` for the old constants, and caches
  keyed by the resolved path. Content is otherwise unchanged.
- [ ] 3.3 `bands` and `grade`, from `benchmark_grade.py` and `benchmark_bands.py`. The scripts
  re-export every moved name, private ones included.
- [ ] 3.4 The aliases in each parent `__init__`, with explicit re-exports, and the dotted imports
  inside `quantum-nematode` rewritten (Decision C). `quantum-nematode` depends on `wormbench`, and
  `quantumnematode/__init__.py` sets the host root.
- [ ] 3.5 The data resolver and the manifest (Decision D); `wormbench_records.py build|sync|check`
  (Decisions E and F), with the stored classifier path mapped to its resolver name.
- [ ] 3.6 The measured route (Decision H) and `IMPORT.md`'s contract for it.
- [ ] 3.7 The CLI (Decision J), with `benchmark_grade.py tracks` calling the same function.
- [ ] 3.8 The example (Decision I): the selection rule fixed and recorded first, then the archive and
  the stored expected grading.

## 4. Tests

- [ ] 4.1 The package's own tests (Decision L): the resolver's order and fall-through, the checksum
  refusal, no implicit fetch (a `file://` fixture), the records and synced copies, the example and
  the CLI. They need nothing but `wormbench`.
- [ ] 4.2 The cross-package tests: the alias identity (each old path imported first, in a fresh
  interpreter), the CLI against the script, and `--simulated` against the captures path.
- [ ] 4.3 Re-point the patches in `test_grading.py`, `test_benchmark_v2_grading.py`,
  `test_benchmark_v3.py`, `test_benchmark_v4.py` and `test_measurement.py`.
- [ ] 4.4 Deliberate bugs, each caught by a test:
  - a wrong resolver order;
  - a skipped checksum;
  - an import-time path;
  - a star re-export in place of an alias;
  - an implicit fetch;
  - a patch left on a script's re-exported name;
  - a stale in-wheel copy;
  - a validity record that falls back to v1;
  - `--simulated` without measurement.
- [ ] 4.5 Tooling: `testpaths`, Pyright's `include` and its 3.10 execution environment, coverage
  `source`, and the CI job that builds the wheel and tests it alone on 3.10 (at the declared minimums)
  and 3.13, joined to the aggregate `Tests` check.

## 5. Identity

- [ ] 5.1 `wormbench_identity_check.py check` and `speedup_identity_check.py check`: every case
  byte-identical.

## 6. Documentation

- [ ] 6.1 AGENTS.md: the package, its commands and the data resolver; and the rule that a fit
  manifest digests `packages/wormbench` too (Decision M).
- [ ] 6.2 `data/benchmark/IMPORT.md`: the `wormbench grade` route and `--simulated`.
- [ ] 6.3 CHANGELOG, under *Unreleased*, and a release checklist in CONTRIBUTING.
- [ ] 6.4 The spec's Purpose, written when the change is archived.

## 7. Merge

- [ ] 7.1 Wait for #541. Then rebase onto `main`, take both baselines again at the new
  `origin/main`, re-run the checks, run the full suite (when no fit holds the cores) and
  pre-commit, and merge.

## 8. Publishing (after the merge; each step confirmed with the maintainer first)

- [ ] 8.1 A Zenodo deposit of the fetched files and `PROVENANCE.md`, with each file's licence. Its
  per-file URLs, pinned to the versioned record, go into the manifest (Decision O).
- [ ] 8.2 A TestPyPI release, installed into a clean Python 3.10 environment, with the example
  graded.
- [ ] 8.3 The PyPI release, tagged `wormbench-v1.0.0`, with the DOI added to the README and both
  `CITATION.cff` files.

## 9. Follow-up change

- [ ] 9.1 Chemotaxis for imported tracks: Chen et al.'s measured field and an import route, with
  their own controls.
