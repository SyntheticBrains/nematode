# Tasks: the `wormbench` package

Not a phase task. The maintainer decided on 2026-11-23 and 2026-11-24 to make the benchmark a
standalone package:

- named `wormbench`, a new package in this repository;
- the small reference files in the wheel, the tracks on Zenodo;
- a measured route for simulated tracks;
- chemotaxis import as a follow-up.

Two independent spec reviews' findings of 2026-11-24 are written into the design. No campaign runs
in this change, no experiment's input changes, and no PD1074 held-out row is read.

**It merges only after PR #541's registered fit, grade and read are done, or after PR #541 records
C.2 as banked.**

Every test or check run during this change pins OMP and torch to one thread and uses one worker. It
runs only when the machine stays within 16 worker processes in total, so never beside a 16-worker
fit.

## 1. Review

- [x] 1.1 `/nematode-review-spec`, by an independent reviewer: 4 blocking and 12 should-fix
  findings, written into the design.
- [x] 1.2 A second independent review: 2 blocking and 7 should-fix findings, written into the design.
- [x] 1.3 A short third look by the second reviewer: ready to implement once its two wording fixes (the measured import path as a sixth content change; band-cohort selection in the builder) were made, as they are.

## 2. Baselines, before any code moves

- [ ] 2.1 A worktree at `origin/main` with its own `uv sync`ed environment, and the change's worktree
  with its own.

- [ ] 2.2 `scripts/analysis/wormbench_identity_check.py baseline|check --build <shared dir>`
  (Decision K):

  - it records the commit and module paths, and refuses a run whose modules lie outside its tree;
  - it copies the three off-tree inputs into both trees and checks the pinned SHA;
  - it generates the capture set once and pins it by SHA256;
  - it passes every Scheer load through `refuse_held_out`.

  The baseline is taken at `origin/main`.

- [ ] 2.3 `speedup_identity_check.py baseline` at `origin/main`, through the module-path guard, with
  `--workers 1`, and `--build` and `--out` in its own subdirectory of the shared directory.

- [ ] 2.4 The two absolute comparisons (C.1's v4 record, the speed-up check's stored jobs) run at
  `origin/main` first. Any mismatch there is reported as drift already on `main`.

## 3. The package

- [ ] 3.1 `packages/wormbench/` with its `pyproject.toml` (Decision A), README, NOTICE,
  `CITATION.cff` and Ruff configuration.
- [ ] 3.2 Move the modules of Decision B, making exactly its six content changes and no others:
  - path handling resolved when used;
  - `track_populations`'s `base`;
  - direct imports of `wormbench.builder`;
  - the package records;
  - the docstring labels;
  - the measured import path, as new functions, with the existing reading functions unchanged.
- [ ] 3.3 `bands`, `builder` and `grade`, with the two `mutant_bands` and the two `_finite` in their
  own modules. The scripts re-export every moved name, private ones included.
- [ ] 3.4 The aliases at the top of each parent `__init__`, the `.pyi` stubs for the old paths, and
  the two dotted imports in the package source rewritten (Decision C). `quantum-nematode` depends on
  `wormbench`, and `quantumnematode/__init__.py` sets the host root.
- [ ] 3.5 The resolver, the manifest and `urls.json` (Decisions D and O).
  `wormbench_records.py build|sync|check` (Decisions E and F), with the stored classifier path mapped
  to its resolver name, `-text` for `_data/**`, and the large-file hook's limit raised.
- [ ] 3.6 The measured route (Decision H): per-animal seeds and labels, duplicate seeds refused,
  and v1 refused.
- [ ] 3.7 The CLI with its `wormbench` provenance block, and `benchmark_grade.py tracks` calling the
  same function (Decisions G and J). The package's own `IMPORT.md`.
- [ ] 3.8 The example (Decision I):
  - the selection rule fixed and recorded first;
  - its rows passed through `refuse_held_out`;
  - then the archive and the stored expected grading.

## 4. Tests

- [ ] 4.1 The package's tests in `packages/wormbench/tests/wormbench_tests/` (Decision L):

  - the resolver;
  - the checksum refusal;
  - no implicit fetch;
  - the example, compared with the 1e-9 tolerance;
  - the CLI.

  A socket-refusing fixture blocks the network, and there is no top-level tests `__init__.py`.

- [ ] 4.2 The repository tests:

  - the alias identity (each old path imported first, in a fresh interpreter);
  - the CLI against the script;
  - `--simulated` against the captures path;
  - `wormbench_records.py check`;
  - the synced copies and the manifest;
  - the held-out refusal.

- [ ] 4.3 Rewrite the validity tests to take their record from a fixture
  (`test_benchmark_v2_grading.py:134–207`, `test_benchmark_v3.py:349–381`,
  `test_benchmark_v4.py:242–261`). Re-point the patches in `test_grading.py:478–479`, and pass
  `test_measurement.py`'s toy library by explicit path.

- [ ] 4.4 Deliberate bugs, each caught by a test:

  - a wrong resolver order;
  - `WORMBENCH_DATA` outranking the host root;
  - a skipped checksum;
  - an import-time path;
  - a star re-export in place of an alias;
  - a missing stub;
  - an implicit fetch;
  - a patch left on a script's re-exported name;
  - a stale in-wheel copy;
  - a validity record that falls back to v1;
  - `--simulated` without measurement;
  - one seed shared by several animals;
  - a held-out row admitted;
  - an identity check run with both stages' modules in one tree.

- [ ] 4.5 Tooling:

  - `testpaths`;
  - Pyright's `include` with a 3.10 execution environment;
  - coverage `source`;
  - `connectome_identity_check.py`'s `SOURCE_PATHS`;
  - the CI job that builds the wheel and tests it alone on 3.10 (at the declared minimums) and 3.13,
    asserting `wormbench.__file__` is in site-packages, joined to the aggregate `Tests` check.

## 5. Identity

- [ ] 5.1 `wormbench_records.py check`, then `wormbench_identity_check.py check` and
  `speedup_identity_check.py check`, in the change's worktree and environment: every case
  byte-identical.

## 6. Documentation

- [ ] 6.1 AGENTS.md:
  - the package, its commands and the data resolver;
  - the rule that a fit manifest digests `packages/wormbench` too (Decision M);
  - the repository's default-version rule beside the package's (Decision G).
- [ ] 6.2 The repository's `data/benchmark/IMPORT.md`: the `wormbench grade` route and
  `--simulated`, and the stale "all five" at line 49 fixed.
- [ ] 6.3 CHANGELOG, under *Unreleased*, and a release checklist in CONTRIBUTING.
- [ ] 6.4 The spec's Purpose, written when the change is archived.

## 7. Merge

- [ ] 7.1 Wait for #541. Then:
  - rebase onto `main`;
  - take both baselines again at the new `origin/main` and re-run the checks;
  - run the full suite, when no fit holds the cores, and pre-commit;
  - merge.

## 8. Publishing (after the merge; each step confirmed with the maintainer first)

- [ ] 8.1 A Zenodo deposit of the fetched files and `PROVENANCE.md`, with each file's licence. Its
  per-file URLs, pinned to the versioned record, are written into `urls.json` (Decision O).
- [ ] 8.2 A TestPyPI release, installed into a clean Python 3.10 environment, with the example
  graded.
- [ ] 8.3 The PyPI release, tagged `wormbench-v1.0.0`, with the DOI added to the README and both
  `CITATION.cff` files.

## 9. Follow-up change

- [ ] 9.1 Chemotaxis for imported tracks: Chen et al.'s measured field and an import route, with
  their own controls.
