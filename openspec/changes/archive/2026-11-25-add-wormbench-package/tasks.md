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

- [x] 2.1 A worktree at `origin/main` with its own `uv sync`ed environment, and the change's worktree
  with its own.

- [x] 2.2 `scripts/analysis/wormbench_identity_check.py baseline|check --build <shared dir>`
  (Decision K):

  - it records the commit and module paths, and refuses a run whose modules lie outside its tree;
  - it copies the three off-tree inputs into both trees and checks the pinned SHA;
  - it generates the capture set once and pins it by SHA256;
  - it passes every Scheer load through `refuse_held_out`.

  The baseline is taken at `origin/main`.

- [x] 2.3 `speedup_identity_check.py baseline` at `origin/main`, through the module-path guard, with
  `--workers 1`, and `--build` and `--out` in its own subdirectory of the shared directory.

- [x] 2.4 The two absolute comparisons (C.1's v4 record, the speed-up check's stored jobs) run at
  `origin/main` first. Any mismatch there is reported as drift already on `main`.

**Done (2026-11-24).** The baseline at `origin/main` (b0008abf) took 4 min on one thread. It shows no drift on `main`: C.1's v3 answer, re-graded at base 525,000,000, equals Logbook 124's committed values in all 21 statistics of all 5 populations, and `BandBuilder` reproduces the vendored v1 bands at 64 and 200. The speed-up check's baseline wrote 13 records.

## 3. The package

- [x] 3.1 `packages/wormbench/` with its `pyproject.toml` (Decision A), README, NOTICE,
  `CITATION.cff` and Ruff configuration.
- [x] 3.2 Move the modules of Decision B, making exactly its six content changes and no others:
  - path handling resolved when used;
  - `track_populations`'s `base`;
  - direct imports of `wormbench.builder`;
  - the package records;
  - the docstring labels;
  - the measured import path, as new functions, with the existing reading functions unchanged.
- [x] 3.3 `bands`, `builder` and `grade`, with the two `mutant_bands` and the two `_finite` in their
  own modules. The scripts re-export every moved name, private ones included.
- [x] 3.4 The aliases at the top of each parent `__init__`, the `.pyi` stubs for the old paths, and
  the two dotted imports in the package source rewritten (Decision C). `quantum-nematode` depends on
  `wormbench`, and `quantumnematode/__init__.py` sets the host root.
- [x] 3.5 The resolver, the manifest and `urls.json` (Decisions D and O).
  `wormbench_records.py build|sync|check` (Decisions E and F), with the stored classifier path mapped
  to its resolver name, `-text` for `_data/**`, and the large-file hook's limit raised.
- [x] 3.6 The measured route (Decision H): per-animal seeds and labels, duplicate seeds refused,
  and v1 refused.
- [x] 3.7 The CLI with its `wormbench` provenance block, and `benchmark_grade.py tracks` calling the
  same function (Decisions G and J). The package's own `IMPORT.md`.
- [x] 3.8 The example (Decision I):
  - the selection rule fixed and recorded first;
  - its rows passed through `refuse_held_out`;
  - then the archive and the stored expected grading.

**Done.** Notes on what differs from the design's wording:

- `benchmark_bands.py`'s own `_finite` stays in the script, since no moved code calls it.
- The package's track contract is in its README, which the wheel carries as its description, rather than in a second `IMPORT.md`.
- The resolver checks a listed file found in `WORMBENCH_DATA`, the package or the fetched cache. The host's root is read as given, as an explicit path is: it is the project's `data/`, whose agreement with the manifest `wormbench_records.py check` enforces.
- The example archive is 1.03 MB, and the wheel 1.34 MB.

## 4. Tests

- [x] 4.1 The package's tests in `packages/wormbench/tests/wormbench_tests/` (Decision L):

  - the resolver;
  - the checksum refusal;
  - no implicit fetch;
  - the example, compared with the 1e-9 tolerance;
  - the CLI.

  A socket-refusing fixture blocks the network, and there is no top-level tests `__init__.py`.

- [x] 4.2 The repository tests:

  - the alias identity (each old path imported first, in a fresh interpreter);
  - the CLI against the script;
  - `--simulated` against the captures path;
  - `wormbench_records.py check`;
  - the synced copies and the manifest;
  - the held-out refusal.

- [x] 4.3 Rewrite the validity tests to take their record from a fixture
  (`test_benchmark_v2_grading.py:134–207`, `test_benchmark_v3.py:349–381`,
  `test_benchmark_v4.py:242–261`). Re-point the patches in `test_grading.py:478–479`, and pass
  `test_measurement.py`'s toy library by explicit path.

- [x] 4.4 Deliberate bugs, each caught by a test (see the note below for how each was checked):

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

- [x] 4.5 Tooling:

  - `testpaths`;
  - Pyright's `include` with a 3.10 execution environment;
  - coverage `source`;
  - `connectome_identity_check.py`'s `SOURCE_PATHS`;
  - the CI job that builds the wheel and tests it alone on 3.10 (at the declared minimums) and 3.13,
    asserting `wormbench.__file__` is in site-packages, joined to the aggregate `Tests` check.

**4.4, run as mutations** (each planted alone, run against the test that should catch it, then restored; the clean code passes all of them). Caught:

- `WORMBENCH_DATA` outranking the host root;
- a skipped checksum;
- an import-time path;
- a star re-export in place of an alias;
- a stale in-package copy;
- a validity check that ignores its record;
- `--simulated` without measurement;
- a 1e-15 shift in a band centre, caught by the identity check itself.

A shared seed and a held-out row are covered by direct tests. A missing stub shows as a Pyright error, which pre-commit runs. No generic test can catch a patch left on a script's re-exported name: the tests that used such patches were rewritten. The single-tree identity run is the check's own guard.

**4.5.** Done locally:

- the wheel, built from its sdist, installed alone in a fresh Python 3.10.22 environment with numpy 1.24.4 and scipy 1.10.1, imports from site-packages with no `quantumnematode` module;
- its 23 tests pass there;
- the CI job `wormbench alone` is written, and has not yet run on CI.

## 5. Identity

- [x] 5.1 `wormbench_records.py check`, then `wormbench_identity_check.py check` and
  `speedup_identity_check.py check`, in the change's worktree and environment: every case
  byte-identical.

**Done.**

- `wormbench_identity_check.py check`: 23 of 23 outputs byte-identical. After the script's `tracks` branch moved onto the package function, the tracks, captures and batched cases were re-checked: 18 of 18 identical.
- `speedup_identity_check.py check`: 59 of 59 jobs identical, including the stored records.

## 6. Documentation

- [x] 6.1 AGENTS.md:
  - the package, its commands and the data resolver;
  - the rule that a fit manifest digests `packages/wormbench` too (Decision M);
  - the repository's default-version rule beside the package's (Decision G).
- [x] 6.2 The repository's `data/benchmark/IMPORT.md`: the `wormbench grade` route and
  `--simulated`, and the stale "all five" at line 49 fixed.
- [x] 6.3 CHANGELOG, under *Unreleased*, and a release checklist in CONTRIBUTING.
- [ ] 6.4 The spec's Purpose, written when the change is archived.

## 7. Merge

- [ ] 7.1 Wait for #541. Then:
  - rebase onto `main`;
  - take both baselines again at the new `origin/main` and re-run the checks;
  - run the full suite, when no fit holds the cores, and pre-commit;
  - merge.

**Done to the merge (2026-11-25).** #541's fit, grade and read finished, and its close-out runs no further `c2_fit.py` stage, so this change may merge first. #541 then resolves one conflict in `.pre-commit-config.yaml`, keeping both edits. `main` (3a4603bc, with numpy 2.5.5) came in by a merge commit, on a tree identical to the rebased one. Both baselines were re-taken there in freshly synced environments:

- `wormbench_identity_check.py`: 23 of 23 outputs identical.
- `speedup_identity_check.py`: 59 of 59 jobs identical.
- No drift on `main`: C.1's v3 answer matches Logbook 124 in all 21 statistics of all 5 populations, and the builder reproduces the v1 bands at 64 and 200.
- `wormbench_records.py check` passes.
- The full suite passes (9,381 passed, 35 skipped, 2 xfailed), and so does `pre-commit run -a`.

The merge waits for the maintainer.

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
