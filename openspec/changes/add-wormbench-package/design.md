# Design: the `wormbench` package

## Context

The grader is spread across these places today:

| Piece | Where | Size |
|---|---|---|
| Readouts, grading statistics, measurement model, chemotaxis statistics, track import, reference loaders | `quantumnematode/benchmark/` | 2,392 lines |
| The roaming/dwelling classifier and HMM loaders | `quantumnematode/validation/roaming_dwelling.py` | 413 lines |
| The per-step record | `quantumnematode/report/behaviour.py`, which already says it "must load without the brain, the agent or the environment" | 74 lines |
| Chen's measured butanone field, read by the chemotaxis grading | `quantumnematode/evaluation/odour_field.py` | 103 lines |
| Band loaders, families' bands at a size, `grade_populations*`, the measured grading, `_finite` | `scripts/analysis/benchmark_grade.py`, beside the batched simulation | 945 lines |
| The bands builder (`BandBuilder`), at any size, from the real tracks | `scripts/analysis/benchmark_bands.py`, beside the band CLI | 1,699 lines |

Every file that moves parses and compiles under Python 3.10. Its modules import with numpy 1.24 and
scipy 1.10. On 64 band-cohort animals they give repr-identical statistics under Python 3.10 with
numpy 1.24 and under Python 3.13 with numpy 2.5 (the first spec review, on macOS).

**Files read by the moved code:**

- **Path constants built at import.** `references.py:27–34` builds them from `parents[4]` and uses
  them as default arguments, and `roaming_dwelling.py:40–42` does the same for
  `data/roaming_dwelling/`. `measurement.py` reads through `BENCHMARK_DATA`. In a source checkout
  `parents[4]` is the repository root, so every in-repository test passes while a wheel is broken.
- **The controls' records.** `benchmark_grade.py`'s validity checks (`_validated`, `v2_is_valid`
  to `v4_is_valid`) read `docs/.../115-benchmark-v2/controls_v2.json` and the v3 and v4 records.
  `_validated` returns False when a file is missing, so without them the default silently falls to
  v1.
- **v1's kappa.** `BandBuilder` reads it from `docs/.../109-benchmark-grading/bands_record.json`,
  for every lawn, uniform-well and mutant band at a size with no vendored bands.
- **The classifier.** Every bands file names it by the repository path
  `data/benchmark/classifier.json`.
- **Paths against the repository root:** `track_populations` resolves a manifest's entries against
  `_REPO` (`benchmark_grade.py:276`), and `_builder` imports `benchmark_bands` through `sys.path`
  (`:96–97`).

## Goals and non-goals

**Goals:**

- an installable grader on Python 3.10 or later, depending on numpy and scipy only;
- one copy of the code, used by the project and by outsiders;
- every committed grading output unchanged;
- outsiders' simulated tracks graded as the project grades its own.

**Non-goals:**

- simulation, controllers, fits and the benchmark's controls;
- chemotaxis grading for imported tracks, which is a follow-up change;
- new families, statistics or bands;
- any submission, leaderboard or attribution workflow, retired with `benchmark-management`;
- publishing, which follows the merge as confirmed steps.

## Decision A: a workspace package with numpy and scipy

`packages/wormbench/pyproject.toml` declares:

- `requires-python = ">=3.10"`;
- `dependencies = ["numpy>=1.24", "scipy>=1.10"]`;
- the licence expression `Apache-2.0 AND CC0-1.0 AND CC-BY-4.0` (PEP 639), for the code and the
  bundled data;
- a neutral description: "Grade simulated or real *C. elegans* tracks against real-worm behaviour".

The root workspace already includes `packages/*`, and `quantum-nematode` adds `wormbench` as a
workspace dependency.

Python 3.10 reached end of life in October 2026, and on it numpy stops at 2.2 and scipy at 1.15. The
floor is kept for reach, at no cost today, and can rise in any later release.

## Decision B: what moves, and what changes in it

| Package module | From |
|---|---|
| `readouts`, `grading`, `measurement`, `chemotaxis`, `track_import`, `references` | `quantumnematode/benchmark/` |
| `classifier` | `validation/roaming_dwelling.py` |
| `records` (`BehaviourStep`) | `report/behaviour.py` |
| `odour_field` | `evaluation/odour_field.py` |
| `bands` | from `benchmark_grade.py`: `load_bands*`, `_band`, `family_bands*`, `mutant_bands`, `classifier`, `_v2_family_map`, `_kinematic_bands_at`, `_activity_bands_at`, `_builder` |
| `builder` | from `benchmark_bands.py`: `BandBuilder` and what it calls, including its own `mutant_bands`, `per_date_kinematic_tallies`, `per_date_activity_tallies`, `KINEMATICS_INDEX` and `ACTIVITY_INDEX` |
| `grade` | from `benchmark_grade.py`: `family_values`, `window`, `grade_populations*`, `_grade_with_share`, `_grade_measured`, `_small_lawn_derived`, `_chemotaxis_reading`, `chemotaxis_fragments`, `activity_values`, `kinematic_values`, `track_populations`, its `_finite`, and the validity checks |
| `data`, `cli` | new |

**Two modules for two `mutant_bands`.** `benchmark_grade.py`'s `mutant_bands(bands, mutant, n)`
and `benchmark_bands.py`'s `mutant_bands(mutant, control, n, rng, wild_type_scale, kappa)` are
different functions. They stay in different modules (`bands` and `builder`). Likewise the two
scripts' `_finite` differ (`benchmark_bands.py:664` turns keys into strings and handles tuples and
numpy integers), so each moves with its own module.

**The content changes, all of them** (everything else moves verbatim):

1. **Path handling** resolves when used (Decision D): no import-time path, `None` defaults, and a lazy
   module `__getattr__` for the old constants.

2. **`track_populations`** takes a `base` directory, against which a manifest's relative entries
   resolve. `benchmark_grade.py` passes `_REPO`, as it resolves them today. The CLI passes the
   manifest's directory.

3. **`_builder`, `_kinematic_bands_at` and `_activity_bands_at`** import `wormbench.builder`
   directly, not `benchmark_bands` through `sys.path`.

4. **The validity checks and `BandBuilder`'s kappa** read the package's records (Decision E).

5. **Docstrings** that name versions "B v2", "B v3" or "B v4" (for example
   `benchmark_grade.py:285, 319, 612, 675, 730`) say "benchmark v2" and so on, since the moved code
   is package source and follows its rule against planning labels.

6. **The measured import path** (Decision H), written as new functions in `track_import`. Today
   the importer cannot carry what measurement needs:

   - `read_csv` reads only `animal`, `t_s`, `x`, `y` and `on_lawn` (`track_import.py:183–191`);
   - `read_wcon` accepts `@` fields but discards them (`:252`);
   - `ImportedTrack` and `to_steps` carry no label or seed;
   - `import_population` cuts to the window first (`:395`).

   The new functions read the labels and the seed, and measure the whole track before windowing.
   The existing functions keep their signatures and outputs, so reading unmeasured tracks stays
   byte-identical, and the identity check covers it.

**Stays in the scripts:**

- `simulate_v2`, `batched_populations`, `capture_populations`;
- the arms and protocols, and the hazard controllers;
- `benchmark_bands.py`'s `build()` and command line, including its read of
  `data/roaming_dwelling/ji2021_readings.json`;
- the controls.

The two scripts re-export every name that moved, private names included, since fits and probes reach
for them (`c2_fit.py` uses `bg._builder` and `bg._kinematic_bands_at`, and the scripts use `_finite`
51 times). So every script runs unchanged.

## Decision C: old import paths alias the moved modules

- **Placement.** Each alias is registered at the top of its own parent package's `__init__`, before
  any other import there:

  - `quantumnematode/benchmark/__init__.py` for the six benchmark modules;
  - `quantumnematode/validation/__init__.py` for `roaming_dwelling` (its module `__getattr__` raises
    for unknown names, so the alias must be registered there);
  - `quantumnematode/report/__init__.py` for `behaviour`;
  - `quantumnematode/evaluation/__init__.py` for `odour_field`.

  Each sets both the `sys.modules` entry and the package attribute.

- **Same object.** `quantumnematode.benchmark.grading is wormbench.grading`. Every name, private ones
  included, and every class is the same object, so `isinstance`, dataclass identity and pickles
  work. Pickles written after the move need `wormbench` installed.

- **Pyright.** An alias in `sys.modules` resolves `from quantumnematode.benchmark import grading`,
  but not the dotted form `from quantumnematode.benchmark.grading import x` (the second review,
  checked on a scratch package: `reportMissingImports`). So each old module path keeps a `.pyi`
  stub (`from wormbench.grading import *`, plus the private names the scripts and tests use).

  - With the stubs, the scratch package checks with 0 errors, and nothing changes at run time.
  - Each dotted import through a stub still gives a `reportMissingModuleSource` warning. That rule
    is set to `none` for `packages/quantum-nematode/tests` in Pyright's configuration, so the check
    is clean, not merely free of errors.
  - The two dotted imports in the package's own source (`report/dtypes.py:10`,
    `evaluation/batched.py:79`) are rewritten to import from `wormbench`.
  - The eight in the tests (`test_benchmark_v2_grading.py:20–21`, `test_grading.py:23`,
    `test_measurement.py:14`, `test_benchmark_v3.py:24,218`, `test_benchmark_v4.py:21–22`) stay as
    they are and resolve through the stubs.

- **No cycle.** `wormbench` never imports `quantumnematode`. A test checks that importing
  `wormbench` leaves no `quantumnematode` module in `sys.modules`.

## Decision D: data resolved when used, checked, and never fetched implicitly

**Resolution happens when a file is used.** No module computes a data path at import. Default
arguments become `None`, and each resolves inside its function.

**The old constants** (`BENCHMARK_DATA`, `SCHEER_TRACKS_PATH` and the rest) stay importable from
outside through a module `__getattr__` that resolves them on access. This works for
`from x import CONSTANT` (the second review checked it). A module cannot read its own lazy constant
by bare name, so no moved code does. They resolve like this:

- `BENCHMARK_DATA` is the directory that would serve the `benchmark` namespace: the host root's
  `benchmark/` when one is set, else `WORMBENCH_DATA`'s, else the wheel's copy. It is kept for the
  in-repository callers that join paths to it (`test_scheer_store.py:106`,
  `benchmark_v2_controls.py:1559`, `patch_leaving_reference.py:38`).
- A file constant resolves as that file would.

**Caches** (`functools.cache` builders) are keyed by the resolved path, so a change of root cannot
return stale bands.

**Order, file by file**, over two namespaces: `benchmark`, now `data/benchmark/`, and
`roaming_dwelling`, now `data/roaming_dwelling/`. A file absent from one source falls through to the
next.

1. **A host root**, set by `wormbench.data.set_root(path)`. `quantumnematode/__init__.py` sets this
   repository's `data/`, but only if `data/benchmark/` exists there, so an installed
   `quantum-nematode` without the repository falls through to the other sources. Importing any
   `quantumnematode` module finishes that `__init__` first, so the project always reads its vendored
   files.
2. **`WORMBENCH_DATA`**: a directory with `benchmark/` and `roaming_dwelling/` subdirectories.
3. **The wheel** (`importlib.resources`, Decision F).
4. **The fetched cache**: `$XDG_CACHE_HOME/wormbench/<data version>/`, or `~/.cache/wormbench/...`.

A process that runs only `wormbench` code, such as a spawned worker that never imports
`quantumnematode`, has no host root. It resolves from the environment variable, the wheel and the
cache. Every file the project reads is either in the wheel (byte-identical, Decision F) or is a
track, which such a worker must be given by an explicit path or by `WORMBENCH_DATA`.

**Checks.** A shipped manifest gives each file's namespace, name, SHA256 and size.

- Every load through the resolver checks the SHA256 and refuses a mismatch, naming the file.
- A load from an explicit path (a test's toy file, a user's own bands) is not checked.
- `wormbench data verify` checks every resolvable file.

**URLs.** Download URLs live in a separate `urls.json`, which `sync` does not regenerate (Decision O).

**No implicit download.** Only `wormbench data fetch` downloads. A grading that needs an unresolvable
file fails with an error naming the file and the fetch command.

**Stored repository paths.** The `classifier` path that every bands file stores
(`data/benchmark/classifier.json`) is read as the resolver name `benchmark/classifier.json`. The
bands files are not edited, since their SHA256 sums are recorded in committed records.

**Tests** use a fixture that saves and restores the host root and the environment variable.

## Decision E: two records, read only as package files

`scripts/analysis/wormbench_records.py build` writes two files into `packages/wormbench/wormbench/_data/`:

- **`validity.json`**: for each benchmark version, its controls' verdict, the SHA256 of the bands
  they validated, and the committed record it came from. Those are
  `115-benchmark-v2/controls_v2.json` and the v3 and v4 records that `CONTROLS_V3_PATH` and
  `CONTROLS_V4_PATH` name today. The rule is the one `_validated` applies now: the verdict is VALID,
  and the recorded SHA256 equals the bands file's.
- **`kappa_v1.json`**: v1's kappa, from `109-benchmark-grading/bands_record.json`.

They are package files, outside the resolver's namespaces. They are always read from the package, in
the repository as in the wheel.

- `wormbench_records.py check` regenerates both from the committed records, and fails on any
  difference.
- It is a precondition of every identity run (Decision K).
- A test in the repository's suite runs it.

So in the repository the outputs cannot change: what the package reads equals what the scripts read
today, and the check proves it each time.

**The tests that patched the controls' paths.** These tests patch `bg.CONTROLS_V*_PATH` and
`bg.BANDS_V*_PATH` to exercise the validity rule and the default version:

- `test_benchmark_v2_grading.py:134–207`;
- `test_benchmark_v3.py:349–381`, including the patch of `mm.file_sha256` at `:355`;
- `test_benchmark_v4.py:242–261`.

After the move no module reads those names, so the patches would do nothing. These tests are
rewritten to pass a validity record through a fixture (an argument of the validity check, or a
record file given to it). The verdict-and-SHA rule itself is tested in `wormbench_records.py`'s
tests.

## Decision F: what the wheel carries, and how

**In the wheel, about 1.5 MB with the example:**

- under `benchmark/`:
  - `bands.json` and `bands_v2.json` to `bands_v4.json`;
  - `classifier.json`;
  - `scheer_tracker_noise.npz`;
  - `lopezcruz2019_reference.json` and `lopezcruz2019_split.json`;
  - `PROVENANCE.md` and the package's own `IMPORT.md` (Decision J);
- under `roaming_dwelling/`:
  - `reference_hmm.json` and `calibration_retry.json`, the classifier loaders' defaults;
  - `PROVENANCE.md`;
- `validity.json` and `kappa_v1.json`;
- the example (Decision I);
- the manifest.

**On Zenodo, fetched:**

- `scheer2023_tracks.npz` and `scheer2023_split.json`;
- `calhoun2014_tracks.npz` and `roberts2016_tracks.npz`;
- `chen2025_tracks.npz`;
- `lopezcruz2019_reorientations.json`.

**How the wheel gets them.** `wormbench_records.py sync` copies the in-wheel `data/` files into
`packages/wormbench/wormbench/_data/`, where they are committed, and regenerates the manifest. The
copies build correctly from the sdist, which `uv build` builds from by default, and they are present
in editable installs.

- **Git:** `.gitattributes` marks `packages/wormbench/wormbench/_data/** -text`, so line-ending
  normalisation cannot change a file whose SHA256 is pinned. The copies are **not** in Git LFS: a
  fresh worktree holds LFS pointers, and a wheel built there would ship a pointer.
- **Pre-commit:** `check-added-large-files` gets a `--maxkb` above the example archive's size.
- **Tests:** a repository test fails if any copy differs from its source or the manifest does not
  match `data/`.

## Decision G: versions and the default

- **`wormbench` 1.x defaults to benchmark v4,** and only a major release changes the default, since
  that changes what an unpinned user's grading means. If `validity.json` ever stops marking the
  pinned default VALID, the CLI refuses to grade without `--bands`, rather than grade on a version
  its own record does not support.
- **A later benchmark version may ship in a 1.x release,** marked VALID in `validity.json`, but
  only behind `--bands`. Its default pin is in the package's code, not derived from the newest VALID
  entry.
- **The repository's rule is unchanged.** `benchmark_grade.py` keeps its own default, the newest
  version whose controls are VALID, through its re-exported validity checks. So the project's
  scripts and the package's CLI may differ in default once a v5 exists, and they are compared only
  with `--bands` given.
- **Provenance goes into the CLI's output only.** `wormbench grade`'s output adds a `wormbench`
  block: the package version, the bands version and the bands' SHA256. `benchmark_grade.py`'s output
  stays as it is today. Its v1 output has no `bands` key, which `test_benchmark_v2_grading.py:164`
  asserts, and the identity requirement holds it to that.
- **The data version** (the Zenodo record) changes only when a fetched file changes.

## Decision H: simulated tracks are measured, as the project's are

`wormbench grade --simulated` passes each animal on Scheer's rig through `measurement.measure`,
with the lawn tracker model in the bands (`bands["measurement"]["lawn"]`), as `capture_populations`
does. Scheer's rig means the small lawn and the uniform well, as `benchmark-grading` requires. Local
search is never measured, and its animals need no seed or labels.

**Per animal**, it needs each second's primitive (`run`, `reversal`, `omega` or `delta`) and an
integer seed:

- in CSV, `primitive` and `seed` columns, the seed constant within an animal;
- in WCON, a per-record `@primitive` array and `@seed`. WCON's core rejects unprefixed fields
  (`track_import.py:252`).

It refuses:

- an animal missing either;
- two animals with the same seed, since they would draw the same noise;
- WCON records merged under one id that carry different `@seed`s.

**Order**, as the captures path has it:

1. lawn membership from the exact positions;
2. the whole track measured with that animal's seed;
3. then the window.

**Bands:** `--simulated` with `--bands v1` is refused, since `bands.json` has no measurement model.
v1 graded unmeasured tracks.

A cross-package test checks that the same steps graded through `--simulated` and through the
project's captures path give byte-identical gradings.

## Decision I: the example

- **The cohort:** 64 PD1074 OD 2 small-lawn animals from **band cohorts**. They are drawn by
  `numpy.random.default_rng(<seed recorded in the archive>)` from band-cohort rows with at least 90%
  finite positions. The builder calls `track_import_check.refuse_held_out` on its rows, so a
  held-out row cannot enter. The selection rule is fixed before any grading.
- **Storage:** one compressed archive, about 0.9 MB. `wormbench example --out <dir>` writes its
  CSVs, sidecars and manifest.
- **The expected grading** is stored. The test compares statistics with
  `math.isclose(rel_tol=1e-9, abs_tol=1e-12)`, and every pass, flag and count exactly, since
  floating-point results can differ in the last bits between macOS and Linux CI. The absolute floor
  keeps a statistic at or near zero from needing a bit-for-bit match.
- **Not a validation.** It is not tested for a pass: band-cohort animals are inside the bands they
  are graded against, so it demonstrates the grader and validates nothing.

## Decision J: the command line and its documentation

- `wormbench grade --manifest <file> --out <json> [--bands v1|v2|v3|v4] [--simulated]`.
  - Relative manifest entries resolve against the manifest's directory.
  - It calls the same package function as `benchmark_grade.py tracks`. The cross-package test grades
    one manifest with absolute paths through both, with `--bands` given. With the CLI's `wormbench`
    provenance block removed, the outputs must be byte-identical.
- `wormbench example --out <dir>`.
- `wormbench data fetch [--dir <path>]` and `wormbench data verify`.
- `wormbench info` prints the package, benchmark and data versions and where each file resolves.

The wheel's `IMPORT.md` is the package's own contract:

- the formats;
- `--simulated`;
- which families an imported population is graded on;
- with no repository commands and no logbook links.

The stale "all five" at the repository's `IMPORT.md:49` is fixed to the six families. The repository
file keeps the repository's commands and links back to the package's contract.

## Decision K: identity before and after

`scripts/analysis/wormbench_identity_check.py baseline|check --build <shared dir>`.

**Two trees, two environments.**

- The baseline runs in a worktree at `origin/main` and the check in the change's worktree, each in
  its own `uv sync`ed environment.
- Both write to one shared `--build` directory outside either tree.
- Each run records `quantumnematode.__file__`, `wormbench.__file__` (on the check) and
  `git rev-parse HEAD`. It refuses to run if a module's path lies outside its own tree, so the two
  stages cannot grade the same code.

**Inputs.**

- The off-tree inputs `speedup_identity_check.py` reads are copied into both trees' `build/` from
  the main checkout's: `build/npe-recovery/simulations.jsonl` (`npe_recovery.py:59`),
  `build/cma-recovery-v2/candidates.jsonl` and `.../attempt2/probe/answers.jsonl`
  (`cma_recovery_v2.py:54, 56, 952`). The copy checks the SHA pinned at `cma_recovery_v2.py:954`.
- The capture set is generated once, pinned by SHA256 and shared by both stages.

**Preconditions:**

- `wormbench_records.py check` passes.
- Every Scheer load either selects its rows only with `band_rows` (`side == "bands"`), as
  `wormbench.builder` does, or passes them through `track_import_check.refuse_held_out`. That script
  function is not importable from package code.

**Cases, band cohorts only:**

- **Real tracks:** the band-cohort Scheer animals through the import round trip, on v1 to v4.
- **Simulated, batched:** the Poisson (`constant`) and `pirouette` negatives on fixed seeds, on v1 to
  v4. This covers v2's and v3's chemotaxis statistics and v2's linear local-search ratio.
- **Captures:** the pinned capture set on v1, unmeasured, and on v4, measured.
- **The fits' objective:** gradings with `presence=False` on v2 and on v4.
- **Mutants:** through `mutant_bands` at a vendored size, and at a size with none, which exercises
  `paired_direction`.
- **The band builder at unvendored sizes:** v1's lawn and uniform well, `_kinematic_bands_at` and
  `_activity_bands_at`. `BandBuilder` at 64 and 200 must reproduce the vendored bands.
- **A committed simulated record:** C.1's v3 answer's v4 populations from Logbook 124's
  `read.json` (`c1_v3_answer`, base 525,000,000). They are re-simulated and re-graded, and must equal
  the committed statistics. No held-out row enters it. Logbook 124's controls read held-out rows
  and are not used.

`speedup_identity_check.py` also runs:

- through the identity check's module-path guard;
- with `--workers 1`;
- with `--build` set to its own subdirectory of the shared directory, since its `baseline.jsonl`
  and `baseline_manifest.json` would otherwise collide, and `--out` in it, so Logbook 118's committed
  `identity.json` is never overwritten.

Its baseline is taken before the move.

**Absolute comparisons run at the baseline too.** Two cases compare today's code against outputs
committed in Logbooks 118 and 124: the C.1 record, and the speed-up check's stored jobs. Code has
changed since those logbooks, always under the byte-identity rule. So both run first at `origin/main`.
A mismatch there is drift already on `main`: it is reported, and not attributed to the move. The
change must then reproduce the baseline's outputs. `connectome_identity_check.py` adds `packages/wormbench` to its `SOURCE_PATHS`,
since the batched evaluator runs `OdourField`.

Any difference blocks the merge.

**The machine.** Identity runs and test runs pin OMP and torch to one thread. They run only when
they keep the machine within 16 worker processes in total, so not beside a 16-worker fit. The
speed-up check is about 59 jobs at 20–30 s, roughly 25 minutes.

After the rebase onto `main` once #541 is in (#541 edits `benchmark_grade.py`), both baselines are
taken again.

## Decision L: testing and tooling

- **Package tests** live in `packages/wormbench/tests/wormbench_tests/`, mirroring
  `quantum-nematode`'s layout. There is no `__init__.py` at `packages/wormbench/tests/`, and the
  standalone job asserts that `wormbench.__file__` lies in site-packages, so it tests the wheel, not
  the source tree. They cover:

  - the resolver's order and fall-through;
  - the checksum refusal;
  - no implicit fetch, tested against a `file://` fixture;
  - the example's grading;
  - the CLI.

  A conftest fixture replaces `socket.socket` with one that raises, so no test can reach the
  network. They need nothing but `wormbench`.

- **Repository tests** live in `quantum-nematode`'s suite:

  - the alias identity, each old path imported first in a fresh interpreter;
  - the CLI against `benchmark_grade.py tracks`;
  - `--simulated` against the captures path;
  - `wormbench_records.py check`;
  - the synced copies and the manifest against `data/`;
  - the held-out refusal in the example builder and the identity check.

- **The validity-rule tests,** rewritten as in Decision E.

- **Other re-pointed patches:**

  - `test_grading.py:478–479` patches `bg.classifier` and `bg.family_values`, and now patches them on
    `wormbench.grade`;
  - `test_measurement.py:174` loads a toy library, and now passes it by explicit path.

  Imports stay as they are.

- **Pyright:** `packages/wormbench` is added to `include`, with an execution environment at
  `pythonVersion = "3.10"`, plus the stubs of Decision C.

- **Ruff:** the package's own `[tool.ruff]` extends the root's, with `target-version = "py310"`, so
  autofixes cannot introduce 3.11+ idioms.

- **Coverage:** `source` adds `wormbench`.

- **CI:**

  - a job builds the wheel, installs it into fresh virtual environments on Python 3.10 (at numpy 1.24
    and scipy 1.10, the declared minimums) and 3.13, and runs the package's tests with
    `-o addopts=""`, without xdist;
  - it joins the aggregate `Tests` check;
  - `.test_durations` is not refreshed in this PR.

## Decision M: relation to existing specs and manifests

- **The module paths existing specs name** stay valid through the aliases:
  `quantumnematode.benchmark.readouts.reversal_onsets` (`benchmark-readouts`),
  `quantumnematode.benchmark.references` (`benchmark-references`), and `benchmark_grade.py`'s
  defaults (`benchmark-controls`). Their requirements are not modified.
- **`benchmark-controls`' rule** that PD1074's held-out cohorts are not read again is honoured: no
  case or example reads them, and both go through `refuse_held_out`.
- **Fit manifests.** A fit manifest built like C.2's digests `packages/quantum-nematode/quantumnematode`
  (`c2_fit.py`'s `PACKAGE`). After the move, such a manifest no longer covers the grader, so every
  later fit manifest also digests `packages/wormbench/wormbench`. AGENTS.md states it. C.2's own
  guarded stages refuse on `main` after this merge, which is why it merges after #541.

## Decision N: licences, provenance and citation

- **Code:** Apache-2.0.
- **Data:** each file keeps its deposit's licence: CC0-1.0 for Scheer, Calhoun and Roberts; CC BY 4.0
  for López-Cruz and Chen.
- **Provenance:** `PROVENANCE.md` ships in the wheel and in the Zenodo record.
- **Package files:**
  - a NOTICE with the data attributions;
  - its own `CITATION.cff`, naming the deposits to cite;
  - release tags of the form `wormbench-v<version>`;
  - a release checklist in CONTRIBUTING.

## Decision O: Zenodo before and after the deposit

`urls.json` holds each fetched file's URL, pinned to a versioned Zenodo record.

- **Until the deposit** (tasks §8), every URL is `null`, and `data fetch` refuses, naming the missing
  record.
- **After it,** the deposit's URLs are written by hand in a follow-up commit. `sync` never touches
  them.
- **Testing:** fetching is tested against a `file://` fixture only.

## Risks

| Risk | Mitigation |
|---|---|
| A second copy drifting | Decision C: there is one copy, and the old paths are aliases |
| The C.2 manifest | The merge order |
| The source-layout trap | Decision D (resolution when used), and the CI job that tests the built wheel alone |
| An identity check that passes vacuously | Decision K: two trees, two environments, recorded module paths, refusal |
| A wheel without the tracks surprising a user | The explicit error, and the example |
| Publishing under the wrong name or licence | The confirmed publishing steps in `tasks.md` §8 |
