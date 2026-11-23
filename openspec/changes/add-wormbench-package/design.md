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

Every file that moves compiles on Python 3.10. Its modules import with numpy 1.24 and scipy 1.10,
and on 64 band-cohort animals they give repr-identical statistics under Python 3.10 with numpy 1.24
and under Python 3.13 with numpy 2.5 (the spec review, on macOS).

**Files read by the moved code** (the review's B2 and S1):

- **Path constants built at import.** `references.py:27–34` builds them from `parents[4]` and uses
  them as default arguments, and `roaming_dwelling.py:40–42` does the same for
  `data/roaming_dwelling/`. `measurement.py` reads through `BENCHMARK_DATA`. In a source checkout
  `parents[4]` is the repository root, so every in-repository test passes while a wheel is broken.
- **The controls' records.** `benchmark_grade.py`'s validity checks read
  `docs/.../115-benchmark-v2/controls_v2.json` and the v3 and v4 records. `_validated` returns
  False when a file is missing, so without them the default silently falls to v1.
- **v1's kappa.** `BandBuilder` reads it from `docs/.../109-benchmark-grading/bands_record.json`,
  for every lawn, uniform-well and mutant band at a size with no vendored bands.
- **The Ji 2021 readings.** `BandBuilder` reads `data/roaming_dwelling/ji2021_readings.json`.
- **The classifier.** Every bands file names it by the repository path `data/benchmark/classifier.json`.

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

The root workspace already includes `packages/*`. `quantum-nematode` adds `wormbench` as a workspace
dependency.

Python 3.10 reached end of life in October 2026, and on it numpy stops at 2.2 and scipy at 1.15. The
floor is kept for reach, at no cost today. It can rise in any later release.

## Decision B: what moves, and what stays

**Moves into `wormbench`:**

| Package module | From |
|---|---|
| `readouts`, `grading`, `measurement`, `chemotaxis`, `track_import`, `references` | `quantumnematode/benchmark/` |
| `classifier` | `validation/roaming_dwelling.py` |
| `records` (`BehaviourStep`) | `report/behaviour.py` |
| `odour_field` | `evaluation/odour_field.py` |
| `bands` | `benchmark_grade.py`: `load_bands*`, `_band`, `family_bands*`, `mutant_bands`, `classifier`, `_v2_family_map`, `_kinematic_bands_at`, `_activity_bands_at`, `_builder`; and `benchmark_bands.py`: `BandBuilder` and what it calls |
| `grade` | `benchmark_grade.py`: `family_values`, `window`, `grade_populations*`, `_grade_with_share`, `_grade_measured`, `_small_lawn_derived`, `_chemotaxis_reading`, `chemotaxis_fragments`, `activity_values`, `kinematic_values`, `track_populations`, `_finite`, the `v*_is_valid` checks |
| `data`, `cli` | new |

Content is unchanged except for import paths and the path handling of Decision D.

**Stays in the scripts:** `simulate_v2`, `batched_populations`, `capture_populations`, the arms and
protocols, the hazard controllers, the band command line and the controls. `benchmark_grade.py` and
`benchmark_bands.py` re-export every name that moved, private names included. Fits and probes reach
for several of them: `c2_fit.py` uses `bg._builder` and `bg._kinematic_bands_at`, and the scripts
use `_finite` 51 times. So every script runs unchanged.

## Decision C: old import paths alias the moved modules

- **Placement.** Each alias is registered in its own parent package's `__init__`:

  - `quantumnematode/benchmark/__init__.py` for the six benchmark modules;
  - `quantumnematode/validation/__init__.py` for `roaming_dwelling`;
  - `quantumnematode/report/__init__.py` for `behaviour`;
  - `quantumnematode/evaluation/__init__.py` for `odour_field`.

  Each sets both the `sys.modules` entry and the package attribute. `validation/__init__.py` has a
  module `__getattr__` that raises for unknown names, so registering the alias there is required,
  not optional.

- **Same object.** `quantumnematode.benchmark.grading is wormbench.grading`. Every name, private ones
  included, and every class is the same object, so `isinstance`, dataclass identity and pickles
  work. Pickles written after the move need `wormbench` installed.

- **Pyright.** Each parent's `__init__` re-exports explicitly
  (`from wormbench import grading as grading`). Code inside `quantum-nematode` that imports a moved
  module by its dotted path is rewritten to import from `wormbench`, about eight sites (for example
  `report/dtypes.py:10` and `evaluation/batched.py`'s imports). With `reportMissingImports` set to
  error, nothing then resolves through a missing file.

- **No cycle.** `wormbench` never imports `quantumnematode`. A test checks that importing
  `wormbench` leaves no `quantumnematode` module in `sys.modules`.

*Alternative rejected:* star re-exports. They drop private names, and they create a second module
object, so module-level state and `isinstance` checks could diverge.

## Decision D: data resolved when used, checked, and never fetched implicitly

**Resolution happens when a file is used.** No module computes a data path at import. Default
arguments become `None`, and the path resolves inside the function. The old constants
(`BENCHMARK_DATA`, `SCHEER_TRACKS_PATH` and the rest) stay importable through a module `__getattr__`
that resolves them on access. Caches (`functools.cache` builders) are keyed by the resolved path, so
a change of root cannot return stale bands.

**Order, file by file**, over two namespaces, `benchmark` (now `data/benchmark/`) and
`roaming_dwelling` (now `data/roaming_dwelling/`). A file absent from one source falls through to the
next.

1. `WORMBENCH_DATA`: a directory with `benchmark/` and `roaming_dwelling/` subdirectories.
2. A host root, set by `wormbench.data.set_root(path)`. `quantumnematode/__init__.py` sets this
   repository's `data/` before anything else imports, so the project always reads its vendored
   files and never downloads.
3. The wheel (`importlib.resources`, Decision F).
4. The fetched cache: `$XDG_CACHE_HOME/wormbench/<data version>/`, or `~/.cache/wormbench/...`.

**Checks.** A shipped manifest gives each file's namespace, name, SHA256, size and download URL. Every
load through the resolver checks the SHA256 and refuses a mismatch, naming the file. A load from an
explicit path (a test's toy file, a user's own bands) is not checked. `wormbench data verify` checks
every resolvable file.

**No implicit download.** Only `wormbench data fetch` downloads. A grading that needs an unresolvable
file fails with an error naming the file and the fetch command.

**Stored repository paths.** The `classifier` path that every bands file stores
(`data/benchmark/classifier.json`) is read as the resolver name `benchmark/classifier.json`. The
bands files are not edited, since their SHA256 sums are recorded in committed records.

**Tests** use a fixture that sets and clears the host root and the environment variable, so no test
depends on xdist's import order.

## Decision E: two records generated from committed ones

`scripts/analysis/wormbench_records.py build` writes two files into the package's data:

- `validity.json`: for each benchmark version, its controls' verdict, the SHA256 of the bands they
  validated, and the committed record it came from (`115-benchmark-v2/controls_v2.json`, and the
  v3 and v4 records the scripts read now);
- `kappa_v1.json`: v1's kappa, from `109-benchmark-grading/bands_record.json`.

The validity checks and `BandBuilder` read these through the resolver. In the repository the host
root is set, so they keep reading the original records, and the outputs cannot change.
`wormbench_records.py check` regenerates both and fails on any difference. A test runs it.

## Decision F: what the wheel carries, and how

**In the wheel, under 1.5 MB:**

- `benchmark/`:
  - `bands.json` and `bands_v2.json` to `bands_v4.json`;
  - `classifier.json`;
  - `scheer_tracker_noise.npz`;
  - `lopezcruz2019_reference.json` and `lopezcruz2019_split.json`;
  - `PROVENANCE.md` and `IMPORT.md`;
- `roaming_dwelling/`:
  - `ji2021_readings.json`;
  - `reference_hmm.json`;
  - `calibration_retry.json`;
  - `PROVENANCE.md`;
- `validity.json` and `kappa_v1.json`;
- the example (Decision I);
- the manifest.

**On Zenodo, fetched:**

- `scheer2023_tracks.npz` and `scheer2023_split.json`;
- `calhoun2014_tracks.npz` and `roberts2016_tracks.npz`;
- `chen2025_tracks.npz`;
- `lopezcruz2019_reorientations.json`;
- `roaming_dwelling/scheer2023_windows.npz`, if a moved reader needs it.

**How the wheel gets them.** `wormbench_records.py sync` copies the in-wheel files from `data/` into
`packages/wormbench/wormbench/_data/`, and they are committed there. A test fails if any copy differs
from its source. Committed copies build correctly from the sdist, which `uv build` builds from by
default, and they are present in editable installs. That is why they are copies rather than a
build-time include from `../../data`. `sync` also regenerates the manifest, and the same test checks
the manifest against `data/`.

## Decision G: versions

- `wormbench` 1.0.0 grades on benchmark v4 by default, with v1 to v3 available and frozen.
- Every grading records its bands version and SHA256.
- **A change of default is a major release**, since it changes what an unpinned user's grading
  means. A new benchmark version may ship in a minor release, but only behind `--bands`.
- The data version (the Zenodo record) changes only when a fetched file changes.

## Decision H: simulated tracks are measured, as the project's are

`wormbench grade --simulated` passes each animal through `measurement.measure`, with the lawn
tracker model in the bands (`bands["measurement"]["lawn"]`), before windowing. The project's
captures path does the same. It applies on Scheer's rig only (the small lawn and the uniform well),
as `benchmark-grading` requires. Local search is never measured.

It needs, per animal:

- each second's primitive, as `run`, `reversal`, `omega` or `delta`, in a `primitive` column of the
  CSV or a WCON field;
- an integer `seed` in the sidecar or the WCON metadata.

It refuses an animal missing either. `IMPORT.md` documents the contract. A test checks that the same
steps graded through `--simulated` and through the project's captures path give byte-identical
gradings.

## Decision I: the example

- **The cohort:** 64 PD1074 OD 2 small-lawn animals from **band cohorts** (`side == "bands"`). They
  are drawn by `numpy.random.default_rng(<seed recorded in the file>)` from the band-cohort rows with
  at least 90% finite positions. The selection rule is fixed before any grading.
- **Storage:** one compressed archive, about 0.9 MB. `wormbench example --out <dir>` writes its
  CSVs, sidecars and manifest.
- **The test:** grade it and compare with a stored expected grading. It is not tested for a pass:
  band-cohort animals are inside the bands they are graded against, so it demonstrates the grader
  and validates nothing.

## Decision J: the command line

- `wormbench grade --manifest <file> --out <json> [--bands v1|v2|v3|v4] [--simulated]`.
  - The manifest's relative entries resolve against the manifest's own directory.
  - `benchmark_grade.py tracks` calls the same package function, so for the same absolute paths both
    write byte-identical files.
- `wormbench example --out <dir>`.
- `wormbench data fetch [--dir <path>]` and `wormbench data verify`.
- `wormbench info` prints the package, benchmark and data versions and where each file resolves.

## Decision K: identity before and after

`scripts/analysis/wormbench_identity_check.py baseline|check`, run with one worker.

- The baseline is taken in a separate worktree at `origin/main`, since the main checkout serves the
  C.2 fit.
- Both runs use the same relative invocation, and paths are normalised in the comparison.
- After the rebase onto `main` once #541 is in (#541 edits `benchmark_grade.py`), both baselines are
  taken again.

**Cases, band cohorts only:**

- **Real tracks:** the band-cohort Scheer animals through the import round trip on v1 to v4.
- **Simulated, batched:** the Poisson (`constant`) and `pirouette` negatives on fixed seeds on v1 to
  v4. This covers v2's and v3's chemotaxis statistics and v2's linear local-search ratio.
- **Captures:** one capture set on v1, unmeasured, and on v4, measured.
- **The fits' objective:** a v2 grading with `presence=False`.
- **Mutants:** through `mutant_bands` at a vendored size, and at a size with none, which exercises
  `paired_direction`.
- **The band builder at unvendored sizes:** v1's lawn and uniform well, `_kinematic_bands_at` and
  `_activity_bands_at`. `BandBuilder` at 64 and 200 must reproduce the vendored bands.
- **A stored v4 record**, re-graded from its inputs.

`speedup_identity_check.py` also runs, with `--workers 1` and `--out` in the scratch directory, so
Logbook 118's committed `identity.json` is never overwritten. Its own baseline is taken before the
move. Any difference blocks the merge.

## Decision L: testing and tooling

- **Package tests** live in `packages/wormbench/tests/`, which is added to the root `testpaths`:

  - the resolver's order and fall-through;
  - the checksum refusal;
  - no implicit fetch, tested against a `file://` fixture;
  - the generated records and the synced copies;
  - the example's grading;
  - the CLI.

  They need nothing but `wormbench`.

- **Tests that need both packages** live under `quantum-nematode`'s tests: the alias identity, the
  CLI against `benchmark_grade.py tracks`, and `--simulated` against the captures path.

- **Re-pointed patches.** Several existing tests patch moved names through the script modules:

  - `test_grading.py:478–479` patches `bg.classifier` and `bg.family_values`;
  - `test_benchmark_v2_grading.py`, `test_benchmark_v3.py` and `test_benchmark_v4.py` patch
    `bg.CONTROLS_V*_PATH` and `bg.BANDS_V*_PATH`;
  - `test_measurement.py:174` patches `mm.BENCHMARK_DATA` with a toy library.

  These patches are re-pointed at the `wormbench` module that reads the name, or at an explicit
  path. Their imports stay as they are.

- **Pyright:** `packages/wormbench` is added to `include`, with an execution environment at
  `pythonVersion = "3.10"`.

- **Ruff:** the package's own `[tool.ruff]` extends the root's, with `target-version = "py310"`, so
  autofixes cannot introduce 3.11+ idioms.

- **Coverage:** `source` adds `wormbench`.

- **CI:** a job builds the wheel, installs it into fresh virtual environments on Python 3.10 (with
  numpy 1.24 and scipy 1.10, the declared minimums) and 3.13, and runs the package's tests with
  `-o addopts=""`, without xdist and with no network. The job joins the aggregate `Tests` check.
  `.test_durations` is not refreshed in this PR.

## Decision M: relation to existing specs and manifests

- **The module paths existing specs name** stay valid through the aliases:
  `quantumnematode.benchmark.readouts.reversal_onsets` (`benchmark-readouts`),
  `quantumnematode.benchmark.references` (`benchmark-references`), and `benchmark_grade.py`'s
  defaults (`benchmark-controls`). Their requirements are not modified.
- **`benchmark-controls`' rule** that PD1074's held-out cohorts are not read again is honoured: no
  case or example reads them.
- **Fit manifests.** A fit manifest built like C.2's digests `packages/quantum-nematode/quantumnematode`
  (`c2_fit.py`'s `PACKAGE`). After the move, such a manifest no longer covers the grader, so every
  later fit manifest also digests `packages/wormbench/wormbench`. C.2's own guarded stages refuse on
  `main` after this merge, which is why it merges after #541.

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

Until the deposit (tasks §8), each fetched file's URL in the manifest is `null`. `data fetch` then
refuses, naming the missing record. The deposit's per-file URLs, pinned to its versioned record,
replace them in a follow-up commit, and fetching is tested only against a `file://` fixture.

## Risks

| Risk | Mitigation |
|---|---|
| A second copy drifting | Decision C: there is one copy, and the old paths are aliases |
| The C.2 manifest | The merge order |
| The source-layout trap | Decision D (resolution when used), and the CI job that tests the built wheel alone |
| A wheel without the tracks surprising a user | The explicit error, and the example |
| Publishing under the wrong name or licence | The confirmed publishing steps in `tasks.md` §8 |
