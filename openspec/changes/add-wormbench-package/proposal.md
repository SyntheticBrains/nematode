## Why

The benchmark is the project's validated asset. Its four versions each passed their controls
(Logbooks 110, 115, 120, 124), and no comparable instrument for simulated *C. elegans* behaviour
was found in the novelty search of 2026-10-30, in the literature-watch digests from W40 to W47, or
in the direction review's sweep of 2026-11-23. On 2026-11-23 and 2026-11-24 the maintainer decided
to make the grader a standalone package, so that anyone can grade their own simulated or real
tracks against the same bands, the same way the project grades its own.

As it stands, an outsider cannot install it:

- **It sits inside `quantum-nematode`.** That package is marked `Private :: Do Not Upload`, pins
  Python 3.13 exactly, describes itself as a quantum brain simulation, and pulls in Qiskit, PyTorch
  and the rest of the platform. The grading code itself needs only numpy, and scipy for building
  bands.
- **The population grading is in scripts.** The band loaders, the families' bands at a size,
  `grade_populations_v2`, `_v3` and `_v4` and the measured grading live in
  `scripts/analysis/benchmark_grade.py`, beside the simulation. The bands builder lives in
  `scripts/analysis/benchmark_bands.py`. The chemotaxis reading needs
  `quantumnematode/evaluation/odour_field.py`.
- **It reads repository files that a wheel would not have.** The version a grading defaults to is
  decided by reading the controls' records under `docs/`. The bands builder reads v1's kappa from a
  record under `docs/`. Every bands file names its classifier by a repository-relative path. Three
  modules build their data paths from `parents[4]` of their own file when they are imported.
- **Grading at most sizes needs the real tracks.** Each family is graded at the population's own
  size: from the vendored bands at the sizes they were built at (64 and 200), and otherwise from
  bands rebuilt from the deposits' tracks, 36 MB, vendored in this repository only.
- **Simulated tracks are graded differently from the project's.** From v2 on, the project passes
  its simulated tracks through the tracker measurement model, while imported tracks are never
  measured (`data/benchmark/IMPORT.md`). An outsider's simulated worm would be graded as if it were
  real.

The maintainer's decisions:

- **Name:** `wormbench`. It was free on PyPI on 2026-11-24.
- **Location:** a new package in this repository's uv workspace, `packages/wormbench/`, which
  `quantum-nematode` depends on, so the project and outsiders grade with the same code.
- **Data:** the small reference files ship in the wheel. The real tracks go to Zenodo under a DOI,
  fetched on request and checked against SHA256 sums shipped in the wheel.
- **Simulated tracks:** `wormbench grade --simulated` applies the same measurement model the project
  applies, given each animal's per-second primitive labels and a seed.
- **Chemotaxis for imported tracks** is a follow-up change. Imported populations are graded on the
  other families, as today.
- **Publishing** to PyPI and Zenodo is confirmed with the maintainer before any upload, and is not
  part of this change's merge.

**This is not NematodeBench.** The `benchmark-management` capability, a submission workflow with a
leaderboard and contributor attribution, was retired on 2026-07-25, and its spec asks that it not be
reintroduced without a new proposal. This change adds no submissions, leaderboard, verification or
attribution. It packages the grader that `benchmark-grading` already specifies.

## What Changes

### 1. The package

`packages/wormbench/` is a uv workspace member: Python 3.10 or later, depending on numpy and scipy
only. It holds the benchmark modules now in `quantumnematode/benchmark/`, the roaming/dwelling
classifier, the per-step record, the odour field, the population grading and band loaders from
`benchmark_grade.py`, and the bands builder from `benchmark_bands.py`. The simulation, the controls,
the band command line and the fits stay where they are. It also holds:

- a data layer that resolves each reference file when it is used (never when a module is imported),
  checks its SHA256, and never downloads except on request;
- two small records generated from the committed controls and band records: which versions are
  VALID, and v1's kappa;
- a command line: `wormbench grade` (with `--simulated`), `wormbench example`,
  `wormbench data fetch` and `wormbench data verify`, and `wormbench info`;
- an example: 64 band-cohort wild-type animals on Scheer's small lawn, written out by
  `wormbench example`, with a stored expected grading.

### 2. One grader for the project and outsiders

`quantumnematode` imports `wormbench`. Every old import path keeps working and resolves to the same
module object, and the scripts re-export every name they moved, private ones included.

### 3. Every committed grading unchanged

An identity check grades a fixed set of cases on `origin/main`, in its own worktree, and again after
the move, and finds every output byte-identical. The cases cover every benchmark version, both
simulated sources, real tracks, mutants, the fits' objective, and every band-builder path at sizes
with and without vendored bands. It reads band cohorts only.

### 4. Tooling and documentation

The new package is added to pytest's paths, Pyright (at Python 3.10), Ruff (targeting 3.10) and
coverage. A CI job builds the wheel and tests it alone on Python 3.10 and 3.13. Also added: a README
with the worked example, `IMPORT.md` extended for the simulated route, the AGENTS.md commands, a
NOTICE with the data attributions, a `CITATION.cff` for the package, and a CHANGELOG line.

## Impact

- **Merge order.** `scripts/analysis/benchmark_grade.py` is one of the 22 files the C.2 fit's
  manifest binds, and that manifest digests `packages/quantum-nematode/quantumnematode`
  (`docs/experiments/logbooks/supporting/133-c2-fit/manifest.json` on `feat/fit-c2`). This change
  **does not merge to `main` until PR #541's registered fit, grade and read are done**, or until #541
  records C.2 as banked. After the merge, C.2's guarded stages refuse on `main`, as intended. A later
  fit manifest digests `packages/wormbench` as well.
- **No registered verdict changes.** No bands, bar, statistic or rule changes; the frozen versions
  stay frozen. The identity check is the evidence.
- **Held-out data.** Nothing in this change reads PD1074's held-out cohorts, which
  `benchmark-controls` forbids reading again. The example and the identity cases use band cohorts.
- **Cost.** About 4–6 engineering days. Compute is the test suite and the identity check, run with one
  worker while a fit holds the machine's cores.
- **Specs.** A new capability, `benchmark-package`. The requirements of `benchmark-grading`,
  `-readouts`, `-references` and `-controls` stay true: the module paths they name keep resolving,
  through the aliases.
