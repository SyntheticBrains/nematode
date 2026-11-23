## ADDED Requirements

### Requirement: A standalone grader

The benchmark's grader SHALL be an installable package, `wormbench`, in `packages/wormbench/`. It
SHALL require Python 3.10 or later, and its only runtime dependencies SHALL be numpy and scipy.
Installed with nothing else from this repository, it SHALL import, grade imported tracks and run its
own tests. It SHALL hold:

- the readouts, the grading statistics, the measurement model, the chemotaxis statistics, the track
  import and the reference loaders;
- the roaming/dwelling classifier, the per-step record and the measured odour field;
- the band loaders, the families' bands at a population's size, and the band builder they call;
- the population grading of every benchmark version.

Importing it SHALL import no `quantumnematode` module.

#### Scenario: Installed alone on the oldest supported Python

- **WHEN** the built wheel is installed into a fresh Python 3.10 environment with numpy 1.24 and
  scipy 1.10, and nothing else from this repository
- **THEN** `import wormbench` SHALL succeed, its tests SHALL pass with every network connection
  refused, `wormbench.__file__` SHALL lie in that environment's site-packages, and `sys.modules`
  SHALL hold no `quantumnematode` module

#### Scenario: Its dependencies are numpy and scipy

- **WHEN** the package's metadata are read
- **THEN** its runtime dependencies SHALL be numpy and scipy, and nothing else

### Requirement: One grader for the project and for outsiders

`quantum-nematode` SHALL depend on `wormbench` and SHALL hold no second copy of any module that
moved. Each old import path SHALL resolve, at run time, to the same module object as its `wormbench`
counterpart, whichever of them is imported first:

- `quantumnematode.benchmark.<module>`;
- `quantumnematode.validation.roaming_dwelling`;
- `quantumnematode.report.behaviour`;
- `quantumnematode.evaluation.odour_field`.

The type checker SHALL resolve both the attribute and the dotted forms of each old path.
`scripts/analysis/benchmark_grade.py` and `scripts/analysis/benchmark_bands.py` SHALL re-export every
name that moved, private names included.

#### Scenario: The old path is the same module

- **WHEN** `quantumnematode.validation.roaming_dwelling` is imported first in a fresh interpreter,
  and then `wormbench.classifier`
- **THEN** they SHALL be the same object, and so SHALL each class and function reached through
  either, private names included

#### Scenario: Dotted imports of old paths type-check

- **WHEN** Pyright checks a test that imports `from quantumnematode.benchmark.grading import ...`
- **THEN** it SHALL report no missing import

#### Scenario: The project's benchmark tests keep their imports

- **WHEN** the existing benchmark tests run with their import statements unchanged, each patch of a
  moved name pointed at the module that reads it, and the validity tests given their record by a
  fixture
- **THEN** they SHALL pass, and a patch left on a script's re-exported name SHALL be caught by a
  deliberate-bug test

### Requirement: Moving the grader changes no output

Moving the grader SHALL leave every output of the project's grading scripts byte-identical. An
identity check SHALL grade a fixed set of cases twice:

- first in a worktree at `origin/main`, then on the change;
- each in its own environment, writing to one shared directory;
- reading band cohorts only.

It SHALL record each run's commit and module paths, and SHALL refuse a run whose modules lie outside
its own tree. Its cases SHALL include:

- real tracks through the import round trip, on every benchmark version;
- the batched negatives on fixed seeds, on every version;
- a capture set pinned by SHA256, unmeasured and measured;
- gradings without presence, on v2 and v4;
- a mutant grading at a vendored size and at a size with none;
- every band-builder path at a size with no vendored bands, and the builder reproducing the vendored
  bands at their sizes;
- a committed simulated record, re-simulated and re-graded.

Any difference SHALL block the merge.

#### Scenario: The identity check passes

- **WHEN** the identity check's baseline is taken at `origin/main` and its check runs on the change
- **THEN** every case's output SHALL be byte-identical, and the speed-up identity check SHALL pass
  without overwriting any committed record

#### Scenario: The check cannot grade one tree twice

- **WHEN** the check stage's `quantumnematode` or `wormbench` module resolves outside the change's
  tree
- **THEN** the check SHALL refuse to run

#### Scenario: No held-out cohort is read

- **WHEN** the identity check, the example builder or any test of this change loads Scheer tracks
- **THEN** it SHALL pass its rows through the held-out refusal, and SHALL read no PD1074 held-out
  row

### Requirement: Reference data resolved when used, checked, and never fetched implicitly

No module of `wormbench` SHALL compute a data path when it is imported. Each reference file SHALL be
resolved when it is used, file by file, from the first of these sources that holds it:

1. a root set by the host;
2. `WORMBENCH_DATA`;
3. the wheel;
4. the fetched cache.

`wormbench` SHALL ship a manifest giving each file's SHA256 and size. It SHALL refuse a file loaded
through the resolver whose SHA256 differs from the manifest's. It SHALL NOT download anything except
on `wormbench data fetch`. A grading that needs a file no source holds SHALL fail with an error
naming the file and the fetch command. `quantum-nematode` SHALL set this repository's `data/` as the
host root.

#### Scenario: A tampered file is refused

- **WHEN** a file loaded through the resolver has a SHA256 that differs from the manifest's
- **THEN** loading it SHALL raise an error naming the file, and nothing SHALL be graded

#### Scenario: Grading at an unvendored size without the tracks

- **WHEN** a population is graded at a size with no vendored bands, and no source holds the tracks
- **THEN** grading SHALL fail with an error naming the missing file and `wormbench data fetch`, and
  SHALL make no network request

#### Scenario: The project reads its vendored files

- **WHEN** any `quantumnematode` module is imported in this repository, whatever `WORMBENCH_DATA`
  holds
- **THEN** every reference file SHALL resolve under this repository's `data/`, and no fetch SHALL be
  needed

### Requirement: The package's default version and its records

Every 1.x release of `wormbench` SHALL grade on benchmark v4 by default. A later version SHALL be
selectable only through `--bands` until a major release. The wheel's validity record and v1's kappa
SHALL be package files, generated from the committed controls and band records by a rule equal to
the scripts' own validity rule, and SHALL be read from the package in the repository as in the
wheel. The in-wheel copies of `data/` files SHALL equal their sources byte for byte, and the manifest
SHALL match `data/`. A test SHALL regenerate and compare all of them, and the identity check SHALL
require that test to pass first.

#### Scenario: The wheel's default needs no repository record

- **WHEN** `wormbench grade` runs from the installed wheel with no `--bands`
- **THEN** it SHALL grade on v4, and SHALL NOT fall back to an older version because a repository
  record is absent

#### Scenario: A stale copy or record is caught

- **WHEN** a file under `data/benchmark/` or a committed controls record changes and the package's
  copy or generated record does not
- **THEN** the test SHALL fail, naming the file

### Requirement: A command-line grader with the project's output

`wormbench grade --manifest <file> --out <json> [--bands v1|v2|v3|v4] [--simulated]` SHALL read the
manifest format of `scripts/analysis/benchmark_grade.py tracks`, resolving relative entries against
the manifest's directory. It SHALL call the same package function as that script. Its output SHALL
add a `wormbench` block holding the package version, the bands version and the bands' SHA256. The
script's output SHALL NOT change.

With `--simulated` it SHALL pass each animal on Scheer's rig through the measurement model before
windowing, as the project measures its simulated tracks: lawn membership from exact positions, then
the whole track measured with the animal's own seed, then the window. Each such animal SHALL carry
its own per-second primitive labels and its own integer seed. Local search SHALL NOT be measured,
and its animals SHALL need neither.

#### Scenario: Same input, same grading

- **WHEN** one manifest of imported tracks, with absolute paths, is graded by `wormbench grade` and
  by `benchmark_grade.py tracks` with the same `--bands`
- **THEN** the two outputs SHALL be byte-identical once the CLI's `wormbench` block is removed

#### Scenario: Simulated tracks are graded as the project grades its own

- **WHEN** one set of simulated steps is graded through `wormbench grade --simulated` and through
  the project's captures path on v4
- **THEN** the two gradings SHALL be byte-identical

#### Scenario: A simulated track without labels, or with a repeated seed

- **WHEN** `--simulated` is given and a measured animal has no primitive labels or no seed, or two
  animals share a seed
- **THEN** grading SHALL fail, naming the animals and the field

#### Scenario: Measured grading on v1

- **WHEN** `--simulated` is given with `--bands v1`
- **THEN** grading SHALL be refused, since v1's bands carry no measurement model

### Requirement: A worked example from band cohorts

`wormbench example --out <dir>` SHALL write a 64-animal cohort and its manifest. The animals SHALL be
band-cohort PD1074 OD 2 small-lawn animals, chosen by a recorded seed from rows with at least 90%
finite positions, whose rows passed the held-out refusal. Grading it on v4 SHALL reproduce a stored
expected grading: every statistic within a relative tolerance of 1e-9, and every pass, flag and count
exactly. It SHALL NOT be presented as a validation, since its animals are inside the bands.

#### Scenario: The example grades with no download

- **WHEN** the example is written and graded on v4 with no fetched data
- **THEN** the grading SHALL match the stored expected grading. It SHALL cover the small-lawn,
  kinematics and activity families at size 64, with D and D_fit undefined because the cohort does
  not supply every family they need
