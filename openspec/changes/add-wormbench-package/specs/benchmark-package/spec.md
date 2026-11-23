## ADDED Requirements

### Requirement: A standalone grader

The benchmark's grader SHALL be an installable package, `wormbench`, in `packages/wormbench/`. It
SHALL require Python 3.10 or later, and its only runtime dependencies SHALL be numpy and scipy.
Installed with nothing else from this repository, it SHALL import, grade imported tracks and run its
own tests. It SHALL hold:

- the readouts, the grading statistics, the measurement model, the chemotaxis statistics, the track
  import and the reference loaders;
- the roaming/dwelling classifier, the per-step record and the measured odour field;
- the band loaders, and the families' bands at a population's size with the builder they call;
- the population grading of every benchmark version.

Importing it SHALL import no `quantumnematode` module.

#### Scenario: Installed alone on the oldest supported Python

- **WHEN** the built wheel is installed into a fresh Python 3.10 environment with numpy 1.24 and
  scipy 1.10, and nothing else from this repository
- **THEN** `import wormbench` SHALL succeed, its tests SHALL pass with no network access, and
  `sys.modules` SHALL hold no `quantumnematode` module

#### Scenario: Its dependencies are numpy and scipy

- **WHEN** the package's metadata are read
- **THEN** its runtime dependencies SHALL be numpy and scipy, and nothing else

### Requirement: One grader for the project and for outsiders

`quantum-nematode` SHALL depend on `wormbench`, and SHALL hold no second copy of any module that
moved. Each old import path SHALL resolve to the same module object as its `wormbench` counterpart:
`quantumnematode.benchmark.<module>`, `quantumnematode.validation.roaming_dwelling`,
`quantumnematode.report.behaviour` and `quantumnematode.evaluation.odour_field`. This SHALL hold
whichever of them is imported first. `scripts/analysis/benchmark_grade.py` and
`scripts/analysis/benchmark_bands.py` SHALL re-export every name that moved, private names
included.

#### Scenario: The old path is the same module

- **WHEN** `quantumnematode.validation.roaming_dwelling` is imported first in a fresh interpreter,
  and then `wormbench.classifier`
- **THEN** they SHALL be the same object, and so SHALL each class and function reached through
  either, private names included

#### Scenario: The project's benchmark tests keep their imports

- **WHEN** the existing benchmark tests run with their import statements unchanged, and with patches
  of moved names pointed at the module that reads them
- **THEN** they SHALL pass, and a patch left pointing at a script's re-exported name SHALL be
  caught by a deliberate-bug test

### Requirement: Moving the grader changes no output

Moving the grader SHALL leave every grading output byte-identical. An identity check SHALL grade a
fixed set of cases, reading band cohorts only, first in a worktree at `origin/main` and then on the
change. The cases SHALL include:

- real tracks through the import round trip on every benchmark version;
- the batched negatives on fixed seeds on every version;
- captures, both unmeasured and measured;
- a grading without presence;
- a mutant grading at a vendored size and at a size with none;
- every band-builder path at a size with no vendored bands;
- the builder at the vendored sizes, which SHALL reproduce the vendored bands.

Any difference SHALL block the merge.

#### Scenario: The identity check passes

- **WHEN** the identity check's baseline is taken at `origin/main` and its check runs on the change
- **THEN** every case's output SHALL be byte-identical, and the speed-up identity check SHALL pass
  without overwriting any committed record

#### Scenario: No held-out cohort is read

- **WHEN** the identity check, the example or any test of this change loads Scheer tracks
- **THEN** it SHALL read band-cohort rows only, and SHALL read no PD1074 held-out row

### Requirement: Reference data resolved when used, checked, and never fetched implicitly

No module of `wormbench` SHALL compute a data path when it is imported. Each reference file SHALL be
resolved when it is used, file by file, from the first of these sources that holds it:

1. `WORMBENCH_DATA`;
2. a root set by the host;
3. the wheel;
4. the fetched cache.

`wormbench` SHALL ship a manifest giving each file's SHA256, size and download URL. It SHALL refuse a
file loaded through the resolver whose SHA256 differs from the manifest's. It SHALL NOT download
anything except on `wormbench data fetch`. A grading that needs a file no source holds SHALL fail
with an error naming the file and the fetch command. `quantum-nematode` SHALL set this repository's
`data/` as the host root before any moved module is used.

#### Scenario: A tampered file is refused

- **WHEN** a file loaded through the resolver has a SHA256 that differs from the manifest's
- **THEN** loading it SHALL raise an error naming the file, and nothing SHALL be graded

#### Scenario: Grading at an unvendored size without the tracks

- **WHEN** a population is graded at a size with no vendored bands, and no source holds the tracks
- **THEN** grading SHALL fail with an error naming the missing file and `wormbench data fetch`, and
  SHALL make no network request

#### Scenario: The wheel decides the default version without the repository

- **WHEN** `wormbench grade` runs from the installed wheel with no `--bands`
- **THEN** it SHALL grade on the newest version that the shipped validity record marks VALID, which
  is v4, and SHALL NOT fall back to an older version because a repository record is absent

#### Scenario: The project reads its vendored files

- **WHEN** `quantum-nematode` is imported in this repository
- **THEN** every reference file SHALL resolve under `data/`, and no fetch SHALL be needed

### Requirement: Shipped records match the committed records they come from

The wheel's validity record and v1's kappa SHALL be generated from the committed controls and band
records. The in-wheel copies of `data/` files SHALL equal their sources byte for byte, and the
manifest SHALL match `data/`. A test SHALL regenerate and compare all of them.

#### Scenario: A stale copy is caught

- **WHEN** a file under `data/benchmark/` changes and its in-wheel copy does not
- **THEN** the test SHALL fail, naming the file

### Requirement: A command-line grader with the project's output

`wormbench grade --manifest <file> --out <json> [--bands v1|v2|v3|v4] [--simulated]` SHALL read the
manifest format of `scripts/analysis/benchmark_grade.py tracks`, resolving relative entries against
the manifest's directory. For the same input it SHALL write the same grading as that script, which
SHALL call the same package function. With `--simulated` it SHALL pass each animal on Scheer's rig
through the measurement model before windowing, as the project measures its simulated tracks, using
each animal's per-second primitive labels and seed. It SHALL refuse an animal that lacks either.
Local search SHALL NOT be measured.

#### Scenario: Same input, same grading

- **WHEN** one manifest of imported tracks, with absolute paths, is graded by `wormbench grade` and
  by `benchmark_grade.py tracks` on the same bands
- **THEN** the two output files SHALL be byte-identical

#### Scenario: Simulated tracks are graded as the project grades its own

- **WHEN** one set of simulated steps is graded through `wormbench grade --simulated` and through
  the project's captures path on v4
- **THEN** the two gradings SHALL be byte-identical

#### Scenario: A simulated track without labels

- **WHEN** `--simulated` is given and an animal has no primitive labels or no seed
- **THEN** grading SHALL fail, naming the animal and the missing field

### Requirement: A worked example from band cohorts

`wormbench example --out <dir>` SHALL write a 64-animal cohort of band-cohort PD1074 OD 2 small-lawn
animals, chosen by a recorded seed from rows with at least 90% finite positions, with its manifest.
Grading it on v4 SHALL reproduce a stored expected grading. It SHALL NOT be presented as a
validation, since its animals are inside the bands.

#### Scenario: The example grades with no download

- **WHEN** the example is written and graded on v4 with no fetched data
- **THEN** the grading SHALL equal the stored expected grading. It SHALL cover the small-lawn,
  kinematics and activity families at size 64, with D and D_fit undefined because the cohort does
  not supply every family they need
