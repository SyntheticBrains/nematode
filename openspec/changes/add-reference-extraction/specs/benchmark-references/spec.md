## ADDED Requirements

### Requirement: Vendored real-worm reference sets with provenance

The repository SHALL vendor, under `data/benchmark/`, derived reference sets from:

- Scheer & Bargmann 2023 (CC0);
- Calhoun 2014 (CC0);
- Roberts 2016 (CC0);
- López-Cruz 2019 (CC BY 4.0).

Each set SHALL be described in `data/benchmark/PROVENANCE.md`:

- its citation, deposit and licence;
- the source file's name, size and published MD5, or the SHA-256 of the downloaded file (for a
  whole-dataset zip, the zip's SHA-256 and each inner file's size);
- the download date and method;
- its frame rate, units and coordinate origin;
- the script and command that derived it.

Only CC0 or CC BY sources SHALL be vendored, and never the original deposit files.

#### Scenario: Every vendored set names its licence

- **WHEN** the files in `data/benchmark/` are listed
- **THEN** each data file SHALL have a `PROVENANCE.md` entry naming its licence and source checksum

### Requirement: Scheer tracks at 1 s with cohort fields

`data/benchmark/scheer2023_tracks.npz` SHALL hold, for every animal of the wild type at OD 2, OD 1, OD 4
and on a uniform lawn, and of tph-1, pdfr-1 and tax-4:

- the midbody position at 1 s, in µm, as int16 (int32 for an arm whose range exceeds int16), with
  −32768 where missing;
- its origin: the lawn centre where the recording has a lawn boundary, otherwise the animal's first
  valid position;
- the authors' per-frame reversal, omega, pause, head-poke, midbody-in-lawn and missing flags at 3
  frames/s;
- the authors' per-bin in- and out-of-lawn run masks and roaming/dwelling labels;
- the lawn boundary as 64 radii, or a sentinel where the arm has no boundary;
- its genotype, recording date, video and source row.

The extraction SHALL read the deposit only through an unpickler that admits numpy, pandas,
`builtins.slice` and `datetime.date`, and replaces the `ssm` classes with inert stand-ins.

#### Scenario: The unpickler refuses other classes

- **WHEN** the extraction reads a pickle that references a class outside the admitted modules
- **THEN** it SHALL raise before constructing that class

#### Scenario: An arm without a lawn boundary

- **WHEN** an arm's recordings carry no `Center_Point` or `Lawn_Boundary_Pts`
- **THEN** its positions SHALL be stored relative to each animal's first valid position, with its
  origin and the boundary sentinel recorded

### Requirement: The extraction reproduces the vendored windows

The 5-s windows re-derived from the 1-s store, for the OD 2 wild-type animals matched to the Fig. 1
file by video name, source file and row, SHALL reproduce `data/roaming_dwelling/scheer2023_windows.npz`:

- at least 95% of the Fig. 1 animals matched;
- `on_food`, `off_lawn` and `labels` identical;
- `speed` within √2 µm / 5 s;
- the classifier's roaming observation identical on at least 99.5% of on-food windows;
- the calibrated classifier's decoded states agreeing on at least 99% of on-food windows.

#### Scenario: The re-derived windows match

- **WHEN** the test suite runs
- **THEN** a test SHALL re-derive the windows from the vendored store and check every condition

#### Scenario: Too few animals match

- **WHEN** fewer than 95% of the Fig. 1 animals match `PD1074_od2_Data`
- **THEN** the extraction SHALL stop and report the matched count

### Requirement: A cohort split fixed from counts

Cohorts SHALL be recording dates. The split SHALL be drawn from the dates' arm membership and animal
counts alone, by the rule of design Decision C (`SPLIT_SEED = 2027`, at most 10,000 draws, with each
arm's share constraint checked for feasibility first). It SHALL be written to
`data/benchmark/scheer2023_split.json` before any behavioural statistic of the store is computed. Every
date SHALL be on one side, `bands` or `held_out`, with all its animals. The record SHALL state that the
classifier was calibrated on animals from both sides.

#### Scenario: The split re-draws identically

- **WHEN** the split rule is re-run on the vendored store's counts
- **THEN** it SHALL produce the vendored split exactly

#### Scenario: An arm with one date has no held-out cohort

- **WHEN** an arm was recorded on a single date
- **THEN** that date SHALL be on the `bands` side, and the split record SHALL say the arm has no
  held-out cohort

#### Scenario: An arm whose dates cannot meet the share

- **WHEN** no subset of an arm's dates holds 20–45% of its animals
- **THEN** the arm SHALL be recorded as share-infeasible, and it SHALL need only one date on each side

#### Scenario: No draw passes

- **WHEN** none of the 10,000 draws meets the constraints
- **THEN** the extraction SHALL stop and report the closest draw, without loosening the rule

### Requirement: The primitives' kinematics checked on food

On the held-out OD 2 wild-type cohorts, the kinematics check SHALL measure reversal duration, reversal
speed, omega duration, forward speed, turning rate and pause fraction from the authors' flags and the
1-s positions. Bouts are built by merging one-frame gaps and dropping runs shorter than two frames.
Each quantity is stratified by distance to the lawn boundary, with intervals bootstrapped over cohorts.

A quantity whose source states a mean, SD and n SHALL be acted on. Its reference span runs from the
lowest source mean − 2 SEM to the highest + 2 SEM. A disagreement SHALL be the inside-lawn stratum's 95%
interval lying wholly outside that span. A disagreement SHALL be recorded and brought to the maintainer
before B.3 registers. This change SHALL NOT edit `data/locomotion/primitives.json`.

#### Scenario: A disagreement is brought back, not applied

- **WHEN** the inside-lawn reversal duration's interval lies wholly outside 2.10–3.71 s
- **THEN** `kinematics.json` and Logbook 105 SHALL record it, and `primitives.json` SHALL be unchanged

#### Scenario: The check reads no graded statistic

- **WHEN** the kinematics check runs
- **THEN** it SHALL read only flags and positions of held-out cohorts, never labels, run masks or
  derived roaming fractions

### Requirement: Search tracks and search targets

Calhoun 2014's tracks SHALL be vendored at 1 s, with the per-frame pirouette, omega and non-omega flags,
in µm if the paper states the scale and in pixels, marked as such, otherwise. Roberts 2016's wild-type
cohorts SHALL be vendored at 1 s in µm with their worm numbers, by-eye states, Viterbi states and cohort
IDs, and the dead-worm tracks beside them. The whole of López-Cruz 2019's `data_2min_bins` sheet SHALL
be vendored, with each bin's mean, SEM and n, and each mutant row linked to its same-dates wild type.

#### Scenario: Calhoun's scale is unstated

- **WHEN** the paper gives no pixel scale
- **THEN** the store SHALL mark its units as pixels, and the loader SHALL carry that unit

#### Scenario: López-Cruz lacks a D37 mutant

- **WHEN** a D37 mutant has no row in the sheet
- **THEN** PROVENANCE.md SHALL record its absence

### Requirement: A loader for the reference sets

`quantumnematode.benchmark.references` SHALL read every vendored set into per-animal tracks in its
stated units and seconds, with cohort fields and split side where the set has them, and depend on
numpy alone.

#### Scenario: Tracks load with their cohorts

- **WHEN** `load_scheer_tracks()` is called
- **THEN** every animal SHALL carry its genotype, date and split side, and its positions SHALL be in
  mm, with NaN where the store holds −32768
