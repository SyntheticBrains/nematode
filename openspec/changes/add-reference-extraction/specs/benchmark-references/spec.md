## ADDED Requirements

### Requirement: Vendored real-worm reference sets with provenance

The repository SHALL vendor, under `data/benchmark/`, derived reference sets from:

- Scheer & Bargmann 2023 (CC0);
- Calhoun 2014 (CC0);
- Roberts 2016 (CC0);
- López-Cruz 2019 (CC BY 4.0).

Each set SHALL be described in `data/benchmark/PROVENANCE.md`:

- its citation, deposit and licence;
- the source file's name, size and published MD5, or the SHA-256 of the downloaded file;
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

- the midbody position at 1 s, in µm from the lawn centre, as int16, with −32768 where missing;
- the authors' per-frame reversal, omega, pause, head-poke, midbody-in-lawn and missing flags at 3
  frames/s;
- the authors' per-bin in- and out-of-lawn run masks and roaming/dwelling labels;
- the lawn boundary as 64 radii;
- its genotype, recording date, video and source row.

The extraction SHALL read the deposit only through an unpickler that admits numpy, pandas,
`builtins.slice` and `datetime.date`, and replaces the `ssm` classes with inert stand-ins. It SHALL
refuse a coordinate outside the int16 range rather than clip it.

#### Scenario: The unpickler refuses other classes

- **WHEN** the extraction reads a pickle that references a class outside the admitted modules
- **THEN** it SHALL raise before constructing that class

#### Scenario: A position outside the store's range is refused

- **WHEN** an animal's position lies more than 32.767 mm from its lawn centre
- **THEN** the extraction SHALL raise, naming the animal

### Requirement: The extraction reproduces the vendored windows

The 5-s windows re-derived from the 1-s store, for the OD 2 wild type, SHALL reproduce
`data/roaming_dwelling/scheer2023_windows.npz`:

- `on_food`, `off_lawn` and `labels` identical;
- `speed` and `angular` within the bound implied by the store's 0.5 µm rounding;
- the calibrated classifier's decoded states agreeing on at least 99% of on-food windows.

#### Scenario: The re-derived windows match

- **WHEN** the default test tier runs
- **THEN** a test SHALL re-derive the windows from the vendored store and check all three conditions

### Requirement: A cohort split fixed from counts

Cohorts SHALL be recording dates. The split SHALL be drawn from the dates' arm membership and animal
counts alone, by the rule of design Decision C (`SPLIT_SEED = 2027`). It SHALL be written to
`data/benchmark/scheer2023_split.json` before any behavioural statistic of the store is computed.
Every date SHALL be on one side, `bands` or `held_out`, with all its animals.

#### Scenario: The split re-draws identically

- **WHEN** the split rule is re-run on the vendored store's counts
- **THEN** it SHALL produce the vendored split exactly

#### Scenario: An arm with one date has no held-out cohort

- **WHEN** an arm was recorded on a single date
- **THEN** that date SHALL be on the `bands` side, and the split record SHALL say the arm has no
  held-out cohort

### Requirement: The primitives' kinematics checked on food

On the held-out OD 2 wild-type cohorts, the kinematics check SHALL measure reversal duration, reversal
speed, omega duration, forward speed, turning rate and pause fraction from the authors' flags and the
1-s positions. Each is stratified by distance to the lawn boundary, with intervals bootstrapped over
cohorts.

For reversal duration, reversal speed and omega duration, a disagreement SHALL be the inside-lawn
stratum's 95% interval lying wholly outside the span of the sources' reported mean ± SD. Each
disagreement SHALL be recorded as a dated D36 amendment, with `data/locomotion/primitives.json` updated
to the on-food value, before B.1 closes.

#### Scenario: A disagreement amends the primitives

- **WHEN** the inside-lawn reversal duration's interval lies wholly outside the published span
- **THEN** `primitives.json` SHALL take the on-food value with Scheer as its source, and the amendment
  SHALL be dated in the roadmap and the tracker

#### Scenario: The check reads no graded statistic

- **WHEN** the kinematics check runs
- **THEN** it SHALL read only flags and positions of held-out cohorts, never labels, run masks or
  derived roaming fractions

### Requirement: Search tracks and search targets

Calhoun 2014's and Roberts 2016's tracks SHALL be vendored at 1 s in µm, with their turn flags where the
deposit provides them. López-Cruz 2019's reorientations per 2-min bin SHALL be vendored for the wild
type, pdfr-1, tax-4 and tph-1 where present, with each bin's mean, SEM and n.

#### Scenario: A deposit without turn flags

- **WHEN** a search deposit holds positions but no turn flags
- **THEN** positions SHALL be vendored alone, and PROVENANCE.md SHALL record the absence

### Requirement: A loader for the reference sets

`quantumnematode.benchmark.references` SHALL read every vendored set into per-animal tracks in mm and
seconds, with their cohort fields and split side, and depend on numpy alone.

#### Scenario: Tracks load with their cohorts

- **WHEN** `load_scheer_tracks()` is called
- **THEN** every animal SHALL carry its genotype, date and split side, and its positions SHALL be in
  mm, with NaN where the store holds −32768
