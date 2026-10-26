## ADDED Requirements

### Requirement: External tracks are graded through the benchmark

The benchmark SHALL read tracks as CSV (with a metadata sidecar) or as WCON 1.0's core, resample them to 1 s, mark lawn membership, and grade them on the 11 wild-type statistics through the same windows, classifier, statistics and bands as the project's own tracks. A WCON field outside the core SHALL be refused with a message naming it.

#### Scenario: A skeleton is reduced to its midpoint

- **WHEN** a WCON record gives a skeleton at each time
- **THEN** the track SHALL use the skeleton's middle point at each time

#### Scenario: A gap is missing data

- **WHEN** neighbouring samples lie more than 1 s apart
- **THEN** the resampled seconds between them SHALL be NaN, and no window touching them SHALL be classified

#### Scenario: Pixels without a scale

- **WHEN** units are pixels and no `mm_per_px` is given
- **THEN** the import SHALL be refused

### Requirement: Lawn membership follows the benchmark's rule

A sample SHALL be on the lawn by the track's on-lawn column if present. Otherwise it SHALL be on the lawn when the boundary distance from the lawn's centre is positive, by the rule Scheer's tracks are cut by. On a uniform well, every sample SHALL be on the lawn.

#### Scenario: A circular lawn

- **WHEN** the metadata give a lawn centre and radius
- **THEN** a sample SHALL be on the lawn exactly when it lies inside that circle

### Requirement: A round trip reproduces the native statistics

Band-cohort Scheer tracks (small lawn and uniform well) and Roberts 2016's off-food tracks from its calibration cohorts, written as CSV and as WCON and re-imported, SHALL reproduce the native path exactly: every lawn tally and statistic, to 1e-12 for floats and exactly for counts, and every off-food worm's detector onsets and valid seconds. No held-out cohort SHALL be written or read.

#### Scenario: A held-out animal is refused

- **WHEN** the check is asked to export an animal whose split side is held out
- **THEN** it SHALL stop before writing anything

### Requirement: The import path does not need the simulator

Importing the track module and grading imported tracks SHALL load no environment, agent or brain module.

#### Scenario: Grading in a clean interpreter

- **WHEN** a fresh interpreter imports the module and grades a small CSV
- **THEN** no `quantumnematode.env`, `quantumnematode.agent` or `quantumnematode.brain` module SHALL be loaded
