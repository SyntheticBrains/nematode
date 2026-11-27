## ADDED Requirements

### Requirement: Tracks written in the formats the grader reads

The package SHALL provide writers for both import formats, `wormbench.track_export.write_csv` and
`write_wcon`, whose files the importer reads back to the same samples in mm and s and the same assay
metadata. `wormbench wcon --tracks <tracks.csv> --out <tracks.wcon> [--simulated]` SHALL write a
CSV's tracks, with its sidecar, as one WCON 1.0 file, read it back, and only then replace the output.
Unless every animal's times and positions, the metadata and, with `--simulated`, each animal's
primitive labels and seed read back unchanged, it SHALL fail and leave the output as it was. It SHALL
refuse an output that is its source or the source's sidecar, or whose name the grader would read as a
CSV.

Neither writer SHALL write a file the importer would read differently from the tracks given:

- tracks with no samples, sharing an animal id as the importer reads ids (as strings), with times
  that are not finite or do not increase, or with infinite positions;
- a mix of simulated and plain tracks, and a simulated track with an unknown label, a label count
  unlike its sample count, samples not 1 s apart, or a seed that is not an integer;
- metadata whose lawn the importer would read back otherwise: a uniform well without a uniform lawn,
  local search with a lawn, or a start that is not finite;
- in WCON, a lawn family's track carrying an `on_lawn` flag, since WCON's core has none, or metadata
  holding no lawn.

The project's round-trip check and the worked example SHALL be written with these writers, and their
outputs SHALL NOT change.

#### Scenario: A CSV written as WCON grades the same

- **WHEN** the worked example's animals are written as WCON by `wormbench wcon` and graded on v4
- **THEN** the grading SHALL equal the CSVs' grading, apart from the manifest path

#### Scenario: Simulated labels and seeds are carried

- **WHEN** a simulated CSV is written as WCON with `--simulated`
- **THEN** each animal's primitive labels and seed SHALL read back unchanged, and the WCON file SHALL
  be measured and windowed exactly as the CSV is

#### Scenario: A track whose grade WCON cannot hold is refused

- **WHEN** a lawn family's CSV marks lawn membership by an `on_lawn` column
- **THEN** `wormbench wcon` SHALL fail, naming the flag, and SHALL write nothing

#### Scenario: The project's files do not change

- **WHEN** `track_import_check.py` and `wormbench_example.py check` run on the package's writers
- **THEN** the round-trip record SHALL be byte-identical to the committed one, and the check SHALL
  pass
