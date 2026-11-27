## 1. Writers

- [x] 1.1 `wormbench/track_export.py`: `metadata_object`, `write_csv`, `wcon_document`,
  `write_wcon` and `csv_to_wcon`, refusing what the importer would misread.
- [x] 1.2 Keep the CSV bytes the script wrote, so the worked example's archive is unchanged.

## 2. Command

- [x] 2.1 `wormbench wcon --tracks <csv> --out <wcon> [--simulated]`, failing with the writer's
  message and leaving no file.

## 3. The project's callers

- [x] 3.1 `track_import_check.write_track` calls the package's writers; drop its own CSV and JSON
  code.
- [x] 3.2 Its record is byte-identical to `supporting/115-benchmark-v2/import_roundtrip_v2.json`, and
  `wormbench_example.py check` passes.

## 4. Tests

- [x] 4.1 `test_track_export.py`: both formats read back as written for each family; the worked
  example grades the same as WCON; pixels written in mm; simulated labels and seeds carried; a
  plain conversion carries no labels; a seedless local-search track; the on-lawn and missing-lawn
  refusals; the importer-misread refusals; CSV input only; the read-back check removes a bad file.
- [x] 4.2 Plant defects and confirm each fails a test.

## 5. Review

- [x] 5.1 An independent review's findings: refuse empty tracks, ids equal as strings, simulated
  tracks the importer refuses, metadata that read back otherwise, and an output that is the source
  or a `.csv`; stage the WCON file and replace the output only after the check; test the simulated
  CSV writer, the on-lawn values, the metadata check, the remaining refusals and a missing source.

## 6. Documentation

- [x] 6.1 The package README (*Writing WCON*), `data/benchmark/IMPORT.md`, AGENTS.md and the
  CHANGELOG.
