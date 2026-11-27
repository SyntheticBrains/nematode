## Why

The Phase 10 synthesis ([Logbook 138](../../../docs/experiments/logbooks/138-phase10-synthesis.md))
assigned WCON export, a SHOULD item of the phase, to the `wormbench` release that follows the close.
WCON is the field's interchange format for worm tracks, and the package reads it, but nothing in the
package writes it:

- **The only writer is in a script.** `scripts/analysis/track_import_check.py` writes one animal at a
  time as CSV or WCON for the importer's round-trip control, and the worked example's archive is
  built through it. An outside user has no way to write the format the grader reads.
- **A naive conversion can change a grade.** The importer marks lawn membership by an `on_lawn`
  column before the boundary, and WCON's core has no such flag. A CSV that relies on the column,
  converted by hand, would be graded by its boundary instead, or not at all.

## What Changes

- **New module `wormbench.track_export`:** `write_csv` and `write_wcon` write a list of tracks, plain
  or simulated, in mm and s, as `wormbench.track_import` reads them; `wcon_document` and
  `metadata_object` give the document and the metadata object; `csv_to_wcon` converts a CSV and
  checks the result reads back unchanged.
- **New command `wormbench wcon --tracks <tracks.csv> --out <tracks.wcon> [--simulated]`.**
- **Refusals:** tracks sharing an animal id, times that do not increase, infinite positions, a mix
  of simulated and plain tracks, and, in WCON, a lawn family's track carrying an `on_lawn` column or
  metadata with no lawn.
- **`track_import_check.py` writes with the package's writers.** Its round-trip record and the worked
  example's archive stay byte-identical.
- **No grading code changes.** `track_import` is not edited; the writers are a new module.

## Capabilities

### Modified Capabilities

- `benchmark-package`: adds the requirement "Tracks written in the formats the grader reads".

## Impact

- New: `packages/wormbench/wormbench/track_export.py`,
  `packages/wormbench/tests/wormbench_tests/test_track_export.py`.
- Changed: `packages/wormbench/wormbench/cli.py` (the `wcon` command),
  `scripts/analysis/track_import_check.py` (its `write_track` calls the package), the package README,
  `data/benchmark/IMPORT.md`, AGENTS.md and the CHANGELOG.
- Released with `wormbench` 1.0.0, in the same pull request as the version, the citation files and
  the CHANGELOG.
