## Why

B.5 is the last open item of the benchmark block. It is part of the roadmap's minimum-viable outcome, *"the benchmark valid … with the 1-s worm, its primitives and assays, and track import. Citable alone."* Today the benchmark grades only the simulator's captures and the project's vendored references. A lab with its own tracks, or another simulator's output, cannot be graded without writing code against our internals. B.5 lets any 1-s-resampleable track be graded through the same windows, classifier, statistics and bands, independently of the simulator.

The maintainer's decisions of 2026-10-26:

- **formats:** CSV and WCON, through a minimal in-repo reader of WCON 1.0's core with no new dependency;
- **lawn membership:** a per-sample column, or a lawn boundary in the metadata (a circle, or a polar boundary as in Scheer's records), or `uniform` for a well with no boundary;
- **local search:** off-food tracks are graded on it, through the reversal-onset detector registered for off-food tracks (Logbook 107);
- **placement:** a package module and a `tracks` subcommand of `benchmark_grade.py`.

## What Changes

- **`quantumnematode.benchmark.track_import`:**

  - `read_csv` and `read_wcon` return `ImportedTrack`s: an animal id, times in seconds, positions in mm, an optional per-sample on-lawn flag, and the assay metadata;
  - `resample_1s` puts each track on the benchmark's 1-s grid by linear interpolation inside its recorded span. Gaps longer than 1 s become NaN, so the grader's own missing-sample rule applies;
  - `lawn_membership` uses the column if present, otherwise `references.boundary_distance` on the boundary (the rule Scheer's tracks are cut by), or all-on for `uniform`;
  - `to_steps` converts a track to the `BehaviourStep`s the grader already reads for simulated captures, from the analysis start given in the metadata.

- **`benchmark_grade.py tracks --manifest <"<family> <file>" per line> --out <json>`** grades imported populations with `grade_populations`, the path every simulated arm uses.

- **The round-trip control** (`scripts/analysis/track_import_check.py`):

  - band-cohort Scheer tracks, small lawn and uniform well, are written out as CSV and as WCON, re-imported and graded;
  - so are Roberts 2016's off-food tracks from its calibration cohorts (C15, C162, C57);
  - each family's statistics must equal the native path's exactly.

  Held-out cohorts are never written or read.

- **Independent of the simulator:** a test checks that importing `track_import` and grading imported tracks loads no environment, agent or brain module.

- **The positioning note** (`data/benchmark/IMPORT.md`): the input contract, plus how the 11 statistics relate to Tierpsy's feature set and the Open Worm Movement Database's.

## Capabilities

### New Capabilities

- `track-import`: grading external tracks through the benchmark.

## Impact

- **Code:** `quantumnematode/benchmark/track_import.py`, the `tracks` subcommand, `scripts/analysis/track_import_check.py`, and tests. No change to grading.
- **Records:** `docs/experiments/logbooks/supporting/114-track-import/roundtrip.json` and Logbook 114.
- **Compute:** minutes. No simulation runs, so it does not contend with C.1's fit.
