## Context

- **The grader** (`benchmark_grade.grade_populations`) grades populations of `BehaviourStep` lists, each already cut to its family's window (`window`: `LAWN_SAMPLES` or `SEARCH_SAMPLES` from an assay's analysis start).
  - **Lawn families:** `grading.simulated_windows` reads `x`, `y` and `on_lawn`. On a uniform well, every sample is on the lawn.
  - **Local search:** `grading.simulated_search` and `simulated_search_presence` read `x` and `y` alone, through the off-food detector.
- **Real Scheer tracks** (`references.ScheerTracks`) hold positions in mm at 1 s about the lawn centre, and a polar boundary per animal (`boundary_radius`). `grading.real_on_lawn` marks a sample on the lawn when `references.boundary_distance > 0`, or everywhere on a uniform well. Band cohorts are those with split side `bands` (`references.BANDS`).
- **Off-food tracks:** Roberts 2016's (`references.load_search_tracks("roberts2016")`) are in mm at 1 s, 10 min each. B.2 calibrated the detector on cohorts C15, C162 and C57 and validated it on C186 and C21. Calhoun 2014's are in pixels with no published scale, so they cannot be imported in mm.
- **WCON 1.0** is JSON with `units` (for t, x and y), `data` (records with `id`, `t`, and `x` and `y` as a point or a skeleton per time) and an optional `metadata` (Javer et al. 2018a, *Nature Methods* 15:645).

## Goals / Non-Goals

**Goals**

- A real or simulated track from outside this repository is graded on the 11 statistics exactly as the project's own are.
- A round trip through each format reproduces the native statistics exactly.
- Grading an imported track runs no simulation, reads no simulator config, and calls no simulator code.

**Non-goals**

- WCON export (a SHOULD item).
- Tierpsy feature extraction. The note positions the statistics against Tierpsy's features; it does not compute them.
- Grading mutants from imported tracks. The mutant rule needs paired dates, so it stays with the vendored references.

## Decisions

1. **The CSV contract.** One row per sample, with the columns:

   - `animal`, `t_s` (seconds), `x` and `y`;
   - optionally `on_lawn` (0/1).

   A sidecar `<file>.json` holds the metadata:

   - `units`: `mm`; `um`, `µm` or `microns`; or `px` or `pixels` with `mm_per_px`;
   - `family` (`small_lawn`, `uniform_lawn` or `local_search`);
   - `analysis_start_s`, the assay window's start;
   - the lawn: `{"centre": [x, y], "radius": r}`, or `{"centre": [x, y], "radii": [...]}` at equally spaced angles from +x, or `"uniform"`.

2. **The WCON core.**

   - **Read:** `units` (t in s; x and y in mm, µm or px with `mm_per_px`) and `data` records, merged by `id` in time order. The benchmark's metadata, the same fields as the CSV sidecar, sit in a top-level `@nematode_benchmark` object, the form WCON gives custom fields.
   - **Skeletons:** a skeleton per time (x and y as lists) is reduced to its midpoint, the middle of its points by index, since the benchmark reads the midbody.
   - **Origin offsets:** a record's offsets are added to its positions.
   - **Ignored:** other `@`-prefixed keys, as a WCON reader may ignore extensions it does not know.
   - **Refused:** any other key outside the core, with a message naming it.
   - **Field names** (offsets, centroid, units) are checked against the WCON 1.0 specification during implementation and cited in `IMPORT.md`.

3. **Resampling.**

   - Each animal is put on t = 0, 1, 2, … s from its first sample, by linear interpolation where neighbouring samples lie at most 1 s apart, and NaN otherwise. The grader's missing-sample rule then excludes any window that touches a gap.
   - The on-lawn column is resampled by nearest sample.
   - A track already at 1 s passes through unchanged, which is what makes the round trip exact. The vendored Scheer tracks are 2,400 samples, `LAWN_SAMPLES`, and the native path (`benchmark_bands.per_date_tallies`) windows each whole track. With `analysis_start_s = 0`, the imported window is the native one.

4. **Lawn membership.** The order is: the column if present, then the boundary, then `uniform`. A boundary's positions are taken relative to its centre, and a sample is on the lawn when `references.boundary_distance(x, y, radii) > 0`, Scheer's own rule. A circle is a constant `radii`.

5. **Steps.** `to_steps` writes one `BehaviourStep` per second, with:

   - `x`, `y` and `on_lawn` (on a uniform well, every step on);
   - zero heading, concentration and gradient, since the benchmark never reads them;
   - `phase` set to `"assay"` from `analysis_start_s`.

   The window is then `window(family, steps, (start, len(steps)))`, as for captures. A track shorter than its family's window is refused, with the grader's own error.

6. **The CLI.** `benchmark_grade.py tracks --manifest <file> --out <json>` takes lines `<family> <path>`, where `path` is a CSV (with its sidecar) or a WCON file. Each animal is one population member, and the population is graded by `grade_populations`, so a family is graded at its own size (bands at 64 or 200, or built on demand).

7. **The round-trip control (`track_import_check.py`).**

   - **The data:** the band-cohort Scheer animals of the wild-type small-lawn arm and the uniform-well arm (split side `bands` only), and Roberts 2016's calibration cohorts (C15, C162, C57).
   - **The trip:** each set is written as CSV and as WCON, re-imported and graded.
   - **The comparison:**
     - **lawn families:** the summed tallies (`grading.tally`) and statistics (`lawn_statistics`) from the native path, `real_on_lawn` and `animal_windows` on the same animals and windows;
     - **local search:** each worm's detector onsets and valid seconds, the inputs the family's statistics are computed from, against `reversal_onsets` and `valid_seconds` on the same positions. Roberts' tracks are 10 min, shorter than the local-search window, so the comparison is at the detector, not at γ.
   - **The gate:** **every compared value is equal** (to 1e-12 for floats, exactly for counts), for both formats.
   - **Held-out data:** an assertion refuses any animal whose split side is held out, before anything is written.
   - **The record:** `roundtrip.json`, with each family's values from each path.

8. **Independence at run time.** A test patches `BatchedEvaluator.run` and `npe_recovery.simulate` to raise, then grades a small synthetic CSV, which must succeed. This is a property of what grading does, not of what it imports. Importing the grader already loads simulator modules: `report/dtypes.py`, home of `BehaviourStep`, imports `brain.arch._brain`, and the `validation` package eagerly imports its agent-side modules. Decoupling those is a refactor of core modules, outside B.5. A lighter import would not make the package lighter to install either, since `qiskit` is a base dependency.

9. **The positioning note (`data/benchmark/IMPORT.md`).**

   - **The input contract:** decisions 1–5.
   - **What the benchmark measures that Tierpsy does not:** state-resolved roaming and dwelling bouts, lawn exits per on-lawn hour, pre-exit enrichment, and local-search decay. Tierpsy's 256 features are per-track kinematics and posture, without state or lawn context (Javer et al. 2018b, *Phil. Trans. R. Soc. B* 373:20170375).
   - **Where the two overlap:** speed and angular speed per window.
   - **OWMD:** it is a store of WCON tracks, so its tracks are importable whenever their metadata give the lawn and the analysis start.

## Risks / Trade-offs

- **[A tracker's centroid is not the midbody]** → stated in the note. Scheer's positions are midbody positions. A centroid differs by under a body width, and the boundary rule is sensitive only near the edge.
- **[Pixel units need a scale]** → refused without `mm_per_px`.
- **[A frame rate other than 1 s]** → resampled by linear interpolation. At 3 frames/s, Scheer's native rate, this is how the vendored 1-s tracks were made.
- **[Calhoun's tracks have no pixel scale]** → they are not used. Pixel inputs need `mm_per_px`, and the note says so.
