# Grading external tracks

The benchmark grades any worm track that can be put on a 1-s grid, from a lab's tracker or from another simulator. The tracks go through the same windows, classifier, statistics and bands as the project's own references and simulations (`quantumnematode.benchmark.track_import`). Grading imported tracks runs no simulation. Importing the grader loads none of the simulator and no PyTorch, so the benchmark works without the `torch` extra.

```bash
uv run python scripts/analysis/benchmark_grade.py tracks --manifest <manifest.txt> --out <grade.json>
```

Each manifest line is `<family> <file>`. A file is a CSV with its metadata sidecar, or a WCON file. Each animal in a file is one member of the family's population, and the population is graded at its own size.

## The input contract

**Metadata.** These go in a CSV's sidecar `<file>.json`, or in a WCON file's top-level `@nematode_benchmark` object:

| Field | Meaning |
|---|---|
| `family` | `small_lawn`, `uniform_lawn` or `local_search` |
| `analysis_start_s` | when the family's window begins, on the track's own clock |
| `lawn` | `{"centre": [x, y], "radius": r}`, or `{"centre": [x, y], "radii": [...]}` (the boundary's radius at equally spaced angles from +x), or `"uniform"`. Not needed for `local_search`, or when the CSV has an `on_lawn` column. |
| `units` (CSV) | `mm`; `um`, `µm` or `microns`; or `px` or `pixels` with `mm_per_px` |

**CSV.** One row per sample, with columns `animal`, `t_s`, `x` and `y`, and optionally `on_lawn` (0/1). Empty or `nan` positions are missing samples.

**WCON 1.0** (Javer et al. 2018a), its core:

- **units:** `units` gives t in s, min or h, and x and y in a physical length unit. WCON has no pixel unit.
- **data:** one record or a list. Records sharing an `id` are merged in time order.
- **positions:**
  - a skeleton at each time is reduced to its middle point, since the benchmark reads the midbody;
  - origin offsets `ox` and `oy` are added;
  - `null` is a missing sample.
- **other fields:** `@`-prefixed keys are ignored. A key outside the core is refused, with its name.

**Resampling.**

- Each animal is put on whole seconds from its first sample, by linear interpolation between samples at most 1 s apart.
- A longer gap is missing data, so no 10-s window touching it is classified.
- A track already at 1 s passes through unchanged.

**Lawn membership.** A sample is on the lawn:

- by the `on_lawn` column, if present;
- otherwise, when its signed distance inside the boundary about the lawn's centre is positive, the rule the vendored Scheer tracks are cut by;
- on a uniform well, always.

**Windows.** A family's window is fixed:

- the lawn families read 40 min (2,400 s) from `analysis_start_s`;
- local search reads 45 min (2,700 s) from food removal.

A track shorter than its window is refused.

**The round trip.** `scripts/analysis/track_import_check.py` writes the vendored band-cohort Scheer animals out as CSV and as WCON, and re-imports them. The animals are 1,013 on the small lawn and 69 on the uniform well. Their tallies and statistics equal the native path's exactly. Roberts 2016's calibration-cohort worms (78) give the same detector onsets and valid seconds ([Logbook 114](../../docs/experiments/logbooks/114-track-import.md)).

## Against Tierpsy and the Open Worm Movement Database

**Tierpsy's 256 features** (Javer et al. 2018b) summarise each track's morphology, posture, path and motion, including its motion modes (forward, backward, paused) and their durations. The benchmark measures something else: behaviour relative to food and its history.

- **what Tierpsy does not define:**

  - roaming and dwelling states and their bout lengths;
  - lawn exits per on-lawn hour and the roaming enrichment before an exit;
  - time on the lawn;
  - the decay of reorientations after food removal.

  None of Tierpsy's features is defined against a lawn boundary or a removal time.

- **the overlap:** speed and angular speed, which the benchmark reads per 10-s window to classify states.

The two are complementary. Tierpsy describes how a worm moves, and the benchmark grades what it does on and off food against batch-matched real-worm bands.

**The Open Worm Movement Database** (Javer et al. 2018a) distributes tracks as WCON. One of its tracks is gradeable here when three conditions hold:

- its metadata give the lawn (a boundary, or uniform) and the analysis start;
- it is long enough for its family's window;
- its positions are in a physical unit.

## References

- Javer A, et al. (2018a). An open-source platform for analyzing and sharing worm-behavior data. *Nature Methods* 15:645–646.
- Javer A, Ripoll-Sánchez L, Brown AEX (2018b). Powerful and interpretable behavioural features for quantitative phenotyping of *Caenorhabditis elegans*. *Phil. Trans. R. Soc. B* 373:20170375.
- WCON specification: `WCON_format.md` in the openworm/tracker-commons repository.
