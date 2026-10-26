# 114: Phase 10 — Track Import: External Tracks Graded Through the Benchmark

**Status**: completed. **B.5 ships.** The benchmark now grades tracks from outside this repository, CSV or WCON, through the same windows, classifier, statistics and bands as its own references and simulations. A round trip through both formats reproduces the native path exactly, and grading imported tracks runs no simulation.

> | Round trip | Animals | CSV | WCON |
> |---|---|---|---|
> | Scheer small lawn, band cohorts | 1,013 | tallies and statistics equal | equal |
> | Scheer uniform well, band cohorts | 69 | equal | equal |
> | Roberts 2016, calibration cohorts (C15, C162, C57) | 78 worms, 2,323 onsets | detector onsets and valid seconds equal | equal |

**Date**: 2026-10-26.

**OpenSpec change**: `add-track-import`.

**Records**: [`supporting/114-track-import/roundtrip.json`](supporting/114-track-import/roundtrip.json), every family's values from both paths. The contract and the positioning are in [`data/benchmark/IMPORT.md`](../../../data/benchmark/IMPORT.md).

## Objective

B.5 is the last item of the benchmark block, and part of the roadmap's minimum-viable outcome: *"the benchmark valid … with the 1-s worm, its primitives and assays, and track import. Citable alone."* Until now, grading meant using the project's simulator or its vendored references.

The maintainer's decisions of 2026-10-26:

- CSV and WCON's core, read in the repository with no new dependency;
- lawn membership by a column or a boundary;
- local search included;
- a package module and a CLI.

## What was built

- **`quantumnematode.benchmark.track_import`:**
  - CSV with a metadata sidecar, and WCON 1.0's core with the metadata in a top-level `@nematode_benchmark` object, WCON's form for custom fields. Field names were checked against the WCON specification;
  - resampling to 1 s, where a gap longer than 1 s is missing data;
  - lawn membership by a column, by a circular or polar boundary under Scheer's rule, or by a uniform well;
  - conversion to the per-second steps the grader reads.
- **`benchmark_grade.py tracks --manifest … --out …`**, beside `batched` and `captures`.
- **`scripts/analysis/track_import_check.py`**, the round trip. It refuses held-out animals and cohorts before writing anything.
- **Tests:**
  - skeleton midpoints, offsets, merged records and refused keys;
  - gaps, interpolation and pixel scales;
  - circular and polar lawns, and the column's precedence;
  - the held-out guard and the manifest's family check;
  - grading with the simulator disabled.

## The round trip

- **The lawn families:** each band-cohort Scheer animal, with its own polar boundary, was written as a CSV and as a WCON file, re-imported and cut to its window. Every animal's tally, and the population's six (small lawn) or three (uniform well) statistics, equal the native path's, which windows the same 2,400 samples by `real_on_lawn` and `animal_windows`.
- **Local search:** Roberts' tracks are 10 min, shorter than the 45-min local-search window, so the comparison is at the detector, the input the family's statistics are computed from. Every worm's onsets and valid seconds are equal.
- **Calhoun 2014's tracks** are pixels with no published scale. They cannot be imported in mm, so they are not used.

## A limit stated, not hidden

Grading imported tracks calls no simulator code: a test grades with the batched evaluator and the simulation function replaced by ones that raise. But importing the grader still loads simulator modules. `report/dtypes.py`, home of `BehaviourStep`, imports the brain base class, and the `validation` package eagerly imports its agent-side modules. Decoupling them is a refactor of core modules, recorded as a possible later change. It would not lighten the installed package either, since `qiskit` is a base dependency.

## Conclusions

1. **B.5 ships.** Any track that can be put on a 1-s grid can be graded on the benchmark's 11 wild-type statistics, given its lawn and analysis start.
2. **The round trip is exact** in both formats, on 1,082 real band-cohort animals and 78 off-food worms.
3. **With B.5, the benchmark block is complete.** The minimum-viable outcome stands: the benchmark is valid (B.4), and its 1-s worm, primitives, assays and track import are all shipped.
