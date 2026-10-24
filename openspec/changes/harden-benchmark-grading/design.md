## Context

- **The direction rule** (`grading.Band.direction`): resolved when |μ| > 1.96σ, with σ the band's scale at the graded n, after κ and any inflation. That σ is the spread of a new population's statistic. A reference's Δ is non-zero when the centre's own uncertainty excludes zero, which is a different and smaller quantity.
- **Per-date paired differences** are what the paired bootstrap resamples, so a paired t-test across them uses the same unit, the recording date. With three to eleven paired dates, a t-test is the standard small-sample test, and it is more conservative than the bootstrap SE (pdfr-1 roaming: bootstrap z −1.99, t −2.14 on 7 df).
- **The bands' builders** (`benchmark_bands.py`) already take n as an argument. Only the CLI fixes n to 64 and 200, and `benchmark_grade.bands_at` refuses other sizes.
- **`fit_local_search`** searches γ on a log grid from 1e-3 to 3 per minute, then refines by golden section between the best grid point's neighbours. A flat curve's least-squares γ runs to the grid's top.
- **López-Cruz's pdfr-1 pair** (rows 91 and 90) is kept in the bands and never drawn into the hold-out. It has no held-out counterpart.

## Goals / Non-Goals

**Goals**

- A direction test that asks the right question, on the reference dates' own unit.
- A grader that can read B.4's real controls at their own sizes.
- No graded statistic that B.4 cannot positive-control.

**Non-goals**

- Changing any band's centre or scale, or any wild-type statistic.
- Pooling band and held-out mutant dates. Those dates are B.4's mutant positive control.
- B.4's multiplicity rule. It is B.4's registration, noted in the tracker.

## Decisions

1. **The paired t-test.**
   - For each paired band date with both arms present, the Δ is the mutant's pooled statistic minus the wild type's.
   - A direction is resolved when |t| exceeds t_{k−1}(0.975), with k paired dates and k ≥ 2. Otherwise it is unresolved. A single paired date cannot resolve a direction.
   - It is written to `bands.json` beside each mutant statistic, as `direction`, `paired_t` and `paired_dates`. The magnitude band is unchanged.
2. **Bands on demand.**
   - `benchmark_bands.build_family_bands(family, n)` and `build_mutant_bands(mutant, n)` build one family's bands at n, from the vendored store, with seed 2030 and 2,000 resamples, and κ from `bands_record.json`.
   - The grader calls them when n is not in `bands.json`, and uses the vendored values when it is. A test checks the two agree at n = 200.
   - López-Cruz's band has no n: it is the band batches' prediction interval at any n.
3. **γ without a decay.**
   - `local_search_statistics` returns NaN for γ when the fitted γ is within 1% of either grid bound, or α ≤ 0. The early/late ratio is unchanged.
   - Real band batches are re-checked: every one keeps its γ, or the change is reported.
4. **pdfr-1's local search reported.**
   - Its Δγ and Δ early/late move to `bands_record.json` under `reported`.
   - pdfr-1's family is two statistics, and its bar is `z_star(2)` = 2.24.
5. **Ji 2021 beside.** The record reports Ji 2021's pdfr-1 and tax-4 patch-assay roaming against its wild type's, from `data/roaming_dwelling/ji2021_readings.json`, as directions only. Its assay differs from Scheer's.

## Risks / Trade-offs

- **[The rebuild changes recorded numbers]** → only the direction fields, pdfr-1's bar and pdfr-1's search move. The centres, scales, κ and the rule's checks are unchanged by construction. The rebuild proves it, and Logbook 109's addendum reports every changed value beside the original.
- **[On-demand bands are slower]** → a family at n takes the bootstrap's 2,000 resamples, seconds on the store. B.4 grades a handful of populations.
- **[The t-test is conservative with few dates]** → it is stated. It is the test that matches the unit the bands resample.
