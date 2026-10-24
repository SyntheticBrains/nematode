## Context

From the B.3 survey of 2026-10-24.

**The store's arms** (`data/benchmark/scheer2023_split.json`):

| Arm | Dates | Animals | Band dates | Held-out dates |
|---|---|---|---|---|
| PD1074 OD 2 (the wild type) | 65 | 1,586 | 41 | 24 (573 animals) |
| PD1074 OD 1 | 1 | 45 | 1 | 0 |
| PD1074 OD 4 | 3 | 83 | 2 | 1 |
| PD1074 uniform lawn | 3 | 100 | 2 | 1 (31 animals) |
| N2 OD 2 | 70 | 1,380 | 42 | 28 |
| tph-1 (MT15434) | 5 | 139 | 3 | 2 |
| pdfr-1 (CX14295) | 16 | 395 | 10 | 6 |
| tax-4 (CX13078) | 19 | 499 | 12 | 7 |

**Same-date controls.**
- tph-1 shares all five of its dates with N2, and none with PD1074.
- pdfr-1 shares 11 of its 16 dates with PD1074: 8 band, 3 held out.
- tax-4 shares 17 of its 19 dates with PD1074: 11 band, 6 held out.
- A date's side is the split's, so a mutant and its control on one date are always on the same side.

**Scheer's protocol** (eLife 88657, Methods):
- Assays ran 1 h at 3 frames/s, and only the last 40 min were analysed.
- The uniform lawn was the same 10 mm well with 15 µL of OD 2 bacteria spread evenly, "as in the small
  lawn foraging assay".
- Flavell 2013's uniform lawn is a different plate, and none of its tracks are vendored.

**The classifier** (`validation/roaming_dwelling.py`, `data/roaming_dwelling/calibration_retry.json`):
- D.1's Gaussian HMM was fitted on an animal-level half of every wild-type date (split seed 2026) and
  gated at κ = 0.63 on the other half.
- It reads 10-s windows of 5-s samples: log speed and angular speed.
- It decodes each run of on-lawn windows separately.
- On real tracks, D.1 marked a window on the lawn from the authors' `in_lawn_run`. On simulated ones,
  `lawn_states.episode_windows` marks it from the worm's on-lawn flag at every step in it.

**Leaving** (`data/roaming_dwelling/scheer2023_leaving.json`):
- An exit is the first wholly off-lawn window after an on-lawn run.
- The wild type gives 0.206 exits per on-lawn hour, and the pre-exit band (30–180 s) is 0.29 above the
  overall roaming fraction.
- Scheer reports leaving as events per animal per 40-min assay (Fig. 1H), counting an event when
  the whole body leaves. The roadmap's published rate is about one event per 95–100 min (Bendesky
  2011). The project's sustained rule counts fewer: a whole 10-s window off the lawn.

**López-Cruz** (`data/benchmark/lopezcruz2019_reorientations.json`):
- The deposit has 58 wild-type rows. Most repeat a control reused for several mutants.
- Grouped by recording-date string they are **29 distinct batches**, median n ≈ 55.
- Fitted over the registered window, their γ has mean 0.115 and SD 0.033. The early/late ratio has
  median 4.2 [IQR 3.6, 4.9].
- pdfr-1 (row 91) has one same-date control (row 90).

**The readouts' scope** (B.2): on-lawn reversal counts are not graded. The local-search readout is
`reversal_onsets` on the off-food phase.

## Goals / Non-Goals

**Goals**

- A pass rule and a distance that read real and simulated worms through the same functions.
- Built from band cohorts alone, so B.4's positive controls are a genuine hold-out.
- Every threshold fixed here, before any simulated worm is graded against it.

**Non-goals**

- The controls themselves (B.4), and track import (B.5).
- Grading reversal counts on the lawn: the detector fails there (B.2).
- Grading density dependence, the quintile ratio, or Flavell's bout timescales. Each lacks either dates
  or a matched simulated assay, so each is reported.
- Thermotaxis and the frozen scenarios.

## Decisions

1. **The classifier is refitted on band dates, and the gate is on band dates too.**
   - The OD 2 wild type's 41 band dates are split two thirds and one third by date (seed 2029).
   - The model is fitted with `GaussianHMM.fit_labelled` on the two thirds, against the authors' labels.
   - Only windows that are both on the lawn by the geometric rule and labelled roaming or dwelling
     enter the fit, as runs of consecutive such windows. The gate decodes the same runs.
   - It is gated at κ ≥ 0.6 on the third: the gate D.1 used.
   - D.1 passed at κ = 0.63, three points above the bar, with twice the fit animals and an
     animal-level split. A date-level split may come in lower. A failure spends no held-out data.
   - The fitted model is what ships, not a refit on all band dates, so the gate's reading belongs to
     the model that is used.
   - If the gate fails, the script stops and the maintainer decides. The registered fallback is D.1's
     model, with its overlap with the held-out dates stated.
   - Mutants and density arms are read by the same model, unchanged.

2. **One window rule, by geometry.**
   - For real worms on the small lawns, a second is on the lawn when the midbody lies inside
     Scheer's boundary for its background slot (`boundary_distance`, positive inside, already
     validated against the authors' centroid distance). On the OD 2 wild type it agrees with the
     per-frame `midbody_in_lawn` flag on about 99% of seconds.
   - **The uniform well has no boundary.** In the store its 64 radii are NaN, its in-lawn flag is 0 on
     every frame, and its positions are relative to the first position, not a lawn centre. The bacteria
     cover the well, so on the uniform well every sample is on the lawn, for real and simulated worms
     alike.
   - For simulated worms, a second is on the lawn when its position lies in the lawn's footprint, in a
     phase with lawns: the capture's existing on-lawn flag.
   - A window is on the lawn when every 1-s sample in its span is, as `episode_windows` reads it, not
     only its three 5-s samples. A window with a missing sample is unclassified.
   - D.1's `in_lawn_run` comes from the authors' processing, which has no simulated counterpart. Its
     agreement with the geometric rule is reported.

3. **Phase-aware readers.** A simulated run is read only from its assay phase's `analysis_from_s`:
   - Scheer's small lawn: minutes 20–60 of 60, the same 40 min as Scheer's recordings;
   - the uniform lawn: the same;
   - López-Cruz: from removal.

4. **Bout means are censoring-aware.** A bout is a maximal run of one state inside one on-lawn run.
   - A bout is completed when it ends in the other state. It is censored when it ends at a run's end:
     leaving the lawn, a missing window, or the recording's end.
   - The mean is the state's total time over its completed bouts: the exponential MLE under right
     censoring.
   - The 40-min window truncates long dwelling bouts, and this reads real and simulated worms alike.
   - The estimator is the mean only for exponential bouts, and roaming bouts are double-exponential
     (Flavell 2013). It is one fixed estimator, applied the same way to both, not a claim about the
     distribution.

5. **The graded statistics.** The definitions below are fixed. On real tracks every value is
   recomputed by this pipeline.

   | Family | Statistic | Definition |
   |---|---|---|
   | Small lawn | roaming fraction | roaming windows / classified on-lawn windows, pooled |
   | | roaming bout mean, dwelling bout mean (s) | Decision 4 |
   | | exits per on-lawn hour | exits / on-lawn hours, pooled |
   | | pre-exit enrichment | roaming fraction 30–180 s before exits, minus the overall |
   | | time on the lawn | on-lawn windows / all windows |
   | Uniform lawn | roaming fraction, roaming bout mean, dwelling bout mean | as above |
   | Local search | γ | `fit_local_search` over 8–44 min after removal, t₀ = 9 min |
   | | early/late ratio | mean onsets per bin, 8–14 min over 34–44 min after removal |
   | Mutants | Δ roaming fraction, Δ exits per on-lawn hour | mutant minus same-date wild type, Scheer |
   | | Δγ, Δ early/late (pdfr-1) | against row 90, López-Cruz |

   - **Local search is graded only by level-free statistics.** López-Cruz's detection thresholds are
     unpublished, and B.2's detector differs from theirs in level.
   - **A statistic that cannot be computed** (a worm that never leaves has no pre-exit band) fails its
     band and enters D at the cap.

6. **Bands by a cluster bootstrap over band dates, at the graded population's size.**
   - For a statistic and a population of n animals, draw band dates with replacement. Take all of each
     drawn date's animals, until n is reached; the last date is subsampled. Compute the pooled
     statistic, 2,000 times (seed 2030).
   - The band's centre μ is the statistic pooled over every band animal. Its scale σ is the bootstrap's
     SD.
   - So σ holds the between-date spread and the sampling noise of n animals.
   - The simulated arms are graded at D38's N = 200. B.4's real positive controls are graded at their
     own size, by the same procedure.
   - **Mutants:** a resample draws paired dates, and Δ is the pooled mutant minus the pooled wild type
     over the drawn dates. The animals counted toward n are the mutant's.
   - **Thin arms** (fewer than five band dates): the uniform lawn, and tph-1 with its three paired
     dates.
     - Their σ is the larger of their own and the OD 2 wild type's for the same statistic at the same
       n.
     - For a mutant's Δ it is the larger of its own and √2 times the wild type's: the spread of a
       difference of two independent batches.
   - **López-Cruz:** a batch is one distinct date string. Where two rows share one, the larger n is
     kept.
     - pdfr-1's same-date control (row 90) is one of the 29. That pair (rows 90 and 91) is kept as
       band data and never drawn into the hold-out. So B.4 has no held-out mutant cohort for local
       search, and says so.
     - The other 28 batches are split by date, a third held out (seed 2031), with the Scheer split's
       procedure.
     - μ is the mean of the band batches' per-batch fits, and σ their SD, at their own size, n ≈ 55.
       That is conservative for N = 200, and stated.
     - μ is not B.2's 0.114: that is one fit of the pooled Fig. 1C curve, and a fit of a pooled curve
       is not the mean of the fits. μ is used because it is on the same scale as σ. B.2's value is
       reported beside it.
     - pdfr-1's one paired batch gives its Δ a σ of √2 times the wild type's.

7. **The pass rule.**
   - z = (s − μ)/σ, with |z| capped at 10.
   - **The wild type is one family** of 11 statistics across three assays. A statistic passes when
     |z| ≤ z\* = Φ⁻¹(1 − 0.05/22) = 2.84, Bonferroni for a 95% family-wise rate. The worm passes the
     wild type when all 11 do.
   - **Each mutant is its own family:**
     - tph-1 and tax-4 have 2 statistics, so z\* = 2.24;
     - pdfr-1 has 4, so z\* = 2.50.
     - **Direction:** where the reference's interval μ ± 1.96σ for Δ excludes zero, the simulated Δ
       must have its sign. Where it includes zero, as for "normal leaving", the magnitude test alone
       applies: it is the equivalence band. σ is the band's scale after any inflation, so a thin arm
       cannot resolve a direction from resampling three dates.
   - **The roadmap's expected directions** (pdfr-1: less roaming, normal leaving, more search; tax-4:
     less roaming, normal leaving; tph-1: more roaming, more leaving) are checked against the
     references and reported. The rule uses the references' own intervals.

8. **The z distance.**
   - D = √(Σ wᵢ zᵢ² / Σ wᵢ) over the wild-type family.
   - Each assay's weight is 1/3, split equally among its statistics: 1/18 for each small-lawn
     statistic, 1/9 for each uniform-lawn one, 1/6 for each local-search one.
   - **D_fit** drops local search and renormalises, to 1/12 and 1/6. It is the distance C.2's sep-CMA
     minimises, since D37 holds local search out of the fit.
   - **W's minimum, the rule registered here and read in B.4:**
     - Split the held-out wild-type dates into two halves by date, 1,000 times.
     - For each split, take z = (s_A − s_B)/σ, with σ the band's scale at a half's size (not √2σ),
       and form D from those z's. It is an empirical floor, so the scale only has to be the one D
       uses.
     - W's registered minimum effect is that distribution's 95th percentile: the distance two real
       batches show by chance.

9. **The rule's own check, before any simulated worm** (two gates, registered here). It is read at
   the operating point: populations of N = 200.
   - **The small lawn.** Split the OD 2 wild type's band dates into halves, 200 times (seed 2032).
     Build the bands on one half at n = 200. From the other half, draw one 200-animal population by
     the bands' own date resampling, and grade it on the six statistics at the wild-type z\* = 2.84.
     **Gate:** a joint pass rate ≥ 0.90.
   - **López-Cruz.** Split the band batches into halves, 200 times (same seed). Build the band on one
     half, and grade each batch of the other half alone on γ and early/late, at the same z\*. That is
     the unit σ describes. **Gate:** a per-batch joint pass rate ≥ 0.90.
   - The uniform lawn has two band dates and cannot be halved, so it is left out and stated.
   - A failure of either gate stops the change before the bands are committed.

10. **The preflight, as description.**
    - Two controllers, each running 64 animals per assay through the batched evaluator, on seeds
      9101–9164, at the D36 pilot's registered settings:
      - ConstantHazards, the Poisson negative: run speed 0.15 mm/s, hazards (0.02, 0.01, 0.01)/s;
      - PirouetteHazards, E.2's positive control: the same, with β = 3.
    - Each is graded at n = 64 and reported.
    - It runs after D36's amendment merges.
    - If ConstantHazards passes the wild type, the benchmark cannot yet tell a Poisson worm from the
      animal, and the maintainer decides before B.4 registers. The bands are not tuned on it.

11. **The uniform lawn is Scheer's**, protocol `uniform_lawn_scheer`:
    - a 10 mm well whose lawn covers it;
    - the worm placed on it at the centre;
    - 60 min, analysed from 20 min.

    - The lawn's radius equals the well's, with `wall_clearance_mm: 0`. The default clearance of 1 mm
      would trip the lawn-fits-the-arena check.
    - The circle arena holds the worm at r ≤ 5 mm. So "always on the lawn" needs the rim to count as
      on the lawn in both runners: `_cells_under` tests `<=`, and implementation checks `lawn_at`.

    It is the uniform-lawn reference with tracks and dates. `uniform_lawn_flavell` stays. Only
    Flavell 2013's published timescales are reported, with no simulated reading.

## Risks / Trade-offs

- **[Bonferroni over correlated statistics is conservative]** → the bands may be wide enough for a
  negative to pass. The preflight and B.4's negatives are there to show it. Tightening after seeing a
  negative would be tuning, so it goes to the maintainer.
- **[Thin arms are inflated, not measured]** → the uniform lawn and tph-1 rest on the OD 2 wild type's
  spread, stated in their rows.
- **[López-Cruz's band is at n ≈ 55]** → wider than a 200-animal band would be. Stated, and the
  conservative direction.
- **[Few exits per date]** → about 3.5 exits per date make per-date leaving statistics meaningless.
  The cluster bootstrap pools dates up to n, so a 200-animal resample holds about 25 exits.
- **[The geometric rule differs from D.1's]** → the agreement is reported, and the classifier is
  refitted under the rule it is used with.
- **[Tracking noise]** → real dwelling speeds carry tracking noise of a few µm per sample, and
  simulated ones carry none. This classifier asymmetry was inherited from D.1. It is named as a
  limit, not corrected.

## Open Questions

None blocking. D38's N = 200 is taken as the simulated population's size, and C may register a
different N with the bands recomputed at it.
