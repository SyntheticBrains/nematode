# 109: Phase 10 — The Benchmark's Grading Rule: Between-Date Bands, a Retry Registered Before It Ran, and Two Non-Learning Worms That Fail

**Status**: completed. **B.3 ships.** The benchmark now has bands, a pass rule and a distance, built from band cohorts alone.

- **The classifier,** refitted on band dates, passes its gate.
- **The rule's own check failed on its first attempt.** Its bands were too narrow. A retry was registered before it ran, and it passes.
- **The preflight passes:** a Poisson worm and a pirouette worm both fail the wild type, by a wide margin.

> | Step | Result | Gate |
> |---|---|---|
> | Classifier on held-back band dates | κ = 0.625 (accuracy 0.915) | κ ≥ 0.6 |
> | Rule's check, attempt 1 | small lawn 0.80; local search 0.86 | ≥ 0.90 each (**failed**) |
> | Rule's check, the registered retry | small lawn **0.925**; local search **0.962** | ≥ 0.90 each |
> | Preflight, ConstantHazards (n = 64) | fails all three families, D = 6.63 | description |
> | Preflight, PirouetteHazards (n = 64) | fails all three families, D = 7.22 | description |

**Date**: 2026-10-24.

**OpenSpec change**: `add-benchmark-grading`.

**Records** (in [`supporting/109-benchmark-grading/`](supporting/109-benchmark-grading/)):

- `attempt1_rule_check.json`: the failed first check and its diagnosis (`scripts/analysis/benchmark_check_diagnosis.py`);
- `bands_record.json`: the retry's calibration, its checks, the López-Cruz band batches and the reported rows;
- `preflight_constant.json`, `preflight_pirouette.json`: the preflight.

The vendored outputs are in `data/benchmark/`: `classifier.json`, `bands.json` and `lopezcruz2019_split.json`.

## Objective

D34 fixes the rule's form:

- bands from batch-matched real-data spread, read with the project's own classifier;
- mutant effects relative to the controller's own wild type;
- composite rules that depend on closed-loop contingencies.

B.3 builds that rule from the band cohorts alone, so that B.4's positive controls and negatives read held-out data the rule never saw.

The maintainer's decisions of 2026-10-24:

1. D36's form is food-dependent kinematics, as its own change ([Logbook 108](108-food-dependent-kinematics.md)), merged before the preflight.
2. The classifier is recalibrated on the band cohorts only.
3. The core lawn, search and mutants are graded. Density and the quintile ratio are reported.
4. The bands are between-date, with a z distance.

## What was built

- **`quantumnematode.benchmark.grading`** reads real and simulated tracks through one path.

  - 1-s positions are read at 5 s and cut into 10-s windows.
  - A window is on the lawn when every 1-s sample in its span is. On the small lawns that is the lawn's boundary, for real worms, or footprint, for simulated ones. On the uniform well every sample is on the lawn.
  - Each animal reduces to a tally of sums, so a population's statistics are ratios of summed tallies and a bootstrap resample is a sum.
  - Bout means are censoring-aware: a state's time over its completed bouts.

- **The statistics.** The wild type is graded on 11:

  | Family | Statistics |
  |---|---|
  | Scheer's small lawn | roaming fraction, roaming bout mean, dwelling bout mean, exits per on-lawn hour, pre-exit roaming enrichment, time on the lawn |
  | Scheer's uniform well (new protocol `uniform_lawn_scheer`) | roaming fraction, roaming bout mean, dwelling bout mean |
  | López-Cruz's local search | γ, and the early/late ratio, both level-free |

  The mutants are tph-1 against N2, and pdfr-1 and tax-4 against PD1074. Each is graded on Δ roaming fraction and Δ exits per on-lawn hour, and pdfr-1 also on Δγ and Δ early/late.

- **The pass rule.**

  - |z| ≤ 2.84 for each wild-type statistic, Bonferroni over 11.
  - Each mutant is its own family: 2.24 for two statistics, 2.50 for four.
  - A mutant's Δ must also take the reference's sign wherever the reference's interval excludes zero.
  - An undefined statistic fails and enters the distance at the cap, |z| = 10.

- **The distance.**

  - D is the weighted RMS of the capped z's. Each assay weighs a third, split equally among its statistics.
  - D_fit leaves local search out, as D37 holds it out of the fit.

- **The scripts.**

  - `benchmark_classifier.py` (the recalibration);
  - `benchmark_bands.py` (the bands and the rule's checks);
  - `benchmark_grade.py` (grades a batched run or a set of captures);
  - `benchmark_check_diagnosis.py` (attempt 1's diagnosis).

## The classifier

The OD 2 wild type's 41 band dates were split by date (seed 2029):

- 27 dates and 679 animals to fit, against the authors' labels;
- 14 dates and 334 animals to gate.

On the gate dates, κ = 0.625 and accuracy 0.915, over 76,991 windows, against the registered 0.6. D.1's model reached 0.63 on an animal-level split, so a date-level split costs it almost nothing. No held-out date was read.

The gate first decoded every on-lawn window rather than only the labelled runs the fit used, against the spec's text. A review caught it. Re-read on the fit's runs, κ moves from 0.62483 to 0.62491 over the same 76,991 scored windows, and the model is unchanged, so the bands are too.

## The bands, and a failed check

The bands come from a cluster bootstrap over band dates at the graded population's size. López-Cruz's band comes from 17 band batches; its 27 wild-type batches were split a third held out, with pdfr-1's control kept in the bands.

**Attempt 1 failed the rule's own check.** Bands from half the band dates graded a 200-animal population drawn from the other half. They passed jointly in 0.80 of 200 splits, against 0.90. López-Cruz's per-batch check passed 0.86. The diagnosis, on band data only:

- **On the small lawn, the z's had an SD of 1.37–1.64, not 1.0.** Part of that is the centre's own uncertainty, which the bootstrap scale left out: adding it lifts the joint pass rate to 0.865. The rest is date-to-date spread that a bootstrap over dates did not hold.
- **López-Cruz's scale was an SD from about nine batches,** read at a normal bar. A Student-t prediction interval passes at 0.948.

**The maintainer chose a registered retry.** It was committed (`64f2e3bb`) before it ran:

1. every Scheer scale holds the centre's uncertainty, √(σ_n² + σ_centre²);
2. each statistic's scale is multiplied by a calibration factor κ, the SD of its z over half-splits on a separate seed (2033), floored at 1;
3. López-Cruz uses a Student-t prediction interval;
4. the check is re-read on fresh splits (seed 2034), at the same gates.

**The retry passes:** the small lawn at 0.925 and local search at 0.962.

- The factors κ came out at 1.19–1.30 once the centre term was in.
- López-Cruz's interval is 1.196 times the plain SD.

### The bands at n = 200

| Statistic | Small lawn: centre (scale) | Uniform well: centre (scale) |
|---|---|---|
| roaming fraction | 0.169 (0.037) | 0.213 (0.089) |
| roaming bout mean (s) | 217 (22) | 328 (28) |
| dwelling bout mean (s) | 1,068 (229) | 1,264 (544) |
| exits per on-lawn hour | 0.71 (0.20) | — |
| pre-exit enrichment | 0.295 (0.070) | — |
| time on the lawn | 0.978 (0.006) | — |

**Local search:** γ = 0.108 per minute (scale 0.040), and early/late = 4.21 (scale 1.62).

**Exits read 0.71 per on-lawn hour, not D.1's 0.21.** The rule is the same: the first wholly off-lawn window after an on-lawn run. D.1's windows came from the authors' out-of-lawn runs, which last longer than one window. Under the geometric rule, a single 10-s window off the lawn is an exit. Real and simulated worms are read alike, and the band is built on this reading.

### The mutants

| Mutant | Δ roaming fraction: centre (scale), direction | Δ exits per on-lawn hour: centre (scale) | Roadmap's expectation |
|---|---|---|---|
| tph-1 (3 paired dates, inflated) | +0.270 (0.052), **more** | +0.07 (0.28), unresolved | more roaming, more leaving |
| pdfr-1 (8 paired dates) | −0.053 (0.046), unresolved | +0.04 (0.27), unresolved | less roaming, normal leaving |
| tax-4 (11 paired dates) | −0.116 (0.045), **less** | +0.31 (0.37), unresolved | less roaming, normal leaving |

pdfr-1 on local search: Δγ = −0.040 (scale 0.057) and Δ early/late = −0.29 (scale 2.29). Neither is resolved.

**Only two directions are resolved:** tph-1 roams more, and tax-4 roams less. Both match the roadmap. pdfr-1's lower roaming is in the expected direction but within its band. No leaving or search difference is resolved, so those are graded by magnitude alone. That is the "normal leaving" equivalence band, and for tph-1's leaving it is weaker than the roadmap's expectation.

## The preflight

Registered settings: seeds 9101–9164, n = 64, run speed 0.15 mm/s, hazards (0.02, 0.01, 0.01)/s, β = 3 for the pirouette rule. The MLP arm's assay configs were used, on the amended primitives.

| Statistic | ConstantHazards: value (z) | PirouetteHazards: value (z) |
|---|---|---|
| small lawn: roaming fraction | 0.95 (+10) | 0.98 (+10) |
| small lawn: exits per on-lawn hour | 234 (+10) | 333 (+10) |
| small lawn: time on the lawn | 0.017 (−10) | 0.016 (−10) |
| small lawn: dwelling bout mean (s) | 65 (−2.4, passes) | undefined (fails) |
| small lawn: pre-exit enrichment | 0.002 (−2.5, passes) | 0.000 (−2.5, passes) |
| uniform well: roaming fraction | 0.82 (+5.5) | 0.91 (+6.3) |
| local search: γ (per min) | 3.0 (+10) | 3.0 (+10) |
| local search: early/late | 1.06 (−1.9, passes) | 1.03 (−2.0, passes) |
| **D / D_fit** | **6.63 / 6.32** | **7.22 / 7.22** |

**Both fail the wild type, in every family, so B.4 may register.**

Two cautions for B.4, as description:

- **Four statistics pass for a Poisson worm:** the dwelling bout mean, pre-exit enrichment, the uniform well's roaming bout mean, and early/late. Each is wide enough that a worm with no state passes it alone. The family rule and the distance carry the discrimination.
- **γ sits at the fit's ceiling of 3 per minute** for both controllers. Their curves are flat, so the decay is undefined, and a flat curve fails at the cap.

## Reported, not graded

- **The geometric rule against the authors' masks:** 98.4% of windows agree with their in-lawn runs, and 99.5% of seconds with their midbody-in-lawn flag.

- **Roaming fraction, ours against the authors' labels:** 0.169 against 0.108 on the small lawn, and 0.213 against 0.148 on the uniform well. Our classifier reads more roaming, as D.1's did.

- **Density, roaming fraction on band dates:**

  | | OD 1 | OD 2 | OD 4 |
  |---|---|---|---|
  | all band dates | 0.333 (1 date) | 0.169 | 0.168 (2 dates) |
  | the one shared date | 0.333 | 0.209 | 0.192 |

  Roaming falls with density, as Scheer reports.

- **γ, two ways:** B.2's fit of the pooled Fig. 1C curve gives 0.114 per minute. The band's centre, the mean of the band batches' fits, is 0.108.

- **Flavell 2013's timescales** (dwelling τ ≈ 482 s; roaming 75 and 520 s) and **the quintile ratio's reference** (2.26–2.58) are reported, not graded.

**Limits.**

- Real dwelling speeds carry tracking noise, and simulated ones carry none.
- The bout means are the exponential estimator, and roaming bouts are double-exponential.
- López-Cruz's band is the spread of single batches of about 55, conservative for 200 animals.
- The uniform well has two band dates, so its rule check cannot be run, and its scales rest on the small lawn's.

## Implementation notes

- **A 40-min track gives 239 complete windows,** not 240. Real and simulated tracks both read 2,400 samples.
- **The uniform well's rim.**
  - A worm held at the circular arena's wall can land about 1e-15 mm outside a lawn whose radius is the well's. That read 9% of steps as off the lawn.
  - Both lawn tests now count a point within 1e-9 mm of a rim as on the lawn.
  - Only a lawn that meets the wall puts a worm there.
- **The off-food detector's parameters** are now a named constant, `readouts.OFF_FOOD_DETECTOR`, pinned to B.2's record.

## The novelty check owed at B's registration

The novelty map's first claim is a graded benchmark of standard on-lawn assays for simulated worms, with real-track positive controls and matched negatives. Three recent works were read against it:

- **NeuroSparkSNT** (Yazır, Research Square rs-10240803, 5 August 2026, CC BY 4.0).
  - An eight-operator semi-Markov phase controller on 114 neurons of Cook 2019's connectome, with fixed weights and nothing fitted.
  - Twelve benchmarks against imaging and behaviour, among them AVA–AVB antagonism and omega dwell time, each on three seeds.
  - Klinokinesis is built in, with Pierce-Shimomura's reversal-to-pirouette probability.
  - No null, no rewiring, no mutant, no lawn, roaming/dwelling or local-search assay, and no closed loop with real-track grading.
  - **It does not occupy the claim.**
- **PGOB** (He, Zenodo, June 2026). It fits BAAIWorm/modWorm and predicts 43 held-out mutants on locomotion features, with no nulls and no lawn assays. **The nearest on mutant hold-out; it does not grade on-lawn foraging statistics.**
- **Morrison & Young 2025** (*PLoS Comput Biol*). Premotor dynamics and dwelling/roaming, open loop. **No graded benchmark and no closed loop.**

The claim stands as first-in-field.

## Conclusions

1. **The benchmark has its rule.** There are 11 wild-type statistics, mutant differences against their own wild type, between-date bands from band cohorts only, a per-statistic pass at a family-wise bar, and a distance.
2. **The rule's first bands were too narrow,** and its own check caught that before anything simulated was graded. The registered retry passes on fresh splits.
3. **Two non-learning worms fail by a wide margin.** Four single statistics do not tell them from the animal, so the family rule and the distance carry the discrimination.
4. **Only two mutant directions are resolved by the data:** tph-1 roams more, and tax-4 roams less.

## Next steps

- [ ] B.4: the controls. Cohort-held-out real tracks and real mutant cohorts must pass. The matched negatives (the open-loop Scheer HMM, and Poisson-timed primitives at wild-type rates) and the trivial ones (a random walk, a correlated random walk, the untrained policy) must fail. W's minimum is read by the registered rule.
