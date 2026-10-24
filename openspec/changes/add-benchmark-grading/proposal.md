## Why

B.1 vendored the reference tracks and their split. B.2 built and validated the readouts. Nothing yet says
what a simulated worm must produce to pass. D34 fixes the rule's form:

- **bands** from batch-matched real-data spread, read with the project's own classifier;
- **mutant effects** relative to the controller's own wild type;
- **composite rules** that depend on closed-loop contingencies.

B.3 builds that rule, from the band cohorts alone, so that B.4's positive controls and negatives read
held-out data the rule never saw.

The maintainer's decisions of 2026-10-24:

1. **D36's form:** food-dependent kinematics. It is its own change, `amend-food-dependent-kinematics`,
   and lands before B.4.
2. **The classifier is recalibrated on the band cohorts only.** D.1's model was fitted across every
   wild-type date, so the held-out dates were never unseen by it.
3. **Core lawn, search and mutants** are graded. Density and the quintile ratio are reported, not
   graded.
4. **Between-date bands, with a z distance.**

## What Changes

- **A classifier from the band cohorts.**
  - The same two-state Gaussian HMM as D.1, fitted on two thirds of the OD 2 wild type's band dates
    against the authors' labels.
  - Gated at κ ≥ 0.6 on the other third, registered before the fit.
  - Vendored as `data/benchmark/classifier.json`. D.1's model and every record read with it are
    unchanged.
- **One window rule for real and simulated tracks.**
  - 1-s positions are read at 5 s.
  - A window is on a lawn when every sample in it lies inside the lawn: by Scheer's boundary for real
    worms, and by the lawn's footprint for simulated ones.
  - Readers are phase-aware: they read only an assay's analysis window.
- **The graded statistics.** On real tracks they are computed by this pipeline, never taken from the
  authors' labels or published values.
  - **Scheer's small lawn, OD 2 wild type** (6 statistics, 40 min):
    - on-lawn roaming fraction;
    - mean roaming and dwelling bout durations, censoring-aware;
    - sustained exits per on-lawn hour, the project's rule (0.21/h), not the authors' 0.65/h;
    - pre-exit roaming enrichment, the composite closed-loop rule;
    - time on the lawn.
  - **The uniform lawn**, Scheer's uniform-lawn arm (3 statistics): roaming fraction and both bout
    means.
  - **Local search**, López-Cruz 2019 (2 statistics, both level-free, since López-Cruz's detection
    thresholds are not published): the decay γ and the early/late ratio.
  - **Mutants**, each against its same-date wild type:
    - tph-1 against N2; pdfr-1 and tax-4 against PD1074;
    - Δ roaming fraction and Δ exits per on-lawn hour on Scheer;
    - pdfr-1's Δγ and Δ early/late on López-Cruz;
    - a direction test wherever the reference's band excludes zero.
- **Bands.**
  - **Between-date, from a cluster bootstrap over band dates,** at the graded population's size. Each
    band holds the date-to-date spread and the sampling noise of n animals.
  - **López-Cruz** has bins, not tracks. Its band is the spread over its band batches: the distinct
    same-date wild-type controls behind its mutants, split a third held out like Scheer's.
  - **Thin arms** (fewer than five band dates) are inflated to the OD 2 wild type's spread, and to √2
    of it for a mutant's difference.
- **The pass rule.**
  - Per statistic, |z| ≤ z\*, with Bonferroni over the family: z\* = 2.84 for the 11 wild-type
    statistics, and per mutant over its own statistics.
  - A worm passes the wild type when every wild-type statistic passes. A mutant passes when each of
    its statistics passes, and every resolved direction matches.
- **The z distance.**
  - D is the weighted RMS of the capped z's: each assay weighted equally, split equally among its
    statistics.
  - D_fit leaves out local search, which D37 holds out of C.2 and C.3.
  - The rule for W's minimum is registered here and read in B.4. It is the distance between two halves
    of the held-out wild-type dates.
- **The rule's own check, on the band cohorts only.** Bands built on half the band dates grade the other
  half. The joint pass rate over 200 half-splits must be at least 0.90, a gate registered before the
  bands are read.
- **A preflight,** as description before B.4 registers. ConstantHazards and PirouetteHazards are graded
  on pilot seeds 9101–9164. A negative that passes here is brought to the maintainer before B.4.
- **A new assay protocol, `uniform_lawn_scheer`:** Scheer's uniform lawn, a 10 mm well spread with
  bacteria, 60 min with the last 40 analysed. It is the only uniform-lawn reference with tracks and
  dates. Flavell 2013's bout timescales (dwelling τ ≈ 482 s; roaming 75 and 520 s) are reported beside
  it from `uniform_lawn_flavell`, not graded.
- **Reported, not graded:**
  - roaming against density (OD 1, OD 2, OD 4);
  - the authors' labels' readings beside ours;
  - the quintile ratio's reference (2.26–2.58).
- **The novelty check owed at B's registration:** NeuroSparkSNT (Research Square 2026), PGOB (Zenodo
  2026), and Morrison & Young 2025, recorded in the logbook.

## Capabilities

### New Capabilities

- `benchmark-grading`: the classifier from the band cohorts, the window rule, the statistics, the
  bands, the pass rule, the z distance and the rule's own check.

### Modified Capabilities

- `assay-protocols`: the generated configs gain `uniform_lawn_scheer`.

## Impact

- **Code**: a new `quantumnematode/benchmark/grading.py`, which holds the statistics, bands, pass rule
  and distance for real and simulated windows.
- **Scripts**:
  - `scripts/analysis/benchmark_classifier.py` (the recalibration and its gate);
  - `scripts/analysis/benchmark_bands.py` (statistics per date, bands, the rule's check);
  - `scripts/analysis/benchmark_grade.py` (grades a batched or captured run);
  - `generate_assay_configs.py` gains the new protocol.
- **Data**: `data/benchmark/classifier.json`, `bands.json`, and `lopezcruz2019_split.json`, with
  `PROVENANCE.md`.
- **Default behaviour**: none. No simulation code changes.
- **Held-out data**: never read. B.4 reads them.
- **Cost**: about 5–7 engineering days. The bands take minutes on the store. The preflight runs two
  controllers × three assays × 64 animals in the batched evaluator, minutes per assay.
- **Seeds**: preflight 9101–9164; smokes 5991–5999. 5101–5356 stay reserved for B.4.
- **Depends on**: `amend-food-dependent-kinematics`, merged before the preflight runs, because it
  changes what the simulated worm does on a lawn.
