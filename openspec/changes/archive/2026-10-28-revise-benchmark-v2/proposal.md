## Why

C.1 passed the wild type ([Logbook 113](../../../docs/experiments/logbooks/113-c1-fit.md)) partly through a route no real worm takes. The benchmark grades how much a worm roams and how long its bouts last, but never how it moves within a state or how often it reverses. An investigation of the whole benchmark followed (the maintainer's request of 2026-10-27: reopen B once, covering every angle). It found:

- **Loopholes:**

  - roaming speed is ungraded. C.1 roams at about 2.6× the real speed at the classifier's own 10-s scale;
  - the on-food reversal rate is ungraded. C.1's detected on-lawn reversals are about 4% of the real rate;
  - the omega rate is ungraded. C.1's dwelling omega hazard is thousands of times Scheer's;
  - only the mean of the roaming-bout distribution is graded, though real roaming bouts are double-exponential;

- **Measurement asymmetries:**

  - real 1-s tracks carry tracking jitter, and a point worm has none. Real dwelling's 5-s angular speed is the value a randomised heading gives;
  - real tracks lose 42% of their samples in omega seconds, so those windows go unclassified. Simulated tracks lose none.

  So a point worm at the real reversal rate reads at z −2.9 on detected reversals.

- **What holds:** an open-loop worm fitted directly to D fails every population, on lawn retention and on local search.

The planning probes behind these numbers are uncommitted, and this change's probe stage re-reads them into a committed record before any registration cites them.

The maintainer's decisions of 2026-10-27:

- **statistics:** per-state speed, on-lawn reversal rate and roaming-bout shape. Local search's rise was considered and dropped, as handling rather than search;
- **a measurement model:** jitter and missingness, measured from Scheer's band cohorts and applied to every simulated track;
- **validation on fresh N2 data:** 26 held-out N2 dates no reading has touched, so PD1074's held-out cohorts are not re-read;
- **weighting:** a separate kinematics family, fitted for every controller;
- **chemotaxis** (decided after the first draft): a fifth family, from Chen et al. 2025's naive N2 butanone tracks (*PLoS Biology*; CC BY 4.0, figshare 10.6084/m9.figshare.24764403). It is held out as a prediction for C.2 and C.3, and fitted for C.1, as local search is under D37. Families are now weighted a fifth each;
- **adversaries:** the blind worm and C.1's unconstrained answer must both fail, as conditions of validity;
- **three changes in sequence:** this one (B v2), then F.0b-i re-run in full on v2, then C.1b on v2.

## What Changes

- **A measurement model** (`quantumnematode.benchmark.measurement`). Before anything is windowed, every simulated track gets:

  - position noise at the tracker's measured jitter;
  - missing seconds at the tracker's measured rates, conditioned on the simulated primitive (omega seconds against the rest).

  Real and imported tracks are never touched.

- **B v2's statistics** (`data/benchmark/bands_v2.json`, built from band cohorts only):

  - **the kinematics family** (small lawn, a quarter of D, in D_fit):

    - median and 90th-percentile 10-s window speed in dwelling, and in roaming;
    - detected reversal onsets per on-lawn minute;
    - Kaplan–Meier roaming-bout survival at 5 min;

  - **the chemotaxis family:**

    - a linear-gradient assay on the 1-s primitives worm, matched to Chen's plates;
    - statistics derivable from positions and the known gradient: up-gradient drift or the chemotaxis index, the unit-free quintile ratio of reorientation rate against dC/dt, and the weathervane regression;
    - bands from the 6 standard-protocol recordings, validated leave-one-recording-out, with the allowance registered first;
    - its own tracker parameters in the measurement model.

    The details that depend on the data are fixed after extraction and before Part A;

  - **the other three families are unchanged.** D now weights five families equally. D_fit, for C.2 and C.3, is small lawn, uniform well and kinematics, while C.1 fits all five;

  - **z\*** is Bonferroni over 20 wild-type statistics, Φ⁻¹(1 − 0.05/40) ≈ 3.02. The κ calibration and the rule checks are extended.

- **Validity** (`scripts/analysis/benchmark_v2_controls.py`), registered before any held-out read or adversary fit:

  - **a reachability oracle:** B.4's matched open-loop worm, at the real rates, through the measurement model, must pass the kinematic statistics it matches;
  - **a fresh positive control:** N2 bands built from N2's band dates, graded on its 26 unread held-out dates;
  - **negatives:** B.4's six, re-graded on v2, must fail;
  - **adversaries:** the blind worm, fitted against v2's D, and C.1's restart-3 answer must both fail the wild type at 3 of 5.

- **W's minimum v2:** the 95th percentile of D, over the small-lawn and kinematics families, between halves of N2's held-out dates. W reads on the same families.

- **Versioning:** v1 (`bands.json`) is kept, and every past verdict stays on v1. `benchmark_grade.py --bands v1|v2` defaults to v2 after registration. The track-import round trip and `IMPORT.md` are updated.

- **The tracker:** declared non-goals with reasons; the digest's items (W's reconstruction and weight-noise checks; H.1 closed).

## Capabilities

### Modified Capabilities

- `benchmark-grading`: the measurement model, the kinematics and chemotaxis families, and v2's bands and weights.
- `benchmark-controls`: B v2's validity, with the oracle, the N2 positive control, the negatives, the adversaries and W's minimum v2.

## Impact

- **Code:** `benchmark/measurement.py` (new); `benchmark/grading.py`, extended for histograms, bout lists, the reversal rate, the chemotaxis statistics and five families; `benchmark_bands.py`; `benchmark_grade.py`; `scripts/analysis/benchmark_v2_controls.py` (new); `track_import_check.py`. With tests.
- **Records:** `docs/experiments/logbooks/supporting/115-benchmark-v2/` and Logbook 115.
- **Compute:** about 1–2 h to build the bands and run the checks, 1–2 h for the controls, and about 3–4 h for the blind adversary (2 restarts, with the chemotaxis assay in each evaluation). Every later fit evaluation is about 25–35% slower.
- **Data:** Chen 2025's deposit (3.5 GB, downloaded by hand into the off-repo archive). Only derived naive-N2 tracks are vendored, with a `PROVENANCE.md` entry.
