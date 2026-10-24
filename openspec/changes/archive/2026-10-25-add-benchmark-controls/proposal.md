## Why

D34: the benchmark is **valid only if** its positive controls pass and every negative fails, and **void** otherwise. B.3 built the rule from band cohorts alone, and its hardening fixed what B.4 would have tripped on (Logbook 109). B.4 reads the held-out cohorts for the first time, so every rule is fixed in a launch record committed before any held-out value is read.

## What Changes

The maintainer's decisions of 2026-10-25 are marked (decided).

- **Positive controls**, held-out real data, each graded at its own size on bands built on demand:

  - **the wild type,** as three kinds of unit:
    - Scheer's small lawn: 24 held-out dates, 573 animals, six statistics;
    - the uniform well: one held-out date, 31 animals, three statistics;
    - López-Cruz's 10 held-out batches, each on γ and early/late, and each required to be present.
  - **the mutants,** each held-out mutant against its held-out same-date wild type:
    - tph-1 against N2 (2 paired dates);
    - pdfr-1 and tax-4 against PD1074 (3 and 6 paired dates);
    - each on Δ roaming and Δ exits, with resolved directions.

- **The allowance across controls** (decided). Fifteen control units are graded, each at its normal bar. Their failure probabilities under real data are taken from the rule's own checks, or from the nominal 5% where no check exists:

  - the small lawn, 0.075;
  - a López-Cruz batch, 0.038;
  - the uniform well and each mutant, 0.05.

  The largest number of failures real data exceed with at most 5% chance is **2**: P(more than 2) = 0.025. **The positive controls pass when at most 2 units fail.** Requiring every unit to pass would void the benchmark by chance 49% of the time.

- **Matched negatives**, matched to band data (decided):

  - **the open-loop HMM worm:** roaming and dwelling switch by the authors' HMM transitions every 10 s, ignoring food. In each state the speed, curvature noise and reversal and omega hazards are the band wild type's in that state, from Scheer's band cohorts.
  - **the Poisson worm:** constant hazards, speed and curvature noise at the band wild type's time-averaged on-lawn values.

- **Trivial negatives**, matched to wild-type speed (decided):

  - **a random walk:** runs at the band wild type's mean on-lawn speed, with each second's curvature drawn uniformly over the full range;
  - **a correlated random walk:** the same speed, with curvature noise set to the wild type's heading persistence;
  - **the untrained policy:** the MLP and connectome arms, each 8 independently initialised networks × 25 animals (decided).

- **Every negative must fail the wild type**, at n = 200 per assay, on animal seeds 5101–5300.

- **W's minimum is read**, by B.3's registered rule, on the held-out wild-type dates.

- **The verdict:** **VALID** when the positive controls pass by the allowance and every negative fails. **VOID** otherwise. On VOID, the statistics and bands are revised in a new change before anything is graded (the roadmap's pivot).

## Capabilities

### New Capabilities

- `benchmark-controls`: the positive controls, the allowance, the negatives, W's minimum and the verdict.

## Impact

- **Code**:
  - `quantumnematode/evaluation/controllers.py` gains the negatives' batched controllers: state-switching hazards, and curvature-noise runs;
  - `scripts/analysis/benchmark_controls.py` reads the held-out data and grades everything;
  - `scripts/analysis/benchmark_negatives_params.py` derives the matched parameters from band cohorts.
- **Data and records:**
  - `docs/experiments/logbooks/supporting/110-benchmark-controls/`: `launch.md`, committed before any held-out read; the matched parameters; the controls' and negatives' gradings; W's minimum;
  - Logbook 110.
- **Held-out data:** read once, after `launch.md` is committed.
- **Cost:** minutes. Each negative takes a few seconds per assay at 200 animals (Logbook 104), plus classification and bands on demand. No long campaign.
- **Seeds:**
  - animals 5101–5300 for every negative;
  - untrained networks 5301–5316 (8 per arm);
  - a pilot on 9201–9264, disjoint;
  - smokes on 5991–5999.
