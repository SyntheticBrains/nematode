## Context

- **Benchmark v2** ([Logbook 115](../../../docs/experiments/logbooks/115-benchmark-v2.md); `bands_v2.json`, SHA-256 `21f591b8…`): 19 statistics in five equal families at z\* 3.0078, with simulated tracks measured on Scheer's and Chen's rigs. Local search grades γ and the linear early/late ratio, on López-Cruz's 17 band batches (factor at m = 11). Chemotaxis grades drift, the log quintile ratio and the weathervane slope on Chen's 6 standard recordings, with leave-one-out at an allowance of 1.

- **The gap** ([Logbook 119](../../../docs/experiments/logbooks/119-c1b-fit.md), addendum):

  - C.1b's answers reverse 42–56% of the time off food, against 17.3% real;
  - they reorient 3–20× too often in local search with no decay, and pass it;
  - from midbody positions, continuous backing looks like forward travel.

- **The planning brief** (2026-11-01, description): on the same real off-food tracks, our detector counts 2–3× López-Cruz's reorientations, so a level against López-Cruz's counts needs a cross-lab factor. An early-window level misses restart 2. On a log scale, early/late fails restart 2 and passes every held-out batch.

- **The probe** (`build/benchmark-v3/probe.json`, description, animals at 99,940,000): one population per answer for C.1b's 8 restarts and the κ probe. Bar ≈ 3.0.

  | Statistic | Held-out real | Reversal-dominated answers caught | Others caught |
  |---|---|---|---|
  | log early/late | z −0.81 to +1.14 (10 López-Cruz batches) | 2, 4, κ (not 6) | 7 |
  | Chen onset rate | z −0.47 to +0.78 (Chen leave-one-out) | 2, κ (not 4, 6) | 0, 1, 3, 5, 7 |
  | Chen direction flips at 0.03 mm | z −0.38 to +0.88 | 2, 6, κ (not 4) | 1, 5, (7 at +3.0) |
  | reversal share (labels) | Roberts 17.3% off food | 2, 4, 6, κ (42–55%) | 0, 1, 3, 5, 7 low (1–4%) |

## Goals / Non-Goals

**Goals:**

- a benchmark that fails reversal-dominated locomotion from positions alone, while held-out real worms pass;
- every new statistic read the same way on real and simulated tracks, through v2's measurement models;
- v1 and v2 frozen and byte-identical.

**Non-Goals:**

- **grading the share of time reversing.** It is reported, and escalated to grading only if a fit is caught by it but passes the three;
- a level against López-Cruz's counts (the cross-lab factor);
- changing the lawn families, kinematics or the measurement models;
- refitting C.1, which is its own change on v3.

## Decisions

1. **The log early/late ratio.**

   - Local search's ratio is graded as log(early/late), over the same bins (early 8–14 min, late 34–44 min after removal).
   - The band: the 17 López-Cruz band batches' mean of log(early/late), and their SD × v3's prediction factor for one new batch at m = 21.
   - Local search's presence test (ratio above 1 at its lower bound) is unchanged.
   - γ is unchanged, on v2's band.

2. **The Chen onset rate:** 60 × the mean of the per-second onset flags the chemotaxis samples already carry (`fragment_samples`). It is pooled over a population's kept seconds, as the other chemotaxis statistics are, by the same off-food detector.

   - **Band:** the 6 standard recordings' mean, with their SD × v3's factor.
   - **Linear z,** since the recordings' spread on a log scale would pass ×/÷12.

3. **The direction-flip fraction.**

   - Over each fragment's consecutive 1-s displacements: the share of pairs, both at least **0.03 mm**, whose directions differ by more than 135°.
   - The floor is v2's own, the on-lawn reversal rate's, and is above the Chen rig's jitter (0.0087 mm per axis).
   - It is pooled over a population's fragments, read on Chen's real fragments and on simulated tracks after the Chen rig model. The band is built the same way as the onset rate's.

4. **The bar:** 21 graded statistics, z\* = 3.0381.

   - The families stay equal, split equally within each, so chemotaxis's five statistics weigh 1/25 each.
   - v3's prediction factors use m = 21 throughout, the local-search bands' included.
   - v2's module tuples are untouched; v3 has its own (`V3_STATISTICS`, `CHEMOTAXIS_STATISTICS_V3`, `LOCAL_SEARCH_STATISTICS_V3`).

5. **The reported reversal share:**

   - where a population's steps carry primitive labels, the share of seconds in a reversal bout, over the local-search and chemotaxis windows;
   - reported beside the grade with Roberts' tracker reference (17.3% of tracked frames in reverse, off food), never graded and never in D;
   - imported tracks report none.

6. **Grading:**

   - `grade_populations_v3` mirrors v2's, with the new statistics, presence unchanged, and the reported share;
   - `benchmark_grade.py --bands v1|v2|v3`;
   - the default is v3 once `controls_v3.json` reads VALID with `bands_v3.json`'s SHA-256, otherwise v2's rule;
   - D_fit is unchanged (the small lawn, the uniform well and kinematics). Whether C.2's fit reads the new statistics is C.2's decision.

7. **The validity map** (registered in Part A; every item must pass for VALID):

   1. **The round trip:** v2's chemotaxis round trip (Chen's fragments, smoothed and passed through the jitter model, `default_rng([475_000_000 + fragment, 2037])`) for the two new statistics, within |z| ≤ 1 of the raw fragments' value.
   2. **The reachability oracle.**
      - The fit: `BatchedMechanistic` fitted on discarded seeds to the three new statistics alone (the RMS of their z, capped at 10), by sep-CMA from the box centre at σ 0.2, CMA seeds 9421 and 9422, cap 100, animals at 470,000,000. The better restart is taken.
      - The check: on 40 populations of 200 (animals at 471,000,000), each new statistic's mean z must lie in [−1, 1], and |z| ≤ z\* must hold in at least 90% of populations.
      - Part B records the fitted answer before the 40 populations run.
   3. **Held-out López-Cruz:** every one of the 10 held-out batches passes log early/late at |z| ≤ z\*.
   4. **Chen leave-one-out:** each standard recording against bands from the other five, on all five chemotaxis statistics and presence, within the allowance. The allowance is recomputed for five statistics at m = 21 by v2's binomial rule.
   5. **The adversaries:** C.1b's restarts 2, 4 and 6 (`119-c1b-fit/fit.json`) and the κ probe (`probe_kappa.json`), each on 5 fresh populations (animals at 472,000,000). Each must fail **at least one of the three new statistics** in at least 3 of 5 populations, whatever presence reads.
   6. **The negatives** fail the wild type at 3 of 5 (animals at 473,000,000):
      - B.4's six;
      - the four open-loop ones, which must also fail chemotaxis;
      - v2's blind-worm answer;
      - C.1's v1 answer.
   7. **The lawn N2 control is carried from v2,** since the lawn families and kinematics are unchanged and v3's bar is more lenient. It is recorded as carried, not re-read.
   8. **Recomputed and reported:** W's minimum on both family sets, and the 3-of-5 probability for a faithful controller.

   **The verdict map:**

   - **VALID:** v3 becomes the default;
   - **VOID:** v2 stays the default, and the failing item and its diagnosis go to the maintainer. If the oracle fails, the run stops before Part B.

8. **The registration** (launch record 120):

   - **Part A** fixes every rule above, and is committed before the bands are built or any control is read. `benchmark_bands.py --version v3` and `benchmark_v3_controls.py` refuse until then.
   - **Part B,** after the oracle's fit, records its answer before the oracle's populations and the reading run.
   - **Disclosure:** the planning brief's and the probe's readings of held-out data (the 10 López-Cruz batches, the Roberts cohorts, Chen's 8 recordings) and of C.1b's answers are seen.

9. **Seeds:**

   | Base | Use |
   |---|---|
   | 470,000,000 | the oracle's fit, k = 10,000·r + 16·(n − 1) + p, CMA 9421 + r |
   | 471,000,000 | the oracle's 40 check populations |
   | 472,000,000 | the adversaries, 5 populations each |
   | 473,000,000 | the negatives and the two answers, 5 populations each |
   | 475,000,000 | the round trip's jitter |
   | 99,940,000 | the probe, discarded |

## Risks / Trade-offs

- **[A continuous reverser that also decays]** → the three statistics could miss it, as the onset rate and flips missed restart 4. The reported share watches for it, and the escalation path is registered.
- **[The Chen onset rate's band is wide (CV 0.42)]** → it catches gross excess only, which is what the probe shows (z +3.7 to +6.1).
- **[The bar loosens slightly (3.0078 → 3.0381)]** → every v2 statistic passes slightly more easily. The negatives are re-graded at the new bar.
- **[The probe saw the held-out data]** → disclosed in Part A. The registered reading uses fresh simulated seeds, and the real held-out values were seen only as description.
- **[Cost]** → the oracle's fit about 1–2 h; everything else under an hour.
