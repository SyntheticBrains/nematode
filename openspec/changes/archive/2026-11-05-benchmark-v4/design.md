## Context

- **Benchmark v3** ([Logbook 120](../../../docs/experiments/logbooks/120-benchmark-v3.md); `bands_v3.json`): 21 statistics in five equal families at z\* 3.0381. On the lawn it grades the small lawn's states, bouts and exits, the uniform well, and kinematics (10-s window speeds and roaming-bout survival). The on-lawn reversal rate is reported, not graded.

- **The gap:**

  - C.2's stage-2 answer moves only by reversals and delta turns, with run speed pinned at zero (Logbook 123);
  - C.1's v3 answer reverses 3% of the time (Logbook 121).

  v3 lets both through.

- **The real reference:** Scheer's per-frame flags on the lawn, PD1074 at OD 2: reversal 14.3%, pause 49.8%.

- **The instrument:** the real tracker's 1-s displacements of a still worm (`scheer_tracker_noise.npz`, 87,035 seconds in 6,213 stretches) have median 0.0054 mm and 95th percentile 0.0193 mm.

- **The planning probe** (`supporting/124-benchmark-v4/probe/`, seeds 600,000,000 + 200·c + i, description): the table in the proposal.

## Goals / Non-Goals

**Goals:**

- a benchmark that fails a worm that does not run forward, and one that reverses too rarely, from positions alone, while real worms pass;
- the new statistics read the same way on real tracks and on simulated tracks after Scheer's rig model, and on imported tracks;
- v1–v3 frozen and byte-identical.

**Non-Goals:**

- **grading the reversal share from primitive labels.** It stays reported: grading it would need a head–tail measurement model;
- **the activity family on the uniform well** (one held-out date), on the mutants, or off food;
- **C.2's encoder changes** (readout box, saturation report): C.2's next change;
- **refitting C.1:** its own change if v4 fails its v3 answer.

## Decisions

1. **The activity tally.**

   - Over each animal's track, an on-lawn second is one whose two samples are present and on the lawn: `grading.real_on_lawn` for real tracks, and the steps' `on_lawn` for simulated ones.
   - The tally counts:
     - on-lawn seconds;
     - pauses: an on-lawn second whose displacement is shorter than **0.02 mm**;
     - flips and pairs: `chemotaxis.direction_flips` on the track with every off-lawn sample blanked;
     - onsets and their seconds: `grading.on_lawn_reversals`.
   - Real tracks are tallied over the whole recording. Simulated tracks are tallied over the small lawn's analysis window after `measure_window`, as kinematics are.

2. **The statistics,** pooled over a population's tallies:

   - pause share = pauses / on-lawn seconds;
   - flip fraction = flips / pairs;
   - onset rate = 60 × onsets / onset seconds.

   The pause floor is the instrument's: the smallest 0.0025-mm step at which at least 95% of the still-worm library's 1-s displacements fall below it (0.02 mm; 95.4%). It is derived in code from the library by this rule, and a test asserts it. The flip and onset floors are v3's and v2's (0.03 mm).

3. **Bands:** `lawn_bands` on the activity tallies of PD1074's OD 2 band dates, at n = 200 and 64.

   - κ is measured per statistic by `measure_kappa` on its own generator, `default_rng(2045)`.
   - The rule check uses 200 half-splits with κ fixed, on `default_rng(2046)`, at v4's bar (`check_small_lawn` with the activity statistic). It reads the family's pass rate, the share of splits in which every graded activity statistic passes, with each statistic's failures beside it.
   - Every v3 band is byte-identical.

4. **The family:** activity is a sixth family in D and in D_fit. Families weigh equally, split equally within each.

5. **The onset rate is conditional.** It is graded if, and only if, the oracle (decision 7.3) brings it within the bar. Otherwise it is reported beside the grade and v4 grades 23 statistics. The bar is z\* = 3.0781 for 24 statistics and 3.0654 for 23.

   - `bands_v4.json` holds all three activity bands with `activity_graded: null` when it is built.
   - After the oracle's check (decision 7.3), `benchmark_bands.py --version v4 --onset graded|reported` sets the field and nothing else. It refuses a choice the check's record does not support.
   - Every v4 grade refuses while the field is null, so no reading uses a provisional bar.
   - The bands' SHA-256 is final after that step, and `controls_v4.json` pins it.

6. **Grading:**

   - `grade_populations_v4` is v3's grade with the activity family added (`ACTIVITY_STATISTICS_V4`);
   - `benchmark_grade.py --bands v1|v2|v3|v4`;
   - the default is v4 once `controls_v4.json` reads VALID with `bands_v4.json`'s SHA-256, otherwise v3's rule;
   - imported tracks carry the activity family, from positions.

7. **The validity map** (registered in Part A; every item must pass for VALID):

   1. **The rule check:** the activity family's half-split pass rate is at least 0.90 at v4's bar, as v2's families were gated.
   2. **The round trip:** Scheer band animals exported to CSV and WCON and imported again give activity tallies equal to the native path's.
   3. **The reachability oracle.**
      - The fit: `BatchedMechanistic` with C.1's 25 parameters, fitted on discarded seeds to the three activity statistics alone (the RMS of their z, capped at 10), on the small lawn. Sep-CMA from the box centre at σ 0.2, CMA seeds 9431 and 9432, cap 100. The better restart is taken.
      - The check: on 40 populations of 200, the pause share's and the flip fraction's mean z must each lie in [−1, 1], with |z| ≤ z\* in at least 90% of populations.
      - The onset rate is read by the same rule at the bar it would be graded at (3.0781), and decides decision 5 alone. The pause share and the flip fraction are then read at the bar that decision gives.
      - Part B records the fitted answer before the 40 populations run. The check (`oracle-check`) writes `oracle_check.json` once.
   4. **Fresh N2:**
      - The units: v2's 26 N2 held-out dates (the same halves, `default_rng(2035)`), each half graded on the activity family against N2's own activity bands from its 42 band dates, on `default_rng([2047, n])`. That gives 2 units.
      - The gate: at most 1 of the 2 units fails, as v2's N2 control allowed 1 of 4. A systematic fault in the bands or the measurement fails both halves. Part C states the chance that both fail by chance, from the rule check's family pass rate at the decided bar.
   5. **The adversary:** C.2's stage-2 answer on the wiring it was fitted on, on 5 fresh populations, must fail at least one graded activity statistic in at least 3 of 5, v3's rule.
   6. **The negatives:** v3's (B.4's six, v2's blind worm, C.1's v1 answer) each fail the wild type at 3 of 5 at v4's bar.
   7. **Reported:**
      - C.1's v3 answer (Logbook 121, restart 6) on its own registered populations (525,000,000 + 200·g + i, g < 5), so its v4 and v3 grades compare like for like;
      - PD1074's held-out cohorts on the activity family, as description: the planning probe read them;
      - W's minimum on both family sets, the fitted set now with activity;
      - the 3-of-5 probability for a faithful controller, by v3's method.

   **The verdict map:**

   - **VALID:** v4 becomes the default.
   - **VOID:** v3 stays the default, and the failing item goes to the maintainer.
   - If the oracle fails the pause share or the flip fraction, the read stops there.

8. **The registration** (launch record 124):

   - **Part A** fixes every rule above and is committed before the bands are built or any control is read. `benchmark_bands.py --version v4` and `benchmark_v4_controls.py` refuse until then.
   - **Part B,** after the oracle's fit, records its answer before its 40 populations run.
   - **Part C,** after the oracle's check, records the onset decision the check gives, committed with `oracle_check.json` and the decided bands. The read and the verdict refuse until then. The bands' SHA-256 changes once, at this step: Part B binds the undecided bands, Part C the decided ones.
   - **Preflights before Part A** (discarded seeds, disclosed):
     - the adversary gate: C.2's stage-2 answer on 5 populations against provisional bands, built in process through the library functions by Part A's rules, since `benchmark_bands.py --version v4` refuses before Part A;
     - one 60-generation oracle restart, which must bring the pause share and the flip fraction within the bar.
   - **Disclosure:** the planning probe and the instrument floor's derivation read PD1074's band and held-out cohorts on the three statistics, and the four controllers.

9. **Seeds:**

   | Base | Use |
   |---|---|
   | 600,000,000 | the planning probe, used (200·c + i) |
   | 610,000,000 | the oracle's fit, k = 10,000·r + 16·(n − 1) + p, CMA 9431 + r: restart 0 spans 610.0–610.32 M and restart 1 612.0–612.32 M. *(Moved from 601 M before Part A: there restart 1 would have spanned 603.0–603.32 M, the adversary's block, the overlap v3's erratum records.)* |
   | 602,000,000 | the oracle's 40 check populations |
   | 603,000,000 | the adversary, 5 populations |
   | 604,000,000 | the negatives, k = 5·i + g for entry i in the registered order |
   | 525,000,000 | C.1's v3 answer, re-read on its own registered populations |
   | 606,000,000 | the adversary preflight, discarded |
   | 607,000,000 | the oracle preflight, discarded (CMA 9433): its restart at 607.0–607.19 M, its answer's 5 populations at k = 9,000 + p (608.8 M) |
   | generators 2045–2047 | κ, the rule check, N2's activity bands |

## Risks / Trade-offs

- **The tracker's noise while a worm moves is not modelled** (v2's non-goal). A slowly moving simulated worm's pause share could therefore be biased against a real one's. The oracle tests whether a fitted mechanistic worm reaches the real share through the model, and the N2 control tests the bands.
- **The onset rate may stay unreachable,** as v2 found for open-loop worms. The fallback is registered. If it applies, the onset rate catches C.1's rare reversals only as reported description, and the flip fraction (C.1 at −1.4 date-SDs) carries less of that error.
- **The bar loosens slightly** (3.0381 → 3.0781). v3's statistics pass slightly more easily; the negatives are re-graded at the new bar.
- **The probe saw PD1074's held-out activity values,** so they are reported, not registered. The registered positive control is N2, whose activity statistics no reading has computed.
- **v4 may fail C.1's v3 answer.** C.1 would then be refitted on v4 before the fidelity table, a cost of about 12–18 h.
