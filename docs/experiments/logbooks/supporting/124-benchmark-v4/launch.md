# Launch record: B v4, on-lawn activity graded from positions

**Part A registered 2026-11-05**, change `benchmark-v4`. It is committed before the v4 bands are built by the guarded builder and before any registered control is read. It fixes decisions 1–9 of the change's design with every threshold and seed. Two later parts change no rule:

- **Part B,** added after the oracle's fit, records the bands, the rule checks, the round trip and the oracle's fitted answer, before the oracle's 40 populations run.
- **Part C,** added after the oracle's check, records the onset decision that check gives, with the decided bands, before the read runs.

A verdict registered here is never re-read under a different rule. Anything else is reported beside it as description.

v1 (`bands.json`), v2 (`bands_v2.json`) and v3 (`bands_v3.json`, SHA-256 `8af38429…`) stay frozen for every past verdict. With v4's code in place, all 59 jobs of the identity check reproduce every v1–v3 statistic, D and presence reading byte for byte, among them v3's grade with and without presence on C.1's v3 answer and on the box centre (`identity.json`).

## Part A: the rules

### The activity tally (decision 1)

Over each animal's track, an **on-lawn second** is one whose two samples are both present and on the lawn: by `grading.real_on_lawn` on a real track, by the steps' `on_lawn` on a simulated or imported one. The tally (`grading.activity_tally`) counts:

- the on-lawn seconds;
- **pauses:** on-lawn seconds whose displacement is shorter than **0.02 mm**;
- **flips and pairs:** `chemotaxis.direction_flips` on the track with every sample not on the lawn blanked. A pair is two consecutive 1-s displacements, both at least **0.03 mm**; it flips when their directions differ by more than **135°**;
- **onsets and their seconds:** `grading.on_lawn_reversals` at its 0.03 mm floor.

Real tracks are tallied over the whole recording. Simulated tracks are tallied over the small lawn's analysis window after Scheer's rig model (`measure_window`), as kinematics are. Imported small-lawn tracks are tallied as real ones are, never measured.

### The statistics (decision 2)

Pooled over a population's tallies:

- **the pause share** = pauses / on-lawn seconds;
- **the flip fraction** = flips / pairs;
- **the onset rate** = 60 × onsets / onset seconds, per on-lawn minute.

**The pause floor is the instrument's.** It is the smallest multiple of 0.0025 mm below which at least 95% of the still-worm library's 1-s displacements fall (`scheer_tracker_noise.npz`, within each noise stretch): **0.02 mm**, with 95.4% below it. `grading.pause_floor` derives it from the library, the builder refuses to build if it is not 0.02 mm, and a test asserts it.

### Bands v4 (decision 3)

`benchmark_bands.py --version v4` writes `data/benchmark/bands_v4.json` and `bands_v4_record.json` here:

- **Copied from v3, byte-identical:** every v3 band, the measurement models, κ and the classifier.
- **The activity bands** at n = 200 and 64: `lawn_bands` on the per-date activity tallies of PD1074's OD 2 band dates, on `default_rng([2030, 6, n])`. The scale is κ × √(sd_n² + sd_centre²), from cluster bootstraps over the dates (2,000 resamples).
- **κ** per statistic by `measure_kappa`, on `default_rng(2045)`: 200 half-splits, each statistic's z SD, floored at 1.
- **The rule checks,** each 200 half-splits at size 200 with κ fixed, read at both bars v4 can have:
  - **the activity family** on `default_rng(2046)`: its pass rate is the share of splits in which every graded activity statistic passes, with each statistic's failures beside it;
  - the small lawn (v1's κ) and kinematics (v2's κ) on seed 2040, and local search on 2034, as v3's, re-read at v4's bar.
- **No bands are written if the activity check fails at both bars.**
- The file is written with `activity_graded: null`: no bar, families or weights. **Every v4 grade refuses while it is null.**

### D v4 and the bar (decisions 4–5)

- **Six families,** equal weight, each split equally among its statistics: v3's five, and **activity** on Scheer's small lawn.
- **The onset rate is graded only if the oracle reaches it** (below). Otherwise it is reported beside the grade.
  - Graded: 24 statistics, z\* = Φ⁻¹(1 − 0.05/48) = **3.0781**.
  - Reported: 23 statistics, z\* = Φ⁻¹(1 − 0.05/46) = **3.0654**.
- The z cap stays at 10, and an undefined statistic enters at the cap and fails.
- **D_fit** is the small lawn, the uniform well, kinematics and activity.
- **A population passes** when every family passes, presence as on v3.

### Grading (decision 6)

- `grade_populations_v4` is v3's grading with the activity family, read on the small lawn's animals. The activity statistics the bands do not grade are reported as `activity_reported`.
- `benchmark_grade.py --bands v1|v2|v3|v4`.
- **The default:** v4 once `controls_v4.json` reads VALID with `bands_v4.json`'s SHA-256; otherwise v3's rule.
- Imported small-lawn tracks carry the activity family.

### The validity map (decision 7)

`benchmark_v4_controls.py` reads every item. **VALID** only when all of these hold:

1. **The rule check:** the activity family's pass rate is at least **0.90** at v4's decided bar. The other families' checks at that bar are reported.

2. **The round trip:** every PD1074 OD 2 band animal, written as a CSV with its sidecar and as WCON on the small lawn's metadata and imported again, gives an activity tally **equal** to the native path's, animal by animal. If it fails, the run stops before the oracle's fit.

3. **The reachability oracle.**

   - **The fit,** only once the bands are built and the round trip has passed:

     - C.1's 25-parameter model (`BatchedMechanistic`, `c1_v3_fit.parameters`) in its unit box;
     - fitted to the three activity statistics alone: the objective is the RMS of their capped z against `bands_v4.json`'s activity bands at n = 200, on the small lawn alone;
     - sep-CMA from the box centre, σ₀ 0.2, population 16, CMA seeds **9431 and 9432**;
     - fit animals at `610_000_000 + 200·k + i`, with k = 10,000·r + 16·(n − 1) + m for restart r, generation n and member m;
     - v2's σ-guarded plateau stops a restart (from generation 40, the mean of the last 20 generations' median objectives no lower than the 20 before by more than 0.05, and σ ≤ σ₀); otherwise it stops at the **cap of 100** generations.

   - **The answer:** the restart with the lower final objective (the mean of its last 20 medians) is taken, and its answer is its mean at the stop, clipped to the box. Part B records it before its populations run.

   - **The check** (`oracle-check`, after Part B): 40 populations of 200 at `602_000_000 + 200·p + i`, p < 40. A statistic passes the rule when its mean z lies in **[−1, 1]** and |z| ≤ the bar in **at least 90%** of populations; an undefined value enters at the cap and fails.

     1. **The onset rate** is read at the bar it would be graded at, 3.0781. If it passes, it is graded (24 statistics); otherwise it is reported (23).
     2. **The pause share and the flip fraction** are read at the bar that decision gives.

   - **If the pause share or the flip fraction fails,** activity is unreachable by C.1's model: nothing else is read, and the change returns to the maintainer.

   - **The decision is mechanical.** `benchmark_bands.py --version v4 --onset graded|reported` sets `activity_graded` and the bar, weights and families that follow from it, and nothing else. It refuses before Part B, without `oracle_check.json`, or against its reading. Part C records it.

4. **Fresh N2,** the registered positive control. No reading has computed N2's activity statistics.

   - **The units:** v2's 26 unread N2 held-out dates (`N2_od2_Data`; all but tph-1's paired dates), split into the same two halves on `default_rng(2035)`. That gives 2 units.
   - **Each unit** is graded on the graded activity statistics against N2's own activity bands, from its 42 band dates at the half's size, on `default_rng([2047, n])` with PD1074's activity κ. It fails if any graded statistic fails. A reported onset rate is reported beside it.
   - **At most 1 of the 2 units may fail,** as v2's N2 control allowed 1 of 4. A systematic fault in the bands or the measurement fails both. Part C states the chance that both fail by chance, from the activity check's pass rate at the decided bar.

5. **The adversary:**

   - C.2's stage-2 answer (`122-c2-encoder/readiness/pilot_staged.json`, `stage2_extended`), on the wiring it was fitted on (`sign_override_exclude: [lopezcruz2019]`);
   - on 5 populations at `603_000_000 + 200·g + i`, g < 5, on the small lawn alone;
   - a population catches it when at least one graded activity statistic fails (|z| > z\*, or undefined);
   - **it must be caught in at least 3 of 5.**

6. **The negatives:**

   - in v3's order: B.4's six (`hmm`, `poisson`, `random_walk`, `correlated_walk`, `untrained_mlp`, `untrained_connectome`), v2's blind-worm answer and C.1's v1 answer;
   - each on 5 populations at `604_000_000 + 200·k + i`, with k = 5·i + g, graded in full on v4, presence included;
   - the untrained networks are B.4's (5301–5308 for the MLP, 5309–5316 for the connectome), each driving 25 of a population's 200 animals;
   - **each must fail the wild type in at least 3 of 5** at v4's bar. Their chemotaxis failures are reported.

7. **Reported, not gated:**

   - **C.1's v3 answer** (Logbook 121, its chosen restart) on its own registered populations, `525_000_000 + 200·g + i`, g < 5, graded in full on v4 with presence. Its v3 statistics' values on v4 must equal its stored v3 grade's, which the record states, so the two grades compare like for like. Its v4 verdict at 3 of 5 decides whether C.1 is refitted, in its own change.
   - **PD1074's held-out cohorts,** pooled, on the activity family at their own size, as description: the planning probe read them.
   - **W's minimum** on both family sets, by v3's rule and seeds: the fitted set (small lawn, kinematics and now activity) between the same halves of N2's unread held-out dates (1,000 splits on seed 2034), the activity difference scaled by PD1074's activity bands at the half's size; the held-out set is v3's. Each minimum is the 95th percentile of its distances, with v3's fitted minimum beside it.
   - **The 3-of-5 probability for a faithful controller,** by v3's method: each family fails a faithful population at its rate (the small lawn's, kinematics', local search's and activity's from their rule checks at v4's decided bar, the uniform well's at B.4's 0.05, and chemotaxis's leave-one-out rate carried from v3); the product of the passes is the chance of passing all six, and the binomial chance of at least 3 of 5 follows.

Otherwise **VOID**: v3 stays the default, and the failing item and its diagnosis go to the maintainer. Every reading is recorded either way.

- **VALID:** `verdict` writes `controls_v4.json` with `bands_v4.json`'s SHA-256, and `benchmark_grade.py` grades on v4 by default. `--bands v3` reproduces every v3 reading.

### Seeds (decision 9)

| Seeds | Use |
|---|---|
| v3's | every v3 band, the measurement models' streams, the chemotaxis starts, presence |
| 2030 (index 6), 2045, 2046 | the activity bands, κ, the activity rule check |
| 2040, 2034 | the other rule checks (lawn and kinematics; local search), as v3's |
| 2035, 2047 | N2's halves (v2's) and N2's activity bands |
| 2034, 2036 | W's splits (fitted; held out), as v3's |
| 610 M; 9431–9432 | the oracle's fit (restart 0 610.0–610.32 M, restart 1 612.0–612.32 M; CMA) |
| 602 M | the oracle's 40 check populations |
| 603 M | the adversary |
| 604 M | the negatives and the two answers |
| 525 M | C.1's v3 answer, re-read on its own registered populations |
| 600 M | the planning probe, used |
| 606 M, 607 M (CMA 9433), 608.8 M | the preflights, discarded |
| 99.6 M, 99.61 M | the implementation smoke, discarded |

All are disjoint from every band and every earlier scheme. The oracle's fit was first planned at 601 M; there restart 1 would have reused the adversary's 603 M block, the overlap v3's erratum records, so it was moved before this record.

### Disclosures

Seen before this record was committed:

- **The planning probe** (`probe/activity_probe.json`, animals at 600 M): PD1074's band and held-out cohorts, per date, on the three statistics at five pause floors, with the authors' per-frame reversal and pause shares; and C.2's stage-2 answer, C.1's v3 answer and the constant-hazard and pirouette negatives on 200 animals each.

- **The pause floor's derivation** from the still-worm library.

- **The activity bands** at n = 200, built in process by the registered functions and generators for the preflights and the smoke (`provisional_bands_v4`, without the rule checks): centres 0.4819, 0.1321 and 2.2841 per minute, scales 0.0322, 0.0125 and 0.2254. The guarded build reproduces them. These are Part B's readings, seen before registration. No rule was changed after seeing them.

- **The implementation smoke** (8 animals at 99.6 M and 99.61 M, as an execution check): the box centre's activity z's (pause −3.8, flips −2.1, onsets −1.4), and C.2's stage-2 answer's (pause +1.2, flips −3.8, onsets −3.6). The N2 and W code ran with the arm swapped to PD1074, so N2's activity statistics were not computed. PD1074's held-out cohorts passed all three statistics against bands at their size (z +0.31, −0.91, −0.55).

- **The preflights** (`preflight.json`, on the provisional bands):

  - **the adversary gate** (606 M): C.2's stage-2 answer was caught in 5 of 5 populations at 3.0781, and in 5 of 5 by the pause share and the flip fraction alone at 3.0654. Its z's: the pause share +3.5 to +3.7, the flip fraction −10 (the cap), the onset rate −9.7;
  - **the oracle** (one restart of 60 generations, CMA 9433, 607 M): it ran to its cap, its median objective falling to 0.83 at generation 60 (the mean of its last 20, 1.51; σ 0.092). Its answer, on 5 populations at 608.8 M, passed all three statistics in 5 of 5, with mean z −0.63 (pause share), +0.88 (flip fraction) and −0.26 (onset rate), and largest |z| 1.55, 1.37 and 1.50. By the registered rule on those 5 populations, the onset rate would be graded.

- **Not seen:** the guarded bands build and its rule checks; the round trip; the oracle's fit at 610 M and its 40 populations at 602 M; N2's activity statistics; the adversary at 603 M; the negatives at 604 M; C.1's v3 answer on v4; W's v4 minimum.

### Cost

From the preflights, on 16 workers:

| Stage | Estimate |
|---|---|
| bands build | a few minutes: the activity bands took 17 s, and the rule checks run at two bars |
| round trip | a few minutes |
| oracle fit | 4.9 s per generation on the small lawn: at most 16 min for both restarts at the cap |
| oracle check | under a minute (40 populations) |
| read | about 10–15 min: the adversary's 5 populations took 9 s, and v3's 16 negative populations 124 s |

### The guards

- `benchmark_bands.py --version v4`, `benchmark_v4_controls.py round-trip` and `oracle-fit` refuse unless this record is committed and unchanged. The builder refuses to rebuild an existing bands file. `round-trip` runs once. `oracle-fit` also refuses unless the bands are built and the round trip passed.
- `oracle-check` refuses unless this record is committed with Part B, and `bands_v4.json`, `bands_v4_record.json`, `round_trip.json` and `oracle_fit.json` are byte-identical to their versions in that commit. It runs once.
- `--onset` refuses before Part B, without `oracle_check.json`, against its reading, or on bands already decided.
- `read` and `verdict` refuse unless this record is committed with Part C, and the decided `bands_v4.json`, its record, the round trip, the fit and `oracle_check.json` are byte-identical to their versions in that commit, with the bands' decision matching the check's. `read` runs once: it refuses if `read.json` or its start marker exists.

## Part B: the bands, the round trip and the oracle's answer

*Added 2026-11-05, after the oracle's fit and before its 40 populations run. It changes no rule.*

### The bands

`data/benchmark/bands_v4.json`, SHA-256 `324445d9…`, built by the guarded builder with `activity_graded: null` (`bands_v4_record.json`). Every v3 band is carried byte-identical, and the activity bands equal the provisional build's.

- **The pause floor:** 0.02 mm, with 95.4% of the library's 87,035 still-worm seconds below it.

- **κ** (`default_rng(2045)`): the pause share 1.221, the flip fraction 1.207 and the onset rate 1.216.

- **The activity bands** (centre ± scale):

  | Statistic | n = 200 | n = 64 |
  |---|---|---|
  | pause share | 0.4819 ± 0.0322 | 0.4819 ± 0.0537 |
  | flip fraction | 0.1321 ± 0.0125 | 0.1321 ± 0.0201 |
  | onset rate (per min) | 2.284 ± 0.225 | 2.284 ± 0.372 |

- **The rule checks,** gate 0.90, at both bars v4 can have:

  | Check | at 3.0781 (24) | at 3.0654 (23) |
  |---|---|---|
  | activity | 0.985 | 0.985 |
  | small lawn | 0.95 | 0.95 |
  | kinematics | 0.98 | 0.975 |
  | local search | 0.992 | 0.992 |

  All pass. The activity check's failures: the pause share in 1 split of 200, the flip fraction in 2, and the onset rate in none.

### The round trip

`round_trip.json`: **passes.** All 1,013 PD1074 OD 2 band animals give activity tallies equal to the native path's, animal by animal, through CSV and through WCON.

### The oracle's answer

`oracle_fit.json`, on the bands above:

| Restart | CMA seed | Stop | Final objective | σ at the stop |
|---|---|---|---|---|
| 0 | 9431 | cap, generation 100 | 0.831 | 0.061 |
| 1 | 9432 | cap, generation 100 | 0.570 | 0.084 |

**Restart 1 is taken.** Its answer, in C.1's 25 parameters:

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| a0 | −6.819 | r_dwell | 0.0623 |
| a_intake | −0.516 | r_roam | 0.2955 |
| a_satiety | −1.165 | beta_reversal | 4.914 |
| b0 | −3.727 | edge | 4.375 |
| b_intake | 3.112 | o_dwell | 0.2396 |
| b_satiety | 1.049 | o_roam | 0.1776 |
| v_dwell | 0.199 | beta_omega | 4.906 |
| v_roam | 0.0935 | search_amplitude | 14.68 |
| sigma_dwell | 22.72 | search_tau_s | 185.8 |
| sigma_roam | 42.77 | v_off_food | 0.1025 |
| kappa_weathervane | 49.99 | sigma_off_food | 48.19 |
| speed_log_sd_dwell | 0.5375 | speed_log_sd_roam | 0.975 |
| kappa_off_food | −38.42 | | |

It is fitted to the three activity statistics alone, on the small lawn, so it says nothing about the other families: it dwells faster than it roams, and it reverses and turns far more often than C.1's v3 answer. It witnesses only that C.1's model can reach the activity statistics.

**Next, under the guard:** `oracle-check` (the 40 populations, which decide the onset rate), then `--onset` and Part C, then `read` and `verdict`.
