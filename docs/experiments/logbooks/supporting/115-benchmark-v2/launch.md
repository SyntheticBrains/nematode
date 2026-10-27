# Launch record: B v2, the benchmark revised once

**Part A registered 2026-10-27**, change `revise-benchmark-v2`. It is committed before the oracles, the N2 read, the negatives and the adversary fits, and it fixes decisions 2–12 of the change's design with every threshold and seed. Part B, added before the N2 read and the adversary fits, records the probe's parameters, the bands, the rule checks and the oracles' results; it changes no rule. A verdict registered here is never re-read under a different rule. Anything else is reported beside it as description.

v1 (`data/benchmark/bands.json`, SHA-256 `44122f90…`) stays frozen for every past verdict (B.4, F.0b-i, C.1). The generalised builder rebuilds it byte-identical.

## Part A: the rules

### The measurement model (decision 02)

- **What it does.** Every simulated track passes through a tracker model before any window is cut (`benchmark/measurement.py`). Animal s draws from `default_rng([s, 2033])` on the lawn assays and `default_rng([s, 2037])` on the chemotaxis assay:
  - each 1-s position gains N(0, σ²) noise per axis;
  - then losses start: at each tracked second, with p_omega if the simulated primitive that second is an omega and p_other otherwise;
  - each loss lasts a length drawn from that context's measured interior runs.
- **Runs, not single seconds.** The design's switch applies: real interior losses run a median of 16 s after an omega start and 29.5 s otherwise, longer than 1 s. So losses are drawn as runs with the real length distributions, and the start rates are per tracked second.
- **The lawn model's parameters**, from `probes.json` (band cohorts):
  - σ = 0.00468 mm, from paused seconds (mean step / √π);
  - after an omega start: p_omega = 0.0484 per tracked second, 215 runs;
  - after any other start: p_other = 0.000168, 402 runs.
- **The chemotaxis model's parameters:**
  - σ = 0.00871 mm, raw positions against the authors' smoothed track;
  - simulated 30-min tracks are cut into fragments drawn from the 4,809 standard-protocol fragment lengths (median 39 s, 90th percentile 167 s), one lost second marking each cut.
- **Scope.** Simulated sources only (`batched`, `captures`). Real and imported tracks are never measured.

### The kinematics family (decision 03)

On the small lawn, fitted (in D_fit) for every controller:

- `dwelling_speed_median`, `dwelling_speed_p90`, `roaming_speed_median` and `roaming_speed_p90`:
  - the classifier's own 10-s window speed over on-lawn classified windows of each state;
  - pooled through a fixed histogram per state (80 log-spaced bins from 10⁻⁴ to 1 mm/s), so tallies add for the date bootstrap;
- `reversals_per_on_lawn_min`: off-food-detector onsets (`readouts.reversal_onsets`) on the whole track, counted where the onset's sample is on the lawn, per minute of valid on-lawn seconds;
- `roaming_survival_5min`: the Kaplan–Meier survival of on-lawn roaming bouts at 30 windows, with a bout at a run's edge censored.

### Local search (decision 04)

Unchanged: γ, the early/late ratio and the presence test, as in v1. The rise is not added.

### D v2 (decision 05)

- **D:** five families, equal weight, each split equally among its statistics:
  - small lawn (6);
  - uniform well (3);
  - local search (2);
  - kinematics (6);
  - chemotaxis (3).
- **D_fit:** small lawn, uniform well and kinematics. C.2 and C.3 fit D_fit; local search and chemotaxis are their held-out predictions. C.1 fits D.
- **The bar:** z\* = Φ⁻¹(1 − 0.05/40) = 3.0233, Bonferroni over 20 statistics. The z cap stays at 10.
- **A population passes** when every family passes. Local search and chemotaxis each pass only where their presence tests hold.

### The chemotaxis family (decision 06)

- **The data:** Chen et al. 2025, naive N2, butanone, vendored as `data/benchmark/chen2025_tracks.npz`. The six standard-protocol recordings (20220830, 20220913, 20220922, 20221108, 20230609, 20230613) make the bands. The two off-protocol recordings (20220804, 20220203) are reported, not graded.
- **The assay:** `chemotaxis_chen` on the 1-s primitives worm.
  - The arena is a 100 mm square plate. The worm is fed on a 25 mm lawn for 2,700 s, then assayed for 1,800 s with no food, the analysis running from the assay's start.
  - The odour is the measured landscape, centred on the plate, sensed as tanh(C / C_max).
  - Each animal starts at a real fragment's first position among those beginning in the first 30 s, drawn from `default_rng([s, 2038])`.
  - Samples outside the imaged field are missing.
  - Every simulated chemotaxis population uses the same animal seeds as its lawn populations.
- **The statistics** (`benchmark/chemotaxis.py`), pooled over valid seconds and identical on real and simulated tracks:
  - `drift_mm_s`: velocity along the local gradient;
  - `log_quintile_ratio`: the logarithm of the off-food detector's onset rate in the lowest dC/dt quintile against the highest;
  - `weathervane_deg_s`: the slope, over run seconds more than 2 s from an onset, of turning rate on the sine of the bearing to the gradient.
- **The bands:** the mean over the six recordings. The scale is their SD times the prediction factor for one new recording at v2's bar, √(1 + 1/6) · t(1 − 0.05/40, df 5) / z\* = 2.002.
- **Presence:** chemotaxis passes only where drift and the log ratio are each above 0 at the 2.5% quantile of 2,000 resamples (seed 2041). The resampled units are animals for a simulated population and fragments for a real recording. A resample where a statistic is undefined counts as absent.
- **The leave-one-recording-out validation:**
  - each standard recording is graded against bands from the other five (t, df 4) at |z| ≤ z\*;
  - its unit fails if any statistic fails, or if its own presence test fails;
  - the allowance is the fewest failures whose binomial tail over 6 units is at most 0.025. Each statistic fails at the bar's two-sided 0.05/20, so a recording fails at the nominal rate 1 − (1 − 0.0025)³ = 0.0075. That gives **an allowance of 1** (P(more than 1) = 0.0008). The nominal rate covers the z tests, not presence;
  - this is cross-validation, not a held-out read: 6 dates, one lab, one odour.

### Bands v2 (decision 07)

- From band cohorts only, by v1's date bootstrap at n.
- The kinematics bands use generator `default_rng([2030, 5, n])`, after v1's indices, so v1's values are reproduced.
- κ is calibrated for the kinematics family on seed 2039, and both rule checks (the small lawn and kinematics) are read on seed 2040, each by 200 splits of band dates at size 200 with a gate of 0.90.
- `benchmark_bands.py --version v2` refuses until this record is committed. It writes no bands if a check fails.

### The oracles (decision 08)

1. **The measurement-model round trip.** For each rig and each new statistic, the round-tripped population value must lie within **|z| ≤ 1** of the raw tracks' value, on that statistic's band scale. If it fails, the run stops.

   - **Scheer:** the 1,013 band-side PD1074 small-lawn animals (41 dates). Each is denoised: interior gaps are interpolated linearly, then a centred 5-s moving average is taken, and the seconds before the first and after the last fix stay missing. The track is then measured with the lawn model at `default_rng([154_000_000 + row, 2033])`, using Scheer's own omega flags. The scale is the kinematics band at n = 1,013. The small lawn's statistics are reported beside it, not gated.
   - **Chen:** the 4,809 standard-protocol fragments, denoised by the authors' `SmoothX`/`SmoothY`, resampled to 1 s as the raw paths are. Only the chemotaxis model's jitter is applied (`default_rng([154_000_000 + fragment, 2037])`), since the fragments' cuts are the real ones. The scale is the chemotaxis band's.

2. **The reachability oracle** (only if the round trip passes):

   - B.4's matched open-loop worm (`hazard_controller("hmm")`) is run through the lawn model on the small lawn, at 40 populations of 200 (animals `150_000_000 + 200·p + i`);
   - for each of the four speed statistics and the reversal rate, the mean z must lie in [−1, 1], and |z| ≤ z\* must hold in at least 90% of populations;
   - bout survival is excluded, since that worm's bouts are exponential. Its mean z is reported.

   If the oracle fails after the round trip has passed, the failure belongs to the matched worm or to the statistic. The record says which, and the run stops before Part B and returns to the maintainer.

### The positive control: fresh N2 (decision 09)

- **The units:**
  - N2 (`N2_od2_Data`) has 26 unread held-out dates: all its held-out dates except tph-1's two paired dates, read in B.4 (20180202, 20180212);
  - they are split into two halves of 13 by `default_rng(2035).permutation`;
  - each half is graded on the small lawn and on kinematics, giving 4 units.
- **The bands:** N2's own, from its 42 band dates, at each unit's animal count.
  - Generator `default_rng([2042, f, n])`, with f = 0 for the small lawn and 1 for kinematics.
  - κ is v1's for the small lawn and v2's for kinematics.
- **The gate:** **at most 1 of the 4 units fails.** P(more than 1) is computed from the rule checks' rates and stated in Part B.
- **The read:** once, under the guard. PD1074's held-out cohorts are not read.

### The negatives and adversaries (decision 10)

- **B.4's six negatives** on B.4's animal seeds (5101–5300; networks 5301–5316), through the measurement model and with the chemotaxis assay:

  - **each fails the wild type;**
  - **each of the four open-loop ones (`hmm`, `poisson`, `random_walk`, `correlated_walk`) fails the chemotaxis family.**

  Their v1 values (B.4's record) and v2 values are reported side by side.

- **The blind worm.** This is C.1 with every sensing term pinned in the unit box at the point where the term is zero:

  | Pinned at 0 (priors from 0) | Pinned at 0.5 (symmetric priors, or unused) |
  |---|---|
  | `beta_reversal`, `beta_omega`, `edge`, `search_amplitude` | `kappa_weathervane`, the intake and satiety slopes; `search_tau_s`, unused at zero amplitude |

  Its other 10 parameters are fitted by sep-CMA against v2's D:

  - from the box centre, σ₀ 0.2, population 16;
  - **2 restarts** on CMA seeds 9401 and 9402;
  - fit animals at `151_000_000 + 200·k + i`, with k = 10,000·r + 16·(n − 1) + m;
  - **a restart stops on the plateau rule only once σ ≤ σ₀**, and otherwise at the cap of 210 generations.

  The best restart, by its final objective, is graded on five populations (`152_000_000 + 200·g + i`) and **must fail the wild type at 3 of 5.**

- **C.1's v1 answer** (restart 3, `supporting/113-c1-fit/fit.json`), graded on five populations (`153_000_000 + 200·g + i`), **must fail the wild type at 3 of 5.**

### Validity v2 (decision 11)

**VALID** when all of the following hold:

1. the measurement-model round trip passes, for both rigs;
2. the reachability oracle passes;
3. the N2 control passes (at most 1 of 4 units fails);
4. the chemotaxis leave-one-recording-out validation passes within its allowance of 1;
5. all six negatives fail the wild type, and the four open-loop ones fail the chemotaxis family;
6. the blind worm fails the wild type at 3 of 5;
7. C.1's v1 answer fails the wild type at 3 of 5.

Otherwise **VOID**, and the statistics are revised in a new change. Every reading is recorded either way.

- **VALID:** v2 becomes the benchmark. `benchmark_grade.py` defaults to it, and the F.0b-i re-run and C.1b follow on it.
- **VOID:** the failing check is the diagnosis, and it goes to the maintainer.

### W's minimum v2 (decision 12)

Both family sets are recorded; each minimum is the 95th percentile of its split distances under v2's family map.

- **The fitted families** (small lawn and kinematics):
  - between halves of N2's 26 unread held-out dates, split 1,000 times by `default_rng(2034)`;
  - on PD1074's v2 band scales at each half's smaller size, rounded to 25.
- **The held-out families** (local search and chemotaxis):
  - **local search:** between halves of López-Cruz's 17 band batches, 1,000 splits by `default_rng(2036)`. The scale is the band batches' SD / √(half size), as in v1.
  - **chemotaxis:** between every 3-against-3 split of the six recordings, all 10. The scale is the recordings' SD / √3.
  - Every local-search split is crossed with every chemotaxis split, giving 10,000 pairs. Each family's own 95th percentile is reported.

W's registration must read its arms on the same family set as the minimum it cites.

### Seeds

| Use | Seeds |
|---|---|
| v1's band generators | `default_rng([2030, index, n])`, indices 0–4 |
| kinematics bands | `default_rng([2030, 5, n])` |
| measurement model | `default_rng([s, 2033])` lawn, `default_rng([s, 2037])` chemotaxis |
| N2 split | 2035 |
| W: fitted split, local search split | 2034, 2036 |
| chemotaxis starts | `default_rng([s, 2038])` |
| κ (kinematics), rule checks | 2039, 2040 |
| chemotaxis presence | 2041 |
| N2 bands | `default_rng([2042, f, n])` |
| oracle populations | `150_000_000 + 200·p + i`, p < 40 |
| blind fit (animals), CMA | `151_000_000 + 200·k + i`; 9401–9402 |
| blind grading | `152_000_000 + 200·g + i`, g < 5 |
| C.1's answer on v2 | `153_000_000 + 200·g + i`, g < 5 |
| round trip | `154_000_000 + row` (Scheer), `154_000_000 + fragment` (Chen) |
| negatives | B.4's: animals 5101–5300, networks 5301–5316 |

All are disjoint from every band and every earlier scheme. The planning probes used 140–144 M and 9331. The smokes used 990,001–990,200 and 199 M, all discarded.

### Disclosures

Everything seen before this record was committed:

- **The probes** (`probes.json`, committed): every measurement the design was built from. These are the tracker's parameters, Chen's registration and pixel scale, C.1's and the matched worm's kinematics on v1, and the blind worm on v1, which failed all five populations (D 3.9–5.7).
- **The chemotaxis statistics on the real recordings** were computed as implementation checks. The quintile ratio is graded as its logarithm because it was seen to be positively skewed on the band recordings.
- **A build smoke of the v2 bands** printed both rule checks (joint pass rates 0.94 and 0.96 against the 0.90 gate) and the leave-one-out (all six recordings pass, no failures). Both are Part B's readings, seen before registration. Neither rule was changed after seeing them.
- **The presence test was added after a grading smoke.** The constant-hazard worm (animals 990,001–990,064), which ignores the odour, sat inside every chemotaxis band (drift z −1.6, log ratio −1.8, weathervane −0.2). That contradicts the spec's "a worm that ignores the gradient SHALL fail".
  - The test's form copies local search's, and its quantile and resample count are local search's.
  - It was measured on the eight real recordings, the constant worm and the pirouette worm (animals 990,001–990,200) before registration. All six standard recordings are present; the constant worm is absent, and the pirouette worm present.
  - Weathervane was left out because 4 of the 6 standard recordings' own bounds on it fall below 0.
  - Off-protocol 20220203 sits at the edge, with its log-ratio bound exactly 0, so it is absent.
- **A controls smoke on discarded seeds (199 M)** ran one 200-animal population of the matched open-loop worm. Its reversal rate was 3.32 per minute against the band's 4.07, the oracle's own quantity on a disjoint population. It also ran the HMM negative and the untrained connectome on 16 animals each (both fail the wild type, and the HMM fails chemotaxis), and C.1 at the box centre, never at its fitted answer.
- **Not seen:** the round trip, the reachability oracle's registered populations, any N2 held-out statistic, the negatives on their registered seeds, any blind fit against v2, and C.1's answer on v2.

### Cost

From the smokes:

| Stage | Estimate |
|---|---|
| bands build | about 4 min |
| oracle | about 10 min (40 populations at 4 s on 16 workers, plus the round trips) |
| read | about 20 min (W's minimum, the six negatives at n = 200, C.1's five populations) |
| blind fit | about 35 s per generation; at most 4.1 h for two restarts at the cap, about 1.6 h if they stop where v1's did (generation 83) |

The total is about 2–5 h.

### The guards

- `benchmark_bands.py --version v2` and `benchmark_v2_controls.py oracle` refuse unless this record is committed and unchanged.
- `read` and `blind` refuse unless it is committed with Part B and `bands_v2.json` and `bands_v2_record.json` are byte-identical to their versions in that commit.
- `read` runs once: it refuses if `read.json` exists.

## Part A, amendment 1: the oracle's round trip (registered 2026-10-27)

**What failed.** The bands were built under Part A: both rule checks pass (0.94 and 0.96), and the chemotaxis leave-one-out passes with no failures. The bands file is SHA-256 `74d4dac0…`. The round trip then ran as registered (`oracle.json`):

- **Chen's rig passed:** every |z| ≤ 0.11.
- **Scheer's rig failed** on 5 of its 6 statistics:
  - reversals per on-lawn minute z −12.6;
  - dwelling speed median −5.1 and p90 −5.3;
  - roaming survival −2.5;
  - roaming speed median −1.3.

By the rule, the reachability oracle was not run and the run stopped. That verdict stands and is not re-read.

**The diagnosis** (description, run after the failure; `build/benchmark-v2/roundtrip_diag.json`). The failure is the denoiser's, not the tracker model's:

| Variant | Reversals per minute (z) | Dwelling speed median (z) |
|---|---|---|
| the 5-s mean alone, unmeasured | 2.09 (−23.8) | −5.5 |
| gap-filled, then the model's losses alone | −0.3 | −0.1 |
| a 3-s mean with the full model | −3.4 | −2.1 |

Every gap-filled reading is within 0.3. The jitter added after smoothing restores about 0.9 detected onsets per minute, so the noise model's realism matters, and a smoothed track is no clean truth to test it on.

**The amendment** (decided by the maintainer). The Scheer round trip is replaced by two checks, each needing no smoothed truth. Chen's round trip and the reachability oracle are unchanged, and reachability runs only if all three checks pass:

1. **The losses.** The band-side tracks, gap-filled (interior gaps interpolated, nothing else changed), pass through the lawn model's losses alone (σ = 0), at `default_rng([154_000_000 + row, 2033])`. Every kinematics statistic must lie within **|z| ≤ 1** of the raw value at the band scale for n = 1,013. **Already seen in the diagnosis to pass (all within 0.3);** it is registered so the record holds it.
2. **The jitter**, not yet seen:
   - **The stretches:** on the authors' paused stretches the worm is still, so the real readings are the tracker's noise. A stretch is a run of at least 10 band-side PD1074 seconds, each with all three frames flagged `pause` and its position present.

   - **The model's counterpart:** each stretch's mean position held still, with the lawn model's jitter, at `default_rng([155_000_000 + k, 2033])` for stretch k.

   - **The readings:**

     - detected reversal onsets per paused minute (the off-food detector, on each stretch alone);
     - the median and 90th percentile of the classifier's 10-s window speed;
     - the 1-s step median and 90th percentile, reported only.

   - **The gate:** each gated reading's model-minus-real difference must be at most **1 band scale** of the statistic it biases, at **n = 200**, the size every controller is graded at:

     - the onset rate on `reversals_per_on_lawn_min`;
     - the window speeds on `dwelling_speed_median` and `dwelling_speed_p90`.

     A still worm is where the noise weighs most on what is graded.

   - **If it fails:** the model's independent Gaussian noise does not reproduce the tracker. The run stops, and the noise model (for instance its time correlation) is revised in a new change.

`oracle` writes `oracle_amendment1.json` once, and the verdict reads it.

## Part A, amendment 2: empirical noise, the reversal rate reported, reachability re-fitted (registered 2026-10-28)

**What failed.** Amendment 1's jitter check (`oracle_amendment1.json`) stopped the run. On band-side paused stretches, the real tracker reads a still worm with fewer false onsets than independent Gaussian noise (2.14 against 3.87 per paused minute) and with more 10-s drift (window speed median 0.0038 against 0.0016 mm/s). That verdict stands.

The revision is change `benchmark-v2-empirical-noise`. Its pilots ran on the library half's dates and on discarded seeds only, and they are description; the check half below has not been read.

### The rules (this amendment registers)

1. **The split.** PD1074's 41 band dates go into a library half and a check half by `default_rng(2043).permutation`: the first 21 to the library, the other 20 to the check.

2. **The noise library.** `data/benchmark/scheer_tracker_noise.npz`, SHA-256 `328bfbce…`. It holds the library half's 6,213 paused stretches (93,248 s): runs of at least 10 s with all three frames flagged `pause` and a position, stored as offsets from each stretch's own mean.

3. **The draw.** The lawn model's noise is a track drawn from the animal's lawn stream, `default_rng([s, 2033])`:

   - it starts at a random stretch's first offset;
   - where a stretch ends, it splices: of 50 stretches drawn uniformly with replacement, it continues at the offset after the nearest match to its last offset. A stretch's own last offset never matches.

   The losses are unchanged.

4. **The scope** (decided by the maintainer, 2026-10-28). The lawn model measures only the families graded against Scheer's rig: the small lawn, the uniform well and kinematics. Local search is graded against López-Cruz's reorientation counts from another rig, which no track measures, so its simulated tracks stay exact, as in v1. Chen's assay keeps its own measured model.

   - **Sized beforehand** on discarded seeds (199.8 M): Scheer's noise on simulated local search moved C.1's v1 answer from γ 0.112 to 0.038, and its early/late ratio from 3.51 to 2.03.

5. **The on-lawn reversal rate is reported, not graded** (the registered fallback).

   - **The floor:** the rate is read with the detector's step floor at 0.03 mm, the smallest 0.01-mm floor at which the real still worm detects reversals at under 1% of the real on-lawn rate (library half: 0.013 against 2.27 per minute).
   - **The fallback:** it was fixed in the change's design (commit `0e4e8407`) before the last reachability refit, and that refit triggered it. No open-loop point worm reaches the rate through the tracker: the 11-parameter worm still read 1.31 against 2.28 per minute.
   - **The family:** kinematics has five graded statistics: the four speeds and roaming survival at 5 min.
   - **The bar:** D v2 has **19 statistics, z\* = 3.0078**. The reversal rate is computed and reported beside the grade.

6. **The bands.** `bands_v2.json` was rebuilt at 19 statistics (SHA-256 `21f591b8…`). Both rule checks pass: the small lawn 0.94, kinematics 0.96. The chemotaxis leave-one-out passes with 0 failures (allowance 1), and the bands record is `bands_v2_record.json`. The records of the earlier builds are kept: Part A's as `bands_v2_record_partA.json`, the others in git history.

7. **The jitter check**, read once on the check half:

   - **The real side:** the check half's paused stretches.
   - **The model side:** for stretch k, a 2,700-s noise track drawn at `default_rng([156_000_000 + k, 2033])`, with the stretch's length cut at a uniform point from the same stream.
   - **The gated readings:** the median and 90th percentile of the classifier's 10-s window speed, which bias the graded dwelling speeds.
     - **The gate:** each model-minus-real difference must lie within the central 95% of 1,000 random halvings of the band dates (21 against 20, `default_rng(2044)`): **median [−0.000414, +0.000386] and p90 [−0.000352, +0.000310] mm/s** (`noise_pilot.json`).
     - **The Gaussian model's misses** at the 0.01 floor were −0.0023 and −0.0046.
   - **Reported only:**
     - the onset rate per paused minute at the 0.03 floor (range [−0.0086, +0.0088]), since it biases no graded statistic;
     - the 1-s step quantiles.

8. **The reachability worm** replaces B.4's matched worm, which misses the kinematics with no noise at all (dwelling median z +7.3).

   - **The structure:** `BatchedStateHazards`, with B.4's two states, transitions and window, plus two options, both off by default: a per-state speed spread (a log SD, per 10-s window) and a coupling of each window's reversal hazard to its speed draw.

   - **The fit** (`reachability_worm.json`): 11 parameters by sep-CMA against the RMS z of the then-gated five statistics, on animals at 158 M, CMA seeds 9411–9412, restarts stopping at generations 95 and 81 (plateau). The parameters as fitted:

     | | dwelling | roaming |
     |---|---|---|
     | speed (mm/s) | 0.01264 | 0.10433 |
     | speed spread (log SD) | 0.716 | 1.207 |
     | curvature SD (°/s) | 15.24 | 41.78 |
     | reversal hazard (/s) | 0.0292 | 0.2950 |
     | omega hazard (/s) | 0.00021 | 0.00055 |

     The reversal coupling β is 1.323.

   - **Its oracle, before registration:** on the fit's 5 discarded check populations (158.9 M), the four speeds' RMS z reads 0.79, 1.31, 0.86, 0.79 and 1.48 (median 0.86).

   - **Registered:** the 40 oracle populations (`150_000_000 + 200·p + i`) are read with this worm through the lawn model under Part A's gate, for **the four speeds**: mean z in [−1, 1], and |z| ≤ z\* in at least 90% of populations. Bout survival stays excluded, its mean z reported.

9. **Unchanged:** every other rule, threshold and seed of Part A and amendment 1, and the losses check, which applies no noise and is re-run. The validity map is Part A's, with the measurement checks as amended.

### Seeds added

| Seeds | Use |
|---|---|
| 2043 | the library/check split |
| 2044 | the gate's halvings |
| 156 M | the check's model side |
| 157 M | the pilot's model side |
| 158 M | the reachability fit |
| 158.9 M | the fit's check populations |
| 9411–9412 | the fit's CMA seeds |
| 199.6 M, 199.7 M, 199.8 M | discarded probes |

All are disjoint from every band and earlier scheme.

### Disclosures

Seen before this amendment, all on the library half or on discarded seeds:

- **The tracker varies by date.** The library half reads 2.28 onsets per paused minute at the 0.01 floor. The check half's 2.01 is implied by amendment 1's committed all-date total; at the 0.03 floor it is not known.

- **The first draw, joined end to end, failed in-sample.** The onset rate was +1.19 and the window-speed p90 +0.0010. An autoregressive model and nearest-offset splicing were then tried in-sample (onsets +0.44 to +0.58 and +0.43). Splicing passed both window speeds.

- **The floor grid** on the library half, with still-worm and on-lawn rates per minute:

  | Floor (mm) | Still worm | On lawn |
  |---|---|---|
  | 0.01 | 2.28 | 4.11 |
  | 0.02 | 0.19 | 3.05 |
  | 0.03 | 0.013 | 2.27 |
  | 0.04 | 0.001 | 1.65 |

- **The splice in-sample at 0.03:** onsets +0.0034, window speed median +0.00013, p90 +0.00006.

- **The three reachability fits:**

  | Worm | Median RMS z on 5 fresh populations | What it missed |
  |---|---|---|
  | 8 parameters | 2.53 | reversals −4.25, roaming p90 −3.57, dwelling median +2.58 |
  | 10 parameters, with the spread | 2.00 | reversals −4.17 |
  | 11 parameters, with the coupling | 1.59 | the reversal rate (−2.75 to −3.17) |

- **Not seen:** the check half's still-worm window speeds, the 40 registered reachability populations, and everything after them.

### Guard

`benchmark_v2_controls.py oracle2` refuses unless this record is committed and unchanged, with `bands_v2.json`, `bands_v2_record.json`, the noise library, `reachability_worm.json` and `noise_pilot.json` byte-identical to their versions in that commit. It runs once, writing `oracle_amendment2.json`.

## Part A, amendment 3: the reachability worm refitted on the graded speeds (registered 2026-10-28)

**What failed.** Under amendment 2 (`oracle_amendment2.json`):

- **Every measurement check passed:** the losses, the jitter check on the check half, and Chen's round trip.
- **Reachability missed on one criterion.** Every one of the 40 populations passed every speed at |z| ≤ z\*, but the roaming speed p90's mean z was −1.07, against Part A's [−1, 1].
- **Why:** the worm had been fitted with the then-graded reversal rate in its objective, which drove its roaming reversal hazard to its ceiling.
- **A lapse in preflight:** before registration only the four speeds' RMS z was checked on the fit's discarded populations, not Part A's per-statistic gate. There the p90 already averaged −0.97.

That verdict stands, and the measurement checks stand as read.

**The amendment** (decided by the maintainer):

1. **The refit.** The same 11-parameter worm and box are refitted by sep-CMA against the RMS z of the four graded speeds only, the statistics amendment 2 registered for reachability:

   - animals at `160_000_000 + 200·k + i`, CMA seeds 9411–9412, σ guard, cap 100;
   - restart 0 stopped at the cap (final objective 1.04), and restart 1 plateaued at generation 78 (0.58) and is chosen (`reach_refit.json`, `reachability_worm.json`).

   The parameters as fitted:

   | | dwelling | roaming |
   |---|---|---|
   | speed (mm/s) | 0.01423 | 0.07733 |
   | speed spread (log SD) | 0.663 | 1.440 |
   | curvature SD (°/s) | 172.5 | 87.0 |
   | reversal hazard (/s) | 0.0225 | 0.2148 |
   | omega hazard (/s) | 0.00017 | 0.00041 |

   The reversal coupling β is 1.290.

2. **Part A's own gate, on 10 discarded populations** (`160_900_000 + 200·g + i`) before registration:

   | Statistic | Mean z | Share within z\* | z SD |
   |---|---|---|---|
   | dwelling speed median | −0.22 | 100% | 0.11 |
   | dwelling speed p90 | +0.22 | 100% | 0.19 |
   | roaming speed median | +0.38 | 90% | 1.32 |
   | roaming speed p90 | −0.63 | 100% | 0.46 |

   Under those spreads, 40 populations fail the share criterion with probability about 0.005, and the mean criterion with about 0.002. Bout survival, excluded from the gate, reads a mean z of −3.31.

3. **Registered:** 40 new populations, `159_000_000 + 200·p + i`, since the 150 M populations have been seen. They are read once with this worm through the lawn model, under Part A's gate for the four speeds: mean z in [−1, 1], and |z| ≤ z\* in at least 90%. The stage is `benchmark_v2_controls.py oracle3`, which writes `oracle_amendment3.json` once and refuses unless this record is committed with the bands, the noise library, `reachability_worm.json` and `reach_refit.json` unchanged.

Every other rule of Part A and amendments 1 and 2 is unchanged.

## Part B: the measured parameters, the bands and the oracles

*Added 2026-10-28, after the bands build and the oracles, before the N2 read and the adversary fits. It changes no rule.*

### The measurement models

- **The lawn model**, from `probes.json` and the library:

  - the noise is resampled from `scheer_tracker_noise.npz` (6,213 stretches, SHA-256 `328bfbce…`), spliced at the nearest offset of 50;
  - losses start at p_omega = 0.0484 and p_other = 0.000168 per tracked second, with lengths drawn from 215 and 402 real interior runs.

  It measures the small lawn, the uniform well and kinematics only.

- **Chen's model:** Gaussian jitter of σ = 0.00871 mm, and fragments drawn from the 4,809 standard-protocol fragment lengths.

### The bands

`data/benchmark/bands_v2.json`, SHA-256 `21f591b8…`, with 19 statistics and z\* = 3.0078 (`bands_v2_record.json`):

- **κ** for kinematics is calibrated on seed 2039;
- **the rule checks** (seed 2040, 200 splits at size 200, gate 0.90): the small lawn 0.94, kinematics 0.96, both pass;
- **the chemotaxis leave-one-out:** every standard recording passes and is present, with 0 failures against an allowance of 1;
- **the off-protocol recordings,** reported: 20220804 present, 20220203 not.

### The oracles

| Check | Read in | Result |
|---|---|---|
| the losses on gap-filled tracks | `oracle_amendment2.json` | pass, every kinematics |z| ≤ 0.15 |
| the jitter on the check half (7,014 stretches) | `oracle_amendment2.json` | pass: window speed median −0.000011, p90 −0.000044 mm/s, inside their ranges |
| Chen's round trip | `oracle_amendment2.json` | pass, |z| ≤ 0.11 |
| reachability, 40 populations at 159 M | `oracle_amendment3.json` | pass: mean z −0.29, +0.04, +0.35, −0.74; shares 100%, 100%, 97.5%, 100% |

Bout survival, excluded from the reachability gate, reads a mean z of −2.47.

### Figures the read uses

- **The N2 control:** the rule checks' failure rates are 0.06 for the small lawn and 0.04 for kinematics. Over the 4 units, that gives **P(more than 1 fails) = 0.014**.
- **The 3-of-5 oracle for v2:** a faithful population passes all five families with probability 0.94 × 0.96 × 0.95 × 0.962 × 0.9925 = 0.818. So a faithful controller passes the wild type at 3 of 5 with probability **0.955**, which the C.1b and C.2 registrations carry.

**Next, under the guard:** `read` (the N2 control, B.4's six negatives, C.1's v1 answer, W's minimum) and `blind` (the blind worm fitted against v2), then `verdict`.
