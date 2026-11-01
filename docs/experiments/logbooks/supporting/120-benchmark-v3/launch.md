# Launch record: B v3, reorientation graded from positions

**Part A registered 2026-11-01**, change `benchmark-v3`. It is committed before the v3 bands are built by the guarded builder and before any registered control is read. It fixes decisions 1–9 of the change's design with every threshold and seed. Part B, added after the oracle's fit, records the bands, the rule checks, the round trip and the oracle's fitted answer, before the oracle's 40 populations and the read run; it changes no rule. A verdict registered here is never re-read under a different rule. Anything else is reported beside it as description.

v1 (`data/benchmark/bands.json`, SHA-256 `44122f90…`) and v2 (`data/benchmark/bands_v2.json`, SHA-256 `21f591b8…`) stay frozen for every past verdict. With v3's code in place, both rebuild byte-identical, and so does the López-Cruz split (`rebuild_check.txt`). All 55 jobs of the identity check reproduce every v1 and v2 statistic, D and presence reading byte for byte (`identity.json`).

## Part A: the rules

### The statistics (decisions 1–3)

Each is computed the same way on real tracks and on simulated tracks after v2's measurement models (`benchmark/grading.py`, `benchmark/chemotaxis.py`).

- **Local search's log early/late ratio:** log(early/late), over v1's bins (early 8–14 min, late 34–44 min after removal), in place of the linear ratio. It is undefined where the ratio is not positive.
  - γ is graded as before.
  - The presence test is unchanged: the early/late ratio above 1 at its own lower bound.
- **Chemotaxis's onset rate per minute** (`onset_per_min`): 60 × the mean of the off-food detector's per-second onset flags, over the pooled seconds the chemotaxis samples keep. Graded on a linear scale.
- **Chemotaxis's direction-flip fraction** (`flip_fraction`): over each fragment's consecutive 1-s displacements, the pairs in which both displacements are at least **0.03 mm**. A pair flips when the two directions differ by more than **135°**. The fraction is the population's pooled flips over its pooled pairs.
- **Imported tracks:** local search is graded on log early/late. Chemotaxis, and so its two new statistics, cannot be imported.

### D v3 (decision 4)

- **Five families,** equal weight, each split equally among its statistics:
  - small lawn (6), uniform well (3) and kinematics (5), as v2;
  - local search (2): γ and the log early/late ratio;
  - chemotaxis (5): drift, the log quintile ratio, the weathervane slope, the onset rate and the flip fraction.
- **The bar:** 21 statistics, z\* = Φ⁻¹(1 − 0.05/42) = **3.0381**. The z cap stays at 10, and an undefined statistic enters at the cap and fails.
- **D_fit** is unchanged: the small lawn, the uniform well and kinematics.
- **A population passes** when every family passes. Local search and chemotaxis each pass only where their presence tests hold, as on v2.

### Bands v3

`benchmark_bands.py --version v3` writes `data/benchmark/bands_v3.json` and `bands_v3_record.json` here:

- **Copied from v2, byte-identical:** the small lawn, uniform well, mutant and kinematics bands, the measurement models and κ.
- **Local search:** from López-Cruz's 17 band batches (`lopezcruz2019_split.json`), each statistic's mean, with a scale of their SD × √(1 + 1/k) · t(1 − 0.05/42, k − 1) / z\*(21). At k = 17 the factor is 1.2205, against v1's 1.1957 for γ.
- **Chemotaxis:** from Chen's 6 standard recordings, each of the five statistics' mean, with a scale of their SD × the same factor at k = 6 (2.0146).
- **The rule checks, at v3's bar,** each with a gate of 0.90:
  - the small lawn (v1's κ) and kinematics (v2's κ), on seed 2040: v2's 200 splits of band dates at size 200, re-read at the new bar;
  - local search, on seed 2034: 200 splits of the band batches into halves, each batch of one half graded against a band from the other, joint on γ and log early/late.
- **No bands are written if a rule check fails.**
- **The reported reference:** Roberts et al. 2016's share of tracked frames in the tracker's reverse state, all cohorts, off food.

### The reported reversal share (decision 5)

- Where a population's steps carry primitive labels, the grade reports the share of seconds in a reversal bout, over the local-search and chemotaxis windows, with Roberts' reference beside it.
- It takes no part in any pass, z or distance. Imported tracks report none.
- **The escalation:** if a fit is caught by the reported share but passes the three new statistics, grading the share is proposed in a new change.

### Grading (decision 6)

- `benchmark_grade.py --bands v1|v2|v3`; `grade_populations_v3` mirrors v2's grading with the new statistics, presence unchanged, and the reported share.
- **The default:** v3 once `controls_v3.json` reads VALID with `bands_v3.json`'s SHA-256; otherwise v2 once its own controls read VALID; otherwise v1.

### The validity map (decision 7)

`benchmark_v3_controls.py` reads every item. **VALID** when all of these hold:

0. **The bands' rule checks** pass. The builder refuses to write bands otherwise.

1. **The round trip.**

   - Chen's 4,809 standard-protocol fragments, smoothed by the authors (`SmoothX`/`SmoothY`) and resampled to 1 s as the raw paths are;
   - they are passed through the chemotaxis model's jitter alone, fragment k at `default_rng([475_000_000 + k, 2037])`, since the fragments' cuts are the real ones;
   - for the onset rate and the flip fraction, the round-tripped value must lie within **|z| ≤ 1** of the raw fragments' value, on the v3 band's scale;
   - the other three statistics are reported;
   - if it fails, the run stops before the oracle's fit.

2. **The reachability oracle.**

   - **The fit,** only once the round trip has passed:

     - C.1's 20-parameter model (`BatchedMechanistic`) in C.1's unit box;
     - fitted to the three new statistics alone: the objective is the RMS of their capped z against v3's bands, on the local-search and chemotaxis assays (each assay is its own run on the animals' own streams, so those two alone give the same animals as the full set);
     - sep-CMA from the box centre, σ₀ 0.2, population 16, CMA seeds **9421 and 9422**;
     - fit animals at `470_000_000 + 200·k + i`, with k = 10,000·r + 16·(n − 1) + m for restart r, generation n and member m;
     - v2's σ-guarded plateau stops a restart (from generation 40, the mean of the last 20 generations' median objectives no lower than the 20 before by more than 0.05, and σ ≤ σ₀); otherwise it stops at the **cap of 100** generations.

   - **The answer:** the restart with the lower final objective (the mean of its last 20 medians) is taken, and its answer is its mean at the stop, clipped to the box. Part B records it before its populations run.

   - **The check:** 40 populations of 200 at `471_000_000 + 200·p + i`, p < 40. For each new statistic:

     - its mean z must lie in **[−1, 1]**;
     - |z| ≤ z\* must hold in **at least 90%** of populations.

     An undefined value enters at the cap and fails. Each population's reversal share is reported.

   - **If the check fails,** the new statistics are unreachable by C.1's model: the read stops, nothing else is read, and the change returns to the maintainer.

3. **Held-out López-Cruz:** every one of the 10 held-out batches passes log early/late at |z| ≤ z\* on v3's band. γ is reported.

4. **Chen's leave-one-out:**

   - each standard recording is graded against bands from the other five (the factor at k = 5) on all five chemotaxis statistics;
   - its unit fails if any statistic fails, or if its own presence test fails;
   - the allowance is the fewest failures whose binomial tail over 6 units is at most 0.025, at the nominal rate 1 − (1 − 0.05/21)⁵ = 0.01185. That gives **an allowance of 1** (P(more than 1) = 0.0020).

5. **The adversaries:**

   - C.1b's restarts 2, 4 and 6 (`119-c1b-fit/fit.json`) and the κ probe's answer (`119-c1b-fit/probe_kappa.json`);
   - each on 5 populations at `472_000_000 + 200·k + i`, with k = 5·i + g for entry i in that order and population g;
   - each graded in full on v3, presence included;
   - a population catches the adversary when at least one of log early/late, the onset rate and the flip fraction fails (|z| > z\*, or undefined), whatever presence reads;
   - **each adversary must be caught in at least 3 of its 5 populations.**

6. **The negatives:**

   - in this order: B.4's six (`hmm`, `poisson`, `random_walk`, `correlated_walk`, `untrained_mlp`, `untrained_connectome`), v2's blind-worm answer (`115-benchmark-v2/blind_v2.json`) and C.1's v1 answer (restart 3, `113-c1-fit/fit.json`);
   - each on 5 populations at `473_000_000 + 200·k + i`, with k = 5·i + g;
   - the untrained networks are B.4's (5301–5308 for the MLP, 5309–5316 for the connectome), each driving 25 of a population's 200 animals;
   - **each must fail the wild type in at least 3 of 5 populations;**
   - **each of the four open-loop ones must also fail the chemotaxis family,** presence included, in at least 3 of 5.

7. **The lawn N2 control is carried from v2** (`115-benchmark-v2/read.json`, passed), since the lawn families and kinematics are unchanged and v3's bar is more lenient. It is recorded as carried, not re-read.

8. **Recomputed and reported, not gated:**

   - **W's minimum** on both family sets, by v2's rule and seeds under v3's family map:
     - the fitted families (small lawn and kinematics) between halves of N2's 26 unread held-out dates, 1,000 splits on seed 2034. Their statistics, bands and weights are unchanged, so this must reproduce v2's 2.97, and the record states whether it does;
     - the held-out families: halves of López-Cruz's 17 band batches (1,000 splits on seed 2036) on γ and log early/late, crossed with every 3-against-3 split of Chen's 6 recordings on the five chemotaxis statistics;
     - each minimum is the 95th percentile of its distances.
   - **The 3-of-5 probability for a faithful controller,** by v2's method:
     - each family fails a faithful population at its rate: the small lawn's, kinematics' and local search's from their rule checks at v3's bar, the uniform well's at B.4's 0.05, and chemotaxis's at the nominal 0.01185;
     - the product of the passes is the chance of passing all five families, and the binomial chance of at least 3 of 5 follows.

Otherwise **VOID**: v2 stays the default, and the failing item and its diagnosis go to the maintainer. Every reading is recorded either way.

- **VALID:** `verdict` writes `controls_v3.json` with `bands_v3.json`'s SHA-256, and `benchmark_grade.py` grades on v3 by default. `--bands v2` reproduces every v2 reading.

### Seeds (decision 9)

| Seeds | Use |
|---|---|
| v2's | every band generator, the measurement models' streams (`[s, 2033]`, `[s, 2037]`), the chemotaxis starts (`[s, 2038]`), presence (2041) |
| 2040, 2034 | the rule checks (lawn and kinematics; local search) |
| 2034, 2036 | W's splits (fitted; local search) |
| 470 M; 9421–9422 | the oracle's fit (animals; CMA) |
| 471 M | the oracle's 40 check populations |
| 472 M | the adversaries |
| 473 M | the negatives and the two answers |
| 475 M | the round trip's jitter |
| 99.94 M, 99.95 M, 99.96 M (CMA 9423) | the probe and the preflights, discarded |
| 99.97 M, 99.98 M, 99.99 M | the controls preflight, discarded |

All are disjoint from every band and every earlier scheme.

### Disclosures

Seen before this record was committed:

- **The planning brief's readings of held-out data** (`brief.md`):

  - López-Cruz's 10 held-out batches: their log early/late against the 17 band batches lies within |z| ≤ 1.14 at v2's factor;
  - Roberts' five cohorts, including the held-out C186 and C21: the detector's onset rates, and the tracker's reverse share;
  - Calhoun's 47 worms;
  - Chen's 8 recordings' onset rates.

- **The probe** (`probe.json`, animals at 99.94 M): C.1b's 8 restarts and the κ probe, against the held-out real data.

- **The preflights** (`preflight.json`):

  - **the adversary gate** (99.95 M, bands at m = 21 built as registered): restarts 2, 4 and 6 and the κ probe were each caught in 5 of 5 populations;
  - **the oracle** (one restart of 60 generations, CMA 9423, 99.96 M): its answer, on 5 populations, passed every new statistic in 5 of 5, with mean z +0.16, −0.05 and +0.05.

- **A dry build of the v3 bands,** in memory and written only to scratch, as an implementation check. It printed:

  - the rule checks: the small lawn 0.945, kinematics 0.975 and local search 0.991;
  - the leave-one-out: 0 failures against an allowance of 1;
  - its local-search and new chemotaxis bands, which equal the preflight's.

  These are Part B's readings, seen before registration. No rule was changed after seeing them.

- **The controls preflight** (`preflight_controls.json`, run by `benchmark_v3_controls.py preflight` on the dry-built bands):

  - **the round trip** with its jitter at 99.97 M: the onset rate z +0.12 and the flip fraction +0.02, both passing;
  - **one oracle generation** at the box centre (99.99 M): 17 s on 16 workers, median objective 2.06;
  - **the negatives and the two answers,** 2 populations each (99.98 M): every one failed the wild type and the chemotaxis family in 2 of 2.

- **Not seen:** the guarded bands build; the round trip at 475 M; the oracle's fit at 470 M and its 40 populations at 471 M; the adversaries at 472 M; the negatives at 473 M; W's v3 minimum.

### Cost

From the preflights, on 16 workers:

| Stage | Estimate |
|---|---|
| bands build | about 1.5 min |
| round trip | a few minutes |
| oracle fit | 17 s per generation on the two assays: at most 57 min for both restarts at the cap |
| read | about 10–15 min (16 negative populations took 124 s) |

### The guards

- `benchmark_bands.py --version v3`, `benchmark_v3_controls.py round-trip` and `oracle-fit` refuse unless this record is committed and unchanged. `round-trip` runs once, and `oracle-fit` also refuses unless the round trip passed.
- `read` and `verdict` refuse unless this record is committed with Part B, and `bands_v3.json`, `bands_v3_record.json`, `round_trip.json` and `oracle_fit.json` are byte-identical to their versions in that commit.
- `read` runs once: it refuses if `read.json` or its start marker exists.

## Part B: the bands, the round trip and the oracle's answer

*Added 2026-11-01, after the oracle's fit and before its 40 populations and the read. It changes no rule.*

### The bands

`data/benchmark/bands_v3.json`, SHA-256 `8af38429…`, built by the guarded builder: 21 statistics, z\* = 3.0381 (`bands_v3_record.json`). They equal the dry build's.

- **The rule checks** at v3's bar, gate 0.90: the small lawn 0.945, kinematics 0.975 and local search 0.991. All pass. At v2's bar the first two read 0.94 and 0.96.

- **The new bands** (centre ± scale):

  | Statistic | Band |
  |---|---|
  | γ (per min) | 0.1084 ± 0.0412 |
  | log early/late | 1.393 ± 0.375 |
  | onset rate (per min) | 1.115 ± 0.943 |
  | flip fraction | 0.0234 ± 0.0204 |

  Drift, the log quintile ratio and the weathervane slope keep v2's centres, with scales at v3's factor.

- **Chen's leave-one-out:** every standard recording passes and is present, 0 failures against an allowance of 1 (largest |z| 1.85). Off protocol, reported: 20220804 present, 20220203 not.

- **The reported reference:** Roberts' reverse state is 17.34% of 2,048,900 tracked frames.

### The round trip

`round_trip.json`: **passes.** The onset rate reads 1.201 per minute raw and 1.307 round-tripped (z +0.11); the flip fraction reads 0.0247 and 0.0252 (z +0.03). Drift, the log quintile ratio and the weathervane slope are reported.

### The oracle's answer

`oracle_fit.json`:

| Restart | CMA seed | Stop | Final objective |
|---|---|---|---|
| 0 | 9421 | plateau, generation 80 | 0.127 |
| 1 | 9422 | cap, generation 100 | 0.090 |

**Restart 1 is taken.** Its answer, in C.1's parameters:

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| a0 | −5.367 | r_dwell | 0.0259 |
| a_intake | −5.000 | r_roam | 0.0038 |
| a_satiety | 3.944 | beta_reversal | 1.820 |
| b0 | −9.092 | edge | 1.360 |
| b_intake | −0.500 | o_dwell | 0.0894 |
| b_satiety | 1.866 | o_roam | 0.0013 |
| v_dwell | 0.160 | beta_omega | 0.534 |
| v_roam | 0.0599 | search_amplitude | 19.53 |
| sigma_dwell | 5.15 | search_tau_s | 409.0 |
| sigma_roam | 57.06 | kappa_weathervane | −26.08 |

It is fitted to the three new statistics alone, so it says nothing about the other families: it dwells faster than it roams, and it steers away from the odour. It witnesses only that C.1's model can reach the new statistics.

**Next, under the guard:** `read` (the oracle's 40 populations, then, if they pass, the rest of the read), then `verdict`.
