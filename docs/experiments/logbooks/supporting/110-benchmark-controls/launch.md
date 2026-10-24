# Launch record: B.4, the benchmark's controls

**Registered 2026-10-25**, change `add-benchmark-controls`. This record is committed before any held-out value is read. `scripts/analysis/benchmark_controls.py read` refuses to run unless it is committed and unchanged. A verdict registered here is never re-read under a different rule. Anything else is reported beside it as description.

## The positive controls (held-out data)

There are 15 units, each graded at its own size on bands built on demand (`benchmark_grade.family_bands`, `benchmark_bands.BandBuilder`).

| Unit | Data | Graded on | Bar | Failure rate under real data |
|---|---|---|---|---|
| small lawn | 24 held-out OD 2 dates, 573 animals | six statistics | 2.84 | 0.075 (B.3's rule check) |
| uniform well | 1 held-out date (20201014), 31 animals | three statistics | 2.84 | 0.05 (nominal: no check possible) |
| López-Cruz batch × 10 | each held-out batch | γ, early/late, and presence | 2.84 | 0.038 each (the rule check) |
| tph-1 | 2 held-out dates paired with N2 | Δ roaming, Δ exits, resolved directions | 2.24 | 0.05 (nominal) |
| pdfr-1 | 3 held-out dates paired with PD1074 | the same | 2.24 | 0.05 |
| tax-4 | 6 held-out dates paired with PD1074 | the same | 2.24 | 0.05 |

pdfr-1's unpaired held-out dates (20180202, 20180212, 20180601) are not used. A mutant's Δ is pooled mutant minus pooled wild type over its paired held-out dates, graded on the mutant's bands at n = its held-out count. The mutants reuse held-out wild-type animals that also form the small-lawn unit, so the units are not independent.

**The allowance.** The positive controls pass when **at most 2** of the 15 units fail. Under a Poisson-binomial of the rates above, P(more than 2 fail) = 0.025, and P(more than 1) = 0.137. If the four nominal units failed at 0.10, it would be 0.049. Which units fail is reported in every case, the small-lawn unit prominently.

## The negatives

Each runs on the three wild-type assays (`scheer_lawn`, `uniform_lawn_scheer`, `food_removal_lopezcruz`, the MLP arm's configs for the hazard negatives), at **n = 200 on animal seeds 5101–5300**.

Matched parameters come from 1,013 band animals on 41 band dates (`negatives_params.json`):

| | Speed (mm/s) | Curvature SD (°/s) | Persistence | Reversal hazard (/s) | Omega hazard (/s) |
|---|---|---|---|---|---|
| dwelling | 0.0220 | 60.8 | -0.059 | 0.0878 | 0.00011 |
| roaming | 0.0671 | 35.1 | -0.005 | 0.0798 | 0.00097 |
| pooled | 0.0267 | 56.9 | -0.054 | 0.0869 | 0.00020 |

The delta hazard is the primitives' minimum (1e-4/s).

1. **The open-loop HMM worm** (`BatchedStateHazards`). The authors' transitions apply every 10 s: dwelling → roaming 0.0212, roaming → dwelling 0.1019, stationary roaming 0.172. Each state uses its own row of the table above. The worm ignores food.
2. **The Poisson worm** (`BatchedCurvatureWalk`, AR(1)): the pooled row, held constant.
3. **The random walk:** the pooled speed, minimum hazards, and each second's curvature uniform over ±50°/s.
4. **The correlated walk:** the pooled speed, minimum hazards, and AR(1) curvature at the pooled SD and persistence. The measured persistence is near zero, so it is close to independent turns. That is recorded as measured, not tuned.
5. **The untrained MLP:** networks seeded 5301–5308, each on 25 animals (5101 + 25j …), pooled.
6. **The untrained connectome:** networks seeded 5309–5316, the same way.

A negative **fails** when it fails the wild type: any family fails, or local search is absent. Each negative's D and D_fit are reported.

## W's minimum

B.3's rule, made exact here:

- **The small lawn.** The 24 held-out OD 2 dates are split into halves by date, 1,000 times (seed 2037). z = (s_A − s_B)/σ, where σ is the small lawn's band scale at the smaller half's size, rounded to the nearest 25 animals.
- **Local search.** In the same splits, the 10 held-out batches are halved, 5 and 5. Each half's mean statistic is compared, with σ = the band batches' SD ÷ √5.
- **The uniform well** has one held-out date and is left out.
- **D** uses the small lawn's and local search's weights, renormalised.
- **W's minimum** is the 95th percentile.

## The verdict

- **VALID**: at most 2 positive-control units fail, **and** all six negatives fail the wild type.
- **VOID**: otherwise. The statistics and bands are then revised in a new change before anything is graded (the roadmap's pivot). Every unit's and negative's reading is recorded either way.

## The pilot (description, band data only)

Seeds 9201–9264 at n = 64, with untrained networks 9265–9280 (8 per arm × 8 animals), graded on the vendored bands. No held-out value was read.

| Negative | Wild type | D | D_fit | Time |
|---|---|---|---|---|
| hmm | fails | 5.59 | 4.58 | 4 s |
| poisson | fails | 7.94 | 8.28 | 4 s |
| random_walk | fails | 7.49 | 7.61 | 4 s |
| correlated_walk | fails | 7.65 | 7.84 | 4 s |
| untrained_mlp | fails | 5.64 | 4.65 | 18 s |
| untrained_connectome | fails | 5.80 | 4.94 | 51 s |

Every negative failed every family. The HMM worm came closest. It matches the wild type's on-lawn roaming fraction (0.115), dwelling bouts and pre-exit enrichment, and fails on staying on the lawn: 18% of windows on it, against 98%, and 13 exits per on-lawn hour.

The pilot took 88 s, so the registered reading takes minutes.
