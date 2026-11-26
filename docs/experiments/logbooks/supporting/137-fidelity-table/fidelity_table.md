# The fidelity table (C.4)

Benchmark v4. Every figure is from `fidelity_table.json`.

| | C.1 (v3) | C.2 |
|---|---|---|
| Body | benchmark_grade.ARMS\['mlp'\]: the pooled on-food reversal | benchmark_grade.ARMS\['connectome_c2_following'\]: the speed-following on-food reversal |
| Fitted | the wild type on v3's five families; no mutant | all six families with tph-1, pdfr-1 and tax-4 at half |
| Recovery control | C.1 (v3)'s reduced recovery: gain 0.985, passed | F.0b-ii: gain 0.635, failed |
| Wild type on v4 | 0 of 5 | 5 of 5 (100% of 15) |
| … small_lawn | 3 of 5 | 5 of 5 |
| … kinematics | 5 of 5 | 5 of 5 |
| … activity | 0 of 5 | 5 of 5 |
| … uniform_lawn | 5 of 5 | 5 of 5 |
| … local_search | 5 of 5 | 5 of 5 |
| … chemotaxis | 5 of 5 | 5 of 5 |
| Median D (D_fit) | 2.06 (2.45) | 1.47 (1.63) |
| Median reversal share (real worms 17.3%) | 3.4% | 5.5% |
| tph-1 | predicted: **fail** (r0 0, r4 0, r6 0 of 5); the wild type passes v4 in 0 of 5 | fitted, 0 of 5 |
| pdfr-1 | not encodable | fitted, 5 of 5 |
| tax-4 | predicted: **fail** (r0 0, r4 0, r6 0 of 5); the wild type passes v4 in 0 of 5 | fitted, 5 of 5 |
| tph-1's leaving, Δ exits per on-lawn hour (reported) | median -0.40 | median +0.03 |
| Held-out dates | not read | N2_unread: median 2.71, against real halves' 95th percentile 2.60 and the band reference's 2.28; PD1074_held_out: median 2.00, against real halves' 95th percentile 1.37 and the band reference's 1.08 |

tph-1's leaving is reported against its band centre (+0.07 per on-lawn hour) and B.4's single held-out reading (+4.04).

C.3: not run in Phase 10.

## The negatives (B v4's controls)

| Negative | Fails the wild type in | Median D |
|---|---|---|
| hmm | 5 of 5 | 5.20 |
| poisson | 5 of 5 | 6.86 |
| random_walk | 5 of 5 | 7.58 |
| correlated_walk | 5 of 5 | 7.72 |
| untrained_mlp | 5 of 5 | 5.85 |
| untrained_connectome | 5 of 5 | 4.30 |
| blind_v2 | 5 of 5 | 4.70 |
| c1_v1 | 5 of 5 | 6.46 |

## The real-worm controls (B v4)

- N2's 26 unread dates, on activity in 2 halves: pass.
- PD1074's 24 held-out dates, on activity: pause_share passes, flip_fraction passes, onset_rate_per_min passes.
- W's minimum, the distance between two halves of real dates (95th percentile): 2.60 on the fitted families, 2.05 on local search and chemotaxis.

## C.2 against C.1 (description)

- C.2's recovery control, F.0b-ii, failed its gate.
- the bodies differ: C.2 on the speed-following on-food reversal, C.1 (v3) on the pooled.
- C.2's mutants are fitted and in-sample; C.1's are predicted.
- each is graded on its own registered populations, so the populations are not paired.
