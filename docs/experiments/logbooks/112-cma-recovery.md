# 112: Phase 10 — sep-CMA Recovers C.1's Model Near the Wild Type

**Status**: completed. **F.0b-i passes.** For four worms near the wild type, fitting C.1's 20-parameter model by sep-CMA from the box centre reproduces each worm's statistics about as closely as the worm's own re-simulations do. The median gain is 0.991 against a bar of 0.75. F.0 clears for C.1, so C.1 may now be fitted to real worms. C.2 waits for F.0b-ii.

> | Truth | Floor | Prior D | Best restart's D | Gain | Other restart's gain |
> |---|---|---|---|---|---|
> | 3004 | 0.34 | 6.91 | 0.35 | 0.999 | 1.005 |
> | 5340 | 0.71 | 7.13 | 0.63 | 1.013 | 0.988 |
> | 9105 | 0.25 | 6.82 | 0.36 | 0.983 | 0.867 |
> | 2324 | 0.78 | 6.99 | 1.98 | 0.807 | 0.688 |
>
> **Median gain 0.991** (the mean of the middle two) ≥ 0.75. **Passes.**

**Date**: 2026-10-26.

**OpenSpec change**: `add-cma-recovery`.

**Records** (in [`supporting/112-cma-recovery/`](supporting/112-cma-recovery/)):

- `launch.md`: Part A, committed in `744ef1aa` before the preflight, and Part B, committed in `7014eb53` before the registered fits;
- `preflight.json`: the candidates, the truths, the oracle and the unfitted start;
- `pilot.json`: the pilot fit, its plateau and the cap;
- `recovery.json`: every fit's trajectory, stop, answer, gain and parameter errors.

The fits' state and simulations are kept off-repo (`build/cma-recovery/`).

## Objective

D38, as amended on 2026-10-25 ([Logbook 111](111-npe-recovery.md), addendum), fits C.1 by sep-CMA on the benchmark's distance, like C.2. F.0 requires the method to recover worms it generated itself before it is trusted on real ones. F.0b-i tests that on C.1's model. F.0b-ii tests it on C.2's encoder.

The maintainer's decisions of 2026-10-25:

- F.0b split, with C.1's model first;
- 4 truths × 2 restarts;
- a cap set by a pilot, with a plateau stop;
- each truth's best restart, with a median gain of at least 0.75;
- every fit starting at the box centre with σ₀ 0.2;
- F.0 clearing per model.

## The design

- **Model, box and simulator:** F.0a's. `BatchedMechanistic` runs at 200 fresh animals on the three wild-type assays.
- **The objective:** D over all 11 statistics, local search included (D37), between a candidate's simulation and the truth's.
- **sep-CMA:** population 16, bounds [0, 1]. A fit's answer is its mean mapped into the box (`xfavorite`).
- **The gain fraction:** (prior D − D) / (prior D − floor). It is 1 at the truth's own noise floor and 0 at the prior's typical distance.
- **The tooling:** `scripts/analysis/cma_recovery.py`, with tests of every rule and a two-generation smoke with a resume.

## The preflight (passed)

The first eight candidates in ascending D to the bands were all kept: no stability repeat flipped a statistic, and every gain was reachable. The pilot truth is 4909; the registered truths are 3004, 5340, 9105 and 2324.

| Check | Result | Bar |
|---|---|---|
| the oracle: each truth's repeats 11–20, scored as a fit | median gain 0.996 | ≥ 0.85 |
| the unfitted start: the box centre, re-simulated | median gain 0.032 | ≤ 0.25 |

## The pilot

One fit on truth 4909 ran 300 generations, ignoring the plateau rule.

| Measure | Value |
|---|---|
| first generation the plateau rule holds | 140 |
| gain there | 0.893 |
| gain at 210 (1.5 × 140) | 0.909 |
| gain at 300 | 0.921 |
| sustained seconds per generation | 30.3 |

The cap was set at **210**: only 0.013 was gained after it. The projection was at most 14.2 h.

## The run

All eight fits stopped by the plateau rule, between generations 83 and 184. None reached the cap. They took 10.9 h of generation time, at about 30–34 s per generation.

| Truth | Restart | Stop | Final objective (S) | Answer's D | Gain |
|---|---|---|---|---|---|
| 3004 | 1 | 157 | 0.35 | 0.35 | **0.999** |
| 3004 | 2 | 184 | 0.36 | 0.31 | 1.005 |
| 5340 | 1 | 166 | 0.80 | 0.79 | 0.988 |
| 5340 | 2 | 167 | 0.61 | 0.63 | **1.013** |
| 9105 | 1 | 114 | 1.14 | 0.36 | **0.983** |
| 9105 | 2 | 83 | 1.46 | 1.12 | 0.867 |
| 2324 | 1 | 165 | 2.86 | 2.72 | 0.688 |
| 2324 | 2 | 157 | 2.20 | 1.98 | **0.807** |

Bold marks the restart the gate scored, chosen by its own final objective. In three of the four truths it was also the restart with the higher gain. For 3004 the two differ by 0.006. Every fit gained at least 0.69.

## Beside the gate (description)

1. **Recovery is in statistic space, not parameter space.** Each answer's error per parameter on the unit box, as a median over the eight fits:

   | Parameters | Median error |
   |---|---|
   | search time τ | 0.03 |
   | weathervane κ_w | 0.05 |
   | roaming speed | 0.12 |
   | search amplitude | 0.16 |
   | the other sixteen | 0.18–0.55 (overall median 0.25, the box centre's expected error) |

   The same pair stood out in F.0a's contraction. With 11 statistics for 20 parameters, most parameters trade off along ridges that leave the statistics unchanged. The dwelling-to-roaming intercept a₀ (0.55) is worse than a guess at the centre, so the fits find the statistics by another route along such a ridge. C.1's fitted parameters are therefore read only where they are constrained: weathervane steering and the search time.

2. **Two restarts on the same data differ by up to about W's minimum.** The median D between the two restarts' answers' re-simulations, per truth, is 0.48, 1.13, 1.11 and 1.92 (median 1.12). W's minimum is 1.42. A single fit per arm would leave W reading fitting noise of about its own minimum, so W's draws per arm, sized from F.0 as registered, need several fits and their spread. This is the number F.0 was asked to give W.

3. **The hardest truth was the one with the highest floor, 2324.** Both restarts plateaued well above its floor: D 2.72 and 1.98 against 0.78. Its first restart descended slowly: its 20-generation mean median fell from 3.9 around generation 70 to 2.8–3.0 by its stop at 165. Restarts are the defence, and C.1's real fit should keep at least two.

## Conclusions

1. **F.0b-i passes.** sep-CMA, configured as C.1's real fit will be, reaches the statistics of worms near the wild type. The median gain is 0.991, and every truth's best restart is at least 0.81.
2. **F.0 clears for C.1.** C.1 may be fitted to real worms. C.2 waits for F.0b-ii.
3. **Only weathervane steering and the search time are recovered as parameters.** Everything else is identified only through the statistics it produces.
4. **Restart-to-restart spread is about 1.1 D,** close to W's minimum. W's design has to carry it.

## Next steps

- [ ] F.0b-ii: C.2's encoder, and its sep-CMA recovery, under the same gate design.
- [ ] C.1: the real fit on the band cohorts, with at least two restarts and the registered plateau rule. C.1's observation-size note (simulate at the observation's size, or add between-date noise) applies.
