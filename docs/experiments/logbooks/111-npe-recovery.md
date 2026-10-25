# 111: Phase 10 — NPE Recovery of C.1's Baseline Fails Its Informative Gate, and the Gate Was Out of Reach

**Status**: completed. **F.0a fails as registered.** NPE's posteriors for C.1's mechanistic baseline are calibrated, but they barely leave the prior, and they fail the informative gate.

The diagnosis, read after the gates as description, finds the gate was out of reach. Two simulations of the *same* parameters already differ by more than the bar. NPE is not trusted for C.1 until F.0a-ii fixes this. The maintainer chose sequential NPE with a gate set against the measured noise.

> | Gate (registered) | Flow 5402 | Flow 5403 | Bar |
> |---|---|---|---|
> | Calibrated: pooled 90% coverage | 0.890 ✓ | 0.892 ✓ | in [0.80, 0.97] |
> | Informative: posterior-predictive median D | **5.32** ✗ | **5.35** ✗ | ≤ 1.42 |
> | (beside) prior-predictive median D | 6.79 | 6.79 | — |
> | (beside) coverage near the wild type | 0.914 | 0.910 | — |
>
> **The diagnosis:** the same parameters re-simulated with fresh animals differ by a median **D of 4.08**. That is above the bar, so no posterior could have passed.

**Date**: 2026-10-25.

**OpenSpec change**: `add-npe-recovery`.

**Records** (in [`supporting/111-npe-recovery/`](supporting/111-npe-recovery/)):

- `launch.md`: the registration, committed in `43118284` before the training simulations;
- `pilot.json`: the cost;
- `recovery.json`: both flows, both gates, coverage and contraction per parameter;
- `diagnosis.json`: the same-parameter floor and nearest neighbours.

The 11,300 simulations' statistics are kept off-repo (`build/npe-recovery/`).

## Objective

D38 fits C.1 by neural posterior estimation, and requires a parameter-recovery positive control first (F.0, the plan review's A6). The concern was a wild-type-only fit whose posterior reports the prior. F.0a builds C.1's model and tests whether NPE recovers it from the benchmark's statistics.

The maintainer's decisions of 2026-10-25:

- two changes, NPE first;
- `sbi`;
- calibration and statistic-space gates.

## What was built

- **`BatchedMechanistic`**, C.1's closed-loop baseline with 20 parameters:

  - roaming/dwelling switching driven by intake and satiety;
  - per-state runs with weathervane steering;
  - klinokinetic reversal and omega hazards;
  - an edge reversal on falling food contact;
  - a search term that decays off food.

  Every draw comes from each animal's own generator.

- **`sbi` 0.27**, in the analysis extra. The lock only added packages.

- **`scripts/analysis/npe_recovery.py`**, with `pilot`, `simulate`, `train` and `diagnose`, and tests of the prior, the summary and both gates on toy problems.

## The run

- **The pilot** projected 4.05 h on 16 workers.
- **The run** took 4.9 h for the 10,100 simulations, at a sustained 29–31 s per simulation against the pilot's 20 s. The pilot's 80 seconds did not reach the laptop's sustained, thermally limited clock, so later pilots should run longer.
- **Training and the 1,200 predictive simulations** took 38 min more.
- **Undefined statistics:** 81% of prior draws left at least one statistic undefined.

## The gates

| | Flow 5402 | Flow 5403 |
|---|---|---|
| pooled coverage | 0.890 | 0.892 |
| informative D (median of medians, 20 truths) | 5.32 | 5.35 |
| prior-predictive D | 6.79 | 6.79 |
| coverage, 25 truths nearest the wild type | 0.914 | 0.910 |
| **verdict** | **fails** | **fails** |

**Contraction is small almost everywhere.** The median over parameters is 0.03.

- Only weathervane steering (κ_w, 0.52) is well constrained.
- The base reversal and omega hazards and the search time sit at 0.10–0.17, and the two speeds at 0.09–0.12.
- The switching terms, klinokinesis, the edge term, the turning noise and the search amplitude sit at 0.00–0.04.

Coverage per parameter lies between 0.83 and 0.95.

## The diagnosis (description, after the gates)

1. **The bar was below the simulator's own noise.**
   - The 20 predictive truths were re-simulated with fresh animals (`3_000_000 + 200·k + i`). D between each truth's two simulations has a median of **4.08**, ranging from 0.23 to 6.54.
   - A posterior that returned the exact truth would score about 4.1 against the 1.42 bar.
   - The bar came from W's minimum, set on real wild-type half-populations, which are well-behaved worms. Prior draws are not.
2. **A few discontinuous statistics carry the noise.**
   - One statistic at the z cap of 10 adds 4.08 to D in local search (weight 1/6 each), 3.33 on the uniform well and 2.36 on the small lawn.
   - Many prior worms barely decay after food removal. Their γ flips between undefined and defined on noise alone, and a flip lands on the cap. That is why the floor sits at 4.08–4.15 for many truths even when every statistic is defined in the first simulation.
   - The uniform well's and the small lawn's bout means, with few completed bouts, do the same.
3. **The information is thin across the prior, not only in the flow.**
   - The nearest of the 10,000 training simulations to each truth, in statistic space, lies at a median D of 2.58. The 20 nearest lie at 3.45.
   - Their parameters spread almost as widely as the prior, weathervane aside.
   - So the summary, at 200 animals, separates few parameters across this prior.
4. **What the posterior did recover.** Its predictive D of 5.3 sits between the prior's 6.8 and the floor's 4.1: about half the reachable gain. It is calibrated everywhere, and near the wild type too.

**The lesson.** The informative gate was added in this change's own spec review, without being evaluated at its registered point. Re-simulating 20 truths, about two minutes, would have shown it was out of reach before a five-hour run. The phase protocol's gate-preflight principle covers gates added in review too. **F.0b's planned statistic-space gate has the same flaw,** and is set against its measured floor before it registers (the maintainer's decision).

## Conclusions

1. **F.0a fails as registered.** NPE on C.1's baseline is calibrated but weakly informative, and not trusted for C.1 yet.
2. **The informative gate was unreachable.** The simulator's same-parameter noise exceeds it, mostly through local-search and bout statistics flipping on and off the cap for unrealistic prior worms.
3. **Across a wide prior, the benchmark's statistics at 200 animals constrain few of C.1's 20 parameters.** Weathervane steering is the clear exception.

## Next steps

- [ ] F.0a-ii (the maintainer's choice):
  - truncated sequential NPE, with extra simulation rounds concentrated near each truth;
  - truths drawn from the region C.1 is fitted in;
  - an informative gate set against each truth's measured same-parameter floor, preflighted on pilot truths before launch.
- [ ] F.0b: sep-CMA recovery on C.2's encoder, its statistic-space gate set against the measured floor and preflighted.
