# 116: Phase 10 — F.0b-i on B v2: sep-CMA Recovers Two of Four Truths, and the Gate Fails on the Fit's Objective

**Status**: completed. **The gate fails**: the median best-restart gain is 0.690, against a bar of 0.75. By the registered verdict map, C.1b does not run. sep-CMA recovered the two truths whose own statistics are stable on B v2's scales (gains 0.914 and 1.007). It did not recover the two that are noisy and far from the wild type (0.466 and 0.056).

The failure belongs to the fit's objective, and two defects of it surfaced, one after the other:

1. **Part A's capped distance** gave sep-CMA no signal on statistics beyond the z cap: the registered pilot stopped the run at gain 0.511.
2. **Amendment 1's uncapped distance** counted an undefined statistic at a flat 10. That was less than a defined statistic far from the truth costs, so the fits for the two noisy truths escaped into parameters where whole families are undefined.

A new change fixes the objective and re-runs the recovery before C.1b.

> | Truth | Floor | Prior D | Best restart's gain | Other restart's gain | Stops (generations) |
> |---|---|---|---|---|---|
> | 1164 | 2.29 | 6.98 | 0.466 | 0.091 | 179, 75 |
> | 1354 | 0.89 | 7.04 | 0.914 | 0.818 | 128, 128 |
> | 6985 | 2.95 | 7.05 | 0.056 | 0.084 | 71, 89 |
> | 9105 | 0.93 | 7.16 | 1.007 | 0.997 | 149, 116 |
>
> **Median gain 0.690** (the mean of the middle two) < 0.75. **Fails.**

**Date**: 2026-10-28 to 2026-10-29.

**OpenSpec change**: `rerun-cma-recovery-v2`.

**Records** (in [`supporting/116-cma-recovery-v2/`](supporting/116-cma-recovery-v2/)):

- `launch.md`: Part A, amendment 1 and Part B;
- `preflight.json`;
- `pilot.json`: Part A's pilot, which stopped the run;
- `pilot_amendment1.json`;
- `recovery.json`: every fit's trajectory, stop, answer, gain and parameter errors;
- `diagnosis.json`: the per-statistic decompositions behind both diagnoses.

The simulations and the fits' state are kept off-repo (`build/cma-recovery-v2/`).

## Objective

F.0b-i ([Logbook 112](112-cma-recovery.md)) cleared sep-CMA for C.1's model on B v1, with a median gain of 0.991. B v2 ([Logbook 115](115-benchmark-v2.md)) changed the objective a fit sees:

- 19 statistics in five families;
- simulated tracks passed through a measurement model;
- kinematics bands that are tight in absolute terms.

C.1b refits C.1 on v2, so the method had to be shown to recover known parameters under v2 first.

The maintainer's decisions of 2026-10-28:

- the truths are re-selected on v2 by F.0b-i's rule;
- the cap is set again by a pilot;
- the objective is v2's 19-statistic distance;
- v2's simulation is made faster first, without changing any number.

## Method

- **The speed-up, number for number.** A fit's objective grades without the presence resamples, the noise library stores its stretch offsets once, and measured steps are rebuilt positionally. On a discarded population every statistic and D came back identical, at 1.76× the speed (35.7 s to 20.2 s on one core).
- **The rest is F.0b-i's design:**
  - C.1's 20-parameter model on v2's four assays through the measurement models;
  - truths re-selected on v2 from the 43 F.0a simulations F.0b-i's rule admits;
  - the oracle and unfitted-start gates;
  - a 300-generation pilot setting the cap;
  - 4 truths × 2 restarts of sep-CMA from the box centre;
  - the gate: the median best-restart gain at least 0.75.
- **The plateau stop gained B v2's σ guard**, so the method tested is the one C.1b will use.
- **Before registration,** the gates' probe ran the preflight whole on discarded seeds. The oracle's gain was 1.027 and the unfitted start's 0.047, both far from their bars.

## Results

### The preflight (passed; `preflight.json`)

- 21 of 43 candidates were eligible on their v2 observations.
- The pilot truth is 4909, and the truths are 1164, 1354, 6985 and 9105. Simulation 3525 was skipped because its γ flipped in a repeat.
- The oracle's median gain is 1.009 (≥ 0.85), and the unfitted start's 0.068 (≤ 0.25).

### Part A's pilot stopped the run (`pilot.json`)

On the capped distance, the answer's gain was 0.507 at the guarded plateau (generation 110) and **0.511 at 300**, against the 0.75 stop. The decomposition (`diagnosis.json`, `capped_pilot`) shows two statistics at the cap:

- **roaming speed's median** sat 30.1 band scales from the truth, since kinematics' n = 200 scale is 0.00097 mm/s;
- **γ** was undefined in all 10 re-simulations.

The capped distance is flat over every candidate that far off, so sep-CMA ranked none ahead of another on those statistics.

### Amendment 1: the fit ranks on the uncapped distance

Every quantity read against a gate stayed on the registered, capped D.

- **The probe** on discarded seeds reached a gain of 0.935 in 60 generations.
- **The amended pilot** (`pilot_amendment1.json`) reached 0.982 at its guarded plateau (generation 181), 0.971 at the cap of 272, and 0.980 at 300.
- **The projection** was at most 31.0 h, over the 30 h threshold. The maintainer decided to launch as registered.

### The registered fits (`recovery.json`)

Every fit stopped on the guarded plateau, between generations 71 and 179. The run took about 15 h (2026-10-29, 07:10–22:25), under the 31 h projection.

- **The truths with stable statistics recovered fully.** 1354's best restart gained 0.914 and 9105's 1.007, and both restarts agree (other gains 0.818 and 0.997).
- **The truths with high floors did not:** 1164 reached 0.466 and 6985 0.056.

The decomposition (`diagnosis.json`, `truths`) shows why:

- **Their own statistics are noisy far beyond the bands.** These worms leave the lawn 24–30 times an hour, against the wild type's 0.7. On the wild type's n = 200 scale, their own repeats differ by a median |z| of 13.0 and 15.1 on exits per on-lawn hour, against 1.3 and 0.1 for the two that recovered. That is what puts their floors at 2.29 and 2.95.
- **Uncapped, undefined is cheaper than far.** A statistic undefined in only one record counts a flat 10, while a defined statistic that far off counts much more. So the fits drifted to parameters that leave statistics undefined:
  - 1164's weaker restart (fit 1) leaves exits undefined in 9 of 10 re-simulations;
  - 6985's restarts leave them undefined in 8 and 10 of 10;
  - where 6985's fit 4 stays defined, it sits about 1,700 scales off.

## Corrections made in the open

- **The seed scheme** first put the candidates at base 172 M, inside the fits' range (170–185 M). This was caught while writing the code, and the candidates moved to 210 M and the pilot to 200 M before registration.
- **The pool** was first drafted as "the first 64". The rule admits only 43, which the spec review found.
- **Amendment 1 changed the fit's objective after a registered stop.** The capped pilot's verdict stands, and the amendment was registered, with its probe disclosed, before the re-run pilot and every registered fit.
- **The uncapped objective introduced the loophole that failed the gate.** It is recorded here as the finding, not as a reading to re-interpret: the gate is read as registered.

## What this establishes, and what it does not

**It establishes** that sep-CMA recovers C.1's model under B v2's objective where the truth's statistics are stable on the bands' scales: gains of 0.914 and 1.007, with both restarts agreeing. It also establishes that the objective must neither be flat beyond its cap nor reward leaving a statistic undefined.

**It does not establish** that the method recovers worms whose statistics are noisy on the wild type's scales. The gate is failed, so F.0 is not cleared for C.1 on v2.

**It does not show** that C.1b would fail on real data either. The real wild type's statistics are, by construction, the scale the bands measure. But the registered gate decides, and it failed.

## Registered consequences

- **The gate (median gain ≥ 0.75):** *unmet-with-reason*. The objective allowed undefined statistics to undercut far ones.
- **C.1b on v2:** *deferred-with-destination*, behind a new recovery change. That change ranks on an uncapped z that grows slowly (for example 10·asinh(|z|/10)), makes an undefined statistic cost more than any defined one can, is checked first on the two failed truths with discarded seeds, and re-runs the recovery (tracker § F.0 and § C).
- **The speed-up (1.76×, identical):** *met*. It carries to every later v2 fit.

## Artefacts

- `scripts/analysis/cma_recovery_v2.py`, with the stages `gate-probe`, `objective-probe`, `preflight`, `pilot`, `run`, `read`, `diagnose` and `smoke`.
- `cma_recovery.run_fit` takes an optional simulator and stop rule.
- `benchmark_grade.grade_populations_v2(presence=...)`.
- The measurement model's speed-ups.
