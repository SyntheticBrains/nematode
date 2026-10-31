# 117: Phase 10 — F.0b-i on B v2, Attempt 2: Ranking Definedness First, sep-CMA Recovers C.1's Model and the Gate Passes

**Status**: completed. **The gate passes**: the median best-restart gain is **0.873**, against a bar of 0.75. By the registered verdict map, F.0 is cleared for C.1 on v2, and C.1b proceeds, fitting with this ranking score.

The only change to the objective from attempt 1 ([Logbook 116](116-cma-recovery-v2.md)) is how a fit ranks its candidates: definedness first, then a compressed distance. Every gate and gain is still read on the registered, capped D. After the probe, the maintainer also raised the restarts from 2 to 4 per truth.

> | Truth | Floor | Prior D | Best restart's gain | Every restart's gain | Better of the first two | Stops (generations) |
> |---|---|---|---|---|---|---|
> | 1164 | 2.29 | 6.98 | **0.773** | 0.675, 0.676, 0.773, 0.627 | 0.676 | 132, 208, 170, 127 |
> | 1354 | 0.89 | 7.04 | **0.968** | 0.784, 0.968, 0.613, 0.856 | 0.968 | 95, 171, 47, 118 |
> | 6985 | 2.95 | 7.05 | **0.778** | 0.746, 0.360, 0.778, 0.607 | 0.360 | 52, 102, 153, 165 |
> | 9105 | 0.93 | 7.16 | **1.040** | 1.000, 1.040, 0.986, 1.000 | 1.040 | 145, 104, 145, 135 |
>
> **Median gain 0.873** (the mean of the middle two) ≥ 0.75. **Passes.**

**Date**: 2026-10-29 to 2026-10-31.

**OpenSpec change**: `rerun-cma-recovery-v2b`.

**Records** (in [`supporting/117-cma-recovery-v2b/`](supporting/117-cma-recovery-v2b/)):

- `launch.md`: Part A, with the probe and the maintainer's decisions disclosed;
- `probe.json` and `probe_restart.json`: the probe on discarded seeds;
- `probe_diagnosis.json`: the probe's per-statistic decompositions;
- `recovery.json`: every fit's trajectory, stop, answer, gain and parameter errors.

The simulations and the fits' state are kept off-repo (`build/cma-recovery-v2/attempt2/`). The reused observations (`build/cma-recovery-v2/candidates.jsonl`) are pinned by their SHA-256 in Part A.

## Objective

Attempt 1 failed because of the fit's objective. Capped, it was flat beyond the z cap. Uncapped, a statistic undefined in only one record cost a flat 10, less than a defined statistic far off, so the fits for the two noisy truths escaped into parameters that leave statistics undefined. C.1b may not run until sep-CMA recovers C.1's model under an objective with neither defect.

The maintainer's decisions of 2026-10-29:

- an objective that grows slowly with z and ranks undefined statistics behind defined ones;
- attempt 1's four registered truths and observations, reused;
- the cap of 272, kept;
- a probe on discarded seeds before registration.

## Method

- **The ranking score:** score = 1,000 × u + D_c.
  - u counts the 19 graded statistics undefined in exactly one of the truth and the candidate.
  - D_c is the weighted RMS of 10·asinh(z/10) over the statistics defined in both, on the n = 200 band scales.
  - Fewer undefined statistics always rank ahead, and near the truth each term is z to within 1.5%.
- **What is unchanged:**
  - attempt 1's truths and observations;
  - its σ-guarded plateau rule and cap;
  - its gate. Every gain is read on the capped D, from 10 fresh re-simulations of each answer.
- **The registered fits:** 4 truths × 4 restarts.
  - Seeds: CMA 5430–5445, animals at 230.0–261.0 M.
  - Each truth's best restart is the one with the lowest final objective. The better of the first two restarts is reported beside it, as description comparable with attempt 1.

## Results

### The probe (`probe.json`, `probe_restart.json`; discarded seeds, base 300 M)

| Truth | Restart | Gain |
|---|---|---|
| 1164 | 1st (CMA 5995) | **0.787** |
| 6985 | 1st (CMA 5996) | **0.981** (attempt 1: 0.056) |
| 9105 | 1st (CMA 5997) | **1.000** |
| 1164 | 2nd (CMA 5998) | 0.523 |

- **The escape is fixed.** No probe answer left a statistic undefined.
- **1164 fell short of the probe's 0.80 bar,** so the change returned to the maintainer. A second restart settled in a different wrong basin; the gate's selection kept the first.
- **The diagnosis** (`probe_diagnosis.json`): 1164's first restart sat 20.8 and 9.7 band scales off on the roaming and dwelling speed medians, where its own repeats sit at 0.22 and 0.16.
- **Ruled out as the cause:** the floor (6985, the highest at 2.95, recovered) and the box's bounds (every truth has 1–3 parameters within 0.05 of one).

**The maintainer's decisions** (2026-10-30), registered in Part A:

- 1164 accepted below the probe's bar;
- 4 restarts per truth instead of 2.

### The registered fits (`recovery.json`)

Every fit stopped on the guarded plateau, between generations 47 and 208, well under the cap of 272. The 2,069 generations took 25.4 h at 44.2 s each, from 2026-10-30 15:00 to 2026-10-31 16:32. The projection was about 20 h, because the fits ran longer than assumed: 129 generations on average, against 110.

- **All four truths recover.** Their best restarts gain 0.773, 0.968, 0.778 and 1.040, so every truth clears 0.75.
- **The two truths attempt 1 failed now pass.** 1164 rose from 0.466 to 0.773, and 6985 from 0.056 to 0.778.
- **Selection by final objective picked the highest-gain restart for every truth.**
- **Parameters are not recovered,** as in F.0b-i on v1. The best restarts' median parameter errors are 0.13–0.23 on the unit cube. Behaviour is recovered, not the generating parameters.

### Each change's share (description)

- **The ranking score:** with 2 restarts alone, the median gain would have been **0.822**, also above 0.75, against attempt 1's 0.690.
- **The two extra restarts** rescued 6985, whose first two restarts reached at best 0.360. They also lifted 1164 from 0.676 to 0.773.
- **The per-restart success rate:** 9 of the 16 restarts reach 0.75.

### A false plateau, for C.1b

Two restarts stopped early, fit 6 (1354, generation 47) and fit 8 (6985, generation 52). Their windows held medians with an undefined statistic, which cost 1,000 each:

| Fit | Undefined medians, last 20 generations | Undefined medians, 20 before that | S_n | S\_{n−20} |
|---|---|---|---|---|
| 6 | 3 | 3 | 255.7 | 208.1 |
| 8 | 2 | 1 | 157.0 | 109.0 |

The plateau rule fires when S fails to fall by 0.05, so a rise in S caused by the 1,000-point terms reads as a plateau.

- **Effect on the gate:** none. Selection ranked both restarts behind the truth's others, and the gate read the others.
- **What C.1b's plan has to add:** a guard against a window that still holds an undefined median, so that a restart is not stopped while it is still bringing statistics into definition.

## Corrections made in the open

- **The probe's bar was not met,** and the maintainer's acceptance of 1164 below it is registered and disclosed in Part A. The registered gate was not changed.
- **The restart count changed after the probe,** on fresh seeds and before any registered fit. The better-of-first-two figure (0.822) is reported so the objective's effect can be read alone.
- **The cost projection was low:** 25.4 h, against about 20 h.
- **The amendment's spec review** found three defects, all fixed before registration:
  - a scenario under the wrong requirement;
  - attempt 1's uncapped-ranking requirement left standing beside its replacement;
  - a design claim the amendment had made false.

## What this establishes, and what it does not

**It establishes** that sep-CMA, ranking definedness first and then a compressed distance, recovers C.1's model under B v2's objective:

- on four registered truths, including the two noisy ones that broke attempt 1;
- with four restarts from the box centre and selection by final objective.

**It does not establish** that parameters are identifiable. It also does not establish that two restarts suffice: 6985's first two reached at best 0.360. C.1b fits with 8.

## Registered consequences

- **The gate (median gain ≥ 0.75):** *met* (0.873).
- **F.0 for C.1 on v2:** *cleared*. C.1b proceeds and fits with this ranking score.
- **For C.1b's plan** (tracker § C):
  - the plateau guard for undefined medians;
  - presence handled, since its wild-type pass needs it and the ranking score leaves it out;
  - restarts chosen on the registered grade;
  - a reachability pilot on the real bands before registration, since both of 1164's probe misses were on the kinematics speed medians.
- **F.0b-ii,** C.2's encoder recovery, is still open.

## Artefacts

- `scripts/analysis/cma_recovery_v2.py`:
  - `ranking_score` and `Objective(ranking=True)`;
  - the pinned observations (`candidates_pinned`);
  - the stages `probe-b`, `diagnose-b`, `probe-b2`, `run-b` and `read-b`;
  - `_run_fits` and `_read_fits` taking a restart count. Attempt 1's `recovery.json` re-reads byte-identical.
- **Compute,** measured while the fits ran (description):
  - the machine's High Power mode is 1.35× faster with identical records;
  - thread pinning gives nothing;
  - 16 workers deliver about 6 cores' worth of work.
