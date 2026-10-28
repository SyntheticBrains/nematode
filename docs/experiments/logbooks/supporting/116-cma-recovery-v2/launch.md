# Launch record: F.0b-i re-read on B v2, sep-CMA recovers C.1's model under v2's objective

**Part A, registered 2026-10-28**, change `rerun-cma-recovery-v2`. It is committed before the registered preflight, and before any simulation this record scores. Part B, appended after the preflight and the pilot and before the registered fits, adds their outcomes and changes no gate or threshold. A gate registered here is never re-read under a different rule. Anything else is reported beside it as description.

The template is F.0b-i's registration ([launch record 112](../112-cma-recovery/launch.md)). Every rule below is F.0b-i's unless it is marked as v2's.

## Part A

### The model and the simulator

- **The model:** `BatchedMechanistic`, 20 parameters, on the unit cube mapped through `npe_recovery.PRIOR`, as F.0a and F.0b-i.
- **A simulation** (v2's), `cma_recovery_v2.simulate_v2(k, z, base)`:
  - 200 fresh animals at `base + 200·k + i` on B v2's four assays: `scheer_lawn`, `uniform_lawn_scheer`, `food_removal_lopezcruz` and `chemotaxis_chen`;
  - through B v2's measurement models (`data/benchmark/bands_v2.json`, SHA-256 `21f591b8…`), graded on B v2 without presence;
  - the candidates' observations are graded with presence, which their eligibility reads.
- **The speed-up** (change `rerun-cma-recovery-v2`, section 1) is behaviour-identical. On a discarded population (99.35 M) every statistic and D came back equal (D 7.0375), at 20.2 s instead of 35.7 s on one core.

### The objective (v2's)

For truth j at z, f(z) = `predictive_distance_v2(observation_j, simulate_v2(z))`. That is the weighted RMS of the capped z (cap 10) over B v2's **19 graded statistics**, on the n = 200 band scales, with the five families equal and split equally within each:

- **the small lawn and the uniform well:** v1's bands at 200;
- **kinematics:** the bands at 200;
- **local search and chemotaxis:** their size-independent bands.

A statistic undefined in both simulations counts 0, and one undefined in only one counts the cap. The on-lawn reversal rate, which B v2 reports and does not grade, takes no part.

### sep-CMA

- **Settings:** `cma.CMAEvolutionStrategy([0.5]·20, 0.2, {CMA_diagonal: True, bounds: [0, 1], popsize: 16, seed})`. The loop never consults `es.stop()`. A fit's answer is `es.result.xfavorite` after its last generation.
- **Seeds:** fit f = 2·j + r, for truth j = 0–3 and restart r = 0–1, uses CMA seed **5420 + f**. The pilot uses **5428**.
- **The plateau rule, guarded** (v2's): generations are counted from 1, m_n is generation n's median objective, and S_n is the mean of m over generations n − 19 to n. A fit stops at the first n ≥ 40 with S_n > S\_{n−20} − 0.05 **and σ ≤ σ₀ (0.2)**, or at the cap. C.1b will fit under the same rule.

### Truths (re-selected on v2)

- **The pool:** the 43 F.0a prior simulations F.0b-i's rule admits (every v1 statistic defined, local-search presence bound above 1.5, v1 D ≤ 5, simulation 8,207 excluded). Each is simulated once on v2 with presence, and that simulation is its v2 observation.
- **Eligible:** every v2 graded statistic defined, local-search presence bound above 1.5, and chemotaxis present, all on that observation. The eligible are ordered by ascending v2 D to the bands.
- **The preflight:** in that order and in batches of 8, each candidate is re-simulated 20 times (repeats 1–20).
  - **floor:** the median distance from its v2 observation to repeats 1–10;
  - **oracle:** the median distance to repeats 11–20;
  - **prior:** the median distance to F.0a's training simulations 0–200, each simulated once on v2, other than itself (200 of them).
- **Kept** when no repeat 1–10 changes any graded statistic's definedness from the observation, and prior − floor ≥ 1.0. Skipped candidates are recorded with the reason.
- **Order:** the first kept candidate is the pilot truth, and the next four are truths 0–3. If the pool runs out first, the run returns to the maintainer.

### The preflight's gates

- **The oracle:** over the four truths, the median of (prior − oracle) / (prior − floor) is at least **0.85**.
- **The unfitted start:** the box centre is simulated 10 times and scored against each truth by its median distance. Its median gain is at most **0.25**.

If either fails, no registered fit runs, and the record returns to the maintainer.

### The pilot

- **The run:** one fit on the pilot truth, CMA seed 5428, for 300 generations, ignoring the plateau rule.
- **What it records:**
  - G_p, the first generation at which the **guarded** rule holds;
  - the sustained seconds per generation, the median over generations 101–300;
  - the answer's gain at G_p, at ⌈1.5·G_p⌉ and at 300, each from 10 fresh re-simulations.
- **The cap:** min(300, ⌈1.5·G_p⌉). It is 300 if the guarded rule never holds, and 300 if the gain at ⌈1.5·G_p⌉ is more than 0.05 below the gain at 300.
- **Stops:**
  - if the gain at 300 is below **0.75**, no registered fit runs;
  - if the projected registered time exceeds **30 h** (v2's; v1's was 20 h, and the maintainer accepted about 22–25 h at planning), the maintainer decides before launch.

### The gate

- **The best restart:** for each truth, the restart with the lower S over its last 20 generations.
- **Scoring:** its answer is re-simulated 10 times, D is the median distance from the truth's observation to them, and gain = (prior − D) / (prior − floor).
- **The gate:** **the median of the four truths' gains, the mean of the middle two, is at least 0.75.**

**Reported, not gated:**

- the other restart's gain;
- the restarts' spread;
- each fit's stop and whether it plateaued;
- per-parameter |answer − truth| on the unit cube;
- each generation's median and σ.

**The verdict map:**

- **pass:** F.0 stays cleared for C.1 on v2, and C.1b proceeds;
- **fail:** C.1b does not run, and the failing truths' diagnosis (stuck, slow, or unreachable) goes to the maintainer.

### Seeds

Animals are `base + 200·k + i`, for i < 200.

| Base | k | Use |
|---|---|---|
| 170,000,000 | 10,000·f + 16·(n − 1) + p | fit f's generation n, member p |
| 170,000,000 | 10,000·f + 5,000 … 5,009 | fit f's answer, re-simulated |
| 200,000,000 | 16·(n − 1) + p; 5,000, 5,100, 5,200 (+ 0–9) | the pilot's evaluations and answers |
| 210,000,000 | c (c < 43) | candidate c's v2 observation |
| 210,000,000 | 1,000 + 20·c + q | candidate c's repeat q + 1 |
| 210,000,000 | 2,000 + i (i ≤ 200) | the prior's references |
| 210,000,000 | 2,300 … 2,309 | the box centre |

The ranges are 170.0–185.0 M, 200.0–201.05 M and 210.0–210.46 M. They are disjoint from each other, from every band, and from every earlier scheme:

- B v2's 140–160.9 M and 199 M;
- C.1's 110–126 M;
- F.0b-i's 50–65 M and 80 M;
- F.0a's 1–3 M;
- the region check's 20–28 M.

**Discarded:**

- the smoke at 99,300,000, CMA seed 5993;
- the review's pool probe and the gates' probe at 99,400,000, on the candidate scheme;
- the speed-up's parity population at 99,350,000.

### Disclosures

Seen before this record was committed, all on discarded seeds:

- **The pool's eligibility** (the spec review's probe, 99.4 M):

  - 22 of the 43 candidates were eligible on one v2 simulation each;
  - seventeen lacked chemotaxis presence, four of them also local-search presence, and three had an undefined statistic;
  - the eligible candidates' v2 D ran 4.66–6.84.

- **The gates' probe** (`build/cma-recovery-v2/gate_probe.json`, 99.4 M), the preflight run whole on discarded seeds:

  - 22 eligible, and the first five eligible all kept: simulations 1164, 4909, 1354, 3525 and 4709;
  - **the oracle's median gain 1.027**, against at least 0.85;
  - **the unfitted start's 0.047**, against at most 0.25;
  - both are far from their bars, so the change did not return to the maintainer.

  The registered preflight draws fresh observations and repeats, so its truths may differ.

- **The smoke:** about 40 s per generation of 16 on 16 workers. At that pace the pilot takes about 3.3 h, and the registered fits at most about 19 h at a cap of 210.

### The guard

`scripts/analysis/cma_recovery_v2.py`:

- `preflight` and `pilot` refuse unless this record is committed and unchanged;
- `run` and `read` also require Part B, with `preflight.json` and `pilot.json` byte-identical to their versions in the commit that last touched this record.

## Part B

*Appended after the preflight and the pilot, before the registered fits.*
