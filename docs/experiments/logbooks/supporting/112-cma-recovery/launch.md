# Launch record: F.0b-i, sep-CMA recovers C.1's model near the wild type

**Part A, registered 2026-10-25**, change `add-cma-recovery`. It is committed before the preflight, and before any simulation this record scores. A gate registered here is never re-read under a different rule. Anything else is reported beside it as description. Part B, appended after the preflight and the pilot and before the registered fits, adds their outcomes and changes no gate or threshold.

## Part A

### The model, the box and the simulator

These are F.0a's, unchanged ([Logbook 111](../../111-npe-recovery.md)):

- **the model:** `BatchedMechanistic`, 20 parameters, on the unit cube mapped through `npe_recovery.PRIOR`;
- **a simulation:** 200 fresh animals on `scheer_lawn`, `uniform_lawn_scheer` and `food_removal_lopezcruz`, graded by `benchmark_grade`.

### The objective

For truth j at z, f(z) = `npe_recovery.predictive_distance(observation_j, simulate(z))`. That is D over the benchmark's 11 statistics on the bands' n = 200 scales, capped at 10. A statistic undefined in both simulations counts 0, and one undefined in only one counts the cap. Local search is included, as D37 requires for C.1.

### sep-CMA

- **Settings:** `cma.CMAEvolutionStrategy([0.5]·20, 0.2, {CMA_diagonal: True, bounds: [0, 1], popsize: 16, seed})`.
- **Termination:** the loop never consults `es.stop()`.
- **The answer:** a fit's answer is `es.result.xfavorite`, the distribution's mean mapped into the box, after its last generation.
- **Seeds:** fit f = 2·j + r, for truth j = 0–3 and restart r = 0–1, uses CMA seed 5410 + f. The pilot uses 5418.
- **The plateau rule:** generations are counted from 1, and m_n is generation n's median objective. S_n is the mean of m over generations n − 19 to n. A fit stops at the first n ≥ 40 with S_n > S\_{n−20} − 0.05, or at the cap.

### Truths

- **Candidates:** F.0a's prior simulations 0–9,999 with every statistic defined, a presence bound above 1.5 and D ≤ 5 to the bands, in ascending D. Simulation 8,207, the planning probe's truth, is excluded.
- **Preflight:** candidates are taken in batches of 8. Each is re-simulated 20 times (repeats 1–20).
- **Per candidate:**
  - floor = the median D from its observation, its own F.0a simulation, to repeats 1–10;
  - oracle = the median D to repeats 11–20;
  - prior = the median D to F.0a's first 200 training simulations other than itself.
- **Kept** when none of repeats 1–10 changes any statistic's definedness from the observation, and prior − floor ≥ 1.0. Skipped candidates are recorded with the reason.
- **Order:** the first kept candidate is the pilot truth, and the next four are the registered truths 0–3.
- **Stop:** if the pool runs out first, the run returns to the maintainer.

### The preflight's gates

- **The oracle:** over the four registered truths, the median of (prior − oracle) / (prior − floor) is at least **0.85**.
- **The unfitted start:** the box centre is simulated 10 times and scored against each truth by its median D, the same way. Its median gain is at most **0.25**.

If either fails, no registered fit runs, and the record returns to the maintainer.

### The pilot

- **The run:** one fit on the pilot truth, CMA seed 5418, for 300 generations, ignoring the plateau rule.
- **What it records:**
  - G_p, the first generation at which the rule holds;
  - the sustained seconds per generation, the median over generations 101–300;
  - the answer's gain at G_p, at ⌈1.5·G_p⌉ and at 300, each from 10 fresh re-simulations.
- **The cap:**
  - min(300, ⌈1.5·G_p⌉);
  - 300 if the rule never holds;
  - 300 if the answer's gain at ⌈1.5·G_p⌉ is more than 0.05 below its gain at 300.
- **Stops:**
  - if the gain at 300 is below **0.75**, no registered fit runs;
  - if the projected registered time exceeds 20 h, the maintainer decides before launch.

### The gate

- **The best restart:** for each truth, it is the restart with the lower S over its last 20 generations.
- **Scoring:** its answer is re-simulated 10 times; D is the median distance from the truth's observation to those re-simulations, and gain = (prior − D) / (prior − floor).
- **The gate:** the median of the four truths' gains, the mean of the middle two, is at least **0.75**.

**Reported, not gated:**

- the other restart's gain;
- the restarts' spread, the median D between their answers' re-simulations, paired by repeat;
- each fit's stop generation and whether it plateaued;
- per-parameter |answer − truth| on the unit cube, against the box centre's expected 0.25;
- each generation's median objective and σ.

**F.0 clears per model** (the maintainer's decision). If this gate passes, C.1 may be fitted to real worms. C.2 waits for F.0b-ii.

### Seeds

Animals are `50_000_000 + 200·k + i`, for i < 200:

| k | Use |
|---|---|
| 10,000·f + 16·(n − 1) + p | fit f's generation n, member p |
| 10,000·f + 5,000 … 5,009 | fit f's answer, re-simulated |
| 100,000 + 20·c + q | candidate c's repeat q + 1 |
| 101,000 … 101,009 | the box centre |

- **The pilot:** base `80_000_000`. Its evaluations are at k = 16·(n − 1) + p, and its answers, in generation order, at 5,000, 5,100 and 5,200 (+ 0–9).
- **The smoke:** CMA seed 5991, base `99_000_000`, discarded.
- **The planning probe:** bases `32_000_000` (k < 480) and `33_000_000` (k < 10), on training simulation 8,207. It used CMA seed 9311.

All of these are disjoint from every band and from F.0a's and the region check's schemes.

### Why `gate_preflight.py` does not apply

There are no panel levels. The protocol's principle is applied by hand, through the oracle and unfitted-start gates above and the pilot's own gain (principles 4 and 6, as added 2026-10-25).

### The read

`scripts/analysis/cma_recovery.py`:

- `preflight` and `pilot` refuse unless this record is committed and unchanged;
- `run` and `read` also require Part B, and require `preflight.json` and `pilot.json` to be byte-identical to their versions in the commit that last touched this record.

### Cost, before the pilot

The planning probe ran 30 generations of 16 at a sustained 27 s per generation on 16 workers. That puts the pilot at about 2.3 h and the registered fits at up to about 18 h at a cap of 300. The pilot measures both.

## Part B

**Appended 2026-10-25**, after the preflight and the pilot, and before any registered fit. It changes no gate or threshold. `preflight.json` (committed in `91f34a11`) and `pilot.json` are bound to this commit.

### The preflight (passed)

The first batch of eight candidates was kept whole: no repeat flipped a statistic, and every candidate's gain was reachable. The pilot truth is simulation 4909 (floor 0.32, prior D 7.05).

| Truth | Simulation | Floor | Oracle D | Prior D | Oracle gain | Box-centre gain |
|---|---|---|---|---|---|---|
| 0 | 3004 | 0.34 | 0.33 | 6.91 | 1.002 | 0.049 |
| 1 | 5340 | 0.71 | 0.68 | 7.13 | 1.004 | -0.060 |
| 2 | 9105 | 0.25 | 0.30 | 6.82 | 0.991 | 0.055 |
| 3 | 2324 | 0.78 | 0.83 | 6.99 | 0.991 | 0.015 |

- **The oracle's median gain** is 0.996, against at least 0.85.
- **The unfitted start's** is 0.032, against at most 0.25.

### The pilot

| Measure | Value |
|---|---|
| G_p, the first generation the plateau rule holds | 140 |
| gain at G_p | 0.893 |
| gain at ⌈1.5·G_p⌉ = 210 | 0.909 |
| gain at 300 | 0.921 |
| sustained seconds per generation | 30.3 |

The gain after ⌈1.5·G_p⌉ is 0.013, within 0.05, so there is no fallback. The gain at 300 clears the 0.75 stop.

- **The registered cap is 210 generations.**
- **Projected registered time:** at most 14.2 h, which is 8 fits × (210 generations plus about one more for re-simulating the answer) at the sustained rate, under the 20 h threshold. Fits that plateau stop sooner.
