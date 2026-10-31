# Launch record: F.0b-i on B v2, attempt 2, the fit ranks definedness first

**Part A, registered 2026-10-30**, change `rerun-cma-recovery-v2b`. It is committed after the probe and before any registered fit. There is no Part B: nothing between the probe and the fits changes a rule. A gate registered here is never re-read under a different rule. Anything else is reported beside it as description.

The template is attempt 1's registration ([launch record 116](../116-cma-recovery-v2/launch.md)). Every rule below is attempt 1's unless it is marked as attempt 2's. Two rules change: the objective a fit ranks on, and the number of restarts per truth (4 instead of 2, decided after the probe).

## Part A

### Why attempt 2

Attempt 1 failed its gate ([Logbook 116](../../116-cma-recovery-v2.md)): the median best-restart gain was 0.690, against at least 0.75. Its diagnosis (`116-cma-recovery-v2/diagnosis.json`) found two defects in the fit's objective, one after the other:

- **capped,** it is flat beyond the z cap of 10, so sep-CMA ranked no far candidate ahead of another;
- **uncapped,** a statistic undefined in only one record cost a flat 10, less than a defined statistic far off. So the fits for truths 1164 and 6985 escaped into parameters that leave exits undefined (8–10 of 10 re-simulations).

### The ranking score (attempt 2's)

For truth j, a fit ranks candidate z on `ranking_score(observation_j, simulate_v2(z))`:

- **score = 1,000 × u + D_c;**
- **u** counts B v2's 19 graded statistics undefined in exactly one of the two records;
- **D_c** is the weighted RMS of 10·asinh(z/10) over the statistics defined in both, with z = Δ / scale uncapped on the n = 200 band scales. The five families are equal and split equally within each, and a statistic undefined in both counts 0.

D_c stays below 1,000 for every physically possible worm: a term reaches 1,000 only at about 10^44 scales. So fewer undefined statistics always rank ahead. Near the truth a term is within 1.5% of z for |z| ≤ 3 (2.957 at 3). It is 8.8 at 10, 18.2 at 30, 30.0 at 100 and 58.3 at 1,700, so it has no flat region.

sep-CMA uses only the ranking, and the plateau rule reads the medians of this score.

**Everything read against a gate stays on the registered, capped D**, as attempt 1 defined it: the floor, the oracle, the prior, every gain and the gate.

### The truths and observations (reused)

- **The truths:** attempt 1's registered four, by its rule: truths 0–3 are simulations 1164, 1354, 6985 and 9105. Their floors, priors and oracle are in `116-cma-recovery-v2/preflight.json`, which is bound.
- **The observations:** the registered v2 observations (base 210,000,000, k = c). They are kept off-repo in `build/cma-recovery-v2/candidates.jsonl`, pinned by **SHA-256 `7728d7f0e7c950be49103adf41cd281a95ad8846011c30ac2b92b6a0ecbc7c2e`**. Every attempt-2 stage refuses if the file differs.

The truths were fixed by attempt 1's registered rule, not chosen after the fact. Reusing them tests the new objective on exactly the cases that broke the old one.

### sep-CMA, the plateau rule and the cap

- **sep-CMA:** attempt 1's settings, from the box centre at σ 0.2, popsize 16.
- **The plateau rule:** attempt 1's guarded rule: stop at the first n ≥ 40 with S_n > S\_{n−20} − 0.05 and σ ≤ 0.2.
- **The cap: 272 generations**, amendment 1's pilot cap (`116-cma-recovery-v2/pilot_amendment1.json`, bound). It was set under the uncapped objective, and it is not re-piloted, for three reasons:
  - near the truth the ranking score matches the uncapped distance to within 1.5%;
  - every attempt-1 fit plateaued well before it (generations 71–179);
  - the guarded plateau stops fits sooner.

### The registered fits (attempt 2's)

- **The fits:** 4 truths × **4 restarts**, fit f = 4j + r for truth j = 0–3 and restart r = 0–3, each answer re-simulated 10 times.
- **Seeds:** CMA seed **5430 + f** (5430–5445), and animals at base **230,000,000** on attempt 1's fit scheme. Generation n's member p is at k = 10,000·f + 16·(n − 1) + p, and the answer's re-simulations at 10,000·f + 5,000 … 5,009. They span 230.0–261.0 M.
- **Cost:** about 20 h, and at most about 49 h if every fit runs to the cap. The probe ran at 54 s per generation. The machine was then switched to its High Power energy mode, and 32 discarded simulations (base 99,500,000) re-timed 1.35× faster (77–81 s against 102–116 s, every record identical), about 40 s per generation, with plateaus near 110 generations.

### The gate and the verdict map (attempt 1's)

- **The best restart:** for each truth, the restart of the four with the lowest S over its last 20 generations, now S of the ranking score.
- **Scoring:** D is the median capped distance from the truth's observation to the answer's 10 re-simulations, and gain = (prior − D) / (prior − floor).
- **The gate:** **the median of the four truths' gains, the mean of the middle two, is at least 0.75.**
- **Reported, not gated:** attempt 1's list, with the runner-up's gain as the other restart's, every restart's gain, and the gain of the better of the first two restarts alone (picked by final objective), which is comparable with attempt 1's two-restart reading.

**The verdict map:**

- **pass:** F.0 is cleared for C.1 on v2, and C.1b proceeds, fitting with this ranking score;
- **fail:** C.1b does not run, and the per-restart gains and each failing truth's per-statistic decomposition go to the maintainer, with more restarts and truths at floor ≤ 1 as the options.

### Seeds

Animals are `base + 200·k + i`, for i < 200.

| Base | k | Use |
|---|---|---|
| 230,000,000 | 10,000·f + 16·(n − 1) + p (f < 16) | fit f's generation n, member p |
| 230,000,000 | 10,000·f + 5,000 … 5,009 | fit f's answer, re-simulated |
| 210,000,000 | c | the reused observations (attempt 1's) |

230.0–261.0 M is disjoint from attempt 1's ranges (170.0–185.0 M, 200.0–201.05 M, 210.0–210.46 M and 220.0–221.05 M), and from every scheme launch record 116 lists. The probe's 300.0–307.0 M is discarded.

### Disclosure: the probe

Seen before this record was committed, on discarded seeds (base 300,000,000; `probe.json`, `probe_restart.json` and `probe_diagnosis.json`, all three bound). Each answer is re-simulated 10 times and read on the capped D.

| Truth | Fit | CMA seed | Stop | Answer D | Floor | Prior D | Gain |
|---|---|---|---|---|---|---|---|
| 1164 | 0 | 5995 | plateau, generation 116 | 3.29 | 2.29 | 6.98 | **0.787** |
| 6985 | 1 | 5996 | plateau, generation 93 | 3.03 | 2.95 | 7.05 | **0.981** |
| 9105 | 2 | 5997 | 80 generations | 0.93 | 0.93 | 7.16 | **1.000** |
| 1164 | 3 | 5998 | plateau, generation 84 | 4.53 | 2.29 | 6.98 | 0.523 |

- **The escape is fixed.** No probe answer leaves a statistic undefined. 6985 rose from attempt 1's 0.056 to 0.981.
- **1164 fell short of the 0.80 bar**, so the probe returned to the maintainer. Its first restart sits 20.8 and 9.7 band scales off on the roaming and dwelling speed medians, where its own repeats sit at 0.22 and 0.16. The maintainer had a second restart fitted (fit 3), which settled in a different wrong basin, off across the small lawn and kinematics. Selected as the gate selects, by final objective (4.06 against 5.00), 1164 reads 0.787.
- **Ruled out:** being near the box's bounds (every truth has 1–3 parameters within 0.05 of one) and the floor (6985, the highest at 2.95, recovered).

**The maintainer's decisions** (2026-10-30), on that evidence:

1. **1164 is accepted below the probe's bar.** The probe was the go/no-go for spending the registered run. The gate is unchanged.
2. **Each truth is fitted with 4 restarts instead of 2.** The shortfall is restart-to-restart variance, and the gate's selection follows the final objective. Four restarts also bring the test closer to the 8 C.1b fits with, and measure the per-restart success rate C.1b's restart count needs.

Both decisions were made after the probe, before any registered fit, on fresh seeds, and change neither the gate nor the verdict map.

### The guard

`scripts/analysis/cma_recovery_v2.py`:

- every attempt-2 stage (`probe-b`, `diagnose-b`, `probe-b2`, `run-b` and `read-b`) refuses unless `candidates.jsonl` matches its SHA-256;
- `run-b` and `read-b` also refuse unless every probed truth reached 0.80 or is one the maintainer accepted below it (`PROBE_B_ACCEPTED`: 1164), reading 1164 as the gate selects between its two probe restarts;
- and unless this record is committed and unchanged, with `116-cma-recovery-v2/preflight.json`, `116-cma-recovery-v2/pilot_amendment1.json`, `probe.json`, `probe_restart.json` and `probe_diagnosis.json` byte-identical to their versions in the commit that last touched it.
