# 119: Phase 10 — C.1b, the Mechanistic Baseline Refitted on Benchmark v2: It Fails the Wild Type on One Steering Gain

**Status**: completed. **C.1 fails the wild type on benchmark v2**: the chosen restart passes 0 of its 5 registered populations, against the 3 required. By the registered verdict map, the failing statistics are the diagnosis, and the roadmap's risk row goes to the maintainer.

The chosen restart nearly reaches the wild type. Its D to the bands is 1.71–1.87, and in two populations every one of the 19 statistics passes. What fails in all five is **chemotaxis presence**: it drifts down the butanone gradient, not up it.

Across all 8 restarts and both pilots, the weathervane gain is negative in 9 of 10 answers, and the one positive answer sits at the box's bound and fails the lawn and kinematics families instead. C.1 has one steering gain for every odour, and the lawn assays and the chemotaxis assay pull it in opposite directions.

> | Restart | Stop (generation) | Registered populations passing | D (5 populations) | Weathervane gain κ (°/s) | Chemotaxis present |
> |---|---|---|---|---|---|
> | 0 | plateau, 133 | 0 | 2.49–2.77 | −14.4 | 5 of 5 |
> | 1 | plateau, 135 | 0 | 3.17–3.50 | −24.2 | 0 of 5 |
> | **2 (chosen)** | plateau, 180 | **0** | **1.71–1.87** | −30.3 | **0 of 5** |
> | 3 | plateau, 119 | 0 | 3.23–3.86 | −24.7 | 5 of 5 |
> | 4 | cap, 248 | 0 | 3.61–4.80 | **+49.9** (bound 50) | 5 of 5 |
> | 5 | plateau, 236 | 0 | 1.90–2.66 | −25.9 | 0 of 5 |
> | 6 | plateau, 123 | 0 | 2.42–2.74 | −24.9 | 0 of 5 |
> | 7 | cap, 248 | 0 | 4.03–4.28 | −18.3 | 5 of 5 |

**Date**: 2026-10-31 to 2026-11-01.

**OpenSpec change**: `fit-c1-v2`.

**Records** (in [`supporting/119-c1b-fit/`](supporting/119-c1b-fit/)):

- `launch.md`: Part A, and Part B with the maintainer's decision;
- `pilot.json`;
- `probe_wide.json`;
- `fit.json`: the verdict, every restart, and the chosen answer's grades and parameters;
- `diagnosis.json`: each restart's κ, chemotaxis readings and failing statistics;
- `identity.json`: the window restriction's check.

## Objective

**Why a refit.** C.1's v1 answer fails benchmark v2 on every population ([Logbook 115](115-benchmark-v2.md)). F.0b-i on v2 cleared sep-CMA for C.1's model, using the definedness-first ranking score ([Logbook 117](117-cma-recovery-v2b.md)).

**What C.1b does.** It refits C.1's 20-parameter model to the real wild type on v2, under rules registered before any fit. Two of those rules are new, both decided by the maintainer on 2026-10-31:

- restarts are selected on populations kept apart from the verdict's;
- a 300-generation reachability pilot runs first.

## Method

- **The objective:** the ranking score against the real bands at n = 200. It is 1,000 × (statistics left undefined) + the weighted RMS of 10·asinh(z/10).
- **The fit:** sep-CMA from the box centre, 8 restarts on CMA seeds 5749–5756.
- **The stop rule:** the plateau rule, guarded by σ ≤ 0.2 and by a 40-generation window free of undefined medians.
- **Selection:** every plateaued restart is graded with presence on 3 selection populations. The restart passing most is chosen, with ties broken by the lower median D.
- **The verdict:** the chosen restart must pass at least 3 of 5 fresh registered populations, graded with presence, with every statistic at |z| ≤ 3.0078 and local search and chemotaxis present. v2's oracle for a faithful controller is 0.955.
- **Exactness:** the change also builds and measures only each family's analysis window. That is byte-identical: all 55 of the identity check's jobs match a baseline from `main`. One simulation is about 3–4% faster.

## Results

### The pilot, and the widened-box probe (both discarded seeds)

- **The pilot** (`pilot.json`):
  - G_p was 165, so the cap is 248, and the projection 17.7 h.
  - Its answer passed **0 of 5** populations. It failed the uniform well's roaming bouts (z ≈ −5), the dwelling speed median (+4.3), the weathervane slope (+4.2) and pre-exit roaming (−3.3).
  - `edge` and `a_intake` were pinned at F.0a's upper bounds.
- **The widened-box probe** (`probe_wide.json`; the maintainer's decision): the same pilot with those two upper bounds raised from 5 to 10.
  - Both parameters settled inside the wider box.
  - Its answer still passed 0 of 5 populations, failing the same statistics, with D 2.79.
  - **The box was not the limit.**
- **The maintainer's decision:** accept the reachability fail and run as registered, on F.0a's box (Part B).

### The registered fits (`fit.json`)

- **Run:** the 1,422 generations took 11.9 h at 30.2 s each, from 01:22 to 14:17 including grading.
- **Stops:** six restarts plateaued (generations 119–236), and two stopped at the cap and could not be chosen.
- **Selection:** no plateaued restart passed any of its 3 selection populations. So selection went to the lowest median D: **restart 2**, at 1.79.
- **The verdict: 0 of 5 registered populations pass.** Restart 2's diagnosis:
  - **chemotaxis presence fails in all 5.** The drift is about −0.021 mm/s (z −2.9 to −3.1), and the log quintile ratio −0.14 to −0.26;
  - pre-exit roaming fails in population 0 (z −3.12), and exits per on-lawn hour in populations 3 and 4 (z −3.16);
  - in populations 1 and 2, **every one of the 19 statistics passes**, and only the presence test fails.
- **Across all restarts:** none passes any of the 40 registered populations. The restart spread is a median D of 2.87 (1.92–3.44).

### Why: one steering gain for two odours (`diagnosis.json`; description)

C.1 steers by one weathervane gain κ, applied to the odour it senses. On the lawn assays that odour is the lawn's; in Chen's assay it is butanone.

- **Negative κ:** seven of the 8 restarts, and both pilots (−12.9 and −12.8), settle at κ between −12.8 and −30.3, steering *away* from the odour. None of them climbs the butanone gradient as real worms do. Their drift is 0.0001–0.004 mm/s against the bands' 0.026, and in restart 2 it is negative, about −0.021.

  - Restarts 0, 3 and 7 keep chemotaxis present, narrowly, but fail its weathervane slope (z +5.7 to +10.0).
  - Restarts 1, 5 and 6 lose presence as well.
  - Restart 2 fits the slope (z +2.2 to +2.8), but loses the drift and with it presence.

- **The one positive κ:** restart 4 settles at **+49.9**, pinned at the box's upper bound of 50. It reproduces chemotaxis (weathervane z −0.2 to +0.6, drift 0.023–0.025 mm/s, z ≈ −0.1), but fails the kinematics speeds, the uniform well's roaming bouts and local search's γ.

- **The reading:** the lawn assays reward steering away from the lawn's odour, which plausibly brings the worm to the edge to leave at the real rate. The butanone assay needs steering toward the odour, and C.1 has no term that lets one gain do both.

  This matches what the model leaves out: real worms leave lawns partly in response to aversive cues at the lawn (for example oxygen and bacterial metabolites), and their odour responses depend on feeding state. C.1 has neither.

  It is a reading of 10 answers, not a registered test.

## Corrections made in the open

- **The reachability gate failed and was accepted.** The pilot's answer reached 0 of 5 populations. The maintainer accepted that and ran as registered, and Part B records the decision (`REACHABILITY_ACCEPTED`). The verdict and its map were not changed.
- **The widened-box probe was a decision after the pilot,** on discarded seeds. It changed nothing registered.
- **The pilot's diagnosis named the wrong tension.** It pointed at the uniform well's bouts and the dwelling speed, with two parameters pinned. The registered fits show those are reachable: restart 2 passes them in every population. The binding constraint is chemotaxis, through κ's sign.

## What this establishes, and what it does not

**It establishes**, under the registered rule, that C.1's 20-parameter model on F.0a's box does not pass benchmark v2's wild type. It also establishes that the gap is narrow: the best restart passes every statistic in two populations, and fails only chemotaxis presence everywhere. The 10 answers point to one steering gain shared by the lawn odour and butanone.

**It does not establish** that a mechanistic baseline cannot pass v2. A model whose steering depends on the odour or on feeding state was not fitted. Nor does it change any registered reading before it. C.1's v1 pass stands as read on v1.

## Registered consequences

- **The verdict (3 of 5):** *unmet-with-reason*. Chemotaxis presence fails in every population of the chosen restart, through a weathervane gain the lawn assays pull negative.
- **The roadmap's risk row** ("the mechanistic baseline cannot pass; the primitives or the benchmark are wrong; fixed before C.2, or the phase ships B and the diagnosis"): *to the maintainer*, with the diagnosis above. The candidate fix is in C.1's mechanism, not in the benchmark or the primitives.
- **C.4:** C.1 on v2 enters as failing the wild type. The restarts' grades are in `fit.json` for the `unidentified` rule, and no restart passes the wild type.

## Artefacts

- `scripts/analysis/c1b_fit.py`: the stages `pilot`, `probe-wide`, `run`, `select`, `grade`, `read` and `smoke`.
- The window restriction:
  - `BatchedTrajectories.to_behaviour_steps(animal, start, stop)` and `primitive_kinds`;
  - `measurement.measure_window`;
  - `grading.analysis_bounds`.
- `speedup_identity_check.py --build/--out`.
- `cma_recovery_v2.simulate_v2(parameters=...)`.
