## Context

F.0b-i's design ([launch record 112](../../../docs/experiments/logbooks/supporting/112-cma-recovery/launch.md)) is the template. This change re-reads it on B v2, and keeps every rule that v2 does not force to change.

A v2 simulation of C.1's model, 200 animals on four assays, takes 36 s on one core, against v1's 11 s. A profile of one population (about 45 s under the profiler) shows where:

- **chemotaxis presence** (2,000 resamples of the pooled statistics): about 14 s;
- **the measurement model:** about 10 s, of which per-step dataclass copies are about 5 s and recomputing the library's stretch offsets at each splice about 2 s;
- **the batched evaluator:** about 13 s over the four assays, and converting its arrays to step objects about 5 s.

## Goals / Non-Goals

**Goals:**

- recovery evidence for sep-CMA on C.1's model, under the objective C.1b will fit;
- the same gates as F.0b-i, read on v2;
- a faster v2 simulation that changes no number.

**Non-Goals:**

- C.1b itself, the next change;
- F.0b-ii (C.2's encoder);
- changing B v2's statistics, bands or measurement model;
- vectorising the batched evaluator or the grading pipeline on arrays. That is a larger refactor, deferred to C.2's cost pilot, which already owns compute.

## Decisions

1. **The speed-up** (every change is behaviour-identical):

   - **`grade_populations_v2(..., presence=True)`.** With `presence=False`, chemotaxis's and local search's presence tests are skipped, and the record carries `presence: null` and no family passes. Every statistic and D are unchanged. The fit's objective uses `presence=False`. Every registered grading (populations scored against a gate, verdicts) keeps `True`.
   - **`NoiseLibrary`** stores its stretch offsets once, at construction, instead of recomputing them at each splice.
   - **`measure`** builds each measured step with one constructor call instead of `dataclasses.replace`. The values, draw order and stream are unchanged.
   - **Parity tests:**
     - measured steps are equal field for field before and after, on a population drawn with the vendored library;
     - one v2 simulation's statistics, D and D_fit are identical with presence on and off;
     - the B v2 records that can be recomputed cheaply (the reachability gate on one discarded population) come back unchanged.
   - **The expected cost** is about 2× faster. The pilot measures the sustained rate.

2. **The simulator and the objective.**

   - **`simulate_v2(k, z, base)`:**

     - C.1's model at the unit-cube point z, on 200 animals at `base + 200·k + i`;
     - `benchmark_grade.simulate_v2`, through B v2's measurement models;
     - graded by `grade_populations_v2(presence=False)`.

     It returns each of the 19 graded statistics' values and D to the bands. With `presence=True`, used only for the candidates' v2 observations, it also returns the chemotaxis and local-search presence bounds that the truths' eligibility reads. Fits and repeats run without presence.

   - **The objective** for truth j at z is `predictive_distance_v2(observation_j, simulate_v2(z))`:

     - the weighted RMS, with B v2's five families equal and split equally within each, of the capped z between two records' statistics on v2's n = 200 band scales;
     - the band scales are the kinematics bands at 200, the chemotaxis and local-search bands, and v1's small-lawn and uniform-well bands at 200;
     - a statistic undefined in both counts 0, and one undefined in only one counts the cap of 10, as v1.

     The reported on-lawn reversal rate takes no part.

3. **The truths, re-selected on v2** by F.0b-i's rule, with v2's statistics:

   - **The pool:** F.0a's prior simulations 0–9,999 that F.0b-i's rule admits (every v1 statistic defined, local-search presence bound above 1.5, v1 D ≤ 5), in ascending v1 D, simulation 8,207 excluded. The **first 64** are simulated once on v2; that simulation is each candidate's v2 observation.
   - **Eligible on v2:** every v2 statistic defined, local-search presence above 1.5 and chemotaxis present. The eligible candidates are ordered by ascending v2 D to the bands. There is no v2 D threshold, since the order already takes the nearest.
   - **The preflight:** in that order and in batches of 8, each candidate is re-simulated 20 times on v2:
     - floor = the median distance from its v2 observation to repeats 1–10;
     - oracle = the median distance to repeats 11–20;
     - prior = the median distance to F.0a's first 200 training simulations, each simulated once on v2 (others than itself).
   - **Kept** on F.0b-i's rules: no repeat 1–10 changes any statistic's definedness, and prior − floor ≥ 1.0.
   - **Order:** the first kept candidate is the pilot truth, and the next four are truths 0–3.
   - **Stop:** if the 64 run out, the run returns to the maintainer.

4. **The gates and the pilot**, F.0b-i's:

   - **the oracle:** the median (prior − oracle) / (prior − floor) over the four truths is at least 0.85;
   - **the unfitted start:** the box centre, simulated 10 times, has a median gain of at most 0.25;
   - **the pilot:**
     - one fit on the pilot truth for 300 generations, ignoring the plateau;
     - the cap is min(300, ⌈1.5·G_p⌉), 300 if the rule never holds, and 300 if the gain at ⌈1.5·G_p⌉ is more than 0.05 below the gain at 300;
     - if the gain at 300 is below 0.75, no registered fit runs;
   - **the projection:** if the registered fits are projected beyond **30 h**, the maintainer decides before launch. v1's threshold was 20 h; the maintainer accepted about 22–25 h for this re-run at planning;
   - **the gate:** each truth's best restart, by the lower mean median objective over its last 20 generations, is re-simulated 10 times. D is the median distance to them, and gain = (prior − D) / (prior − floor). **The median gain over the four truths must be at least 0.75.**

5. **The plateau stop gains B v2's σ guard.** A fit stops on the plateau rule only once σ ≤ σ₀ (0.2), and otherwise at the cap. C.1's v1 fit had two restarts stop at generations 40 and 43 on a false plateau (σ 0.27–0.36), and the tracker requires C.2's registration to guard against it. C.1b will use the guarded rule, so the method validated here is the method C.1b fits with. **The pilot's G_p is read with the guard**: the first generation at which the rule holds with σ ≤ σ₀.

6. **sep-CMA's settings** are F.0b-i's: start at the box centre, σ₀ 0.2, population 16, bounds [0, 1], the answer `xfavorite`, and `es.stop()` never consulted. There are 4 truths × 2 restarts, with fit f = 2j + r on CMA seed **6410 + f**, and the pilot on **6418**.

7. **Seeds** (animals at `base + 200·k + i`, i < 200):

   - **registered fits, base 170,000,000:**
     - k = 10,000·f + 16·(n − 1) + p for fit f's generation n, member p;
     - 10,000·f + 5,000 to 5,009 for its answer, re-simulated;
   - **candidates and references, base 172,000,000:**
     - k = c for candidate c's v2 observation (c < 64);
     - 1,000 + 20·c + q for its repeat q + 1;
     - 3,000 + i for the i-th prior reference (i < 201);
     - 4,000 to 4,009 for the box centre;
   - **the pilot,** base 171,000,000: evaluations at 16·(n − 1) + p, and its answers at 5,000, 5,100 and 5,200 (+ 0–9);
   - **the smoke:** CMA seed 5993, base 99,300,000, discarded.

   All are disjoint from every band and earlier scheme. B v2 used 140–160.9 M and 199 M.

8. **The registration**, as F.0b-i's:

   - **Part A** (decisions 2–7, every threshold and seed) is committed before the preflight;
   - **Part B** (the preflight's and the pilot's outcomes) is committed before the registered fits;
   - `preflight` and `pilot` refuse without Part A, and `run` and `read` without Part B and the bound records.

9. **The verdict map:**

   - **pass:** F.0 stays cleared for C.1 on v2, and C.1b proceeds;
   - **fail:** C.1b does not run, and the failing truths' diagnosis (stuck, slow, or unreachable) goes to the maintainer.

## Risks / Trade-offs

- **[v2's truths sit further from the bands]** → C.1's model may not get as close to the v2 wild type as to v1's. The truths are the nearest the pool offers, and the gain is relative to each truth's own floor and prior, so distance alone does not fail the gate.
- **[The guard lengthens fits]** → restarts that would have stopped on a false plateau run to the cap. The pilot's projection shows the cost, and the 30 h threshold bounds it.
- **[Chemotaxis noise]** → the chemotaxis statistics on 200 animals are noisy (drift's floor is part of each truth's floor). The floor and the oracle absorb it, as they did v1's local search.
- **[The speed-up hides a change]** → only behaviour-identical changes are allowed, each pinned by a parity test before any registered simulation.
