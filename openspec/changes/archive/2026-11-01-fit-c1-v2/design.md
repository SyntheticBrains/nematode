## Context

- **C.1 on v1** ([Logbook 113](../../../../docs/experiments/logbooks/113-c1-fit.md); launch record 113):
  - sep-CMA, 8 restarts, the plateau rule, a cap of 210;
  - the chosen restart was the plateaued one with the lowest final objective;
  - its verdict was at least 3 of 5 registered populations.
- **Benchmark v2** ([Logbook 115](../../../../docs/experiments/logbooks/115-benchmark-v2.md)): 19 statistics in five families, with chemotaxis and local search passing only where present. A faithful population passes all five families with probability 0.818, so a faithful controller passes 3 of 5 with probability 0.955.
- **F.0b-i on v2** ([Logbook 117](../../../../docs/experiments/logbooks/117-cma-recovery-v2b.md)) passed with the ranking score, 4 restarts and the σ-guarded plateau:
  - 9 of 16 restarts reached a gain of 0.75;
  - 2 stopped falsely while undefined medians sat in their window;
  - 1164's misses were on the kinematics speed medians.
- **Cost:** after Logbook 118 and High Power mode, a generation of 16 simulations takes about 29 s on 16 workers.

## Goals / Non-Goals

**Goals:** C.1's v2 fit to the real wild type under rules registered before any fit, with a verdict that is the benchmark's own and selection kept apart from it.

**Non-Goals:**

- mutant grading, which is C.4's, under the registered encoding table (`113-c1-fit/encodings.md`);
- changing the model, its box or its priors (F.0a's box, set without search data);
- the v1 path, which stays as it is.

## Decisions

1. **The objective** (`c1b_fit.BandObjective`): the ranking score of `cma_recovery_v2.ranking_score`, with the truth replaced by the bands.
   - The "truth" record holds each graded statistic's band centre at n = 200 (`benchmark_grade.family_bands_v2`). The scales are the same bands' scales.
   - u counts the statistics undefined in the candidate; the centres are always defined.
   - D_c is the weighted RMS of 10·asinh(z/10), for z = (value − centre) / scale.
   - The simulation is `cma_recovery_v2.simulate_v2` (v2's four assays through its measurement models, without presence), at 200 animals.
   - The registered D, capped, is reported beside each answer.
2. **sep-CMA:** F.0b-i's settings: the box centre, σ₀ 0.2, popsize 16, bounds [0, 1], `es.stop()` never consulted, and the answer is `xfavorite`.
   - 8 restarts, r = 0–7, on CMA seeds 5749 + r. These are C.1's band's next eight; C.1 used 5741–5748.
   - Animals at `400_000_000 + 200·k + i`, on F.0b-i's scheme: k = 10,000·r + 16·(n − 1) + p, and answers at 10,000·r + 5,000 + q.
3. **The plateau rule, guarded twice** (`c1b_fit.defined_stop`):
   - a restart stops at the first n ≥ 40 with S_n > S\_{n−20} − 0.05;
   - and σ ≤ σ₀;
   - **and no generation from n − 39 to n has a median of 1,000 or more**, that is, a median candidate with a statistic undefined. Both windows the rule compares must be free of the definedness term.
   - Otherwise it stops at the cap.
4. **The reachability pilot**, on discarded seeds, before Part B:
   - one restart, CMA seed 5740, animals at `430_000_000 + 200·k + i`, run 300 generations whatever the rule says;
   - **G_p** is the first generation at which the doubly guarded rule holds;
   - **D** (the registered, capped D, median over 10 re-simulations) is read at G_p, at ⌈1.5·G_p⌉ and at 300, with the answers' re-simulations at k = 5,000, 5,100 and 5,200 (+ 0–9);
   - **the cap**, by F.0b-i's rule translated to D, is min(300, ⌈1.5·G_p⌉). It is 300 if the rule never holds, and 300 if D at ⌈1.5·G_p⌉ exceeds D at 300 by more than **0.27**. That is F.0b-i's 0.05 of gain times the median prior − floor of F.0b-i on v2's four truths, 5.42 (`116-cma-recovery-v2/preflight.json`);
   - **reachability:** the answer at 300 is graded with presence on 5 discarded populations (`432_000_000 + 200·g + i`). If it passes fewer than 3, the change returns to the maintainer with the failing statistics before Part B. This is not a verdict: one restart can land in a wrong basin, as 7 of F.0b-i's 16 did;
   - **the projection:** 8 × (cap + 1) × the sustained seconds per generation (the median over generations 101–300). Over 20 h, the maintainer decides before launch.
5. **Selection on separate populations:**
   - every restart that stopped on the plateau is graded with presence on 3 selection populations, `421_000_000 + 200·(3r + s) + i` for s < 3;
   - the chosen restart passes the most of them, ties go to the lower median D over the three, and then to the lower r;
   - a restart stopped by the cap cannot be chosen, as in C.1 (D38). If none plateaus, no verdict is read, and the run returns to the maintainer.
6. **The verdict:**
   - every restart, plateaued or stopped by the cap, is graded with presence on the 5 registered populations, `420_000_000 + 200·g + i` for g < 5. C.1's v1 fit graded every restart, and the encoding table's `unidentified` rule reads every restart that passes the wild type;
   - a population passes when every family passes: each statistic at |z| ≤ z\* = 3.0078, and local search and chemotaxis only where present;
   - **the chosen restart passes the wild type when at least 3 of its 5 populations pass;**
   - the other restarts' grades are reported, and feed C.4's `unidentified` rule.
   - **The verdict map:**
     - **pass:** C.1 (v2) enters C.4;
     - **fail:** the failing statistics are the diagnosis, and the roadmap's risk row ("the primitives or the benchmark are wrong; fixed before C.2") goes to the maintainer.
7. **Reported, not gated:**
   - every restart's stop, final score, selection and registered grades, and median D over 10 re-simulations;
   - the restart spread over plateaued pairs;
   - the chosen answer's parameters.
8. **The window restriction** (exact):
   - `to_behaviour_steps(animal, start, stop)` builds only steps start to stop, each with its step index and its signed speed from the previous step;
   - `measure_window(steps, first, primitives, seed, model)` draws the noise, lost seconds and fragment cuts over the whole episode, exactly as `measure` does, and applies them only to the window's steps;
   - `benchmark_grade.simulate_v2` builds and measures each lawn family's window, and local search's window unmeasured, with `analysis_steps`'s length check kept. Chemotaxis already measures only its window, and now builds only it;
   - **identity:** `speedup_identity_check.py baseline` on `main` before the edit, then `check` after. `--build build/c1b-fit/identity` and `--out supporting/119-c1b-fit/identity.json` keep this change's baseline and record apart from Logbook 118's.
9. **The registration:** launch record 119.
   - **Part A** fixes everything above, and is committed before the pilot.
   - **Part B**, after the pilot and before any fit, records the cap, the projection and the pilot's reachability, and changes no rule.
   - **The guard:** `pilot` refuses unless Part A is committed and unchanged, and every other registered stage also needs Part B. Each binds the registered mutant encoding table, `113-c1-fit/encodings.md`, byte-identical, as C.1's guard did, since C.4 grades C.1b's mutant predictions under it.
   - **The smoke:** CMA seed 5999, animals at 99,050,000, its grading at 99,060,000, all discarded, so it never touches a registered population.

## Risks / Trade-offs

- **[The model cannot reach v2's wild type]** → the pilot shows it first, on discarded seeds. A pass at its single restart is not required for the fits to be worth running, but a fail goes to the maintainer.
- **[Grading costs a little]** → 3 selection populations per plateaued restart (at most 24 simulations) and 5 registered populations per restart (40), about 3 minutes on 16 workers.
- **[The doubly guarded rule never holds on real data]** → fits run to the cap, cannot be chosen, and no verdict is read. The pilot shows whether the rule holds before registration.
- **[The cap tolerance is borrowed]** → 0.27 is F.0b-i's gain tolerance put on a D scale from committed truths. It decides only between the cap and 300.
