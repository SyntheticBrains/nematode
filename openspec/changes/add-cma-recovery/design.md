## Context

- **C.1's model** is `BatchedMechanistic`, with 20 parameters. F.0a's box (`npe_recovery.PRIOR`) maps the unit cube onto their ranges, and none of its ranges uses search data.
- **The simulator** is `npe_recovery.simulate`: 200 animals on `scheer_lawn`, `uniform_lawn_scheer` and `food_removal_lopezcruz`, graded.
- **The distance** is `npe_recovery.predictive_distance`: D over the 11 statistics on the bands' n = 200 scales, capped at 10. A statistic undefined in both simulations counts 0, and one undefined in only one counts the cap.
- **F.0a's 10,000 prior simulations** are off-repo, with their parameters (`build/npe-recovery/simulations.jsonl`).
- **`region_check.json`** (Logbook 111, addendum) records the pool near the wild type: 43 simulations with every statistic defined, D ≤ 5 to the bands, and a presence bound above 1.5. It also records their floors.
- **`cma` 4.5** is installed. `CMA_diagonal: True` gives sep-CMA.
- **The planning probe**, uncommitted, used training simulation 8,207 as its truth (floor 0.62). Over 30 generations of 16, sep-CMA's population median fell from 6.91 to 4.30, and its re-simulated mean gained 0.51. It ran at about 27 s per generation, sustained.
- **D37:** C.1 is fitted on all 11 statistics, and only C.2 and C.3 hold local search out.

## Goals / Non-Goals

**Goals**

- A registered test that sep-CMA, configured as C.1's real fit will be, reaches the statistics of worms near the wild type.
- A gate that its oracle passes and its unfitted start fails, shown before launch.
- The seed-to-seed spread that W's draws per arm are sized from.

**Non-goals**

- C.1's real fit, which comes after F.0 clears.
- C.2's encoder, which is F.0b-ii.
- Recovery in parameter space, which is impossible in principle with 11 statistics for 20 parameters. It is reported, not gated.

## Decisions

01. **The objective.** For truth j and candidate z in the unit cube:

    - f(z) = `predictive_distance(observation_j, simulate(z))`;
    - every evaluation uses fresh animals;
    - this is the D C.1's real fit will minimise against the bands, with the truth's statistics in place of the band centres.

02. **sep-CMA.**

    - **Settings:** `cma.CMAEvolutionStrategy(x0 = 0.5·1, σ0 = 0.2, {CMA_diagonal: True, bounds: [0, 1], popsize: 16, seed})`.
    - **Termination:** cma's own criteria are disabled; only the plateau rule and the cap stop a fit.
    - **The answer:** a fit's answer is the distribution's mean when it stops.
    - **Seeds:** fit f = 2·j + r (truth j = 0–3, restart r = 0–1) uses CMA seed 5410 + f, and the pilot uses 5418.

03. **The plateau rule.**

    - m_g is generation g's median objective, and S_g is the mean of m over generations g − 19 to g.
    - A fit stops at the first g ≥ 39 with S_g > S\_{g−20} − 0.05, that is, less than 0.05 of improvement over 20 generations, or at the cap.
    - Whether a fit stopped by the rule or by the cap is reported, not gated.

04. **Truths (`preflight`).**

    - **Candidates:** the 43 pool simulations in ascending D to the bands. Training simulation 8,207, which the planning probe used, is excluded.
    - Each candidate is re-simulated 20 times with fresh animals:
      - its **floor** is the median D from its observation to repeats 1–10;
      - its **oracle score** is the median D to repeats 11–20.
    - A candidate is **kept** when:
      - no repeat changes any statistic's definedness from the observation;
      - prior_j − floor_j ≥ 1.0, where prior_j is the median D to F.0a's first 200 training simulations other than itself.
    - **Order:** the first kept candidate is the pilot truth, and the next 4 are the registered truths. Candidates are preflighted in batches of 8 until 5 are kept.
    - If the pool runs out first, the run stops and returns to the maintainer.

05. **The preflight's gates**, before Part B:

    - **The oracle:** over the 4 registered truths, the median of (prior_j − oracle_j) / (prior_j − floor_j) is at least **0.85**.
    - **The unfitted start:** the box centre, simulated 10 times, is scored against each truth the same way, and its median gain is at most **0.25**. A fit that never moved would fail.

    If either fails, the run stops before Part B. `gate_preflight.py` has no panel levels here, so this is the principle applied by hand (protocol principle 6, as added 2026-10-25).

06. **The pilot (`pilot`).**

    - One fit (CMA seed 5418) on the pilot truth runs to 300 generations, whatever the rule says. It is the campaign's configuration, at full load for over two hours.
    - It records:
      - G_p, the generation where the plateau rule first fires;
      - the sustained seconds per generation;
      - its mean's gain at G_p and at generation 300, each from 10 fresh re-simulations.
    - **The registered cap** is min(300, ⌈1.5·G_p⌉), or 300 if the rule never fires.
    - **Stops:**
      - if the pilot's gain at generation 300 is below 0.75, the run stops before Part B: the gate would be reachable for the oracle but not for the method;
      - if the projection exceeds 20 h, the maintainer decides before launch.

07. **The gate (registered in Part A).**

    - For each truth, the restart with the lower S at its stop is chosen.
    - Its mean is re-simulated 10 times: post_j is the median D to the truth's observation, and gain_j = (prior_j − post_j) / (prior_j − floor_j).
    - **The median of gain_j over the 4 truths is at least 0.75.**
    - **Reported, not gated:**
      - the other restart's gain, and the two restarts' D to each other (the spread that sizes W);
      - each fit's stop generation and whether it plateaued;
      - per-parameter |mean − truth| on the unit cube, against a random guess's 0.25;
      - every generation's m_g and σ.

08. **The launch record, in two parts.**

    - **Part A**, committed before `preflight`, holds every rule, seed, threshold and stop above.
    - **Part B**, committed after `preflight` and `pilot` and before `run`, adds the truths, the preflight's numbers, the pilot's G_p, the cap and the projection. It changes no gate or threshold.

    `read` refuses unless `launch.md` is committed and unchanged, with `preflight.json` and `pilot.json` byte-identical to their versions in the commit that last touched `launch.md` (B.4's guard).

09. **Seeds and simulation indices.** Animals are `50_000_000 + 200·k + i`:

    | k | Use |
    |---|---|
    | 10,000·f + 16·g + p (g < 300, p < 16) | fit f's evaluations |
    | 10,000·f + 5,000 … 5,009 | fit f's mean, re-simulated |
    | 100,000 + 20·c + q (q < 20) | candidate c's repeats |
    | 101,000 … 101,009 | the box centre, re-simulated |

    - **Range:** the largest k is 101,009, so animals stay below 70.2 million.
    - **The pilot:** base `80_000_000`, with evaluations at k = 16·g + p and its two means at 5,000–5,009 and 5,100–5,109.
    - **The planning probe:** bases `32_000_000` (k < 480) and `33_000_000` (k < 10), recorded as used.

10. **Parallelism and resumption.** One fit runs at a time on 16 single-thread workers, one per candidate in a generation. After each generation, the CMA state is pickled and the generation is appended to `build/cma-recovery/`, so an interrupted fit resumes at its last generation.

## Risks / Trade-offs

- **[The descent is slow and the cap stops fits short]** → the pilot measures it. If the pilot's gain at 300 generations is below 0.75, the run stops before anything registers.
- **[The plateau rule fires on noise during a slow descent]** → the cap is 1.5 times the pilot's plateau generation, and the pilot reports what it gained after the rule first fired. The rule only stops fits that have flattened.
- **[Recovery holds near the wild type only]** → stated. C.1's real fit lands there, which is why the truths sit there.
- **[The two restarts share x₀]** → they differ by CMA seed only, as the real fit's will. Their spread is a property of the method as it will be used.
- **[Cost]** → at most about 18 h registered, and less if fits plateau. Above 20 h projected, the maintainer decides.
