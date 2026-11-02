## Context

- **Benchmark v3** ([Logbook 120](../../../docs/experiments/logbooks/120-benchmark-v3.md)) is VALID and the default. It grades 21 statistics in five equal families at z\* 3.0381, with local search and chemotaxis passing only where present. A faithful controller passes 3 of 5 populations with probability 0.977.

- **C.1b on v2** ([Logbook 119](../../../docs/experiments/logbooks/119-c1b-fit.md)) failed: its fits were reversal-dominated off food. v3 catches every one of C.1b's nine answers.

- **The probes** (2026-11-02, description, discarded seeds; records to `supporting/121-c1-v3-fit/probes/`). Each is one sep-CMA restart against v3's ranking score and C.1b's guarded stop, its answer graded with presence on 5 populations:

  | Probe | Model | Start | Stop | Passing | D | What fails |
  |---|---|---|---|---|---|---|
  | readiness | 20 parameters | centre | 86 | 0 | 3.42–3.56 | dwelling speeds z +7, dwelling bouts, weathervane |
  | warm from C.1b | 20 | C.1b's answer, σ 0.1 | 103 | 0 | 2.49–2.70 | the three new statistics (reverses 47%) |
  | widened box | 20, 5 bounds widened | centre | 163 | 0 | 3.09–3.54 | dwelling speeds, lawn bouts (runs to the new bounds) |
  | off-food kinematics | 22 | centre | 136 | 0 | 2.50–2.74 | Chen's onset rate, flips, weathervane |
  | speed spread | 24 | centre | 146 | 0 | 1.28–1.46 | weathervane z +4 (in 3 populations, alone) |
  | full | 25 | centre | 133 | 0 | 2.50–2.81 | another basin: kinematics, lawn bouts |
  | full, warm | 25 | the 24's answer, σ 0.1 | 126 | **4** | 0.85–1.04 | one roaming-bout length |
  | full, box A | 25 | centre | 218 | **3** | 1.08–1.33 | a roaming-bout length in 2 |
  | full, box B | 25 | centre | 172 | 0 | 1.44–1.53 | the two roaming speeds |

  **Two paired sensitivity tests** on the speed-spread answer (3 populations each):

  - the search term barely matters;
  - the on-food weathervane gain keeps the worm on the lawn (time on the lawn z −9.5 at half strength);
  - an off-food gain at a quarter of it fixes the weathervane (z +4.1 → +0.7) and passes 1 of 3 populations.

  **The benchmark side checks out:** Chen's naive worms were "directly removed from food and washed for three times before testing" (Chen et al. 2025, Methods), as the simulated assay assumes.

- **Cost:** about 33 s per generation on 16 workers.

## Goals / Non-Goals

**Goals:**

- C.1's fit to v3's wild type, with a model whose added mechanisms each answer a named failure;
- a recovery check on that model;
- rules registered before any registered reading, and selection kept apart from the verdict.

**Non-Goals:**

- mutant grading, which is C.4's;
- changing the benchmark;
- further mechanisms;
- grading the reversal share, which stays reported.

## Decisions

1. **The model's two new options** (`BatchedMechanistic`). Each is off unless its parameters are given, so the 20-parameter model is unchanged without them.

   - **Off-food kinematics:** wherever food contact is 0, the run speed is `v_off_food` and the curvature noise is `sigma_off_food`, whatever the state. On food the state's values apply.
   - **The speed spread:**
     - at the first second and every 10 s after (the classifier's window), each animal draws z;
     - its run speed is then the speed above × exp(`speed_log_sd_s`·z) for its state s, so that speed is the median;
     - z is the sine partner of that second's Box–Muller curvature normal: independent of it, and no extra draw;
     - both spreads are given together, or neither.
   - **`kappa_off_food`** is unchanged: the weathervane gain wherever food contact is 0.
   - **Identity:** `speedup_identity_check.py check` against `build/benchmark-v3/identity`; all 55 jobs must be byte-identical. Tests cover each option's scope.

2. **C.1 (v3): 25 parameters.** C.1's 20 on F.0a's box (`npe_recovery.PRIOR`), plus:

   | Parameter | Range | From |
   |---|---|---|
   | `v_off_food` | uniform 0.005–0.20 mm/s | the state speeds' prior |
   | `sigma_off_food` | uniform 5–60 °/s | the curvature priors' |
   | `speed_log_sd_dwell`, `speed_log_sd_roam` | uniform 0–1.5 | v2's reachability box |
   | `kappa_off_food` | uniform −50 to 50 | the weathervane gain's prior |

   None is set from search data (D38). The mechanisms were chosen by probes on all five families, including local search and chemotaxis. That is disclosed, and it is why C.1's mutant predictions, not its wild-type fit, are its held-out test.

3. **The mutant encodings,** extending `113-c1-fit/encodings.md` and registered in Part A. That table is bound byte-identical by C.1b's records, so it is not edited.

   - **tax-4** sets κ_w = κ_off = β_r = β_o = 0.
   - **tph-1** is unchanged (a_intake = b_intake = 0).
   - **pdfr-1** is not encodable.
   - The four kinematic parameters are not chemosensory, and every mutant keeps them.

4. **The reduced recovery** (F.0b-i's rules on the 25-parameter model, on v3's D):

   - **The truths:** the two passing probe answers, full-warm (t = 0) and full-box A (t = 1).

   - **Each truth's record:**

     - the observation, its first re-simulation;
     - the floor: the median D from the observation to re-simulations 1–10;
     - the oracle: the same over re-simulations 11–20;
     - the prior D: the median D from the observation to 200 prior draws of the 25-parameter box (`default_rng(5781)`).

   - **Gain** = (prior D − D) / (prior D − floor).

   - **Before the fits:**

     - the oracle's median gain must be at least 0.85;
     - the box centre's median gain, re-simulated 10 times, at most 0.25;
     - each truth's prior D must exceed its floor by at least 1.0.

     If any check fails, the recovery stops and returns to the maintainer.

   - **The fits:** 2 restarts per truth (fit = 2t + j):

     - sep-CMA from the box centre, σ₀ 0.2, population 16, CMA seeds 5777 + fit;
     - ranked by the definedness-first score against the truth's observation, on v3's scales;
     - the doubly guarded plateau, cap 330.

   - **The gate:** per truth, the restart with the lower final objective is re-simulated 10 times, and its gain read. The median gain over the two truths (their mean) must be at least **0.75**. If it falls short, no registered fit runs until the maintainer decides.

   - The other restart's gain, the stops and the parameter errors are reported.

5. **The registered fit** (`c1_v3_fit.py`): C.1b's design on v3.

   - **The objective:** the ranking score against v3's band centres at n = 200, over v3's four assays through its measurement models, without presence.
   - **sep-CMA:** 8 restarts r = 0–7 from the box centre, σ₀ 0.2, population 16, bounds [0, 1], CMA seeds 5769 + r. The answer is `xfavorite`.
   - **The stop:** the plateau rule once σ ≤ σ₀ and no median of 1,000 or more lies in the last 40 generations; otherwise **the cap of 330**. The cap is ⌈1.5 × 218⌉ rounded up to 330, from the probes' longest plateau.
   - **Selection:**
     - every plateaued restart is graded with presence on 3 selection populations;
     - the restart passing most is chosen, ties going to the lower median D and then the lower r;
     - a capped restart cannot be chosen. If none plateaus, no verdict is read.
   - **The verdict:**
     - every restart is graded with presence on 5 registered populations;
     - a population passes when every family passes (|z| ≤ 3.0381, local search and chemotaxis present);
     - **the chosen restart passes the wild type when at least 3 of its 5 pass.**
   - **The verdict map:**
     - **pass:** C.1 (v3) enters C.4;
     - **fail:** the failing statistics are the diagnosis, and the roadmap's C.1 risk row goes to the maintainer.
   - **Reported:**
     - every restart's stop, final score, selection and registered grades, and median D over 10 re-simulations;
     - the restart spread;
     - the chosen parameters;
     - each restart's reversal share against Roberts' 17.3% (the probes' passing answers reverse 2–6%).

6. **Seeds.** Each restart's block spans 2,000,000 animals (k = 10,000·r + …), so every block below starts at least that far apart.

   | Base | Use |
   |---|---|
   | 500,000,000 + 200·k + i | the fit: k = 10,000·r + 16·(n − 1) + m; answers at k = 10,000·r + 8,000 + q (q < 10), above the cap's 5,279 |
   | 520,000,000 + 200·g + i | the registered populations, g < 5 |
   | 521,000,000 + 200·(3r + s) + i | the selection populations, s < 3 |
   | 530,000,000 + 200·k + i | the recovery's fits, k = 10,000·fit + …, answers at 10,000·fit + 8,000 + q |
   | 540,000,000 + 200·(100·t + q) + i | the truths' observation (q = 0) and re-simulations (1–20) |
   | 541,000,000 + 200·j + i | the prior draws, j < 200 |
   | 542,000,000 + 200·q + i | the box centre's re-simulations, q < 10 |
   | 86,000,000 and 86,970,000; CMA 5990 | the smoke, discarded |
   | 87,000,000–98,970,000; CMA 5760–5768 | the probes, discarded |

7. **The registration:** launch record 121.

   - **Part A** fixes every rule above, discloses the probes, and is committed before the recovery.
   - **Part B**, after the recovery and before any fit, records its result and changes no rule.
   - **The guards:**
     - `recovery-preflight` and `recovery` refuse unless Part A is committed and unchanged;
     - `run`, `select`, `grade` and `read` need Part B, with the recovery record bound and passing;
     - every stage binds `113-c1-fit/encodings.md` byte-identical.

## Risks / Trade-offs

- **[The mechanisms were chosen on the benchmark]** → It is disclosed. Each one answers a named failure with a measured cause, and the mutant predictions remain C.1's held-out test. No mechanism was added after the 25-parameter probes.
- **[Box-centre restarts find the passing basin about 1 time in 3]** → With 8 restarts and selection, the chance of none is about 4% (crude, from 3 restarts). If none passes, the fail map applies.
- **[Recovery on fitted truths]** → The truths sit near the real wild type, which is where F.0b-i's truths were chosen to be. Using two of them, not four, is the maintainer's reduced choice.
- **[Parameters at their bounds]** → The probe answers touch a few bounds (search amplitude, a switching rate, one κ near −50). The box is not widened. A bound is reported where it holds.
- **[Under-reversal]** → C.1 passes v3 while reversing 2–6%, against 12–17% in real worms. v3 cannot grade it from positions. It is reported beside the verdict.
- **[Cost]** → about 20 h in all. The fits are resumable, and the recovery gates them.
