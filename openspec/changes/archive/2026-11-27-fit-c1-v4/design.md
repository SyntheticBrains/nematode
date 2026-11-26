## Context

- **Benchmark v4** ([Logbook 124](../../../docs/experiments/logbooks/124-benchmark-v4.md)) is VALID and the default. It grades 24 statistics in six equal families at z\* 3.0781, and each mutant at its own bar (2.2414).
  - A faithful controller passes the wild type in 3 of 5 populations with probability 0.975.
  - It passes tph-1 (one graded statistic) and tax-4 (two) in 3 of 5 with probabilities 0.99985 and 0.99888, taking each z as standard normal and independent.
  - So it passes the joint verdict below with probability about 0.974.
- **C.1 (v3)'s registered answer** ([Logbook 121](../../../docs/experiments/logbooks/121-c1-v3-fit.md), restart 6 in `supporting/121-c1-v3-fit/fit.json`):
  - It passes v3 at 3 of 5, and fails v4 at 0 of 5 on the on-lawn onset rate.
  - Its states are ordered, dwelling at 0.012 mm/s and roaming at 0.060.
  - `b0` sits at its upper bound (unit 0.996), and `r_dwell` is near its floor (0.0018/s).
- **The diagnosis** ([Logbook 125](../../../docs/experiments/logbooks/125-c1-v4-diagnosis.md), probes in `supporting/125-c1-v4-fit/probes/`, discarded seeds):
  - Real worms make most on-lawn onsets while dwelling, and C.1's one dwelling rate also sets reversals off food. `r_dwell_on_food` separates the two.
  - **The passing route was two runs, not one:**
    - The first ran 100 generations from C.1 (v3)'s answer without the new parameter, and passed 0 of 5. Its σ fell from 0.094 to 0.017.
    - The second started from that answer with `r_dwell_on_food` at 0.03, a value its sensitivity check chose. It ran at a reset σ of 0.1, stopped on the plateau at generation 101, and passed 4 of 5 (D 1.28–1.44).
    - Neither had the mutants in its objective, and both used the pooled body.
    - Between them the parameters moved far: `r_dwell` to unit 0.000, `b0` from 0.996 to 0.862, `search_tau_s` from 0.649 to 0.164. The second answer sits at an L2 distance of 0.81 from this change's x₀.
  - **The registered route has never been run:** one stage, from C.1 (v3)'s coordinates with the new rate at its box's centre, with the mutants and the following body.
  - From the box centre, three runs passed 0 of 5. The parked design is `supporting/125-c1-v4-fit/parked_fit_design.md`.
- **A probe before Part A** (`supporting/136-c1-v4-fit/probes/separation.json`, discarded seeds at 647 M, description): the start re-simulated 10 times, and Logbook 125's passing answer as an observation with 10 re-simulations, all on the following arm with both mutants.
  - **Its distance from the start:** the answer's floor is 0.78 and the start's distance to it 1.69, so the separation is 0.91, short of the 1.0 the recovery needs. The maintainer chose one truth, the pilot's answer (2026-11-26).
  - **The mutants go the wrong way at both points:**
    - tph-1's Δ roaming fraction is −0.19 at the start and −0.22 at the answer, against real worms' +0.27 (z −9.0 and −9.5).
    - tax-4's Δ exits per on-lawn hour is +12.6 and +10.6, against +0.31 (z at the cap).
  - **The cost:** 35 s per evaluation with the mutants.
  - The start's wild-type D is 1.87 (median of 10), and the answer's 1.35.
- **The decisions since the deferral:**
  - **D37, amended 2026-11-10:** C.1's refit before C.4 fits the wild type on all six families together with tph-1 and tax-4, together half the objective, on the registered encoding table "unchanged". It holds out Scheer's held-out recording dates, and keeps one tax-4 leave-one-out as description. The note of 2026-11-24 dropped the leave-one-out for C.2 only, and the maintainer kept it for C.1 on 2026-11-26.
  - **The encoding table's `unidentified` rule** (`supporting/113-c1-fit/encodings.md`, `c1_fit.mutant_verdict`): across the restarts that pass the wild type, a mutant whose verdict differs among them is `unidentified`. C.4 grades the predictions under it.
  - **The body:** Logbooks 131 and 132 set the speed-following on-food reversal for C.1's refit.
  - **The start:** the direction decision of 2026-11-23, confirmed 2026-11-26, is a warm start under a D38 amendment, with a new pilot.
  - **The recovery** (2026-11-26): the gain is measured from the start, on one truth, the pilot's answer.
- **The grader:** it lives in `wormbench`, and `benchmark_grade` re-exports the functions the fit calls.

## Goals / Non-Goals

**Goals:**

- C.1's registered answer on benchmark v4, its wild type and encodable mutants fitted, for C.4;
- a recovery control that tests the refit's own procedure from its own start;
- rules registered before any registered reading, with selection kept apart from the verdict.

**Non-Goals:**

- new mechanisms (the model is Logbook 125's 26 parameters);
- pdfr-1, which C.1 cannot encode;
- changing the benchmark or the encoding table;
- grading the reversal share, which stays reported;
- C.4 itself, which is its own change.

## Decisions

### 1. D38, amended (2026-11-26)

The roadmap's D38 gains a dated amendment, the maintainer's decision, in this change.

- **C.1's refit on v4 starts from C.1 (v3)'s registered answer** at σ₀ 0.1, its new parameter at its prior box's centre.
- **Why the 2026-10-25 rule no longer binds:**
  - It set the search box and x₀ without search data, so that the start did not encode held-out families. Under D37 as amended, C.1's refit fits every family, so the v3 answer's use of local search and chemotaxis holds nothing out.
  - From the box centre, the model did not reach the basin (Logbook 125).
- **What it costs:** the fit is a local refinement of a registered answer, and is reported as one.
- **The recovery** is measured from that start, on one truth: F.0's prior-relative gain credits a warm start's nearness to its truth before any search.
- **The mutants:**
  - The registered verdict reads the chosen restart.
  - C.4's C.1 mutant rows apply the encoding table's `unidentified` rule across every restart that passes the wild type on its first 5 registered populations. They are reported beside the verdict, and never re-read as it.
  - The mutants are fitted, so these are not posterior-predictive intervals. The leave-one-out's tax-4 is the one prediction.
- C.2's procedure is unchanged.

### 2. The model: 26 parameters

| Coordinates | Parameters | Box |
|---|---|---|
| 0–24 | C.1 (v3)'s 25 (`c1_v3_fit.parameters`) | their registered boxes |
| 25 | `r_dwell_on_food` | log 0.001–0.5 per s, `r_dwell`'s prior |

- `r_dwell_on_food` is a dwelling animal's reversal rate wherever food contact is above 0, in place of `r_dwell`. It applies before the klinokinesis, edge and search terms. Off food, and while roaming, the state's rate applies.
- **The states are not re-ordered.** A warm start begins with them in order.

### 3. The start

- **x₀:** C.1 (v3)'s chosen answer, its 25 unit coordinates read from `supporting/121-c1-v3-fit/fit.json`, with coordinate 25 at 0.5 (0.0224 per s).
  - The new parameter has no fitted value, so it starts at its prior box's centre, as every C.1 coordinate did before.
  - Logbook 125's 0.03 came from search on v4, and it is not used.
- **The search:** σ₀ 0.1 (the warm probes' value), with sep-CMA at population 16, bounds [0, 1] and the diagonal covariance.

### 4. The body: the speed-following arm

- `generate_assay_configs.py` gains the arm `internal_following`.
- Its six assay configs are the `internal` arm's, with `continuous.on_food_reversal_speed: follows_run_speed` as their only difference.
- `benchmark_grade.ARMS["mlp_following"]` names them, and every stage simulates on that arm.
- Every other generated config stays byte-identical: the existing test re-derives each one and compares it with the committed file.

### 5. The objective

- **One evaluation:**
  - 200 animals on v4's assays, through its measurement models, without presence;
  - then tph-1 and tax-4, each on the small lawn on the same animals, under the registered encodings (`c1_v3_fit.mutant_parameters`, so every mutant keeps `r_dwell_on_food`);
  - each mutant's differences from that wild type, on `roaming_fraction` and `exits_per_on_lawn_hour`.
- **The score:** the definedness-first ranking score against v4's band centres at n = 200, with the mutants combined by C.2's formula.
  - `undefined` counts the wild type's undefined statistics, and `dc` is the rest of the score.

  - Each mutant's term t_m is the RMS of 10·asinh(z/10) over the encoding table's graded statistics. For tph-1 that is `roaming_fraction`; for tax-4, `roaming_fraction` and `exits_per_on_lawn_hour`.

  - z is capped at 10, and an undefined difference counts at the cap.

  - With n_f = 6 families, n_m = 2 mutants and w = n_f / n_m:

    score = 1,000·undefined + √((n_f·dc² + w·Σ t_m²) / (n_f + w·n_m)).

  - The mutants together carry half the weight.
- A test checks this against `c2_fit.Objective(..., mutants=("tph-1", "tax-4")).distance`. The leave-one-out uses n_m = 1, so w = 6.

### 6. The stop

A restart stops at the first generation n ≥ 40 at which all of these hold:

- **the plateau:** the mean of the last 20 medians is above the mean of the 20 before, minus 0.05;
- **σ:** σ ≤ 0.1, the registered σ₀.
  - The guarded plateau of Logbooks 116–117 compares the global step size with box fits' 0.2. A warm start at 0.1 meets that from its first generation.
  - At 0.1 the guard binds only if σ grows: every warm probe was below it from generation 1.
- **definedness:** no median of 1,000 or more in the last 40 generations;
- **tightened:** every median in the last 40 generations below 100. That differs from the definedness condition only when a median falls between 100 and 1,000: about half the candidates undefined, or a defined score above 100.

Otherwise the restart stops at the cap. A capped restart cannot be chosen, and its grades are reported.

### 7. The pilot

- **The run:** one restart from x₀, configured exactly as a registered restart, on discarded seeds (CMA 9467), with a provisional cap of 400.
- **Its grade:** its answer is the restart's `xfavorite`, graded with presence on 5 discarded populations, the wild type and both mutants.
- **The go rule,** fixed in Part A: the pilot stops on the plateau, and its wild type, tph-1 and tax-4 each pass in at least 3 of the 5.
- **The recovery's preflight, checked on discarded seeds** with the stage's own function, for its truth, the pilot's answer:
  - its observation and 20 re-simulations, and the start's 10 re-simulations;
  - the separation, oracle and stability checks of decision 8.
- **The cap:** max(300, ⌈1.5 × the pilot's stop generation⌉), rounded up to a multiple of 10, and at most 500.
  - The floor of 300 is about 1.5 times Logbook 125's two-run route (201 generations).
- **The cost bar:** the projected registered compute is (2 recovery restarts + 12 registered restarts + the leave-one-out) × the pilot's stop generation × its mean seconds per generation. It must be at most 48 h.
  - The same at the cap, every restart running to it, is reported beside the projection as its upper bound.
  - The preflight's 231 simulations and the grading add about 1 h.
- **If anything fails** (the go rule, the preflight check, the cost bar, or a pilot reaching 400 without stopping), nothing further registers, and the refit returns to the maintainer.
- **Recorded in Part B:**
  - the pilot's medians, σ, stop and answer (the truth);
  - its five grades and the go rule;
  - the preflight check;
  - the cap;
  - its seconds per generation and the projected cost.

### 8. The recovery (this refit's F.0 control)

- **The truth:** the pilot's answer, bound by its Part B record.
- **Two distances, every check and the gate read on both:**
  - **combined:** C.2's recovery distance, with C.1's mutants. It is the registered capped D on the six families, plus each mutant's RMS capped z over its graded statistics, combined at half by decision 5's formula, without the compression and the undefined count.
  - **wild type:** the registered capped D on the six families alone.
  - **Why both:** at x₀ the mutant terms sit near their cap (tph-1 at z −9.0, tax-4's exits at the cap), so they dominate the combined distance. A recovery that matched only the truth's mutants could pass on the combined distance alone, and the wild-type reading closes that.
- **The truth's record,** on each distance:
  - the observation (q = 0) and 20 re-simulations, the observation kept in the preflight's record so the recovery fits against the committed one;
  - **the floor:** the median distance from the observation to re-simulations 1–10;
  - **the oracle:** the same over 11–20;
  - **the start D:** the median distance to the registered start re-simulated 10 times;
  - **the prior D:** the median distance to 200 prior draws of the 26-parameter box (`default_rng(9470)`), reported only;
  - **the band reference:** the distance from the observation to the bands' centres, and its gain from the start, with a named outcome where that gain is at least 0.75: the gate then cannot tell recovering the truth from fitting the bands;
  - **the gate line:** the D at which the gain is 0.75, floor + 0.25·(start D − floor).
- **The gain from the start:** (start D − D) / (start D − floor). The prior-relative gain, (prior D − D) / (prior D − floor), is reported beside it.
- **The preflight's checks,** before any recovery fit:
  - **separated:** on each distance, the start D is above the floor by at least 1.0;
  - **the oracle:** on each distance, its gain from the start is at least 0.85;
  - **stable:** no statistic's definedness differs between the observation and any of re-simulations 1–10, wild type or mutant, as in C.2's preflight. A flip counts at the cap and would inflate the floor and the start D.
  - If a check fails, the recovery stops and returns to the maintainer.
- **The fits:**
  - 2 restarts (fit = j), each by the registered procedure from x₀;
  - each ranked by the ranking score against the truth's observation, the truth's mutant differences in at half;
  - with the registered stop and cap, and CMA seeds 9463 + fit.
- **The gate:**
  - The restart with the lower final objective (the mean of its last 20 medians) is re-simulated 10 times, and its gain from the start is read on each distance's median.
  - **Both gains must be at least 0.75.** Otherwise no registered fit runs unless an amended Part A records the maintainer's decision to run despite the gate (decision 13).
- **What it tests:** whether the registered procedure, from the registered start, closes the distance to a known worm in the region it is aimed at, under the simulator's own noise.
  - The truth sits beside the wild type and its mutant bands, so the gate cannot tell recovering the truth from fitting the bands. The band reference's gain from the start is reported for that reason, as in F.0b-ii.
  - Each parameter error is read in whichever state order is closer (`unit_errors`, with `r_dwell_on_food` unpaired).

### 9. The registered fit and its verdict

- **The restarts:** 12, r = 0–11, from x₀ at σ₀ 0.1, CMA seeds 9451 + r, with the registered stop and cap.
  - The answer is `xfavorite`, re-simulated 10 times for its median D and score.
  - The restarts share one start. They differ in the search's randomness and in their animals, and their spread is reported as that.
- **Selection:**
  - Every plateaued restart is graded with presence on 5 selection populations, the wild type and both mutants.
  - A selection population counts when the wild type, tph-1 and tax-4 all pass in it.
  - The restart with the most such populations is chosen. Ties go to the lower median score over its 10 re-simulations, then to the lower r.
  - A capped restart is not chosen. If no restart plateaus, no verdict is read.
- **The grade:** every restart on 15 registered populations of 200, every family with presence.
  - Each mutant runs on the small lawn on its population's own animals.
  - It is graded on its table statistics, at its bar, in its resolved direction.
- **The verdict:** read on the chosen restart's first 5 registered populations.
  - **C.1 (v4) passes when the wild type passes in at least 3 of the 5, and tph-1 and tax-4 each in at least 3 of the 5.**
  - A population passes the wild type when every family passes at |z| ≤ 3.0781, with local search and chemotaxis present.
  - **The verdict is in-sample:** every graded statistic and both mutants are fitted.
- **The mutant rows under the `unidentified` rule,** reported beside the verdict:
  - The rule takes every restart whose first 5 registered populations pass the wild type in at least 3, and reads each mutant at 3 of 5.
  - If the verdicts differ among those restarts, the row is `unidentified`. Otherwise it is their shared verdict.
  - With one such restart, the row is its verdict. With none, it is the chosen restart's verdict, with the wild-type failure stated.
- **The verdict map,** each branch with what C.4 receives:
  - **pass:** C.1 (v4) enters C.4 as an in-sample row. Its mutant rows are the `unidentified` rule's, beside the leave-one-out's tax-4.
  - **the wild type passes and a mutant fails:** C.1 (v4) fails its verdict. C.4 receives its wild type as passing, and its mutant rows as read.
  - **the wild type fails:** the roadmap's risk row "The mechanistic baseline cannot pass" goes to the maintainer. C.4 receives the failing statistics as the diagnosis.
  - **no restart plateaus:** no verdict is read, and the refit returns to the maintainer. C.4 has no C.1 (v4) row until the maintainer decides.
  - **the recovery's gate fails, or the pilot stops the refit:** no registered fit runs, and the refit returns to the maintainer. C.4 keeps C.1 (v3)'s failing v4 row.
- **Reported beside, never re-read as the verdict:**
  - populations 6–15's pass rates, and the shares over all 15;
  - every restart's stop, final score, median D, selection and registered grades;
  - the restart spread: the median pairwise capped D between plateaued restarts' re-simulations;
  - the chosen parameters, and any held at a bound;
  - each population's reversal share, against Roberts' 17.3%;
  - the cost of each stage;
  - the held-out reading (decision 10);
  - the leave-one-out (decision 11).

### 10. The held-out reading

C.2's reading (`c2_fit.held_out_reading`), on the chosen restart's 15 registered populations, as description. No new simulation.

- **The rule:** W's rule.
  - 1,000 splits of a date set into halves (seed 2034).
  - PD1074's band scales at the half's size, rounded to 25.
  - The statistics: the small lawn, kinematics and activity's graded statistics.
  - Each half's distance from the other real half (W's minimum), from each of the answer's populations, and from 15 band-cohort populations (`default_rng(2048)`).
- **The sets:** N2's 26 held-out dates (W's minimum's own), and PD1074's 24 held-out dates.
- **Reported:** each distance's median and 95th percentile.
- **A named outcome:** a pass whose answer's median distance to a held-out half is above the band reference's 95th percentile still passes, and C.4 states the held-out distance as a limit.
- **The authority:** D37's amendment of 2026-11-10 holds out these dates for C.1's refit, and Part A authorises this one read.
  - **Both sets have been read before.**
    - N2's held-out dates were read by benchmark v4's registered fresh-N2 control and by C.2's registered reading (launch record 133).
    - PD1074's were read by B.4's controls, by v4's planning probe (Logbook 124), by C.2's reading, and by an unregistered likelihood check on 2026-11-24, disclosed to the maintainer that day.
    - Both were read by this change's first smoke (disclosed below).
  - benchmark-controls' "PD1074's held-out cohorts SHALL NOT be read again" is scoped in this change to the benchmark's own validation: a registered fit's held-out reading under D37 is the exception.
  - Held-out rows load only in `read`. The smoke passes band-cohort stand-ins (`c1_v4_fit.smoke_sets`), which a test checks.

### 11. The tax-4 leave-one-out

- **After `read`:** one restart from x₀ at σ₀ 0.1 (CMA 9468) by the registered procedure, with tax-4 out of the objective. tph-1 alone is then half (n_m = 1, w = 6).
- **Its grade:** its answer is graded on 15 populations of its own, the wild type and both mutants.
- **Reported, both branches always shown** beside the registered fit's tax-4 row:
  - **prediction passed:** its tax-4 passes in at least 3 of its first 5 populations;
  - **prediction failed:** otherwise.
- **The encoding table's step 5:** if the leave-one-out stops at the cap, or its wild type fails its first 5, its tax-4 is still reported, with the cap or the wild-type failure stated.
- **It gates nothing,** and it runs whatever the registered verdict.

### 12. Seeds

The fit block of restart r spans 2,000,000 animals: k = 10,000·r + 16·(n − 1) + m. Answers sit at k = 10,000·r + 8,000 + q, q < 10, above every generation's index under a cap of at most 500 (16·499 + 15 = 7,999). `cma_recovery.answer_k`, whose offset is 5,000, is not used.

| Base | Use |
|---|---|
| 647,000,000 + 200·k + i | the separation probe before Part A, k ≤ 110 (discarded) |
| 650,000,000 + 200·k + i | the registered fit, r < 12 (ends below 674 M) |
| 675,000,000 + 200·g + i | the registered populations, g < 15 |
| 676,000,000 + 200·(5r + s) + i | the selection populations, s < 5 |
| 677,000,000 + 200·g + i | the leave-one-out's populations, g < 15 |
| 680,000,000 + 200·k + i | the recovery's fits, k = 10,000·fit + …, fit < 2 (ends below 684 M) |
| 690,000,000 + 200·q + i | the truth's observation (q = 0) and re-simulations (1–20) |
| 691,000,000 + 200·j + i | the prior draws, j < 200 |
| 692,000,000 + 200·q + i | the start's re-simulations, q < 10 |
| 694,000,000 + 200·k + i | the leave-one-out's fit, k = 16·(n − 1) + m |
| 696,000,000 + 200·k + i | the pilot's fit (discarded) |
| 698,000,000 + 200·g + i | the pilot's grades, g < 5 (discarded) |
| 698,100,000 + 200·q + i | the pilot's preflight check of its truth (discarded) |
| 698,200,000 + 200·q + i | the pilot's preflight check of the start (discarded) |
| 699,000,000–699,400,199 | the smoke's fits, simulations and grade (discarded) |

- **CMA seeds:**
  - 9451–9462, the registered restarts;
  - 9463–9464, the recovery's fits;
  - 9467, the pilot;
  - 9468, the leave-one-out;
  - 9469, the smoke;
  - 9465 and 9466 unused.
- **The prior draws:** `default_rng(9470)`.
- Every block is disjoint from every earlier one in the tracker's seed table and every launch record: Logbook 125's probes used 620–646 M and CMA 9441–9448, and C.2's work starts at 700 M with CMA 9449 and 9471 upwards. A test enumerates the blocks.
- The tracker's seed-table row is written with Part A.

### 13. The registration: launch record 136

- **Part A**, before the pilot:
  - it fixes every rule above;
  - it discloses Logbook 125's probes, the separation probe and the smoke's held-out read (below);
  - it authorises the held-out reading under D37;
  - it cites the D38 amendment.
- **Part B**, after the pilot and before the recovery's preflight: the pilot's result, the go rule, the preflight check, the cap and the projected cost. It changes no rule.
- **Part C**, after the recovery and before the fit: the recovery's result. It changes no rule.
- **The guards:**
  - **The bound records:**
    - Every stage binds the start's record, the encoding table, the six configs, `bands_v4.json` and `data/locomotion/primitives.json` (the speed coupling's knots) byte-identical to their versions at the launch record's last commit.
    - `pilot` needs Part A.
    - `recovery-preflight` and `recovery` also need Part B with the pilot's record bound and its `go` met (the go rule, the preflight check and the cost bar).
    - `run`, `select`, `grade`, `read` and `loo` also need Part C, with the recovery's records bound and its gate passing.
  - **The code:** every stage refuses if the fit's code differs from Part A's commit, or holds an untracked file there.
    - It covers the package sources (`packages/quantum-nematode/quantumnematode`, `packages/wormbench/wormbench`), the dependency lock and project files (`uv.lock` and the three `pyproject.toml`), and the scripts on the fit's computational path (`c1_v4_fit`, `c1_v3_fit`, `c2_fit`, `cma_recovery`, `cma_recovery_v2`, `npe_recovery`, `npe_region_check`, `benchmark_grade`, `benchmark_bands`, `benchmark_v2_controls`).
    - Modules imported only for names off that path (`c1b_fit`, `c1_fit`, `c2_encoder_probe`, `body_kinematics_eval`, `benchmark_controls`) are not bound.
  - **Part A's commit** is the last commit that changed Part A's own text. Part B's and Part C's commits leave it in place.
  - **An amendment:** a dated amendment of Part A, the maintainer's decision, re-pins the code at its own commit. That is the path for a code fix, a decision after a failed pilot (a new pilot under its own record and seeds), or a decision after a failed recovery gate.
    - The last adds to Part A the decision heading `### Decision: run despite the recovery's gate`, on a line of its own, which `run` and the later stages read in place of the gate's pass. A mention of it inside a sentence does not count.
  - **Committing Part B:** Part B is committed together with `pilot.json`, and Part C with the recovery's two records, since each part's guard compares its records with the launch record's last commit.
  - Each stage writes its record once, and refuses to overwrite it.
  - `smoke` is unguarded and discarded.
- **Disclosed before Part A:**
  - **Logbook 125's probes**, discarded.
  - **The separation probe** (above), discarded. It led the maintainer to one truth.
  - **The first smoke's held-out read.** On 2026-11-26 the smoke read N2's unread and PD1074's held-out dates through C.2's default sets, with 3 splits, against a discarded two-generation answer, before any authorisation. Nothing was committed and no rule depends on it. The smoke now passes band-cohort stand-ins.

## Risks / Trade-offs

- **[A route never run]** The registered single-run route from x₀ differs from Logbook 125's two-run route, with its σ reset and 0.03 start. → The pilot runs it exactly, and its go rule reads the result before anything registers.
- **[The mutants go the wrong way at the start]** tph-1's knockout lowers roaming at x₀ (z −9.0), where real worms roam more. The fit must reverse it without trading the wild type, as C.2's mutant-fitted pilots did. → The pilot's go rule reads the wild type and both mutants together.
- **[A local refinement]** The fit starts from a registered answer and searches near it. → It is disclosed. The recovery is read from the same start, with the band reference's gain beside it.
- **[One truth]** The gate rests on one known worm, from the same route as the fit. → It is F.0b-ii's design, chosen by the maintainer after the separation probe.
- **[A gate passed on the mutants]** At x₀ the mutant terms dominate the combined distance. → Every check and the gate are read on the wild type's distance as well.
- **[The mechanism was chosen on the benchmark]** → Every candidate's record is committed, and the leave-one-out's tax-4 is the one prediction.
- **[Thin margins]** The warm probe held the onset rate at z −2.6 to −3.0, against a bar of 3.0781. → The pilot's 5 populations are the readiness evidence.
- **[Under-reversal off food]** The off-food reversal share stayed near 5%, against Roberts' 17.3%. → It is reported.
- **[Bounds]** `b0` starts at its upper bound and `r_dwell` near its floor. → The box is not widened, and bounds are reported.
- **[Held-out cohorts read again]** → Under D37's authority, scoped by the benchmark-controls delta, read only in `read`.
- **[Cost]** About 35 s per evaluation. At a 200-generation route the projection is about 31 h (15 restarts × 200 × 38 s), plus the pilot. → The cost bar of 48 h is read in Part B, and every fit resumes from its saved state.
