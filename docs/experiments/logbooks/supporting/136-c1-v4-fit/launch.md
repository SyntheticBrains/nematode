# Launch record: C.1 refitted on benchmark v4 from C.1 (v3)'s answer, with its encodable mutants

**Part A registered 2026-11-26**, change `fit-c1-v4`. It is committed before the pilot, the recovery and every fit, and fixes every rule, threshold and seed in the change's design.

- **Part B**, added after the pilot and before the recovery's preflight, records the pilot's result.
- **Part C**, added after the recovery and before the fit, records the recovery's result.

Neither part changes a rule. A verdict registered here is never re-read under a different rule; anything else is reported beside it as description.

## Part A: the rules

### The authority

- **D38, amended 2026-11-26** (the maintainer's decision, in this change):
  - C.1's refit on v4 starts from C.1 (v3)'s registered answer.
  - Its recovery's gain is measured from that start.
  - Its mutant rows follow the encoding table's `unidentified` rule.
- **D37, amended 2026-11-10:** C.1's refit fits the wild type on all six families with tph-1 and tax-4, together half the objective, on the registered encoding table unchanged. It holds out Scheer's held-out recording dates, and keeps one tax-4 leave-one-out as description. The maintainer kept the leave-one-out for C.1 on 2026-11-26.
- **Logbooks 131 and 132:** the refit runs on the speed-following on-food reversal.

### The model and the start

- **The model:** `BatchedMechanistic` with C.1 (v3)'s 25 parameters on their registered boxes (`c1_v3_fit.parameters`), plus `r_dwell_on_food` on `r_dwell`'s prior, log 0.001–0.5 per s.
  - `r_dwell_on_food` is a dwelling animal's reversal rate wherever food contact is above 0, in place of `r_dwell`.
  - The states are not re-ordered.
- **The start:**
  - **x₀:** the 25 unit coordinates of restart 6's `answer` in `supporting/121-c1-v3-fit/fit.json` (bound), with coordinate 25 at 0.5 (0.0224 per s, the new box's centre).
  - **The search:** σ₀ 0.1, sep-CMA (diagonal), population 16, bounds [0, 1]. The answer is the restart's `xfavorite`.
- **The mutants:** C.1 (v3)'s registered encodings (`supporting/113-c1-fit/encodings.md`, bound). Every mutant keeps `r_dwell_on_food`.
  - tph-1 sets `a_intake` = `b_intake` = 0.
  - tax-4 sets `kappa_weathervane` = `kappa_off_food` = `beta_reversal` = `beta_omega` = 0.
  - pdfr-1 is not encodable.

### The body

Every stage simulates on `benchmark_grade.ARMS["mlp_following"]`: the six `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives_following_assay_*.yml` configs (bound). Each is its `internal` counterpart with `continuous.on_food_reversal_speed: follows_run_speed`, its only difference, and the knots come from `data/locomotion/primitives.json` (bound).

### The objective

- **One evaluation:**
  - 200 animals on v4's assays, through its measurement models, without presence;
  - then tph-1 and tax-4, each on the small lawn on the same animals;
  - each mutant's differences from that wild type, on `roaming_fraction` and `exits_per_on_lawn_hour`.
- **The score:** the definedness-first ranking score against v4's band centres at n = 200 (`bands_v4.json`, bound), with the mutants at half:
  - score = 1,000·undefined + √((n_f·dc² + w·Σ t_m²) / (n_f + w·n_m)), with n_f = 6, n_m = 2 and w = 3.
  - t_m is the RMS of 10·asinh(z/10) over the encoding table's graded statistics: tph-1's `roaming_fraction`, and tax-4's `roaming_fraction` and `exits_per_on_lawn_hour`.
  - Each z is against v4's Δ band at n = 200, in its resolved direction, capped at 10. An undefined difference counts at the cap.

### The stop and the cap

- **The stop:** a restart stops at the first generation n ≥ 40 at which all of these hold:

  - the mean of the last 20 medians is above the mean of the 20 before, minus 0.05;
  - σ ≤ 0.1;
  - no median in the last 40 generations is 1,000 or more;
  - every median in the last 40 generations is below 100.

  Otherwise it stops at the cap.

- **The cap:** max(300, ⌈1.5 × the pilot's stop generation⌉), rounded up to a multiple of 10, and at most 500.

### The pilot (discarded seeds)

- **The run:** one restart from x₀ (CMA 9467), configured as a registered restart, with a provisional cap of 400.
- **Its grade:** its answer graded with presence on 5 populations, the wild type and both mutants.
- **The go rule:** the pilot stops on the plateau, and its wild type, tph-1 and tax-4 each pass in at least 3 of the 5.
- **The preflight check, on its own answer:** the recovery's separation, oracle and stability checks (below), on discarded seeds, with the stage's own function.
- **The cost bar:** the projected registered compute is (2 recovery restarts + 12 registered restarts + the leave-one-out) × the pilot's stop generation × its mean seconds per generation. It must be at most 48 h. The same at the cap is reported beside it as the upper bound.
- **If any of these fails,** nothing further registers, and the refit returns to the maintainer.

### The recovery (this refit's F.0 control)

- **The truth:** the pilot's answer, bound by Part B.

- **Two distances, every check and the gate read on both:**

  - **combined:** the registered capped D on the six families, with each mutant's RMS capped z over its graded statistics combined at half (w = 3), without compression or the undefined count;
  - **wild type:** the registered capped D on the six families alone.

  At x₀ the mutant terms sit near their cap, so on the combined distance alone a recovery that matched only the truth's mutants could pass.

- **The record,** on each distance:

  - the observation and 20 re-simulations, the observation kept in the preflight's record;
  - **the floor:** the median distance from the observation to re-simulations 1–10;
  - **the oracle:** the same over 11–20;
  - **the start D:** the median distance to x₀ re-simulated 10 times;
  - **reported:**
    - the prior D, from 200 draws of the 26-parameter box (`default_rng(9470)`);
    - the band reference's distance and its gain from the start, with a named outcome where that gain is at least 0.75: the gate then cannot tell recovering the truth from fitting the bands;
    - the gate line, floor + 0.25·(start D − floor).

- **The gain from the start:** (start D − D) / (start D − floor). The prior-relative gain is reported beside it.

- **The checks, before any recovery fit:**

  - **separated:** on each distance, start D − floor ≥ 1.0;
  - **the oracle:** on each distance, the oracle's gain from the start ≥ 0.85;
  - **stable:** no statistic's definedness, wild type or mutant, differs between the observation and any of re-simulations 1–10.

  If a check fails, the recovery stops and returns to the maintainer.

- **The fits:** 2 restarts from x₀, CMA seeds 9463 and 9464. Each is ranked by the ranking score against the observation, with its mutant differences in at half, and runs with the registered stop and cap.

- **The gate:** the restart with the lower final objective (the mean of its last 20 medians) is re-simulated 10 times. **Its gain from the start must be at least 0.75 on each distance's median.** Otherwise no registered fit runs, unless an amended Part A records the maintainer's decision to run despite the gate (*The guards*).

### The registered fit and its verdict

- **The restarts:** 12, r = 0–11, from x₀, CMA seeds 9451 + r, with the registered stop and cap. Each answer is re-simulated 10 times.
- **Selection:**
  - Every plateaued restart is graded with presence on 5 selection populations.
  - The restart with the most populations in which the wild type, tph-1 and tax-4 all pass is chosen. Ties go to the lower median score over its 10 re-simulations, then to the lower r.
  - A capped restart is not chosen. If none plateaus, no verdict is read.
- **The grade:** every restart on 15 registered populations, every family with presence.
  - Each mutant runs on the small lawn on its population's own animals.
  - It is graded on its table statistics, at its bar of 2.2414, in its resolved direction.
- **The verdict, on the chosen restart's first 5 registered populations:** C.1 (v4) passes when the wild type passes in at least 3, and tph-1 and tax-4 each pass in at least 3.
  - A population passes the wild type when every family passes at |z| ≤ 3.0781, with local search and chemotaxis present.
  - The verdict is in-sample.
  - A faithful controller passes it with probability about 0.974.
- **The mutant rows,** for C.4, under the encoding table's `unidentified` rule:
  - The rule takes every restart whose first 5 registered populations pass the wild type in at least 3, and reads each mutant at 3 of 5.
  - If those verdicts differ, the row is `unidentified`; otherwise it is their shared verdict.
  - With none, the row is the chosen restart's verdict, with the wild-type failure stated.
  - The rows are reported beside the verdict, and never re-read as it.
- **The verdict map:**
  - **pass:** C.1 (v4) enters C.4 as an in-sample row, with its mutant rows and the leave-one-out's tax-4.
  - **the wild type passes and a mutant fails:** C.1 (v4) fails. C.4 receives its wild type as passing, and its mutant rows as read.
  - **the wild type fails:** the roadmap's risk row "The mechanistic baseline cannot pass" goes to the maintainer. The failing statistics are the diagnosis.
  - **no restart plateaus:** no verdict is read, and the refit returns to the maintainer.
  - **the pilot or the recovery stops the refit:** no registered fit runs. C.4 keeps C.1 (v3)'s failing v4 row until the maintainer decides.
- **Reported beside, never re-read as the verdict:**
  - populations 6–15's pass rates;
  - every restart's stop, scores, selection and grades;
  - the restart spread, the median pairwise capped D between plateaued restarts' re-simulations;
  - the chosen parameters and any at a bound;
  - each population's reversal share, against Roberts' 17.3%;
  - each stage's cost.

### The held-out reading (authorised here)

- **This record authorises one read**, in `read`, as description, under D37 as amended. It reads N2's 26 held-out dates and PD1074's 24 held-out dates by C.2's rule (`c2_fit.held_out_reading`):

  - 1,000 splits into halves (seed 2034), scaled by PD1074's band scales at the half's size rounded to 25;
  - the small lawn's, kinematics' and activity's graded statistics;
  - each half's distance from the other half, from each of the chosen restart's 15 registered populations, and from 15 band-cohort populations (`default_rng(2048)`).

- **Reported:** each distance's median and 95th percentile.

- **A named outcome:** a pass whose answer's median distance to a held-out half is above the band reference's 95th percentile still passes, and C.4 states the distance as a limit.

- **Earlier reads:**

  - N2's held-out dates by benchmark v4's registered fresh-N2 control and C.2's registered reading (launch record 133);
  - PD1074's by B.4's controls, v4's planning probe (Logbook 124), C.2's reading and an unregistered likelihood check on 2026-11-24, disclosed to the maintainer that day;
  - both by this change's first smoke (*Disclosures*).

  benchmark-controls' no-reread rule is scoped in this change to the benchmark's own validation.

- No other stage loads a held-out row.

### The tax-4 leave-one-out (description)

- **The run:** after `read`, one restart from x₀ (CMA 9468) by the registered procedure, with tax-4 out of the objective. tph-1 alone is then half (w = 6).
- **Its grade:** its answer is graded on 15 populations of its own.
- **Its tax-4 is reported beside the registered fit's,** both labels always shown: **prediction passed** at 3 or more of its first 5 populations, **prediction failed** otherwise. A cap or a wild-type failure in its first 5 is stated beside it.
- It gates nothing, and runs whatever the verdict.

### Seeds

The fit block of restart r spans 2,000,000 animals: k = 10,000·r + 16·(n − 1) + m. Answers sit at k = 10,000·r + 8,000 + q, q < 10. Every block below is disjoint from every block in the tracker's seed table and every earlier launch record.

| Base | Use |
|---|---|
| 647,000,000 + 200·k + i, k ≤ 110 | the separation probe (disclosed, discarded) |
| 650,000,000 + 200·k + i | the registered fit, r < 12 |
| 675,000,000 + 200·g + i | the registered populations, g < 15 |
| 676,000,000 + 200·(5r + s) + i | the selection populations, s < 5 |
| 677,000,000 + 200·g + i | the leave-one-out's populations, g < 15 |
| 680,000,000 + 200·k + i | the recovery's fits, fit < 2 |
| 690,000,000 + 200·q + i | the truth's observation (q = 0) and re-simulations (1–20) |
| 691,000,000 + 200·j + i | the prior draws, j < 200 |
| 692,000,000 + 200·q + i | x₀'s re-simulations, q < 10 |
| 694,000,000 + 200·k + i | the leave-one-out's fit |
| 696,000,000 + 200·k + i | the pilot's fit (discarded) |
| 698,000,000 + 200·g + i | the pilot's grades, g < 5 (discarded) |
| 698,100,000 + 200·q + i | the pilot's check of its answer (discarded) |
| 698,200,000 + 200·q + i | the pilot's check of x₀ (discarded) |
| 699,000,000–699,400,199 | the smoke (discarded) |

- **CMA seeds:** restarts 9451–9462; the recovery 9463–9464; the pilot 9467; the leave-one-out 9468; the smoke 9469. 9465 and 9466 are unused.
- **The prior draws:** `default_rng(9470)`.

### The guards

- **The bound records:** every stage needs this record committed, with these byte-identical to their versions at its last commit:
  - `121-c1-v3-fit/fit.json`;
  - `113-c1-fit/encodings.md`;
  - the six configs;
  - `data/benchmark/bands_v4.json`;
  - `data/locomotion/primitives.json`.
- **The code:** every stage refuses if the code differs from Part A's commit, or holds an untracked file there.
  - It covers `packages/quantum-nematode/quantumnematode`, `packages/wormbench/wormbench`, `uv.lock`, the three `pyproject.toml`, and the scripts on the fit's computational path: `c1_v4_fit`, `c1_v3_fit`, `c2_fit`, `cma_recovery`, `cma_recovery_v2`, `npe_recovery`, `npe_region_check`, `benchmark_grade`, `benchmark_bands` and `benchmark_v2_controls`.
  - Modules imported only for names off that path (`c1b_fit`, `c1_fit`, `c2_encoder_probe`, `body_kinematics_eval`, `benchmark_controls`) are not bound.
- **Part A's commit:** the last commit that changed Part A's own text. Part B's and Part C's commits leave it in place.
- **An amendment:** a dated amendment of Part A, the maintainer's decision, re-pins the code at its own commit. It is the path for:
  - a code fix;
  - a decision after a failed pilot (a new pilot, under its own record and seeds);
  - a decision after a failed recovery gate. That one adds to Part A the decision heading that `c1_v4_fit.DECIDED_MARKER` names, on a line of its own, which `run` and the later stages read in place of the gate's pass. This part carries no such heading.
- **The parts:**
  - `pilot` needs Part A.
  - `recovery-preflight` and `recovery` also need Part B, with `pilot.json` bound and its `go` true. That means the go rule, the check and the cost bar all held.
  - `run`, `select`, `grade`, `read` and `loo` also need Part C, with both recovery records bound and the gate passing.
- **Each record is written once.** Part B is committed together with `pilot.json`, and Part C with the recovery's two records.

### Disclosures

Everything seen before this part was written:

- **Logbook 125's probes** (620–646 M, CMA 9441–9448, discarded), all committed:
  - the readiness runs;
  - the four candidates, with the on-food-dwelling route passing 4 of 5;
  - the box-centre runs;
  - the diagnosis and the sensitivity checks.
- **The separation probe** (`probes/separation.py.txt` and `separation.json`, 647 M, discarded):
  - Logbook 125's passing answer sits 0.91 above its floor (0.78) in distance from x₀ (1.69), short of 1.0. The maintainer then chose one truth, the pilot's.
  - tph-1's Δ roaming fraction is −0.19 at x₀ and −0.22 at that answer, against real worms' +0.27 (z −9.0 and −9.5).
  - tax-4's Δ exits per hour is +12.6 and +10.6, against +0.31.
  - x₀'s wild-type D is 1.87 (median of 10).
  - It costs 35 s an evaluation with the mutants.
- **An arm check:** C.1 (v3)'s answer ran on 16 animals on the pooled and the following arm (99,660,000–99,660,015, inside Logbook 126's discarded profile block). It was an execution check at n = 16; its D (1.07 and 2.03) is not read.
- **The smokes** (699 M, discarded), three runs of two generations each.
  - **The first read N2's unread and PD1074's held-out dates.** It did so through `c2_fit`'s default sets, over 3 splits, against a two-generation answer, before this authorisation. Nothing was committed, and no rule here depends on it.
  - The smoke now passes band-cohort stand-ins (`c1_v4_fit.smoke_sets`), which a test checks.
  - The later two read band cohorts only.
- **Not seen:** every pilot, recovery, registered and leave-one-out simulation.

### Cost

- **The rate:** 35 s an evaluation, so about 38 s a generation at population 16 on 16 workers.
- **The projection:** at a 200-generation route the registered compute is about 31 h: 15 restarts × 200 × 38 s.
- **The extras:** the pilot adds about 2 h, the preflight about 30 min and the grading under 30 min.
- The cost bar is read in Part B.
