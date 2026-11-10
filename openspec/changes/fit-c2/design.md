## Context

- **Benchmark v4** ([Logbook 124](../../../docs/experiments/logbooks/124-benchmark-v4.md)) is VALID and the default.

  - It grades 24 wild-type statistics in six equal families at z\* 3.0781, and each mutant's lawn difference at 2.24.
  - D_fit is the small lawn, the uniform well, kinematics and activity.
  - W's minimum on the fitted families is 2.597 (95th percentile; median 1.389). It is the distance between two halves of N2's 26 unread held-out dates.

- **C.2's procedure options** are in the encoder ([Logbook 130](../../../docs/experiments/logbooks/130-c2-encoder-procedure.md); `c2_encoder_probe.PROCEDURE_SETTINGS`): per-ligand tying, 137 parameters, the soft cap and pool centres, the graded box and inhibition floor 0.09, PDFR-1's and SER-5's signs, and OU speed noise.

- **The speed-following reversal** ([Logbook 132](../../../docs/experiments/logbooks/132-speed-following-reversals.md)) is `continuous.on_food_reversal_speed: follows_run_speed`, unset in every config.

- **The evidence** ([Logbook 131](../../../docs/experiments/logbooks/131-c2-clean-start.md), discarded seeds, description):

  | Step | Fitted | Start | Generations | Its answer |
  |---|---|---|---|---|
  | B04 stage 1 | the 77 sense-to-output parameters, D_fit's families; 64 animals, population 24 | the graded box's centre | 150 | responsive; on the lawn 81% |
  | 2a | all 137, D_fit's families | stage 1's answer, the speed noise's timescale at 1 s, σ₀ 0.1 | 300 | D_fit 1.77 |
  | 2b | all six families | 2a's answer, σ₀ 0.05 | 150 | D 1.69, five families |
  | 3 | the six plus the three mutants at half the objective | 2b's answer, σ₀ 0.05 | 150 | the wild type 0 of 5 (the onset rate alone) |
  | failure probability, under the coupled body | 3's objective plus 10 × Σ Φ((abs(z) − bar) / SD) | the third continuation's state | 50 | **the wild type 5 of 5, then 10 of 10; every mutant 15 of 15** |

  Lessons from the route:

  - an isotropic restart after a stage scatters the search: its first medians were 417 and 709, while restarted from the stage's own steps the first was 17.7;
  - stages without the mutants let serotonin's producers fall silent;
  - a distance cannot see a bar a few tenths away, but a failure probability under measured noise can.

- **Rates as measured** (16 workers, population 48, 200 animals, readouts included):

  - 77.7 s a generation on two assays (2a: 23,303 s over 300, its log's timer having restarted at the generation-33 resume);
  - 309.5 s on six with the three mutants (the speed-following refit: 15,477 s over 50).

- **Owed:** slow speed noise's positive control. Logbook 123 failed it on a worm that did not run, and Logbook 129 deferred its registered re-run to this change.

## Goals / Non-Goals

**Goals:**

- slow speed noise's positive control, re-run on an answer whose speed output responds;
- C.2's fit to the wild type and its mutants, by a procedure fixed before any scored run and identical for every later W arm;
- a readiness pilot of that exact procedure, a recovery control, then the registered fit;
- what C.2 holds out, stated and read: Scheer's held-out dates and one tax-4 leave-one-out.

**Non-Goals:**

- C.1's refit, which gets its own change before C.4, with this change's held-out reading reused;
- W's registration, whose compute is decided from the pilot's measured cost;
- C.3;
- any benchmark change;
- the per-step runner, since every stage runs on the batched evaluator.

## Decisions

### 0. Slow speed noise's positive control

- **Before Part A,** the registered control (`c2_encoder_probe.read_speed_noise_control`, Logbook 123's rule) runs on Logbook 131's answer under the procedure settings, on the speed-following arm.
- **Its points:** the answer with its timescale replaced by each of the control's timescales, and without the OU noise. Without the noise, the timescale entry is dropped from the vector, so the pool centres keep their positions.
- **It passes** when the speed output responds, the roaming 90th percentile rises by the control's bar at some timescale, and at the shortest every kinematics statistic is within the control's band of independent noise.
- **The responsiveness bound reads the speed output before the soft cap,** 2·atanh(m/2). The rule bounds the readout's median within ±2, which a linear readout can leave; under the cap the output itself never leaves ±2, so the bound would hold by construction. Both medians are recorded.
- **If it fails,** the change returns to the maintainer before Part A, since every stage relies on the noise.
- **Its seeds:** 888,000,000 + i, with the control's own reference episode. About 30 min.

### 1. The grading arm

`generate_assay_configs.py` gains the arm `encoded_following`. Its six assay configs are the encoded arm's, with `environment.continuous.on_food_reversal_speed: follows_run_speed`, and every other key is identical. `benchmark_grade.ARMS["connectome_c2_following"]` names them, and every stage of `c2_fit.py` simulates on that arm. The committed-configs test and the assay smoke pick the arm up from the generator's `ARMS`.

### 2. The objective

For a candidate's record r against a truth T (the bands' centres, or F.0b-ii's observation):

- **The ranking score:** `cma_recovery_v2.ranking_score(T, r, scales, family_map)` = 1,000·u + D_c over the stage's families. Scales are the bands' at n = 200.

- **The mutants at half the objective,** where the stage fits them:

  - each mutant's graded statistics are the encoding table's (`113-c1-fit/encodings.md`), read from it, not from `c2_encoder_probe.mutant_reading`, which grades both lawn statistics for every mutant:

    | Mutant | Graded | Reported only |
    |---|---|---|
    | tph-1 | the roaming fraction, direction more | leaving (exits per on-lawn hour) |
    | pdfr-1 | the roaming fraction and leaving, by magnitude | |
    | tax-4 | the roaming fraction, direction less; leaving, by magnitude | |

    Logbook 131's objective, failure probability and tph-1 grade included tph-1's leaving, whose band reference the table excludes as unreliable. Its passes on both statistics imply passes on the roaming fraction alone.

  - each mutant's term t_m is the RMS of 10·asinh(z/10) over its graded statistics, or 10 if none is defined;

  - z is its difference from the wild type on the same animals, against T's difference on the mutant band's scale;

  - with n_f families and n_m mutants, each mutant weighs w = n_f / n_m, and score = 1,000·u + sqrt((n_f·D_c² + w·Σ t_m²) / (n_f + w·n_m)).

- **The saturation guard** adds 500 when over half the motor neurons sit at abs(h) > 0.99 in over half the steps.

- **The graded responsiveness guard** adds 300·(1 − s/0.5) when s < 0.5, where s is the smaller of the speed and reversal outputs' spread between their 5th and 95th percentiles, over 8 recorded animals.

- Both guards read the network through `ConnectomeTopology.forward_observer`, on the wild type's simulation only, before any mutant runs.

- **The failure probability,** in the final stage only, adds 10·Σ Φ((abs(z) − bar) / SD):

  - the sum runs over every defined statistic, the 24 wild-type ones and each fitted mutant's;
  - z is uncompressed against T;
  - the bars are read from the bands (`bands["bars"]`: 3.07809 for a wild-type statistic, 2.24140 for a mutant's);
  - SD is the noise step's (decision 4).

  Its sum is about 10 × the expected number of statistics failing a population.

### 3. The procedure

Every stage runs on the speed-following arm, with population 48 and 200 animals a candidate, at least D38's 200. Every stage has both guards, sep-CMA (`CMA_diagonal`) and bounds [0, 1]. The answer of each stage is its mean (`xfavorite`) clipped to the box.

| Stage | Fits | Families | Mutants | Start | Generations |
|---|---|---|---|---|---|
| **1** | the 77 parameters of the wiring, sensory, readout and noise blocks and the pool centres; the other 60 held at 0.5 | D_fit's four (two assays) | none | x₀ = the graded box's centre, 0.5 everywhere, so every sensory gain is 0; σ₀ 0.2 | 150 |
| **2a** | all 137 | D_fit's four | none | stage 1's answer, the speed noise's timescale set to 1 s; σ₀ 0.1, fresh | 300 |
| **2b** | all 137 | all six | tph-1, pdfr-1, tax-4 at half | 2a's search state | 200 |
| **noise** | none: 2b's answer on 30 populations | all six | all three | | |
| **final** | all 137 | all six | all three at half, plus the failure probability | 2b's search state | 50 |

- **The search state passes** from 2a to 2b and from 2b to the final stage. Each later stage starts a new strategy at the previous stage's answer, with its σ and per-parameter steps (`CMA_stds` = stds/σ), and its own CMA seed. Stage 1 to 2a cannot carry the state, since the dimension changes, so 2a starts fresh as the curriculum's did.

- **The timescale at 1 s** is the curriculum's choice. With the timescale at its centre (8.7 s), B04's second stage stalled at D about 5 with its speed distributions far off.

- **Fixed budgets:** every stage runs its generation count. The plateau rule (`cma_recovery.plateau_reached`: from generation 40, the last 20 medians' mean not 0.05 below the 20 before) is read on the final stage and reported, not used as a stop.

  - This matches W.2's note: every arm runs a fixed count, so arms are compared at equal budget.
  - The final stage of the evidence was still improving at 50 generations, and passed.

- **The final stage of the evidence** was still improving at 50 generations and passed, but it started from a far better-tuned point than B's final stage will (the next table).

- **How B differs from the route as run.** The route is readiness evidence. B fixes a procedure in advance, and the pilot tests it:

  | | The route (Logbook 131) | Procedure B |
  |---|---|---|
  | stage 1 | 64 animals, population 24 (B04) | 200 animals, population 48, D38's minimum |
  | after 2a | 500 generations: 2b 150, stage 3 150, four continuations of 50 | 250: 2b 200, the final stage 50 |
  | the mutants | from stage 3 | from 2b, the maintainer's choice, so serotonin's producers are shaped while the search is broad |
  | search state | isotropic restarts at 2b and stage 3 (σ₀ 0.05); carried from stage 3 on | carried from 2a on |
  | the final stage's start | σ 0.010, steps 0.0016–0.022, after four tuned stages | 2b's end state |
  | the body | the pooled on-food reversal speed until the last stage, then a patched line clipped to [0.049, 0.084] | the package's speed-following knots throughout |
  | the noise | the largest of four measurements (15, 10, 10 and 10 populations, on three answers) | one measurement of 30 populations, on 2b's answer |
  | tph-1 | graded and fitted on both lawn statistics | the encoding table's roaming fraction alone |

### 4. The noise step

- 2b's answer is graded on 30 populations of 200, without presence. Presence leaves every statistic's value unchanged, and grading without it is 1.76× faster.
- Each statistic's SD between populations (ddof 1, undefined populations dropped) is floored at 0.05. A statistic defined in fewer than 2 populations takes the floor.
- The record keeps every population's values.
- Thirty populations put an SD's sampling error near 13%. The SDs also move between answers: pre-exit enrichment's ran 1.00–1.39 across Logbook 131's measurements, more than that error. The risk is stated below.

### 5. Readouts and state

- **Every 10 generations, and at each stage's generation 0,** the stage's mean is graded on the readout animals: the same 200 at every readout, all six families and the three mutants, without presence.
- **Every statistic's value and z is recorded,** with D, D_fit, the families passing, time on the lawn, the output spreads and each mutant's reading.
- **State is saved atomically after every generation:** the strategy, the stage, and the medians and best scores. A resumed run refuses a saved state whose recorded settings differ from the run's.
- **Simulation records** go to a JSON-lines store keyed by k. A stored k at another vector or base is refused, and a torn last line is dropped (`c1_v3_fit.run_jobs`'s rule).

### 6. The answer, the verdict and what is reported

- **The answer** is the final stage's mean at generation 50, clipped to the box.
- **The registered grade:** the answer on 5 registered populations of 200, all six families with presence (`grade_populations_v4`), and each mutant on the small lawn on its population's own animals.
- **The verdict:** C.2 **passes** when the wild type passes in at least 3 of the 5, and tph-1 and tax-4 each pass in at least 3 of the 5.
  - A population passes the wild type when every family passes, with presence for local search and chemotaxis.
  - A mutant passes a population when each of its table-graded statistics passes at the mutant bar, in its resolved direction.
  - pdfr-1, whose directions are not resolved, is graded and reported. It is not part of the verdict.
  - **The verdict is in-sample:** every graded statistic and both resolved mutants are fitted. What C.2 holds out is read beside it (decisions 8 and 9).
- **The verdict map:**
  - **pass:** C.2 enters C.4, and W's arms run this procedure, with W's compute decided from the measured cost before W registers;
  - **fail:** the failing statistics and each stage's readouts are the diagnosis. The roadmap's C.2 risk row goes to the maintainer, and no W arm registers on this procedure.
- **Reported beside, never re-read as the verdict:**
  - the answer on 10 more populations, a pass rate for the wild type and each mutant;
  - the held-out reading (decision 8);
  - every stage's readouts;
  - the final stage's plateau reading;
  - the reversal share against Roberts' 17.3%;
  - the cost per stage;
  - **the restart spread:** the median over g < 15 of the capped D between the pilot's population g and the registered answer's population g, on W's fitted families (the small lawn, kinematics, activity) beside W's minimum, and on all six beside it.
- **Description-only outcomes, named in advance:**
  - **a pass with a poor held-out reading** (the answer's median distance to a held-out half above the band-cohort reference's 95th percentile): C.2 passes, and C.4 states the held-out distance as a limit of the fit;
  - **a pass with the leave-one-out's tax-4 failing:** C.2 passes, and C.4's tax-4 row reads as fitted, not predicted.

### 7. The readiness pilot

- **The registered procedure,** run once on discarded seeds before any registered stage, configured exactly as the fit.

- **Its go rule, fixed in Part A:** the pilot's answer, graded with presence on 5 discarded populations, passes the wild type in at least 3, and tph-1 and tax-4 each in at least 3.

- **Its kill rules,** to save the machine, again from the curriculum. The pilot stops and returns to the maintainer if:

  - the mean of stage 1's last three readouts (generations 130, 140 and 150) is unresponsive (spread below 0.5) or on the lawn less than 75% of the time. B04's readouts swung from 0.44 to 0.83 between generations 50 and 60, and plateaued at 0.80–0.87, so one readout is too noisy to kill on;
  - the mean of 2a's last three readouts has D_fit above 2.5.

  In the registered runs these checkpoints are reported, not stops.

- **If the go rule fails,** the procedure is diagnosed and Part A is amended before anything registered runs.

- **It also gives:**

  - F.0b-ii's truth;
  - the cost per stage, for W's sizing;
  - the final stage's plateau reading;
  - the pass rate on 10 more populations.

### 8. The held-out reading

The fitted worm against held-out recording dates, on W's own rule (`benchmark_v4_controls.w_minimum_v4`'s families, splits, sizes and scales):

- **The rule:**
  - for each of 1,000 splits of a date set into halves (seed 2034), the scale is PD1074's band scale at the half's size rounded to 25;
  - the families are the small lawn, kinematics and activity (its graded statistics);
  - for each split, three distances to each half:
    - from the other real half, which is W's minimum;
    - from each of the answer's 15 graded populations (the verdict's 5 and the pass rate's 10);
    - from each of 15 populations of 200 animals drawn without replacement from PD1074's band cohorts on the small lawn (`default_rng(2048)`), the reference a faithful simulator would match;
  - no new simulation.
- **The sets:**
  - **N2's 26 unread held-out dates** (`benchmark_v2_controls.n2_unread_dates`). This is W's minimum's own set, so its real-against-real 95th percentile reproduces 2.597, which a test checks.
  - **PD1074's 24 held-out dates,** the band strain, beside it. Disclosed: these cohorts are the source of the primitives' on-food reversal values and the speed coupling's knots, and B.4 read them.
- **Reported, as description:** each distance's median and 95th percentile. The comparison that reads is the answer's against the band-cohort populations'. A sample of 200 band animals spans many dates, as a simulated population has no date effect, so the two are matched in size and structure. Two real halves differ also by their dates. On N2 both the answer and the band populations carry the PD1074-to-N2 strain offset.
- **Not read:**
  - the uniform well's single held-out date (31 animals);
  - chemotaxis, which has no held-out recording;
  - López-Cruz's 10 held-out batches, which are batch means, not tracks, and are left to C.4.
- **The function is shared:** C.1's refit before C.4 reads its answer the same way.
- **W's held-out-family minimum** (2.05, local search and chemotaxis) no longer has a held-out family to read against, since C.2 fits all six. W's registration re-defines its readings.

### 9. The tax-4 leave-one-out

- After the registered fit, the same procedure runs on its own seeds with tax-4 out of the objective. tph-1 and pdfr-1 are then together half (w = 3 each), and the failure probability omits tax-4's statistics.
- tax-4 is read at every readout, and its answer is graded on 5 + 10 populations as a prediction, beside the registered fit's fitted tax-4.
- **Description only:** it gates nothing and changes no verdict. It is the one mutant C.4 can read as predicted.
- It needs about 27 h, and runs after the registered fit's read.

### 10. F.0b-ii, C.2's recovery control

- **The truth:** the pilot's answer, z_T.

- **The observation:** z_T simulated on all six assays and the three mutants at 200 animals (q = 0). It replaces the bands' centres and the mutant bands' centres as T in decision 2. The scales are unchanged.

- **The recovery distance** D_rec is the registered capped D (`predictive_distance_v2(capped=True)`) on the six families. The three mutants enter as the objective enters them, each as the RMS of its capped z against the truth's difference, together half.

- **The preflight** (`recovery-preflight`, run once):

  | Quantity | From |
  |---|---|
  | floor | the median D_rec from the observation to re-simulations 1–10 |
  | oracle | the median over re-simulations 11–20 |
  | prior D | the median over 200 uniform draws of the box (`default_rng(5805)`) |
  | box centre | the median over 10 re-simulations of the procedure's x₀ |

  Gain = (prior D − D) / (prior D − floor). Before the recovery fit, each of these must hold, or the recovery returns to the maintainer:

  - prior D − floor ≥ 1.0;
  - the oracle's gain ≥ 0.85;
  - the box centre's gain ≤ 0.25.

- **The truth's stability:** if any of re-simulations 1–10 flips a statistic between defined and undefined against the observation, the recovery returns to the maintainer. The truth is fixed, so no next candidate is taken.

- **The wild-type reference,** reported beside the gate: a record at the bands' centres and the mutant bands' centres, scored by the same gain.

- **Preflighted on a stand-in** (Logbook 131's answer; discarded seeds; `build/c2-fit/standin_preflight.py` and `standin_gate_reach.py`, committed to Part A's records):

  | Quantity | Stand-in | Bar |
  |---|---|---|
  | floor | 0.36 (re-simulations 1–4: 0.31–0.44) | |
  | the oracle's gain | 0.996 | ≥ 0.85 |
  | prior D | 6.41, the median of 20 draws (4.37–8.59) | prior D − floor ≥ 1.0: 6.05 |
  | the box centre's gain | 0.074 (D 5.95) | ≤ 0.25; fails only for a prior D above 7.47 |
  | the wild-type reference's gain | 0.872 (D 1.13) | reported |
  | definedness flips | none in 8 | none |
  | the gate's line | D_rec ≤ 1.87 | |

  Every pre-fit check passes with room. The gate is reachable, but not with room:

  - **B02's answer,** from another lineage and fitted on the pooled body, gains 0.80 and 0.81;
  - **the truth's own lineage** (the third continuation) gains 0.96–0.97.

  A recovery that reaches the wild-type region as far from the truth as B02 sits about 5 points above the bar. The recovery fits the truth's observation, not the bands, so it should land nearer than that.

- **The recovery fit:** the registered procedure from the box centre, every stage, budget, guard and noise step identical, against the observation, on its own seeds.

- **The gate:** its answer re-simulated 10 times. **The median gain must be at least 0.75**, F.0b-i's bar. If it falls short, no registered fit runs until the maintainer decides.

- **Reported:**

  - each stage's readouts against the truth;
  - the answer's distance from z_T, by block;
  - the answer's own wild-type grade against the bands, as description.

- **One truth, not F.0b-i's four.** The cost is the reason: each recovery is a full run, about 30 h. The restart-to-restart spread is the limit it accepts, as the D38 amendment states.

### 11. Seeds

Each run of the procedure has a block of 20,000,000 animal seeds from its base B, and four CMA seeds, one per stage.

| Within a block | Use |
|---|---|
| B + 4,000,000·(s − 1) + 200·(48·(n − 1) + m) + i | stage s = 1–4 (1, 2a, 2b, final), generation n, member m; 2a's 14,400 candidates reach 2,880,000 |
| B + 16,000,000 + i | the readout animals, the same at every readout |
| B + 17,000,000 + 200·p + i | the noise populations, p < 30 |
| B + 18,000,000 + 200·g + i | the grade: g < 5 the verdict (or the go rule), 5 ≤ g < 15 the pass rate; in the recovery, the answer's re-simulations, g < 10 |

| Run | Base | CMA | Status |
|---|---|---|---|
| the pilot | 900,000,000 | 9504–9507 | discarded |
| F.0b-ii's recovery fit | 920,000,000 | 5801–5804 | registered |
| the truth's observation and re-simulations | 940,000,000 + 200·q + i, q ≤ 20 | | registered |
| the prior draws | 941,000,000 + 200·j + i, j < 200 | `default_rng(5805)` | registered |
| the box centre | 942,000,000 + 200·q + i, q < 10 | | registered |
| the registered fit | 960,000,000 | 5811–5814 | registered |
| the tax-4 leave-one-out | 980,000,000 | 5821–5824 | registered, description |
| the speed-noise control | 888,000,000 + i | | the control's reading |
| the stand-in preflight | the truth 885,000,000 + 200·q + i (q < 9), prior draws 886,000,000 + 200·j + i (j < 20, `default_rng(9512)`), box centre 887,000,000 + 200·q + i (q < 2) | | discarded |
| the smoke | 99,880,000–99,889,999, on its own layout (below) | 9508–9511 | discarded |

- **The smoke's layout,** 8 animals and population 6, fits its own range:

  - stage s at 99,880,000 + 2,000·(s − 1) + 8·(6·(n − 1) + m) + i, for two generations;
  - readouts at + 8,000;
  - 3 noise populations at + 9,000 + 8·p;
  - 2 grade populations at + 9,500 + 8·g.

  The fit's layout would have put the smoke inside C.1's registered block (110–125 M).

- Every block is clear of every earlier registration and probe; the highest so far is 880,000,000 + 200·g + i.

- The CMA seeds lie in C.2's reserved band (5801–5864) or above the probes' 9503.

- Tests check that no two blocks overlap.

### 12. The registration and the guards

Launch record 133 has three dated parts:

- **Part A, before the pilot:**

  - every rule above, and the seeds;
  - the go rule and the kill rules;
  - the recovery's preflight checks and gate;
  - the verdict and its map;
  - the readings reported beside it.

  It discloses Logbook 131's probes and binds `113-c1-fit/encodings.md` byte-identical.

- **Part B, after the pilot, before the recovery's preflight:** the pilot's result, the go rule read, its cost per stage and its plateau reading. It changes no rule.

- **Part C, after the recovery, before the fit:** the preflight's and the recovery's results. It changes no rule.

**What a five-day chain reads, bound.** Part A records a manifest of SHA-256 digests:

- `c2_fit.py` and every analysis script it imports;
- the arm's assay configs;
- `bands_v4.json`, `primitives.json` and the neuromodulator data files;
- the encoding table.

Every stage checks the manifest at start and at each resume, and refuses on a mismatch. It also requires `benchmark_grade.v4_is_valid()`. Every record states HEAD and the manifest's digest.

The guards, through `cma_recovery.launch_committed`:

- `pilot` refuses unless Part A is committed and unchanged;
- `recovery-preflight` needs Part B, with `pilot.json` bound and its go rule passing, and refuses if its record exists;
- `recovery` needs the preflight's record committed, unchanged and passing;
- `run`, `grade`, `read` and `loo` need Part C, with both recovery records bound and the gate passing;
- `smoke` is unguarded, on discarded seeds. It runs the held-out reading's code on two band-cohort date sets, never the held-out ones.

### 13. The roadmap amendments

- **D37**, amended 2026-11-10 (the maintainer's decision):
  - C.2's fit and C.1's refit before C.4 fit the wild type on all six families, together with their encodable mutants (C.2: tph-1, pdfr-1, tax-4; C.1: tph-1, tax-4), the mutants together half the objective, under the registered encoding table unchanged.
  - The reason: wild-type data do not identify C.2's perturbation responses (Logbook 128).
  - Local search is no longer held out for C.2.
  - Both hold out Scheer's held-out recording dates, read beside the distance between two halves of them, and one tax-4 leave-one-out, as description.
  - C.4's mutant rows compare fitted mutants, with tax-4's leave-one-out the one prediction.
  - C.3 is unchanged: local search stays held out for it.
- **D38**, amended 2026-11-10:
  - C.2's procedure is staged, with each stage's families, budget, x₀ and σ₀ (or carried search state) pre-registered, and every stage a fixed generation count.
  - The plateau criterion is read on the final stage and reported, not used as a stop, so W's arms run at equal budget.
  - F.0b-ii recovers one truth, the readiness pilot's answer, by the registered procedure from the box centre, before the registered fit.
    - One truth, against F.0b-i's four, because a run costs about 31.5 h.
    - The truth sits beside the wild type, so the gate cannot tell recovering it from fitting the wild type. A record at the bands' centres is scored beside the gate. F.0b-ii therefore tests whether the procedure reaches the region from the box centre, and the spread between runs. It does not test identifying the truth.
  - C.2's mutants are fitted, so its mutant readings are not posterior-predictive intervals.
    - With one registered run, the encoding table's rule for a single passing restart gives each mutant's verdict, and its 15 populations' pass rates are its predictive spread.
    - The leave-one-out's tax-4 is one run, read as description.
- **The tracker's W.2 note** (a fixed count set from checkpoints at 250, 500 and 750 of a 1,000 cap) is superseded by this procedure's per-stage counts, in a dated update.

## Risks / Trade-offs

- **[Procedure B is untested as a whole]**
  - 2b has to find every family and the mutants at once from 2a's state. In the curriculum, 2b without the mutants reached five families.
  - Mitigation: the pilot runs the exact procedure with a go rule and kill rules. A failed pilot costs at most 30 h and changes no registered reading.
- **[Noise measured on 2b's answer, not the final one]**
  - The final stage moves the answer, and its SDs may differ there. Logbook 131's ranged across answers: pre-exit enrichment 1.00–1.39, the onset rate 0.09–0.15.
  - The floor of 0.05 bounds a near-zero SD, but not a statistic whose SD shrinks at the final answer. The final stage's readouts record every statistic, so a drift is visible.
- **[One restart]**
  - A single fit can land in a weaker basin.
  - The pilot and the registered fit are two runs of the same procedure, so their answers' distance is reported as a restart spread. The verdict is read on the registered run alone.
- **[One recovery truth]**
  - It cannot show recovery across the box. It shows whether the procedure finds a known answer's behaviour from the box centre, near the wild type, where the fit works.
- **[The mutants are fitted]**
  - C.2's tph-1, pdfr-1 and tax-4 passes are fits, not predictions. Only the leave-one-out's tax-4 is predicted, and as description.
  - C.4's mutant rows say so, as D37 amended states.
- **[The stage-1 subset, the timescale and the bars were chosen on probes]**
  - They are disclosed in Part A.
  - Choices tuned on the real wiring can favour it even when every W arm gets them. That is a limit for W to state, and W reports each arm's stage-1 result.
- **[N2 against PD1074]**
  - The held-out reading's primary set is N2's, W's own. A strain difference raises the answer's and the band populations' distances to it alike, and the PD1074 reading beside it shows the size.
- **[Fixed budgets against D38's plateau criterion]**
  - A stage still improving at its cap is reported as such, and the final stage of the evidence was.
  - The amendment states the trade: equal budgets for W's arms.
- **[F.0b-ii near the wild type]**
  - It cannot fail a fit that merely reaches the wild type, as the amendment states. The registered fit's own verdict and the pilot's go rule carry that test instead.
- **[Cost]**
  - About 124 h on one machine: three runs, the leave-one-out, the control and the preflight.
  - Every stage is resumable, and each part's guard stops a later stage from running on a failed earlier one.
