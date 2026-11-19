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

- **The lawn guard** (amended 2026-11-13, decision 3b) adds 300·(0.95 − t)/0.95 when t < 0.95, where t is the wild type's time on the small lawn in the same simulation, an undefined time counting as 0. It applies in every stage.

- **The failure probability,** in the final stage only, adds 10·Σ Φ((abs(z) − bar) / SD):

  - the sum runs over every defined statistic, the 24 wild-type ones and each fitted mutant's;
  - z is uncompressed against T;
  - the bars are read from the bands (`bands["bars"]`: 3.07809 for a wild-type statistic, 2.24140 for a mutant's);
  - SD is the noise step's (decision 4).

  Its sum is about 10 × the expected number of statistics failing a population.

- **The hinge,** in the final stage only (amended 2026-11-19, decision 3e), adds 5·Σ max(0, abs(10·asinh(z/10)) − target)²:

  - the sum runs over every defined wild-type statistic and each fitted mutant's graded statistics;
  - the target is 0.3 inside each pass rule: 2.77809 for a wild-type statistic, 1.94140 for a mutant's. This is Logbook 131's hinge.

- **Chemotaxis presence,** in 2b and the final stage (amended 2026-11-19, decision 3e), adds Σ over its two lower bounds b of 5·max(0, 2 − b/s)² + 10·Φ(−b/s):

  - the bounds are the registered grade's (2,000 resamples of the candidate's animals, the 2.5% quantile), for the drift and the log quintile ratio, so those stages' candidates are graded with presence;
  - s is each bound's SD over the fifth pilot's 15 grade populations: 0.000177 mm/s and 0.0443;
  - b/s is floored at −10, and an undefined bound takes the floor.

### 3. The procedure

Every stage runs on the speed-following arm. Stage 1 runs with population 24 and 64 animals a candidate, as the route's did (amended 2026-11-12, decision 3a). Every later stage runs with population 48 and 200 animals a candidate, at least D38's 200. Every stage has the three guards (the lawn guard from decision 3b), sep-CMA (`CMA_diagonal`) and bounds [0, 1]. The answer of each stage is its mean (`xfavorite`) clipped to the box.

| Stage | Fits | Families | Mutants | Start | Generations |
|---|---|---|---|---|---|
| **1** | the 77 parameters of the wiring, sensory, readout and noise blocks and the pool centres; the other 60 held at 0.5 | D_fit's four (two assays) | none | x₀ = the graded box's centre, 0.5 everywhere, so every sensory gain is 0; σ₀ 0.2 | 150 |
| **2a** | all 137 | D_fit's four | none | stage 1's answer, the speed noise's timescale set to 1 s; σ₀ 0.1, fresh | 300 |
| **2b** | all 137 | all six, plus chemotaxis presence (decision 3e) | tph-1, pdfr-1, tax-4 at half | 2a's answer; a new isotropic search, σ₀ 0.05 (decision 3d; until 2026-11-16 2a's search state) | 200 |
| **noise** | none: 2b's answer on 30 populations | all six | all three | | |
| **final** | all 137 | all six, plus chemotaxis presence (decision 3e) | all three at half, plus the failure probability and the hinge (decision 3e) | 2b's search state | 50 |

- **The search state passes** from 2b to the final stage: a new strategy at 2b's answer, with its σ and per-parameter steps (`CMA_stds` = stds/σ), and its own CMA seed. Stage 1 to 2a cannot carry the state, since the dimension changes, so 2a starts fresh as the curriculum's did. 2b starts fresh too, at σ₀ 0.05 (decision 3d), as the curriculum's 2b did.

- **The timescale at 1 s** is the curriculum's choice. With the timescale at its centre (8.7 s), B04's second stage stalled at D about 5 with its speed distributions far off.

- **Fixed budgets:** every stage runs its generation count. The plateau rule (`cma_recovery.plateau_reached`: from generation 40, the last 20 medians' mean not 0.05 below the 20 before) is read on the final stage and reported, not used as a stop.

  - This matches W.2's note: every arm runs a fixed count, so arms are compared at equal budget.
  - The final stage of the evidence was still improving at 50 generations, and passed.

- **The final stage of the evidence** was still improving at 50 generations and passed, but it started from a far better-tuned point than B's final stage will (the next table).

- **How B differs from the route as run.** The route is readiness evidence. B fixes a procedure in advance, and the pilot tests it:

  | | The route (Logbook 131) | Procedure B |
  |---|---|---|
  | stage 1 | 64 animals, population 24 (B04) | the same from 2026-11-12 (decision 3a); first registered at 200 animals and population 48 |
  | the first half | one start | up to six starts, the first that holds the lawn and reaches 2a's bar kept (decisions 3a and 3c) |
  | the lawn | no guard | a guard below 0.95 on the lawn, in every stage (decision 3b) |
  | after 2a | 500 generations: 2b 150, stage 3 150, four continuations of 50 | 250: 2b 200, the final stage 50 |
  | the mutants | from stage 3 | from 2b, the maintainer's choice, so serotonin's producers are shaped while the search is broad |
  | search state | isotropic restarts at 2b and stage 3 (σ₀ 0.05); carried from stage 3 on | an isotropic restart at 2b (σ₀ 0.05), carried from 2b on (decision 3d; carried from 2a on until 2026-11-16) |
  | the final stage's start | σ 0.010, steps 0.0016–0.022, after four tuned stages | 2b's end state |
  | the body | the pooled on-food reversal speed until the last stage, then a patched line clipped to [0.049, 0.084] | the package's speed-following knots throughout |
  | the noise | the largest of four measurements (15, 10, 10 and 10 populations, on three answers) | one measurement of 30 populations, on 2b's answer |
  | tph-1 | graded and fitted on both lawn statistics | the encoding table's roaming fraction alone |

### 3a. The first half's starts (amended 2026-11-12)

The first readiness pilot was stopped at 2a's end by its kill rule: D_fit 3.13 on the mean of its last three readouts, against 2.5. Its worm held the lawn 80–92% of the time through all 300 generations of 2a. Everything since ran on discarded seeds, each rule fixed before its run (records under `133-c2-fit/probes/`).

| Start (stage 1) | 2a | On the lawn, 2a's generations 40–60 | 2a against the bar (D_fit 2.5, 0.97 on the lawn) |
|---|---|---|---|
| B04's, 64 animals and population 24 | the curriculum's, on the pooled body | 0.993 | reaches it |
| B04's | the same replayed on the registered arm | 0.997 | reaches it by generation 100 |
| B04's | another CMA seed | 0.982 | reaches it by generation 210 (D_fit 2.4986) |
| screen (b)'s, 64 and 24 | its own, restarted at generation 100 | 0.975 | reaches it by generation 230 |
| screen (a)'s, 200 and 48 | its own, restarted at generation 100 | 0.912 | does not by generation 250 (3.53) |
| the first pilot's, 200 and 48 | its own | 0.854 | does not by generation 300 (3.13) |
| start (c)'s, 64 and 24 | its own | 0.847 | does not hold the lawn; stopped at generation 95 for the pilot's launch (D_fit 4.33), not read |

- **What it shows.**
  - The body is not the cause: the curriculum's 2a replayed on the registered arm keeps the curriculum's pace.
  - Two of five independent stage-1 answers lead to a 2a that holds the lawn, and every 2a that holds the lawn by generation 60 reaches the bar. Those that do not hold it by then never do.
  - A 2a that reaches the bar takes up to 230 generations, so 2a's 300 stay.
  - Stage 1's size is not shown to matter: 2 of 3 starts at 64 and 24 hold the lawn, and 0 of 2 at 200 and 48.
- **The amendment.**
  - **Stage 1 runs at 64 animals and population 24.** A start costs about 0.8 h against 3.2 h. It needs D38's note (decision 13): stage 1 only finds a starting point, and every later stage, readout and grade stays at 200 animals.
  - **The first half is started up to six times in sequence,** each start from the box centre with its own CMA seeds and candidate animals, read on the run's readout animals.
  - **A start is dropped at 2a's generation 60** when its mean time on the lawn over the readouts at generations 40, 50 and 60 is below 0.95. The bar sits between the lowest reading of a 2a that reached the bar (0.975) and the highest of one that did not (0.912).
  - **The first start that holds the lawn** runs 2a to its 300 generations, and the procedure continues from it.
  - **A run in which no start holds the lawn has no answer.** The pilot stops. The recovery's gate is not met. The registered fit's verdict is fail (the maintainer's decision, 2026-11-12).
  - **An undefined time on the lawn counts as 0** in the reading.
  - **F.0b-ii's preflight gains a check:** the truth's observation is itself at 0.95 or above on the lawn. A truth below the bar could not be recovered by a start the procedure keeps.
  - The rule is the same in every run of the procedure, W's arms included, and each run reports how many starts it dropped.
- **Why the rule cannot bias a verdict.** It selects a start on a fitted statistic of the fit's own readout animals, before any graded population is simulated. A kept start that fails is still caught by 2a's checkpoint and by the verdict. A dropped start that would have reached the bar costs a restart.
- **Six starts.** At 2 in 5, at least one of six holds the lawn with probability 0.95. The rate rests on five starts, so every run's own count is reported.
- **Cost.** A dropped start costs about 2.1 h (stage 1 0.8 h, extrapolated from 1.6 h on 8 shared workers; 60 generations of 2a 1.3 h). At 2 in 5 a run expects 1.5 dropped starts, 3.2 h, a little more than the smaller stage 1 saves (2.4 h). Five dropped starts before a kept one add 10.5 h, and six dropped starts cost 12.6 h. At 2 in 5, a run keeps no start with probability 0.047, and at least one of the pilot, the recovery and the fit does with probability 0.13 (0.17 with the leave-one-out).
- **Not selected on, recorded:** the speed output's 5th-to-95th spread at stage 1's end. It read 1.84 and 1.92 for the two starts that held the lawn and 3.20, 3.38 and 3.65 for the three that did not, a nearly on-off speed output. A two-against-three split is perfect for one random measure in five, and many were looked at.
- **A lawn guard was tried instead of restarts** (`probes/guard_probe.py.txt`): the first pilot's 2a again with a penalty below 0.95 on the lawn. It missed its lawn-holding bar at generations 40–60 (0.949), then held the lawn (0.996 at generation 80) with D_fit falling to 4.47, and was stopped at generation 88 for the pilot's launch. It would count time on the lawn twice in the objective, and it is one run, so restarts are the remedy registered. Resumed, it would say whether a guard is a deterministic alternative, which matters for W's arms. It was resumed, and decision 3b registers the guard beside the restarts.

### 3b. The lawn guard (amended 2026-11-13)

The second readiness pilot (base 1,100,000,000; `pilot_second/`) kept no start. All six were dropped at 2a's generation 60, at 0.720, 0.880, 0.795, 0.921, 0.888 and 0.833 on the lawn, and it stopped with no answer after 15.8 h. Without a guard, 2 of 11 independent starts have held the lawn. Everything since ran on discarded seeds, each rule fixed before its run (records under `133-c2-fit/probes/`):

| Run | Guarded | On the lawn | Against the bar (D_fit 2.5, 0.97 on the lawn) |
|---|---|---|---|
| the guard probe (decision 3a's), resumed during the second pilot's starts 4 and 5 | 2a | 0.949 over generations 40–60, 0.978 or above from generation 60 | reaches it at generations 200–220 (D_fit 2.44) |
| the second pilot's start 0, its 2a again | 2a | 0.974 or above from generation 20 | reaches it at generation 290 (D_fit 2.54, 2.36 and 2.26) |
| the second pilot's start 3, its 2a again | 2a | 0.958 or above from generation 30 | near D_fit 4.5 from generation 90; still running when this was amended |
| a first half from the box centre | stage 1 and 2a | stage 1 ends at 0.979, the first of 12 stage-1 answers at 0.95 or above; 2a at 0.976 or above from generation 10 | **reaches it at generation 120** (D_fit 2.48, 2.98 and 1.81) |

- **What it shows.**
  - With the guard, every run came to hold the lawn, against 2 of 11 starts without it. By the registered reading at 2a's generations 40–60, 3 of the 4 held it. The fourth, the guard probe, began from the first pilot's stage-1 answer, whose unguarded 2a had stayed off the lawn for 300 generations. It read 0.949 there, and 0.978 or above from generation 60. Starts 0 and 3 also began off the lawn (0.886 and 0.914 at generation 0) and held it from generations 20 and 30.
  - From an unguarded stage 1, the guarded 2a reached the bar in 2 of the 3 runs (generations 220 and 290), with start 3 still running. With the guard from stage 1, the one first half run reached it at generation 120, 100 generations sooner than the first of them.
  - Holding the lawn is not enough on its own. The second pilot's start 3 held it from generation 30 and stalled.
- **The amendment** (the maintainer's decision, 2026-11-13):
  - **the lawn guard enters the objective in every stage,** 300·(0.95 − t)/0.95 below 0.95 on the lawn, as the probes ran it;
  - **the restarts and their rule are unchanged:** up to six starts, each dropped at 2a's generation 60 below 0.95 on the lawn over generations 40–60;
  - **the third pilot runs on new seeds** (base 1,140,000,000, CMA 9591–9594, restarts 9651–9660). It doubles as the replicate of the guarded first half.
- **Why it cannot bias a verdict.** The band's centre is 0.978 on the lawn, with a scale of 0.006 at 200 animals. Below 0.95 the statistic sits 4.6 scales below the centre and already fails at 3.08, so at 200 animals the guard reorders only candidates that fail. At stage 1's 64 animals (scale 0.010) it reaches the passing range's lower edge, 0.947–0.950. It counts time on the lawn twice only where the statistic already fails. The final grade does not read it.
- **In F.0b-ii the guard keeps its absolute 0.95.** The preflight admits a truth at 0.95 or above on the lawn. A truth near 0.95 would have candidates at the truth itself pay the guard whenever their 200 animals read below 0.95, which would pull the recovery off it. A kept start has held the lawn at 0.95 or above, and every guarded 2a read 0.97–0.99, so this is a stated limit, not a check. The preflight's record reports the truth's time on the lawn.
- **Cost:** none in simulation, since it reads a statistic every candidate already has. A run that keeps its first start saves the 2.3 h a dropped start cost in the second pilot.
- **It acts while the search's mean holds the lawn.** Candidates off the lawn still pay it: about 10% of start 0's candidates at 2a's generations 260–273, and 30–60% of start 3's at generations 170–188. The procedure records each generation's share of candidates paying it.

### 3c. Restarts on 2a's bar (amended 2026-11-14)

The third readiness pilot (base 1,140,000,000; `pilot_third.json`) kept its first start, the first start any pilot has kept: it read 0.965, 0.974 and 0.989 on the lawn at 2a's generations 40, 50 and 60. 2a's kill rule then stopped it at 2a's end, with mean D_fit 3.83 over generations 280–300 (3.87, 3.85, 3.76) against 2.5, while it held the lawn at 0.97–0.99 from generation 30. It ran 9.9 h: stage 1 1.6 h on 8 workers, and 2a 8.3 h (56 generations on 8 workers, then 244 on 16 at about 87 s).

Every guarded 2a so far, each from its own start:

| Run | Guarded | Mean D_fit over 2a's generations 130–150 | Against 2a's bar |
|---|---|---|---|
| the guarded first half (decision 3b) | stage 1 and 2a | not run: stopped at the bar | reaches it at generation 120 |
| the guard probe | 2a | 2.92 | reaches it at generations 200–220 |
| the second pilot's start 0, again | 2a | 3.90 | reaches it at generation 290 |
| the second pilot's start 3, again | 2a | 4.62 | does not: 4.23 over generations 280–300 |
| the third pilot's start 0 | every stage | 4.65 | does not: 3.83 |

- **What it shows.**
  - The lawn guard works: every guarded run held the lawn.
  - 2a's descent now decides. Three of five reach the bar, two of them late in 2a's budget (generations 220 and 290). Two stall near 4 and stay flat for more than 100 generations.
  - Both stalls fail the same statistics: the dwelling-speed tail far too fast (z about 10), the on-lawn pause share too high (z 4–7) and the on-lawn onset rate too low (z −4 to −6). The third pilot's answer reverses for 70% of its seconds on the lawn.
  - No earlier readout separates the two groups cleanly: at generation 100 the latest of the three that reached the bar reads 4.45, and one of the stalls 4.63.
  - The dopamine wiring gain differs between the groups: −1.49 and −1.36 in the stalls, against +0.04, +0.89 and +0.84 in the three that reached the bar. A check on discarded seeds (`probes/dopamine_flip.py.txt`, its rule fixed first) mirrored that gain alone at each answer. Every answer whose gain was above 0.8 in size broke (D_fit 7.0–8.7), stalls and reachers alike, and the answer at +0.04 did not change. The gain carries weight in every answer and does not switch between the two outcomes, so it is not a remedy.
- **The amendment** (the maintainer's decision, 2026-11-14):
  - **2a reads each start twice more,** each reading a mean over three readouts: the descent at generations 130, 140 and 150, where D_fit must be at most 4.25, and the bar at generations 280, 290 and 300, where it must be at most 2.5. A start that fails any of 2a's three readings is dropped, and the next begins from the box centre, as at the lawn reading. There are still up to six starts in all, and the first to pass all three is kept. The rule holds in every run.
  - **An undefined D_fit fails a reading,** as an infinitely distant one.
  - **2a's kill rule becomes the bar reading.** The pilot no longer stops on 2a, since a start that misses the bar is dropped in every run. Stage 1's kill rule is unchanged.
  - **Each start has its own animals for all of 2a,** since any start may now run 2a to its end. Each run's block grows to 40,000,000 animals, and all four runs move to new bases (decision 11).
  - **The fourth pilot runs on new seeds:** base 1,200,000,000, CMA 9661–9664, restarts 9671–9680.
- **Where 4.25 comes from.** It is set on these five runs: at generation 150 the two stalls read 4.62 and 4.65, and the slowest of the three that reached the bar read 3.90. A drop never makes an answer wrong, only later, so a bar chosen on these runs risks time, not a verdict. It would have dropped neither run that reached the bar.
- **Why the rule cannot bias a verdict:** as decision 3a's. It selects a start on fitted statistics of the run's own readout animals, before any graded population is simulated.
- **Odds and cost.** At 3 in 5, all six starts miss with probability about 0.004. A start dropped at the lawn costs about 2.3 h, at the descent about 4.4 h (stage 1 0.9 h and 150 generations of 2a at about 85 s), and at the bar about 8 h. Both stalls in this sample would have been dropped at the descent. At 2 in 5 a run expects 0.67 dropped starts, about 3 h; five dropped at the bar would add about 40 h.
- **F.0b-ii's preflight gains a check:** the truth's observation has D_fit at most 2.5 against the bands. 2a's readings read D_fit against the bands in every run, the recovery's too, so a truth past the bar could not be recovered by a start the procedure keeps. It mirrors decision 3a's check that the truth holds the lawn.
- **For W:** W's arms run this procedure, so a fit's cost grows by its dropped starts. A null wiring may stall more often than the true one, and that rate is unmeasured. Each arm reports its starts.

### 3d. 2b starts its own search (amended 2026-11-16)

The fourth readiness pilot (base 1,200,000,000; `pilot_fourth.json`, 36.6 h) dropped its first start at the lawn (0.899) and its second at the descent (4.53). It kept its third, which passed all three of 2a's readings (0.997 on the lawn, 3.30 at the descent, 1.85 at the bar). Its answer then failed the go rule: the wild type passed in 0 of 5 populations, and tph-1, pdfr-1 and tax-4 in 5 of 5 each. In all five, local search's log early/late ratio failed (z −3.5 to −3.7), as did chemotaxis's flip fraction (4.5 to 4.6) and weathervane slope (6.3 to 8.3), the small lawn's pre-exit enrichment (−6.0 to −7.1) and the on-lawn onset rate (−3.1 to −3.5). Over 2b's 200 generations, carrying 2a's search state with the mutants at half, the mutants moved from generations 100–110: tph-1 swung about its band, and tax-4 stayed above its bar until generation 200. Local search's ratio stayed between −3.62 and −3.74, and the flip fraction between 4.41 and 5.02. Logbook 131's 2b moved both from a similar 2a answer.

The pilot's 2b differed from 131's in two ways at once: the mutants from its first generation, and 2a's carried search state where 131's restarted isotropically at σ₀ 0.05. A screen separated them (`probes/two_b_screen.py.txt`). From the pilot's 2a answer, each arm ran a new isotropic search at σ₀ 0.05, population 48 and 200 animals, on the registered arm and objective, for 60 generations: arm A without the mutants, arm B with them at half. Its rule was fixed before either ran: an arm moves when local search's log early/late z, averaged over generations 40, 50 and 60, is at least −3.45. The pilot's 2b read −3.64 and −3.62 there, and the ratio's SD between populations is 0.065.

| Arm | Generations 40, 50, 60 | Mean | |
|---|---|---|---|
| A, without the mutants | −3.35, −2.92, −2.89 | −3.06 | moves |
| B, the mutants at half | −3.34, −3.16, −3.14 | −3.22 | moves |

- **What it shows.** The result points to the carried search state, not the mutants. With a fresh search, both arms moved local search within 60 generations from the point where the carried 2b had not moved it in 200. Each arm is one CMA seed on its own animals, so this is consistent with 2a's steps, adapted to its four families, which leave local search and chemotaxis out, holding 2b where 2a left them. The mechanism was not measured. A moved further than B (−3.06 against −3.22), consistent with the mutants slowing it, but B moves.
- **The amendment** (the maintainer's decision, 2026-11-16, by the screen's rule):
  - **2b starts a new isotropic search at 2a's answer,** σ₀ 0.05, on the run's third CMA seed, with all six families and the mutants at half as before;
  - **the final stage still carries 2b's search state;**
  - **the pilot runs a fifth time on new seeds:** base 1,360,000,000, CMA 9682–9685, restarts 9686–9695.
- **Why it cannot bias a verdict.** It changes how 2b's search starts, on a rule read on discarded seeds before any registered simulation, and it holds in every run.
- **Stated, not remedied.** The flip fraction, the weathervane slope, pre-exit enrichment and the on-lawn onset rate did not pass in either arm's 60 generations. The weathervane slope fell (from 9.1 to 4.5 and 4.1) and the others barely moved. 2a's answer already failed pre-exit enrichment (−4.0) and the onset rate (−4.3) at D_fit 1.84, since D_fit's RMS can pass with two statistics failing. 131's flip fraction passed near its generation 150, and the registered 2b runs 200. The pilot's go rule reads whether they pass.
- **Cost.** None: a 2b that starts its own search costs what a carried one did. The fourth pilot measured 2b at 346 s a generation, about 19.2 h.
- **For W:** every arm's 2b starts its own search at σ₀ 0.05.

### 3e. The hinge and chemotaxis presence (amended 2026-11-19)

The fifth readiness pilot (base 1,360,000,000; `pilot_fifth.json`, 47.1 h) dropped its first four starts at the descent (4.30, 5.14, 5.63, 4.77). It kept its fifth (0.988 on the lawn, 2.28 at the descent, 1.97 at the bar). Its answer failed the go rule: the wild type passed in 0 of 5 populations, and tph-1, pdfr-1 and tax-4 in 5 of 5 each. The on-lawn onset rate failed in all 15 populations (z −4.1 to −4.5), and chemotaxis presence in 3 of the first 5. Everything since ran on discarded seeds, each rule fixed before its run (records under `133-c2-fit/probes/`):

| Run | What it asked | Reading |
|---|---|---|
| the onset sweeps | does a one- or two-parameter move at the answer raise the onset rate with everything else passing? | no: every gain of 1 in z took the dwelling speeds over their bars |
| the trace of Logbook 131's route | how did 131 reach the onset rate? | in its own basin, by a quadratic hinge stage before its failure stage, with room on the dwelling speed's median; the failure probability is 10.000 for any onset z below about −3.55 |
| the hinge screen: 50 generations from the pilot's final state, the hinge added | does the onset rate's mean z at generations 30–50 reach −3.0781, with nothing else failing and the mutants passing? | **yes**: −3.01. Its answer passed the wild type in 2 of 5 fresh populations, chemotaxis presence failing in 3 |
| the chemotaxis sweep | do 131's sensory gains restore presence at the screen's answer? | no |
| the presence screen: 50 generations from the hinge screen's state, the presence term added | is chemotaxis present in at least 2 of the readouts at generations 30–50, with the onset rate, everything else, local search and the mutants holding? | **yes**: present at every readout from generation 10. Its answer passed the wild type in 4 of 5 fresh populations, every mutant in 5 of 5 |

- **What it shows.**
  - The final stage's failure probability is flat far past a bar, so it gave the onset rate nothing at z −4.3 and charged every move that cost a statistic near its bar. The hinge pushes there.
  - The objective could not see presence. The log quintile ratio's band is wide: from the fifth pilot's answer to the hinge screen's, the ratio fell from 0.097 to 0.048 with its z moving from −1.69 to −1.76, while its lower bound fell below 0 in 3 of 5 populations.
  - On the pilot's readout animals the ratio read 0.10–0.34 through 2a, which does not score chemotaxis, and 2b, which scores it only by z, traded it away (the drift from 0.009 to 0.003 mm/s).
  - Once scored, presence came back by generation 10 and stayed. On the rule's means over generations 30–50 the onset rate and everything else held, though single readouts failed pre-exit enrichment and the onset rate.
- **The amendment** (the maintainer's decision, 2026-11-19, by the screens' rules):
  - **the final stage adds the hinge** (decision 2);
  - **2b and the final stage add the presence term** (decision 2), their candidates graded with presence;
  - **every readout's wild type is graded with presence** (decision 5);
  - **F.0b-ii's preflight adds a check:** the truth's observation, graded with presence, has both lower bounds at least 2 s above 0 (decision 10);
  - **the pilot runs a sixth time on new seeds:** base 1,540,000,000, CMA 9926–9929, restarts 9930–9939.
- **Why it cannot bias a verdict.** It changes the objective of two stages, on rules read on discarded seeds before any registered simulation, and it holds in every run. The verdict reads presence as it always has. F.0b-ii's preflight gains a check; its gate is unchanged.
- **Stated, not remedied.**
  - Each screen continued the fifth pilot's basin past its final stage, which no registered run does. The sixth pilot runs from the box's centre, so its basin may differ.
  - The term in 2b is not screened there.
  - Pre-exit enrichment sits near its bar in the presence screen's answer (−3.10 and −3.07 in its grade). It is the noisiest statistic (SD 1.14).
  - Presence's scales come from one answer.
  - Each described grade is one CMA seed.
  - The presence term can outweigh the guards: 1,460 at the floor, against the lawn guard's 300, the saturation guard's 500 and an undefined statistic's 1,000, and about 190 at a lower bound near −0.18, where the fifth pilot's 2a answer read. A 2b starting there would rank on presence alone until it cleared. The term is registered as screened, uncapped.
- **Cost.** Grading a candidate with presence cost the presence screen about 28% a generation (404 s against 315 s). 2b is then about 22.9 h and the final stage about 5.5 h, so a run is about 37 h with its first start kept.
- **For W:** every arm's 2b and final stage score presence, and every final stage takes the hinge.

### 4. The noise step

- 2b's answer is graded on 30 populations of 200, without presence. Presence leaves every statistic's value unchanged, and grading without it is 1.76× faster.
- Each statistic's SD between populations (ddof 1, undefined populations dropped) is floored at 0.05. A statistic defined in fewer than 2 populations takes the floor.
- The record keeps every population's values.
- Thirty populations put an SD's sampling error near 13%. The SDs also move between answers: pre-exit enrichment's ran 1.00–1.39 across Logbook 131's measurements, more than that error. The risk is stated below.

### 5. Readouts and state

- **Every 10 generations, and at each stage's generation 0,** the stage's mean is graded on the readout animals: the same 200 at every readout, all six families and the three mutants, with presence on the wild type (decision 3e; until 2026-11-19 without presence). No reading or kill rule reads presence.
- **Every statistic's value and z is recorded,** with D, D_fit, the families passing, time on the lawn, the output spreads and each mutant's reading.
- **State is saved atomically after every generation:** the strategy, the stage, and the medians and best scores. A resumed run refuses a saved state whose recorded settings differ from the run's.
- **Simulation records** go to a JSON-lines store keyed by k. A stored k at another vector is refused, and a torn last line is dropped (`c1_v3_fit.run_jobs`'s rule).

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
  - **the neural reading** (added 2026-11-14, the maintainer's decision): the answer's in-silico single-neuron stimulation responses against Creamer et al.'s measured ones, as Logbook 134's M-a read Logbook 131's answer, with a paired test against the unfitted network and a wiring-only control. Its rules are fixed in Part A, and it is description, never graded. *(Amended 2026-11-16, the maintainer's decision, after Logbook 135: the answer's share is of the readout's reliability ceiling, 0.423, with r(training, test) = 0.173 beside it as copying the training animals; the paired test's named interval resamples the stimulated neurons' 74 classes, with the neuron resample beside it; Logbook 135's calibrated references on held-out worms are quoted, not computed, and a calibrated reading is left to W.)*;
  - **the restart spread:** the median over g < 15 of the capped D between the pilot's population g and the registered answer's population g, on W's fitted families (the small lawn, kinematics, activity) beside W's minimum, and on all six beside it.
- **Description-only outcomes, named in advance:**
  - **a pass with a poor held-out reading** (the answer's median distance to a held-out half above the band-cohort reference's 95th percentile): C.2 passes, and C.4 states the held-out distance as a limit of the fit;
  - **a pass with the leave-one-out's tax-4 failing:** C.2 passes, and C.4's tax-4 row reads as fitted, not predicted.

### 7. The readiness pilot

- **The registered procedure,** run once on discarded seeds before any registered stage, configured exactly as the fit.

- **Its go rule, fixed in Part A:** the pilot's answer, graded with presence on 5 discarded populations, passes the wild type in at least 3, and tph-1 and tax-4 each in at least 3.

- **Its kill rules,** to save the machine, again from the curriculum. The pilot stops and returns to the maintainer if:

  - the mean of a start's last three stage-1 readouts (generations 130, 140 and 150) is unresponsive (spread below 0.5) or on the lawn less than 75% of the time. B04's readouts swung from 0.44 to 0.83 between generations 50 and 60, and plateaued at 0.80–0.87, so one readout is too noisy to kill on;
  - no start of its six is kept by 2a's readings (decisions 3a and 3c, amended 2026-11-12 and 2026-11-14).

  Until 2026-11-14 the pilot also stopped when the mean of 2a's last three readouts had D_fit above 2.5. That is now 2a's bar reading, which drops a start in every run (decision 3c). In the registered runs, stage 1's checkpoint is reported, not a stop.

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
- It needs about 26 h with its first start kept (27 h before the amendment of 2026-11-12), and runs after the registered fit's read.

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

- **The truth clears presence's target** (amended 2026-11-19, decision 3e): the observation, graded with presence, has both lower bounds at least 2 s above 0 (0.000353 mm/s and 0.0886). 2b and the final stage score presence against that target, not against the truth, so a truth below it would be pulled from. Above the target only the chance part pulls, 0.23 a bound at the target and 0.013 at 3 s. Only the observation is graded with presence, since its statistics are the same without it.

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

- **One truth, not F.0b-i's four.** The cost is the reason: each recovery is a full run, about 29 h. The restart-to-restart spread is the limit it accepts, as the D38 amendment states.

### 11. Seeds

Each run of the procedure has a block of 40,000,000 animal seeds from its base B, and four CMA seeds, one per stage (amended 2026-11-14, decision 3c: until then a block was 20,000,000 and only 2a's first 60 generations were each start's own). The first start of the first half takes the run's first two CMA seeds, and start a = 1–5 takes the run's restart seeds R + 2·(a − 1) for stage 1 and R + 2·(a − 1) + 1 for 2a (amended 2026-11-12, decision 3a).

| Within a block | Use |
|---|---|
| B + 250,000·a + 64·(24·(n − 1) + m) + i | stage 1 of start a, generation n, member m; a start's 3,600 candidates reach 230,400 |
| B + 20,000,000 + 2,880,000·a + 200·(48·(n − 1) + m) + i | 2a of start a, every generation n; a start's 14,400 candidates fill its 2,880,000, so the last start's end at B + 37,280,000 |
| B + 4,000,000·(s − 1) + 200·(48·(n − 1) + m) + i | stage s = 3 (2b, at B + 8,000,000) and s = 4 (the final stage, at B + 12,000,000), generation n, member m; the kept start's alone |
| B + 16,000,000 + i | the readout animals, the same at every readout |
| B + 17,000,000 + 200·p + i | the noise populations, p < 30 |
| B + 18,000,000 + 200·g + i | the grade: g < 5 the verdict (or the go rule), 5 ≤ g < 15 the pass rate; in the recovery, the answer's re-simulations, g < 10 |

| Run | Base | CMA | Status |
|---|---|---|---|
| the first pilot, stopped at 2a | 900,000,000 | 9504–9507 | discarded, on the first layout |
| the second pilot, which kept no start | 1,100,000,000 | 9531–9534, restarts 9541–9550 | discarded |
| the third pilot, whose kept start missed 2a's bar | 1,140,000,000 | 9591–9594, restarts 9651–9660 | discarded, on the 20,000,000 layout |
| the fourth pilot, whose answer failed the go rule | 1,200,000,000 | 9661–9664, restarts 9671–9680 | discarded |
| the fifth pilot, whose answer failed the go rule | 1,360,000,000 | 9682–9685, restarts 9686–9695 | discarded |
| the pilot (amended 2026-11-19, decision 3e) | 1,540,000,000 | 9926–9929, restarts 9930–9939 | discarded |
| the screen that diagnosed the fourth pilot | 1,400,000,000–1,400,575,999 and 1,410,000,000–1,410,575,999; readouts on 1,216,000,000 + i reused | 9922, 9923 | discarded |
| a timing check of the simulation's kernels | 1,420,000,000–1,420,000,207 | | discarded |
| the runs that diagnosed the first pilot | 1,010,000,000–1,055,000,199, and 890,000,000 + i for the checks of its worm | 9516–9525 | discarded |
| the runs that diagnosed the second pilot | 1,060,000,000–1,065,000,199, 1,090,000,000–1,095,000,199 and 1,130,000,000–1,135,000,199 | 9571, 9574, 9581–9582 | discarded |
| the check that diagnosed the third pilot | 1,170,000,000 + 1,000·p + i, p < 4 | | discarded |
| F.0b-ii's recovery fit | 1,240,000,000 (920,000,000 until 2026-11-14) | 5801–5804, restarts 5831–5840 | registered |
| the truth's observation and re-simulations | 940,000,000 + 200·q + i, q ≤ 20 | | registered |
| the prior draws | 941,000,000 + 200·j + i, j < 200 | `default_rng(5805)` | registered |
| the box centre | 942,000,000 + 200·q + i, q < 10 | | registered |
| the registered fit | 1,280,000,000 (960,000,000 until 2026-11-14) | 5811–5814, restarts 5841–5850 | registered |
| the tax-4 leave-one-out | 1,320,000,000 (980,000,000 until 2026-11-14) | 5821–5824, restarts 5851–5860 | registered, description |
| the speed-noise control | 888,000,000 + i | | the control's reading |
| the stand-in preflight | the truth 885,000,000 + 200·q + i (q < 9), prior draws 886,000,000 + 200·j + i (j < 20, `default_rng(9512)`), box centre 887,000,000 + 200·q + i (q < 2) | | discarded |
| the smoke | 99,880,000–99,889,999, on its own layout (below) | 9508–9511, restarts 9561–9566 | discarded |

- **The smoke's layout,** 8 animals and population 6, fits its own range:

  - stage s = 3 and 4 at 99,880,000 + 1,000·(s − 1) + 8·(6·(n − 1) + m) + i, for two generations;
  - four starts, stage 1 at + 200·a and 2a, 22 generations, at + 4,000 + 1,100·a. Its starts fail the lawn (bar 2.0, which no worm reaches), the descent (bar −1) and the bar (−1) in turn, at 2a's generations 10, 20 and 22, and the fourth passes all three, so the smoke runs every path;
  - readouts at + 8,500;
  - 3 noise populations at + 9,000 + 8·p;
  - 2 grade populations at + 9,500 + 8·g.

  The fit's layout would have put the smoke inside C.1's registered block (110–125 M).

- Every block is clear of every earlier registration and probe. Before this change the highest was 880,000,000 + 200·g + i. These are now earlier blocks too:

  - the first pilot (900,000,000–916,000,199) and the runs that diagnosed it (890,000,000–890,000,199 and 1,010,000,000–1,055,000,199);
  - the second pilot (1,100,000,000–1,118,002,999) and the runs that diagnosed it;
  - the third pilot (1,140,000,000–1,158,002,999) and the check that diagnosed it (1,170,000,000–1,170,003,199);
  - the fourth pilot (1,200,000,000–1,237,279,999, every start's 2a reserved), the screen that diagnosed it and the timing check (1,400,000,000–1,420,000,207, in three blocks);
  - the fit-first investigation's reserved animals (1,500,000,000–1,501,000,199, Logbook 134).

- The CMA seeds lie in C.2's reserved band (5801–5864) or, for discarded runs, in 9504–9695 and the screen's 9922–9923. They are clear of the seeds the Phase 10 table gives other work: 9601–9616, the fit-first investigation's 9701–9722, 9731, 9732 and 9741–9759, and the neural readout's ceiling's 9770–9779, 9781–9799 and 9801–9805 (Logbook 135).

- The neural reading's paired test resamples on `default_rng(9811)` (classes) and `default_rng(9812)` (neurons), clear of the same (amended 2026-11-16; `default_rng(9781)` until then, which Logbook 135's null rewirings use).

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
- **D38**, a note of 2026-11-12 (decision 3a), the maintainer's decision:
  - C.2's stage 1 runs at 64 animals per evaluation, below D38's 200. It only finds 2a's starting point, and is started up to six times. Every later stage, every readout and every grade stays at 200.
  - The first half's restarts, their selection rule and their count are pre-registered like a stage's budget.
- **D38**, a note of 2026-11-13 (decision 3b), the maintainer's decision: C.2's objective carries a lawn guard in every stage, paid only below 0.95 on the lawn, where the statistic already fails at 200 animals. It is pre-registered like the other guards and W's arms inherit it.
- **D38**, a note of 2026-11-14 (decision 3c), the maintainer's decision: a start of C.2's first half is also dropped when 2a's D_fit lags at generation 150 (above 4.25) or misses 2a's bar at its end (above 2.5), in every run. A run is kept on its first start to pass all three of 2a's readings. The readings, their bars and the count of starts are pre-registered like a stage's budget, and W's arms inherit them.
- **D38**, a note of 2026-11-16 (decision 3d), the maintainer's decision: C.2's 2b starts a new isotropic search at 2a's answer (σ₀ 0.05) instead of carrying 2a's search state, which held local search and chemotaxis where 2a left them. The start is pre-registered like a stage's σ₀, and W's arms inherit it.
- **D38**, a note of 2026-11-19 (decision 3e), the maintainer's decision: C.2's final stage adds a hinge, 5·Σ max(0, abs(10·asinh(z/10)) − target)² with the target 0.3 inside each pass rule. 2b and the final stage add a term on chemotaxis presence's two lower bounds, their candidates graded with presence. Both are pre-registered like the other terms, and W's arms inherit them.
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
  - 2b has to find every family and the mutants at once, from 2a's answer. In the curriculum, 2b without the mutants reached five families. In the fourth pilot, a 2b that carried 2a's search state moved the mutants and not local search or chemotaxis. From the same point, a fresh search moved local search with and without the mutants (decision 3d). In either arm's 60 generations, the flip fraction, the weathervane slope, pre-exit enrichment and the onset rate did not pass.
  - In the fifth pilot, the fresh 2b passed local search and the mutants but traded chemotaxis presence away, and the final stage could not raise the onset rate from z −4.3 (decision 3e). Two screens from its answer, one CMA seed each, moved both once the hinge and a presence term entered the objective. A registered run starts from the box's centre, and its basin may differ.
  - Mitigation: the pilot runs the exact procedure with a go rule and kill rules. A failed pilot changes no registered reading. From 2026-11-14 it costs at most about 48 h with no answer (six starts dropped at 2a's bar), or, from 2026-11-19, about 77 h when five are dropped there before a kept start runs to its answer. The second pilot cost 15.8 h and dropped all six starts at the lawn; the third cost 9.9 h; the fourth 36.6 h and the fifth 47.1 h, each with an answer that failed the go rule.
- **[Noise measured on 2b's answer, not the final one]**
  - The final stage moves the answer, and its SDs may differ there. Logbook 131's ranged across answers: pre-exit enrichment 1.00–1.39, the onset rate 0.09–0.15.
  - The floor of 0.05 bounds a near-zero SD, but not a statistic whose SD shrinks at the final answer. The final stage's readouts record every statistic, so a drift is visible.
- **[The first half is unreliable]**
  - Without the lawn guard, 2 of 11 starts held the lawn, and the second pilot dropped all six. With it, every one of five runs held the lawn, and 3 of the 5 reached 2a's bar (decision 3c). The rates rest on five runs, two of them from the box centre, so every run's own count is reported.
  - The descent's bar of 4.25 was set on the five runs it would have read. A bar too strict drops starts that would have reached 2a's bar, and costs time, not a verdict.
  - A run with no answer is a failure of the search, not evidence about the wiring. W has to state the same for an arm whose starts all drop, and reports each arm's count.
  - The lawn-holding rule was read off four of the lineages it now selects. Start (c) is its one reading ahead of time.
  - The guard was chosen on probes run on the real wiring. Every W arm gets the same guard. An arm that cannot hold the lawn pays it in every stage, and fails the statistic as it would without it. W's arms report their time on the lawn.
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
  - About 114 h on one machine with every first start kept, and about 127 h at 2 starts in 5: three runs, the leave-one-out, the control and the preflight (124 h before the amendment of 2026-11-12).
  - From 2026-11-14 (decision 3c), each run expects about 3 h more for starts dropped at the descent or the bar. At the third pilot's measured 2a (about 87 s a generation), the four runs come to about 119 h with every first start kept. A run that drops five starts at the bar adds about 40 h.
  - Every stage is resumable, and each part's guard stops a later stage from running on a failed earlier one.
