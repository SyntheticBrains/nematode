# Launch record: C.2 fitted to the wild type and its mutants by a staged procedure

**Part A registered 2026-11-10**, change `fit-c2`. It is committed before the readiness pilot, F.0b-ii and the registered fit, and fixes every rule, threshold and seed in the change's design. Part B, added after the pilot and before F.0b-ii's preflight, records the pilot. Part C, added after F.0b-ii and before the registered fit, records the recovery. Neither changes a rule. A verdict registered here is never re-read under a different rule. Anything else is reported beside it as description.

**Part A amended 2026-11-12,** the maintainer's decision, before any registered run. The first readiness pilot was stopped at 2a's end by its kill rule. The amendment changes the first half of the procedure only: stage 1's size, and up to six starts of the first half with one rule that keeps a start (*The first half's starts*, below). Everything seen before it is listed under *Disclosures*. No registered simulation had run, and no verdict, gate or bar after 2a changes.

**Part A amended again 2026-11-13,** the maintainer's decision, before any registered run. The second readiness pilot kept no start of six and stopped with no answer. The amendment adds a lawn guard to the objective in every stage (*The objective*, below) and moves the pilot to new seeds. The restarts and their rule are unchanged. Everything seen before it is listed under *Disclosures*. No registered simulation had run, and no verdict, gate or bar changes.

**Part A amended a third time 2026-11-14,** the maintainer's decision, before any registered run. The third readiness pilot kept its first start, which held the lawn, and stopped at 2a's end with D_fit 3.83 against 2a's bar of 2.5. The amendment does three things:

- 2a reads each start twice more and drops one that lags at generation 150 or misses 2a's bar at its end (*The first half's starts*, below). 2a's kill rule becomes that last reading, in every run.
- Each start gets its own animals for all of 2a, so every run moves to a new, larger block (*Seeds*).
- A neural reading is declared as description beside the verdict (*The neural reading*).

Everything seen before it is listed under *Disclosures*. No registered simulation had run, and no verdict, gate or bar after the first half changes.

**Part A amended a fourth time 2026-11-16,** the maintainer's decision, before any registered run. The fourth readiness pilot kept its third start, which passed all three of 2a's readings (D_fit 1.85 at 2a's end). It then failed its go rule: the wild type passed in 0 of 5 populations, and every mutant in 5 of 5. Its 2b, which carried 2a's search state, never moved local search or chemotaxis. A two-arm screen on discarded seeds, its rule fixed before it ran, found that a new isotropic search at 2a's answer moves local search with the mutants in the objective. The amendment does three things:

- 2b starts a new isotropic search at 2a's answer, σ₀ 0.05, with the mutants at half as before (*The procedure*). The final stage still carries 2b's search state.
- The pilot moves to new seeds (*Seeds*).
- The neural reading states its share against the readout's measured ceiling, 0.423, and its paired test resamples neuron classes, on new generators (*The neural reading*).

Everything seen before it is listed under *Disclosures*. No registered simulation had run, and no verdict, gate or bar changes.

**Part A amended a fifth time 2026-11-19,** the maintainer's decision, before any registered run. The fifth readiness pilot kept its fifth start (D_fit 1.97 at 2a's end) and failed its go rule: the wild type passed in 0 of 5 populations, and every mutant in 5 of 5. The on-lawn onset rate failed in all 15 populations (z −4.1 to −4.5), and chemotaxis presence in 3 of the first 5. Two screens on discarded seeds, each with its rule fixed before it ran, found what the final stage could not see. A hinge moves the onset rate where the failure probability is flat, and a term on chemotaxis presence's lower bounds restores presence. With both, the screens' answer passed the wild type in 4 of 5 fresh populations and every mutant in 5 of 5. The amendment does four things:

- The final stage adds Logbook 131's hinge to its objective (*The objective*).
- 2b and the final stage add a presence term, and their candidates are graded with presence (*The objective*). Every readout's wild type is graded with presence too (*Readouts and state*).
- F.0b-ii's preflight adds a seventh check: the truth's own lower bounds clear the presence term's target (*F.0b-ii*).
- The pilot moves to new seeds (*Seeds*).

Everything seen before it is listed under *Disclosures*. No registered simulation had run. No verdict, gate or bar changes; F.0b-ii's preflight gains a check, and its gate is unchanged.

**Part B added 2026-11-21,** after the sixth readiness pilot and before F.0b-ii's preflight. The pilot passed its go rule: the wild type in 3 of 5 populations, tph-1 and tax-4 each in 5 of 5. Its stage-1 kill at the first start was overridden by the maintainer, as Part B's disclosures state. Part B records it and changes no rule (*Part B: the readiness pilot*, at the end).

## Part A: the rules

### The controller and its arm

- **The controller:** the encoded connectome under the procedure's options (`c2_encoder_probe.PROCEDURE_SETTINGS`):
  - per-ligand tying, 137 parameters;
  - the soft-capped readout with pool centres;
  - the graded box with the inhibition floor at 0.09;
  - PDFR-1's and SER-5's cited signs;
  - slow (OU) speed noise.
- **The arm:** `benchmark_grade.ARMS["connectome_c2_following"]`. Its configs are the encoded arm's with `continuous.on_food_reversal_speed: follows_run_speed` as their only difference ([Logbook 132](../../132-speed-following-reversals.md)). The body is in the config.
- **Every stage, reading and control runs on it,** at 200 animals a candidate and population 48, except stage 1 (64 animals a candidate and population 24).

### The objective

For a candidate's record against a truth T (the bands' centres; in F.0b-ii, the truth's observation):

- **the ranking score** on the stage's families: 1,000 × the statistics undefined in one record + the weighted RMS of 10·asinh(z/10), z on the bands' scales at n = 200;
- **the mutants**, where the stage fits them:
  - each mutant's term is the RMS of 10·asinh(z/10) over the statistics the encoding table grades it on, z its difference from the wild type on the same animals against T's, capped at 10, undefined at the cap;
  - with n_f families and n_m mutants, each mutant weighs n_f / n_m, so the mutants together are half;
- **the saturation guard:** + 500 when over half the motor neurons sit at abs(h) > 0.99 in over half the steps;
- **the graded responsiveness guard:** + 300·(1 − s/0.5) when s < 0.5, s the smaller of the speed and reversal outputs' 5th-to-95th-percentile spreads over 8 recorded animals;
- **the lawn guard,** in every stage (added 2026-11-13): + 300·(0.95 − t)/0.95 when t < 0.95, t the time on the small lawn, an undefined time counting as 0. The three guards read the wild type's simulation only;
- **the failure probability,** in the final stage only: + 10·Σ Φ((abs(z) − bar) / SD) over every defined wild-type statistic and every graded mutant statistic, with the bands' bars (3.07809 and 2.24140) and SD from the noise step;
- **the hinge,** in the final stage only (added 2026-11-19): + 5·Σ max(0, abs(10·asinh(z/10)) − target)² over every defined wild-type statistic and every graded mutant statistic, the target 0.3 inside each pass rule (2.77809 and 1.94140). The failure probability is flat far past a bar (with the onset rate's SD of 0.112 it is 9.999 at z −3.5 and 10.000 below about −3.55), so a statistic failing there is pushed back by the hinge alone;
- **chemotaxis presence,** in 2b and the final stage (added 2026-11-19): for each of presence's two lower bounds b (the drift and the log quintile ratio, as the registered grade computes them: 2,000 resamples of the candidate's animals, the 2.5% quantile), with s its SD between populations, + 5·max(0, 2 − b/s)² + 10·Φ(−b/s). b/s is floored at −10, and an undefined bound takes the floor. s is the SD of the bound over the fifth pilot's 15 grade populations: 0.000177 mm/s for the drift and 0.0443 for the ratio. The ranking score cannot see presence: the ratio's band is wide. From the fifth pilot's answer to the hinge screen's, the ratio fell from 0.097 to 0.048 (the means over their grades' 15 and 5 populations), its lower bound falling below 0 in 3 of the 5, while its z moved from −1.69 to −1.76.

**The graded statistics** are the encoding table's (`113-c1-fit/encodings.md`, bound):

| Mutant | Graded | Reported only |
|---|---|---|
| tph-1 | the roaming fraction, direction more | leaving (exits per on-lawn hour) |
| pdfr-1 | the roaming fraction and leaving, by magnitude | |
| tax-4 | the roaming fraction, direction less; leaving, by magnitude | |

### The procedure

sep-CMA (`CMA_diagonal`), bounds [0, 1]. Each stage's answer is its mean (`xfavorite`) clipped to the box.

| Stage | Fits | Families | Mutants | Start | Generations | CMA seed |
|---|---|---|---|---|---|---|
| **1** | the 77 parameters of the wiring, sensory, readout and noise blocks with the pool centres; the other 60 at 0.5 | D_fit's four | none | the graded box's centre, 0.5 everywhere; σ₀ 0.2; 64 animals a candidate, population 24 | 150 | the start's first |
| **2a** | all 137 | D_fit's four | none | stage 1's answer, the speed noise's timescale set to 1 s; σ₀ 0.1 | 300 | the start's second |
| **2b** | all 137 | all six, plus chemotaxis presence (from 2026-11-19) | the run's, at half | 2a's answer; a new isotropic search, σ₀ 0.05 (amended 2026-11-16; until then 2a's search state) | 200 | third |
| **noise** | none: 2b's answer on 30 populations | all six | the run's | | | |
| **final** | all 137 | all six, plus chemotaxis presence (from 2026-11-19) | the run's, at half, plus the failure probability and the hinge (from 2026-11-19) | 2b's search state | 50 | fourth |

2b's and the final stage's candidates are graded with presence (from 2026-11-19). Their statistics are the same as without it; only presence's lower bounds are added.

- **The search state passes** from 2b to the final stage: a new strategy at 2b's answer, with 2b's σ and per-parameter steps (`CMA_stds` = stds/σ). *(Amended 2026-11-16: it passed from 2a to 2b too until then.)*
- **Fixed budgets:** every stage runs its count. The plateau rule (`cma_recovery.plateau_reached`) is read on the final stage and reported, never a stop.

### The first half's starts

Stage 1 and 2a are started up to **six** times in sequence. The rule is the same in every run of the procedure.

- **Each start** begins from the box's centre, with its own CMA seeds and its own candidate animals. Every start is read on the run's readout animals.

- **2a reads each start three times,** each reading a mean over three of 2a's readouts, and drops the start at the first reading it fails. The next start then begins. The second and third readings were added 2026-11-14:

  | Reading | 2a's readouts | The start passes when |
  |---|---|---|
  | the lawn | generations 40, 50 and 60 | time on the lawn is at least 0.95 |
  | the descent | generations 130, 140 and 150 | D_fit is at most 4.25 |
  | the bar (`reach` in the records) | generations 280, 290 and 300 | D_fit is at most 2.5 |

- **The first start to pass all three is kept.** The procedure continues from its 2a, and no later start runs.

- **A run in which no start is kept has no answer.** No later stage runs. *(2026-11-13: without the lawn guard 2 of 11 starts held the lawn, and the second pilot kept none of six. With it, 3 of 4 runs held the lawn at generation 60, as Disclosures state. 2026-11-14: every guarded run held the lawn, and 3 of 5 reached 2a's bar; at 3 in 5 a run keeps no start of six with probability about 0.004. Every run's count is reported.)*

- **An undefined time on the lawn counts as 0,** and an undefined D_fit fails its reading.

- **Recorded for every start:** every readout, every generation's median, best score and step size (from 2026-11-13, also the share of its candidates paying the lawn guard), every reading it took, and the search's mean where it was dropped.

- **Reported, not selected on:** each start's output spreads at stage 1's end.

### The noise step

- 2b's answer is graded on 30 populations of 200 without presence.
- Each statistic's SD of z between them (ddof 1, undefined populations dropped) is floored at 0.05. A statistic defined in fewer than 2 populations takes the floor.

### Readouts and state

- **Every 10 generations, at each stage's generation 0 and at its last,** the stage's mean is graded on the same 200 readout animals, on all six families and the three mutants, with presence on the wild type (from 2026-11-19; until then without presence). No reading or kill rule reads presence.
- **Recorded:** every statistic's value and z, the families passing, time on the lawn, the output spreads, each mutant's reading, presence's lower bounds, and the distance to the run's truth. From 2026-11-19, every generation of 2b and the final stage also records its best candidate's presence term, and the final stage its hinge.
- **State is saved atomically after every generation.** A resumed run refuses a saved state whose settings differ from its own, and a stored candidate is refused at another vector.

### The readiness pilot

- **The registered procedure,** run once on discarded seeds before any registered stage, configured exactly as the fit.
- **Its go rule:** its answer, graded with presence on 5 populations, passes the wild type in at least 3, and tph-1 and tax-4 each in at least 3.
- **Its kill rules.** The pilot stops and returns to the maintainer if:
  - a start's stage 1 has its smaller output spread below 0.5, or holds the lawn less than 75% of the time, on the mean of its last three readouts;
  - no start of the six is kept.
- **2a's bar is not a kill rule** (from 2026-11-14). It was one until then: the pilot stopped when 2a's D_fit was above 2.5 on the mean of its last three readouts. It is now 2a's third reading, which drops the start in every run.
- **In the registered runs stage 1's checkpoint is reported, not a stop.**
- **If the go rule fails,** the procedure is diagnosed and this Part A is amended before anything registered runs.
- **Reported:** 10 more populations' pass rates, the cost per stage, the final stage's plateau reading.

### F.0b-ii, C.2's recovery control

- **The truth:** the pilot's answer.

- **The observation:** the truth on all six assays and the three mutants at 200 animals (q = 0). It replaces the bands' and the mutant bands' centres in the objective, with the scales unchanged.

- **The recovery distance** D_rec: the registered capped D on the six families, with each mutant's RMS capped z over its graded statistics, against the truth's differences, together half.

- **The preflight** (`recovery-preflight`, run once). Gain = (prior D − D) / (prior D − floor):

  | Quantity | From |
  |---|---|
  | floor | the median D_rec from the observation to re-simulations 1–10 |
  | oracle | the median over re-simulations 11–20 |
  | prior D | the median over 200 uniform draws of the box, `default_rng(5805)` |
  | box centre | the median over 10 re-simulations of 0.5 everywhere |
  | wild-type reference | a record at the bands' and the mutant bands' centres, reported |

  It passes only if all seven hold. Otherwise the recovery returns to the maintainer.

  - prior D − floor ≥ **1.0**;
  - the oracle's gain ≥ **0.85**;
  - the box centre's gain ≤ **0.25**;
  - **stability:** no statistic flips between defined and undefined across re-simulations 1–10;
  - **the truth holds the lawn:** the observation's time on the lawn is at least 0.95, the bar a kept start must reach (added 2026-11-12);
  - **the truth reaches 2a's bar:** the observation's D_fit is at most 2.5. 2a's readings read D_fit against the bands in every run, the recovery's too, so a truth past the bar could not be recovered by a start the procedure keeps (added 2026-11-14);
  - **the truth clears presence's target:** the observation, graded with presence, has both lower bounds at least 2 s above 0 (0.000353 mm/s and 0.0886). 2b and the final stage score presence against that target in every run, not against the truth, so a truth below it would be pulled from. Above the target only the chance part pulls, weakly: 0.23 a bound at the target and 0.013 at 3 s (added 2026-11-19). Only the observation is graded with presence; its statistics are the same as without it.

- **The recovery fit:** the registered procedure from the box centre against the observation, every stage, budget, guard and noise step identical.

- **The gate:** the answer re-simulated 10 times. **The gain at its median D_rec must be at least 0.75.** If it falls short, or the recovery keeps no start and so has no answer, no registered fit runs until the maintainer decides.

- **The limit, stated in advance:** the truth sits beside the wild type, so the gate cannot tell recovering it from fitting the wild type. F.0b-ii tests whether the procedure reaches the region from the box centre, and the spread between runs. It does not test identifying the truth.

- **Reported:** each stage's readouts against the truth, the answer's distance from the truth by block, and the answer's own wild-type grade against the bands.

### The registered fit and its verdict

- **The answer** is the final stage's mean at generation 50, clipped to the box.
- **The registered grade:**
  - the answer on 5 registered populations of 200, every family with presence;
  - each mutant on the small lawn on its population's own animals, graded on its table statistics at the mutant bar in its resolved direction.
- **The verdict: C.2 passes when the wild type passes in at least 3 of the 5, and tph-1 and tax-4 each in at least 3 of the 5.**
  - **A fit that keeps no start has no answer, and fails.**
  - A population passes the wild type when every family passes, with presence for local search and chemotaxis.
  - pdfr-1 is graded and reported. It is not part of the verdict.
  - **The verdict is in-sample:** every graded statistic and both resolved mutants are fitted.
- **The verdict map:**
  - **pass:** C.2 enters C.4, and W's arms run this procedure, with W's compute decided from the measured cost before W registers;
  - **fail:** the failing statistics and each stage's readouts are the diagnosis. The roadmap's C.2 risk row goes to the maintainer, and no W arm registers on this procedure.
- **Reported beside, never re-read as the verdict:**
  - 10 more populations' pass rates;
  - the held-out reading;
  - every readout;
  - the plateau reading;
  - the reversal share against Roberts' 17.3%;
  - the cost per stage;
  - **the restart spread:** the median over g < 15 of the capped D between the pilot's population g and the registered answer's population g, on W's fitted families and on all six;
  - **the neural reading** (added 2026-11-14), below.
- **Description-only outcomes, named in advance:**
  - **a pass with a poor held-out reading** (the answer's median distance to a held-out half above the band reference's 95th percentile): C.2 passes, and C.4 states the held-out distance as a limit;
  - **a pass with the leave-one-out's tax-4 failing:** C.2 passes, and C.4's tax-4 row reads as fitted, not predicted.

### The held-out reading

W's rule (`benchmark_v4_controls.w_minimum_v4`):

- 1,000 splits of a date set into halves (seed 2034);
- PD1074's band scales at the half's size rounded to 25;
- the small lawn, kinematics and activity's graded statistics.

For each split, the distance to each half from three things:

- the other real half, which is W's minimum;
- each of the answer's 15 graded populations;
- each of 15 populations of 200 animals drawn without replacement from PD1074's band cohorts (`default_rng(2048)`), what a faithful simulator would match.

**The sets:**

- **N2's 26 unread held-out dates,** W's minimum's own (its real-against-real 95th percentile reproduces 2.597, which a test checks);
- **PD1074's 24 held-out dates,** beside it. These cohorts are the source of the primitives' on-food reversal values and the speed coupling's knots, and B.4 read them.

**Reported** as each distance's median and 95th percentile, as description. No new simulation. The uniform well's single held-out date, chemotaxis (no held-out recording) and López-Cruz's held-out batches are not read.

### The tax-4 leave-one-out

- **After the registered fit,** the same procedure on its own seeds with tax-4 out of the objective. tph-1 and pdfr-1 are then together half, and the failure probability omits tax-4.
- **tax-4 is read at every readout,** and its answer's tax-4 on 15 populations is reported beside the registered fit's.
- **It gates nothing.**

### The neural reading (added 2026-11-14)

Handed to C.2 by the fit-first investigation ([Logbook 134](../../134-fit-first-neural-readings.md)), the maintainer's decision. It is read once, on the registered fit's answer after `read`, as description. It is never graded, gates nothing and changes no verdict or map. No answer, no reading.

- **The responses,** as Logbook 134's M-a computed them for Logbook 131's answer:

  - the encoded brain under the procedure's settings, with the answer written;
  - one unit added to neuron s's starting state for the settling, from rest;
  - neuron j's response is its settled activity after the forward's four iterations;
  - over Creamer et al.'s 154 named neurons, as [responder, stimulated], with the diagonal masked.

  The responses, the metric, chance and sign agreement are computed by `investigation/spike/ma_c2/reading.py.txt` as committed with this record (SHA-256 `dd6d8706…a9b0`), with the answer's vector in place of `hinge4.json`'s. The measured arrays are `investigation/spike/session0/exports/measured_stams.npz` (`1c98ae81…7b28`), and the 154 names are `investigation/spike/session0/creamer_154_names.json` (`bbf27cd5…e244`).

- **The paired test and the wiring-only control are new code.** That driver has neither. They are written to the rules below after this Part A and before the registered answer exists, tested on Logbook 131's answer, and committed with Part B or Part C, before `run`.

- **The metric:** Creamer's `nan_corr` against the test matrix, one Pearson correlation over every entry finite in both matrices.

- **Reported:**

  - the answer's r, with its Fisher interval and its share of the readout's reliability ceiling: √rel_S = **0.423** (0.369 to 0.471), with rel_S = Cov(T, S) / Var(S) over the entries finite in both the training matrix T and the test matrix S ([Logbook 135](../../135-neural-readout-ceiling.md)). The ceiling assumes the two matrices measure one truth with independent noise. Beside it, r(T, S) = 0.173, the score of copying the training animals. *(Amended 2026-11-16: until then Part A called 0.173 the data's ceiling, which Logbook 135 showed is a reference, not a ceiling.)*;
  - **the unfitted floor:** the box centre's r, recomputed (0.0653 in Logbook 134);
  - **chance:** the 97.5th percentile of 1,000 joint relabellings of the answer's matrix, rows and columns together (`default_rng(9731)`, as M-a);
  - **the paired test** against the box centre: d = r(answer) − r(box centre), with a 95% percentile interval from 10,000 resamples of the stimulated neurons' classes with replacement (`default_rng(9811)`).
    - **The classes** are each stimulated neuron's class among the 302 neurons' (`quantumnematode.connectome.neuron_classes`), 74 among the 154, as Logbook 135's. A drawn class brings every stimulated neuron in it, and a class drawn c times counts c times.
    - Each resample keeps the same drawn columns for both networks and the measured matrix, and r is computed over the drawn columns' entries, repeats kept.
    - Named in advance, on the class interval: **fitting raised it** if the interval lies above 0, **lowered it** if it lies below 0, and **no detectable change** otherwise.
    - **Beside it, as description:** the same interval from 10,000 resamples of the 154 stimulated neurons themselves (`default_rng(9812)`).
    - *(Amended 2026-11-16: until then the neuron resample was the named one, on `default_rng(9781)`, which Logbook 135's null rewirings use. A class's left and right neurons respond alike, so resampling neurons counts one class as two independent draws and narrows the interval. Logbook 135 resampled classes for every interval. Its paired difference for Logbook 131's answer, which has been read, is a reason to read the two the same way, not the reason for the unit.)*;
  - **the wiring-only control:** the connectome's own contacts, with no network. C[j, s] is the count of chemical synapses from s to j plus gap junctions between s and j, from the wiring the encoded brain is built on, unsigned. It is scored by the same metric, with C·C (two-contact paths) beside it. Named in advance: if either reaches the box centre's r, the reading grades the wiring more than the network, and both networks' r are read that way;
  - sign agreement on the test entries whose measured size is above the median, as M-a;
  - **references, quoted from Logbook 135 and not computed here** (added 2026-11-16): on its held-out-worm split, the nearest to a reading of the whole test matrix (it scores the entries finite in both matrices), the unfitted network calibrated with the row/column-mean floor scored 0.092 (H0-cal), and Logbook 131's answer calibrated likewise 0.084 (C.2-cal). On that split the transpose floor does not exist, so both calibrations drop it. Logbook 135's 0.132 for H0-cal is on held-out stimulations and does not compare with this reading. A calibrated reading of the registered answer is left to W's registration, where Logbook 135 places it.

- **Its limits, stated in advance:**

  - the measured arrays come from a fuller record than the CC-BY deposit;
  - the network settles within one step, against responses integrated over 30 s;
  - one answer, so it says nothing about fitting in general;
  - the ceiling and both references come from Logbook 135's reading of the same test matrix, so the share and the comparison rest on one cohort of 30 test animals.

### Seeds

Each run of the procedure has 40,000,000 animal seeds from its base B (animals `B + offset + i`). Start a = 0–5 of the first half has its own stage 1 and all of its own 2a. *(Amended 2026-11-14: until then a block was 20,000,000, and only 2a's first 60 generations were each start's own, at B + 12,500,000 + 576,000·a; the kept start took 2a's later generations at B + 4,000,000.)*

| Within a block | Use |
|---|---|
| B + 250,000·a + 64·(24·(n − 1) + m) + i | stage 1 of start a, generation n, member m; a start's 3,600 candidates reach 230,400 |
| B + 20,000,000 + 2,880,000·a + 200·(48·(n − 1) + m) + i | 2a of start a, every generation n; a start's 14,400 candidates fill its 2,880,000, and the last start's end at B + 37,280,000 |
| B + 4,000,000·(s − 1) + 200·(48·(n − 1) + m) + i | 2b (s = 3, from B + 8,000,000) and the final stage (s = 4, from B + 12,000,000), the kept start's alone |
| B + 16,000,000 + i | the readout animals, the same at every readout |
| B + 17,000,000 + 200·p + i | the noise populations, p < 30 |
| B + 18,000,000 + 200·g + i | the grade: g < 5 the verdict (or the go rule), 5 ≤ g < 15 the pass rate; in the recovery, the answer's re-simulations, g < 10 |

The first start takes the run's first two CMA seeds. Start a = 1–5 takes R + 2·(a − 1) for stage 1 and R + 2·(a − 1) + 1 for 2a, R the first of the run's restart seeds.

| Run | Base | CMA | Status |
|---|---|---|---|
| the first readiness pilot, stopped at 2a | 900,000,000, on the first layout | 9504–9507 | discarded |
| the second readiness pilot, which kept no start | 1,100,000,000 | 9531–9534, restarts 9541–9550 | discarded |
| the third readiness pilot, whose kept start missed 2a's bar | 1,140,000,000, on the 20,000,000 layout | 9591–9594, restarts 9651–9660 | discarded |
| the fourth readiness pilot, whose answer failed the go rule | 1,200,000,000 | 9661–9664, restarts 9671–9680 | discarded |
| the fifth readiness pilot, whose answer failed the go rule | 1,360,000,000 | 9682–9685, restarts 9686–9695 | discarded |
| the readiness pilot (amended 2026-11-19) | 1,540,000,000 | 9926–9929, restarts 9930–9939 | discarded |
| F.0b-ii's recovery fit | 1,240,000,000 (920,000,000 until 2026-11-14) | 5801–5804, restarts 5831–5840 | registered |
| the truth's observation (q = 0) and re-simulations (1–20) | 940,000,000 + 200·q + i | | registered |
| the prior draws | 941,000,000 + 200·j + i, j < 200 | `default_rng(5805)` | registered |
| the box centre | 942,000,000 + 200·q + i, q < 10 | | registered |
| the registered fit | 1,280,000,000 (960,000,000 until 2026-11-14) | 5811–5814, restarts 5841–5850 | registered |
| the tax-4 leave-one-out | 1,320,000,000 (980,000,000 until 2026-11-14) | 5821–5824, restarts 5851–5860 | registered, description |
| the neural reading | chance's relabellings `default_rng(9731)`, M-a's own null reused; the paired test's class resample `default_rng(9811)` and neuron resample `default_rng(9812)` (`default_rng(9781)` until 2026-11-16) | | description |
| slow noise's control | 888,000,000 + i, reference episode 888,000,200 | | the control's reading |
| the stand-in preflight | 885,000,000–887,100,999 | `default_rng(9512)` | discarded |
| the smoke | 99,880,000–99,889,999: 8 animals, population 6, two generations a stage outside 2a, 22 in 2a, four starts | 9508–9511, restarts 9561–9566 | discarded |
| the runs that diagnosed the first pilot | 1,010,000,000–1,055,000,199, and 890,000,000 + i for the checks of its worm | 9516–9525 | discarded |
| the runs that diagnosed the second pilot | 1,060,000,000–1,065,000,199 and 1,090,000,000–1,095,000,199 (its starts 0 and 3 again), 1,130,000,000–1,135,000,199 (the guarded first half) | 9571, 9574, 9581–9582 | discarded |
| the check that diagnosed the third pilot | 1,170,000,000 + 1,000·p + i, p < 4 (`probes/dopamine_flip.py.txt`) | | discarded |
| the screen that diagnosed the fourth pilot (`probes/two_b_screen.py.txt`) | arm A 1,400,000,000 and arm B 1,410,000,000, each + 200·(48·(g − 1) + m) + i for g ≤ 60 (to 1,400,575,999 and 1,410,575,999); readouts on the fourth pilot's readout animals, 1,216,000,000 + i, reused for reading | 9922 (A), 9923 (B) | discarded |
| a timing check of the simulation's kernels, no reading | 1,420,000,000–1,420,000,207 | | discarded |
| the runs that diagnosed the fifth pilot: the onset sweeps and the trace of Logbook 131's route (`probes/onset_sweep.py.txt`, `onset_sweep2.py.txt`, `lineage_131.py.txt`), on common animals | 1,430,000,000 + 200·p + i, p < 3 (to 1,430,000,599) | | discarded |
| the hinge screen (`probes/hinge_screen.py.txt`) | candidates 1,440,000,000 + 200·(48·(g − 1) + m) + i, g ≤ 50 (to 1,440,479,999); its grade 1,441,000,000 + 200·p + i, p < 5; readouts on the fifth pilot's readout animals, 1,376,000,000 + i, reused for reading | 9924 | discarded |
| the chemotaxis sweep (`probes/chemotaxis_sweep.py.txt`) | 1,450,000,000 + 200·p + i, p < 3 (to 1,450,000,599) | | discarded |
| the presence screen (`probes/presence_screen.py.txt`) | candidates 1,460,000,000 + 200·(48·(g − 1) + m) + i, g ≤ 50 (to 1,460,479,999); its grade 1,461,000,000 + 200·p + i, p < 5; readouts on the fifth pilot's readout animals, reused for reading | 9925 | discarded |

The tests check that every block is disjoint from the others, from the five earlier pilots' and their diagnoses', from the timing check's, from the seed table's earlier blocks (which reach 880,001,999), and from the fit-first investigation's reserved animals (1,500,000,000–1,501,000,199). The fourth and fifth pilots' blocks are taken whole, to 1,237,279,999 and 1,397,279,999, every start's 2a included. They also check that no run takes a CMA seed the seed tables give earlier pilots, probes or tests: 9601–9616, the fifth pilot's 9682–9695, the screens' 9922–9925, the investigation's 9701–9722, 9731, 9732 and 9741–9759, and the neural readout's ceiling's 9770–9779, 9781–9799 and 9801–9805 ([Logbook 135](../../135-neural-readout-ceiling.md)), among them. The paired test's two generators are checked the same way.

### Disclosures

Seen before this record was committed, all on discarded seeds:

- **Logbook 131's probes and curriculum** (`131-c2-clean-start/probes/`, each script as run): the route this procedure fixes, adapted after each reading. The stage-1 subset, the 1-s timescale, the guards, the failure probability, its weight and floor, and every stage's budget come from it. Procedure B differs from the route as the design's table states:

  - up to six starts of the first half, the first that holds the lawn kept (amended 2026-11-12; stage 1 was first registered at 200 animals and population 48);
  - 250 generations after 2a, not 500;
  - the mutants from 2b;
  - the search state carried from 2a (until 2026-11-16, when 2b began restarting isotropically at σ₀ 0.05, as the route's 2b did; it is still carried from 2b to the final stage);
  - the package's speed-following knots throughout;
  - one noise measurement of 30 populations;
  - tph-1 on the roaming fraction alone.

- **Logbook 132's confirmation:** Logbook 131's answer passes 10 of 10 under the arm's body.

- **The stand-in preflight** (`probes/standin_preflight.py.txt` and `.json`, `probes/standin_gate_reach.py.txt` and `.json`), Logbook 131's answer standing in for the truth:

  | Quantity | Stand-in | Bar |
  |---|---|---|
  | floor | 0.36 | |
  | the oracle's gain | 0.996 | ≥ 0.85 |
  | prior D (20 draws) | 6.41 (4.37–8.59) | separation 6.05 ≥ 1.0 |
  | the box centre's gain | 0.074 | ≤ 0.25; fails only for a prior D above 7.47 |
  | the wild-type reference's gain | 0.872 | reported |
  | definedness flips | none in 8 | none |

  B02's answer, from another lineage, gains 0.80 and 0.81. The truth's own lineage gains 0.96–0.97. The bars were set before these were read, and are unchanged by them.

- **Slow noise's positive control** (`speed_noise_control.json`), run under its registered rule before this record. Its result is below.

- **The spec review** (independent reviewer and author, 2026-11-10). It found tph-1's leaving graded against the table, and the control owed. Both are fixed here.

- **The script's checks:**

  - 27 tests cover every spec scenario, with 18 deliberate bugs in `c2_fit.py` each caught by its test (`probes/mutants.py.txt`, `probes/mutants.log.txt`);
  - the held-out reading's real halves reproduce W's minimum to 1e-12;
  - the smoke ran every stage, the noise step, the grade and the reading on discarded seeds.

- **The maintainer's decisions (2026-11-10):**

  - procedure B;
  - a readiness pilot first;
  - F.0b-ii at one truth before the fit, its limit stated;
  - symmetric fit splits;
  - a full tax-4 leave-one-out.

- **Not seen:** every registered simulation of F.0b-ii (its fit at 1,240 M, and 940–942 M; 920 M until 2026-11-14), the registered fit (1,280 M; 960 M until then) and the leave-one-out (1,320 M; 980 M until then).

**Seen before the amendment of 2026-11-12,** all on discarded seeds, each run's rule fixed before it ran (`pilot_first.json`, and each script as run with its record under `probes/`):

- **The first readiness pilot** (base 900,000,000, the first layout, stage 1 at 200 animals and population 48). Stage 1 passed its checkpoint (0.87 on the lawn). 2a ended at D_fit 3.13 on the mean of its last three readouts, against 2.5, and the pilot stopped. Its worm held the lawn 80–92% of the time through 2a's 300 generations.

- **Every first half run since,** on the registered arm and objective unless stated. "Reaches the bar" is the mean of three consecutive 2a readouts at D_fit 2.5 or below with at least 0.97 on the lawn:

  | Start (stage 1) | 2a | On the lawn, 2a's generations 40–60 | 2a against the bar |
  |---|---|---|---|
  | B04's (Logbook 131; 64 animals, population 24, the pooled body) | the curriculum's, on the pooled body | 0.993 | reaches it |
  | B04's | the curriculum's replayed on the registered arm (`replay.py`) | 0.997 | reaches it by generation 100 |
  | B04's | another CMA seed (`replay_seed.py`, `continue_seed.py`) | 0.982 | reaches it by generation 210, at D_fit 2.4986 |
  | screen (b)'s: 64 animals, population 24 (`screens.py`) | its own, restarted from its generation-100 answer (`continue_screen.py`) | 0.975 | reaches it by generation 230 |
  | screen (a)'s: 200 animals, population 48 | its own, restarted likewise | 0.912 | does not by generation 250 (3.53) |
  | the first pilot's: 200 and 48 | its own | 0.854 | does not by generation 300 (3.13) |
  | start (c)'s: 64 and 24 (`restart_c.py`) | its own | 0.847 | does not hold the lawn. Stopped at 2a's generation 95 for the pilot's launch, at D_fit 4.33 and 0.861 on the lawn over generations 70–90; whether it would reach the bar was not read |

- **What the amendment takes from them:**

  - the body is not the cause, since the curriculum's 2a keeps its pace on the registered arm;
  - 2 of 5 independent stage-1 answers led to a 2a that holds the lawn;
  - every 2a at 0.975 or above over generations 40–60 reached the bar, and neither of the two at 0.912 or below that ran on did. **The 0.95 bar and its three generations were read off the first six rows.** Start (c) ran after the rule was fixed;
  - stage 1's size is not shown to matter (2 of 3 at 64 and 24, 0 of 2 at 200 and 48). It is amended on cost: about 0.8 h a start against 3.2 h.

- **Seen and not used:**

  - the pace rule the screens ran under (the mean of 2a's generations 80–100 at D_fit 3.1 or below) failed two lineages that later reached the bar. It is not used;
  - the speed output's spread at stage 1's end read 1.84 and 1.92 for the two starts that held the lawn, and 3.20, 3.38 and 3.65 for the three that did not. A two-against-three split is perfect for one random measure in five, and many were looked at. It is reported for every start, and selects nothing;
  - **a lawn guard** (`guard_probe.py`): the first pilot's 2a again, from the same stage-1 answer, with 300·(0.95 − t)/0.95 added below 0.95 on the lawn. It read 0.949 over generations 40–60 (0.916, 0.951, 0.980), which its own rule counts as not holding. It then held the lawn (0.980, 0.991 and 0.996 at generations 60, 70 and 80) with D_fit falling (5.53, 4.88, 4.47), and was stopped at generation 88 for the pilot's launch; whether it would reach the bar was not read. It is not registered: it would count time on the lawn twice in the objective, and it is one run on another CMA seed than the pilot's;
  - forensics on the stage-1 answers and on 2a's moves (`block_moves.py`, `key_params.py`) found nothing that tells the starts apart in their parameters;
  - **checks of the first pilot's stalled worm** (animals 890,000,000 + i): it crossed the lawn's edge 0.42 times a minute, against real worms' 0.06 (`edge_check*.py`, `lawn_time_check.py`); no stage-1 answer reverses more when the food signal falls (`klinokinesis_check.py`); and sensing food 0.25 or 0.5 mm ahead of the tracked point, with no refit, did not reduce the crossings (`head_sense_probe.py`).
  - The two runs stopped for the pilot's launch (`restart_c.py`, `guard_probe.py`) keep their saved states and resume exactly. `export_stopped.py` wrote their records.

- **The amended script's checks:**

  - 40 tests in the fast tier, with 45 further deliberate bugs in the start loop and its callers each caught (`probes/mutants_amend.py.txt` and its log);
  - the smoke runs a dropped start and a kept one on real simulation, stage 1 at its own size and 2a past the reading;
  - the lawn-holding function, run on every recorded 2a above, sorts them as the table does;
  - **an independent review** (2026-11-12, a reviewer without the author's reading) found no defect in the start loop, the seeds or the handling of a run with no answer. It named seven deliberate bugs the first tests passed, each now caught, stale cost figures, now corrected, and the preflight's new check that the truth itself holds the lawn.

**Seen before the amendment of 2026-11-13,** all on discarded seeds, each run's rule fixed before it ran (`pilot_second/`, the record split by start, and each script as run with its record under `probes/`):

- **The second readiness pilot** (base 1,100,000,000, Part A as amended 2026-11-12). Every start was dropped at 2a's generation 60, and the pilot stopped with no answer after 15.8 h:

  | Start | CMA | Stage 1: on the lawn, last three readouts | Stage 1: dopamine wiring gain | 2a: on the lawn, generations 40–60 |
  |---|---|---|---|---|
  | 0 | 9531, 9532 | 0.846 | +0.95 | 0.720 |
  | 1 | 9541, 9542 | 0.878 | +1.40 | 0.880 |
  | 2 | 9543, 9544 | 0.806 | +1.82 | 0.795 |
  | 3 | 9545, 9546 | 0.902 | −1.50 | 0.921 |
  | 4 | 9547, 9548 | 0.882 | +1.50 | 0.888 |
  | 5 | 9549, 9550 | 0.885 | +1.71 | 0.833 |

  Starts 0–3 had the machine to themselves: stage 1 took 0.88–0.92 h and 2a's 60 generations 1.34–1.40 h. Starts 4 and 5 shared it with the resumed guard probe.

- **A dopamine lead, read and failed** (`dopamine_trace.py`, a read-only log of the pilot's state). Found after the pilot's third start: the two stage-1 answers whose 2a held the lawn had a negative dopamine wiring gain (−1.96 and −1.44), and the six that did not a positive one (+0.95 to +1.96). The prediction that start 3 (−1.50) would hold was written down before its reading, and failed at 0.921. The guard probe reached the bar at +0.84. The gain selects nothing.

- **The lawn guard probe, resumed** during the pilot's starts 4 and 5 (2026-11-12 at about 22:05, from generation 88). It ended at 05:35 on 2026-11-13, after the pilot. It held the lawn, at 0.978 or above from generation 60, and **reached the bar at generations 200–220** (D_fit 2.63, 2.37 and 2.33; 0.992 on the lawn). Its own rule had read 0.949 at generations 40–60 as not holding, and its stated branch for that, "the guard cannot rescue it", was wrong.

- **The second pilot's starts 0 and 3, their 2a again under the guard** (`guard_rep.py`, from their stage-1 answers in `pilot_starts.json`). The rule was fixed 2026-11-13 before they ran: the guard is a remedy worth registering if the probe and at least two of starts 0, 1 and 3 reach the bar.

  - **Start 0** held the lawn, at 0.974 or above from generation 20. It **reached the bar at generation 290** (D_fit 2.54, 2.36 and 2.26, a mean of 2.39; 0.987 on the lawn).
  - **Start 3** held the lawn, at 0.958 or above from generation 30, and stalled near D_fit 4.5 from generation 90. It was paused for 2.9 h at generation 178, to keep the machine within its cores, and resumed exactly. It was still running when this was committed. Its record is added in Part B.
  - **Start 1 was not run,** the maintainer's decision: with the guard in stage 1, its question no longer decides the amendment. The rule is therefore not read. With start 0 at the bar and start 3 stalled, the rule turned on start 1 unless start 3 reaches the bar by its end. Its failure branch ("not reliable enough to replace restarts") agrees with keeping the restarts.

- **A first half from the box centre, with the guard in stage 1 and 2a** (`guarded_first_half.py`). Its rule, fixed 2026-11-13 before 2a ran: the guard belongs in stage 1 too if this 2a reaches the bar by generation 150.

  - Stage 1 ended on the lawn at 0.979 over its last three readouts, the first of 12 stage-1 answers at 0.95 or above.
  - 2a's generation-0 readout, with the timescale reset to 1 s, read 0.852 on the lawn. From generation 10 it read 0.976 or above.
  - **2a reached the bar at generation 120** (D_fit 2.48, 2.98 and 1.81, a mean of 2.42; 0.993 on the lawn). By its rule, the guard belongs in stage 1 too.

- **What the amendment takes from them:**

  - **holding the lawn:** with the guard, 3 of 4 runs held it by the registered reading at 2a's generation 60, and the fourth soon after. Without it, 2 of 11 starts held it;
  - **reaching the bar:** from an unguarded stage 1, the guarded 2a reached it in 2 of 3 runs (generations 220 and 290), with start 3 still running. The one first half guarded from stage 1 reached it at generation 120, 100 generations sooner than the first of them. That is one run, and the pilot is its replicate;
  - **the guard enters every stage,** the maintainer's decision. It needs no simulation. It still acts while the search's mean holds the lawn, since candidates off the lawn pay it: about 10% of start 0's at 2a's generations 260–273, and 30–60% of start 3's at generations 170–188. Each generation's share is recorded;
  - **the earlier objection, that it counts time on the lawn twice,** holds only below 0.95. The band's centre is 0.978 with a scale of 0.006 at 200 animals, so below 0.95 the statistic is 4.6 scales off and already fails at 3.08. There the guard reorders only failing candidates. At stage 1's 64 animals (scale 0.010) it reaches the passing range's lower edge, 0.947–0.950. The final grade does not read it;
  - **in F.0b-ii the guard keeps its absolute 0.95.** The preflight admits a truth at 0.95 or above. For a truth near 0.95, candidates at the truth itself would pay the guard whenever their animals read below 0.95, pulling the recovery off it. A kept start has held the lawn at 0.95 or above, and every guarded 2a read 0.97–0.99, so this is a stated limit, not a check;
  - **the restarts stay,** as a safety net.

- **The amended script's checks (2026-11-13):**

  - 47 tests in the fast tier, with 22 deliberate bugs in the guard, its record and the pilot's seeds each caught, and the first amendment's 45 caught again on the new code (`probes/mutants_guard.py.txt` and its log);

  - the smoke runs every stage with the guard on real simulation, a dropped start and a kept one;

  - **an independent review** (2026-11-13, a reviewer without the author's reading) found no defect in the guard, in which records it scores, or in the new seed blocks. It found:

    - the pilot's first restart seeds (9601–9610) overlapping seeds the Phase 10 table gives Logbook 132's tests and the sign-scale calibration, now moved to 9651–9660, with a test;
    - lint failures in the test file;
    - the second pilot's record not yet renamed;
    - the guard probe's resumption misdated.

    It also named the F.0b-ii limit, the guard's action while the mean holds the lawn, and stale cost figures. Each is now stated, and the share of candidates paying the guard is recorded.

**Seen before the amendment of 2026-11-14,** all on discarded seeds, each run's rule fixed before it ran (`pilot_third.json`, and each script as run with its record under `probes/`):

- **The third readiness pilot** (base 1,140,000,000, Part A as amended 2026-11-13). Start 0 held the lawn at 0.965, 0.974 and 0.989 at 2a's generations 40, 50 and 60, and was kept, the first start any pilot has kept. Starts 1–5 never ran. 2a's kill rule stopped it at 2a's end: mean D_fit 3.83 over generations 280–300 (3.87, 3.85, 3.76), against 2.5, at 0.97–0.99 on the lawn from generation 30. Its answer's failing statistics at 2a's end: the dwelling-speed 90th percentile (z 9.6), the on-lawn pause share (z 7.2), the on-lawn onset rate (z −5.7) and the roaming-speed 90th percentile (z −3.8). It reverses for 70% of its seconds on the lawn. It ran 9.9 h: stage 1 1.6 h on 8 workers (beside start 3's guarded 2a), and 2a 8.3 h, with 56 generations on 8 workers and 244 on 16 at about 87 s, after a stop and an exact resume when start 3's run ended.

- **The second pilot's start 3, its 2a under the guard, finished** (`probes/guard_rep_3.json`): it did not reach the bar, ending at mean D_fit 4.23 over generations 280–300 with 0.967 on the lawn. With start 0 at the bar, its rule's reading turned on start 1, which was not run.

- **Every guarded 2a against 2a's bar,** each from its own start:

  | Run | Mean D_fit, generations 130–150 | Against 2a's bar |
  |---|---|---|
  | the guarded first half | not run: it stopped at the bar | reaches it at generation 120 |
  | the guard probe | 2.92 | reaches it at generations 200–220 |
  | the second pilot's start 0, again | 3.90 | reaches it at generation 290 |
  | the second pilot's start 3, again | 4.62 | does not: 4.23 |
  | the third pilot's start 0 | 4.65 | does not: 3.83 |

  Both stalls fail the dwelling-speed tail (z about 10), the on-lawn pause share (z 4–7) and the on-lawn onset rate (z −4 to −6). No earlier readout separates the groups cleanly. At generation 100, the latest of the three to reach the bar read 4.45, and one stall 4.63.

- **A dopamine lead, checked at the answers** (`probes/dopamine_flip.py.txt` and `.json`). Both stalls ended with the dopamine wiring gain strongly inhibitory (−1.49, −1.36), and the three that reached the bar near zero or excitatory (+0.04, +0.89, +0.84). The rule, fixed before it ran: mirror that gain alone at each answer on four populations of 200. The sign holds the stalls if mirroring lowers both stalls' D_fit by at least 1.0 and their dwelling 90th percentile's z by at least 3. **The reading: not a one-parameter switch at the answer.** Mirroring broke every answer whose gain was above 0.8 in size (D_fit 7.0–8.7), stalls and reachers alike, and left the one at +0.04 unchanged (2.20 to 2.26). The as-is values reproduced each answer's recorded D_fit (3.83, 4.24, 2.20, 1.88, 2.26). The gain carries weight in every answer and selects nothing.

- **What the amendment takes from them:**

  - **holding the lawn is solved:** every guarded run held it;
  - **2a's descent decides,** and 2 of 5 stall. A stall is flat for more than 100 generations, so a longer 2a would not rescue it;
  - **the descent's bar of 4.25 was read off the table's generation-150 column:** the two stalls at 4.62 and 4.65, and the slowest reacher at 3.90. A drop costs time, never a verdict. It would have dropped neither run that reached the bar;
  - **2a's bar of 2.5 is the earlier kill rule's own,** on the same three readouts. It now drops a start in every run instead of stopping the pilot. A registered run's kept start therefore always enters 2b at the bar.

- **The fit-first investigation's handover** (Logbook 134, merged into this branch 2026-11-14 before this amendment), read before it:

  - W's size is about 1,200–2,900 machine-hours at about 30 h a fit. W is not part of this record;
  - the neural reading, adopted here as description. Logbook 131's answer read r 0.053 and the unfitted box centre 0.065 against the test matrix. The 19 degree-preserving rewirings read a median of 0.005. The paired test and the wiring-only control were added in review of the handover, before any registered answer exists;
  - its reserved seeds, now earlier blocks.

- **The amended script's checks (2026-11-14):**

  - 57 tests in the fast tier;
  - **deliberate bugs** (`probes/mutants_restart.py.txt` and its log):
    - 49 in the readings, the layout, the seeds and the preflight, 48 caught. The other, leaving out the save after a reading, is equivalent: a resumed run takes the reading again from the saved readouts;
    - the first amendment's 35 and the second's 16 caught again where their code survives;
    - 12 more ported to the code this amendment rewrote, each caught;
  - **the smoke on real simulation** (8.7 min) dropped its first three starts at the lawn, the descent and the bar in turn and kept the fourth, which ran every later stage, the noise step, the grade, the preflight's arithmetic and the held-out reading;
  - **an independent review** (2026-11-14, a reviewer without the author's reading) found no defect in the loop's order or its resumes, in the seed arithmetic, or in the smoke's bars. It found:
    - **F.0b-ii could fail on a truth past 2a's bar,** since 2a's readings read D_fit against the bands, the recovery's too. The preflight now checks it (*F.0b-ii*, above);
    - a type error in the test file;
    - **no test of a reading over more than one readout.** Two bugs survived the suite: a reading taken at its first readout, and a reading of its last readout alone. A test now reads each over two readouts that disagree, and catches both;
    - a resume test that could not tell a resume from a restart, which now checks;
    - stale cost figures, a seed row missing chance's generator, and the neural reading's new code not stated. Each is now corrected or stated.

**Seen before the amendment of 2026-11-16,** all on discarded seeds, each run's rule fixed before it ran (`pilot_fourth.json`, and each script as run with its record under `probes/`):

- **The fourth readiness pilot** (base 1,200,000,000, Part A as amended 2026-11-14), 36.6 h from launch to its record:

  | Start | CMA | 2a's readings | Outcome |
  |---|---|---|---|
  | 0 | 9661, 9662 | the lawn 0.899 | dropped at the lawn, after 2.2 h |
  | 1 | 9671, 9672 | the lawn 0.957; the descent 4.53 | dropped at the descent, after 4.2 h |
  | 2 | 9673, 9674 | the lawn 0.997; the descent 3.30; the bar 1.85 | kept |

  - **The kept start's stages:** stage 1 0.81 h, 2a 6.04 h, 2b 19.24 h, the final stage 3.95 h. The final stage did not plateau.
  - **Its go rule failed.** The wild type passed in 0 of the 5 populations, and tph-1, pdfr-1 and tax-4 in 5 of 5 each. Over all 15 populations the wild type passed in none, tph-1 in 14, pdfr-1 and tax-4 in all 15.
  - **What failed, in all 5 populations:** local search's log early/late ratio (z −3.5 to −3.7), with γ failing in 4 (z 3.4 to 10) and local search's presence failing in 2; chemotaxis's flip fraction (4.5 to 4.6) and weathervane slope (6.3 to 8.3); the small lawn's pre-exit enrichment (−6.0 to −7.1); and the on-lawn onset rate (−3.1 to −3.5). Leaving failed once (−3.35).
  - **2b moved the mutants, not local search or chemotaxis.** Over its 200 generations tph-1's roaming difference left −5.5 at generation 100 and swung between −2.7 and 3.8 after it, ending at 0.2. tax-4's left the cap of 10 at generation 110 and stayed above its bar (2.32 to 4.89) until generation 200, at 1.75. Local search's log early/late ratio stayed between −3.62 and −3.74, and the flip fraction between 4.41 and 5.02.
  - **2a's answer already failed** pre-exit enrichment (−4.0) and the onset rate (−4.3), at a D_fit of 1.84 that passed 2a's bar.

- **The screen that diagnosed it** (`probes/two_b_screen.py.txt`, as run; records `two_b_screen_a.json` and `two_b_screen_b.json`). As run, the script read the fourth pilot's record as `pilot.json` (now `pilot_fourth.json`), and the readout animals from `RUNS["pilot"]`, then the fourth pilot (1,216,000,000 + i). Against the amended script it would read the fifth pilot's, so it does not re-run unchanged; its records state the animals it read.

  - **What it separates.** The pilot's 2b differed from Logbook 131's in two ways at once. It fitted the mutants from its first generation, and it carried 2a's search state where 131's 2b restarted isotropically at σ₀ 0.05.

  - **How it ran.** Two arms from the pilot's 2a answer, each a new isotropic search at σ₀ 0.05, population 48 and 200 animals a candidate, on the registered arm and objective, for 60 generations. Arm A fitted the six families without the mutants, and arm B with them at half. Both were read on the pilot's own readout animals, and each arm's generation-0 readout reproduced the pilot's 2b generation-0 readout exactly.

  - **The rule, fixed before either ran.** L is local search's log early/late z, averaged over the readouts at generations 40, 50 and 60, and an arm moves at L ≥ −3.45. For scale: the pilot's 2b read −3.64 and −3.62 at generations 40 and 60, 131's 2b a mean of −3.28 over 40–60, and the ratio's SD of z between populations is 0.065.

    - If B moves, 2b restarts isotropically at σ₀ 0.05 with the mutants.
    - If only A moves, 131's order returns: 2b without the mutants, then a stage adding them.
    - If neither moves, the procedure goes back to the maintainer.

  - **The reading:**

    | Arm | Generation 40 | 50 | 60 | L | |
    |---|---|---|---|---|---|
    | A, without the mutants | −3.35 | −2.92 | −2.89 | −3.06 | moves |
    | B, the mutants at half | −3.34 | −3.16 | −3.14 | −3.22 | moves |

    B moves, so the amendment takes the first branch.

  - **Reported, not selected on:**

    - at generation 60, D_fit 2.37 (A) and 2.73 (B), and on the lawn 0.994 and 0.985;
    - neither arm brought the flip fraction (4.82 and 4.21; 131's passed near generation 150) or pre-exit enrichment (−4.0 and −5.1) to passing, and the weathervane slope fell to 4.5 and 4.1 without passing;
    - B's tph-1 reached −2.1 and its tax-4 7.7 by generation 60;
    - the share of candidates with γ defined rose from about a quarter to 83% (A) and 71% (B);
    - A ran 6.3 h and B 8.8 h. B was stopped once, at its generation 37 when A finished, and resumed exactly on 16 workers (`probes/screen_a_then_b16.sh.txt`).

  - The screen's candidate stores (11–13 MB each) are kept off the repository with the pilots' run states.

- **The neural readout's ceiling** ([Logbook 135](../../135-neural-readout-ceiling.md), merged into this branch 2026-11-16 before this amendment), read before it:

  - the readout's ceiling is 0.423, not 0.173;
  - the calibrated unfitted network is the reference to beat: 0.132 on held-out stimulations, 0.092 on held-out worms;
  - paired and resampled by class, Logbook 131's answer is not above its unfitted network (−0.012 to −0.013 on the three splits);
  - its null rewirings use generators 9781–9799, so Part A's paired test shared 9781.

- **A timing check of the simulation's kernels** (`probes/kernel_timing.py.txt` and `.json`). One candidate at arm A's answer on 200 discarded animals (1,420,000,000 + i), on one core, was profiled and re-run with two kernels compiled. Its records were compared for identity only. Nothing in the procedure changes.

- **What the amendment takes from them:**

  - **the result points to the carried search state, not the mutants.** From the same point, both arms moved local search within 60 generations, where the carried 2b had not moved it in 200. Each arm is one CMA seed on its own candidate animals, so this is consistent with the state holding it back, not a measurement of why;
  - **the mutants stay in 2b,** by the rule: B moved, so the mutants keep shaping serotonin's producers while the search is broad, as the registration chose;
  - **A moved further than B** (−3.06 against −3.22), consistent with the mutants slowing local search, on one seed each. That is reported, not selected on;
  - **2b starts its own search** at σ₀ 0.05 on the run's third CMA seed, as the screen's arms did. The final stage still carries 2b's state, as the route carried it from stage 3 on;
  - **stated, not remedied:** the flip fraction, the weathervane slope, pre-exit enrichment and the on-lawn onset rate did not pass in either arm's 60 generations. 131's flip fraction passed near generation 150, and the registered 2b runs 200. The pilot's answer may still fail on any of them, and its go rule reads it.

- **The amended script's checks (2026-11-16):**

  - 60 tests in the fast tier;
  - **deliberate bugs** (`probes/mutants_fresh.py.txt` and its two logs):
    - the first run, on the amended code with the third amendment's bugs again, caught 120 of 122 applicable, and 29 were not applicable (earlier pilots' seed lines). One miss was the third amendment's equivalent bug (a reading not saved, which a resumed run takes again). The other was real: dropping the pilot's kill rules passed the suite. A test now checks that the pilot alone has them;
    - on the final code, all 36 of this amendment's caught: 23 in 2b's start, the pilot's seeds and the neural reading's generators, 12 earlier pilot-seed bugs ported to the new line, and the review's below;
  - **the smoke on real simulation** (9.5 min) dropped its first three starts at the lawn, the descent and the bar in turn and kept the fourth. Its 2b started its own search (σ 0.049 after its first generation, from 0.05), and the final stage carried 2b's state, then the noise step, the grade, the preflight's arithmetic and the held-out reading ran;
  - **an independent review** (2026-11-16, a reviewer without the author's reading) found no defect in the code, the seed blocks, the CMA seeds or the generators, and confirmed the disclosed figures against the records except as below. It found:
    - **a test gap:** with 2b's state not handed on, the final stage carried 2a's σ and steps and the suite still passed. The hand-off test now checks the handed σ is 2b's own, and catches it;
    - the held-out-worm calibration misdescribed (Logbook 135's drops the transpose floor there), the fourth pilot's 2b ranges misquoted (now −3.62 to −3.74 and 4.41 to 5.02), the mutants' timing, and a cost total that counted the pilot's dropped starts (now about 120 h);
    - "did not move" where "did not pass" holds (the weathervane slope fell), stale proposal text, and claims stronger than one seed per arm supports. Each is now corrected or stated;
    - that the archived screen script reads `pilot.json` and `RUNS["pilot"]` as they were, now stated beside it.

**Seen before the amendment of 2026-11-19,** all on discarded seeds, each rule fixed before its run (the trace of Logbook 131's route, a description, had none; `pilot_fifth.json`, and each script as run with its record under `probes/`). As run, the scripts read the fifth pilot's record as `pilot.json` (now `pilot_fifth.json`), and the readout animals from `RUNS["pilot"]`, then the fifth pilot (1,376,000,000 + i). Against the amended script they would read the sixth pilot's, so they do not re-run unchanged; their records state the animals they read.

- **The fifth readiness pilot** (base 1,360,000,000, Part A as amended 2026-11-16), 47.1 h from launch to its record:

  | Start | CMA | 2a's readings | Outcome |
  |---|---|---|---|
  | 0 | 9682, 9683 | the lawn 0.998; the descent 4.30 | dropped at the descent, after 4.1 h |
  | 1 | 9686, 9687 | the lawn 0.979; the descent 5.14 | dropped at the descent, after 4.2 h |
  | 2 | 9688, 9689 | the lawn 0.963; the descent 5.63 | dropped at the descent, after 4.3 h |
  | 3 | 9690, 9691 | the lawn 0.991; the descent 4.77 | dropped at the descent, after 4.4 h |
  | 4 | 9692, 9693 | the lawn 0.988; the descent 2.28; the bar 1.97 | kept |

  - **The kept start's stages:** stage 1 0.89 h, 2a 6.85 h, 2b 17.87 h, the final stage 4.32 h. The final stage did not plateau.
  - **Its go rule failed.** The wild type passed in 0 of the 5 populations, and tph-1, pdfr-1 and tax-4 in 5 of 5 each. Over all 15 populations the wild type passed in none, and every mutant in all 15.
  - **What failed:** the on-lawn onset rate in all 15 populations (1.32 a minute against the bands' 2.28; z −4.1 to −4.5). Chemotaxis presence failed in 3 of the first 5 and 5 of the 15, the log quintile ratio's lower bound below 0. Pre-exit enrichment failed in 2 of the first 5 (−3.09 and −3.15), and the dwelling bout once (3.65). Local search passed in every population.
  - **Across the final stage's candidates** the onset rate's z moved with the pause share (r −0.90) and the dwelling speed's median (r +0.92). The search raised onsets only with a busier dwelling worm, which took the dwelling speed's 90th percentile over its bar.
  - **2b moved the mutants and lost chemotaxis.** tph-1 and tax-4 failed at 2b's start. tax-4 passed from generation 50 (failing once more at 180), and tph-1 from generation 190. On the readout animals, through 2a, which does not fit chemotaxis, the log quintile ratio read 0.10–0.34 at generations 0–290 (−0.11 at 300) and the drift 0.007–0.012 mm/s. Through 2b the drift fell to 0.003 and the ratio ranged between −0.02 and 0.16.

- **The onset sweeps** (`probes/onset_sweep.py.txt` and `onset_sweep2.py.txt`, with their records). At the pilot's answer, on three common populations:

  - one parameter at a time: the reversal, speed, omega and delta readouts' biases, CEP's food-contact gain, serotonin's fixed-sign strength and the reversal output's noise;
  - then the reversal bias at +0.10 with each of five moves that slow the dwelling worm.

  The rule: a point opens a direction when, in all three populations, the onset rate passes, every other D_fit statistic passes and tph-1 passes. No move from the pilot's answer opened one. Every move that raised the onset rate by at least 1 in z took the dwelling speeds over their bars, and a slower speed bias added pausing, not reversals. Logbook 131's answer, a reference among the points, met the rule in all three (onset rate z −2.53 to −2.74, every mutant 3 of 3). Read literally, the rule's first branch names it, but that branch is about a move at the pilot's answer, and 131's answer sits 5.07 from it in unit coordinates. The points a quarter and half of the way to it failed badly.

- **The trace of Logbook 131's route** (`probes/lineage_131.py.txt` and its record, the same animals). 131's lineage stayed in one basin, 0.54 apart from its curriculum's stage-3 answer to its last answer. The stage-1 answers of the four kept starts and 131's B04 are 3.19–4.07 apart, and those of every pilot start 3.19–4.57 (`probes/basin_distances.py.txt` and its record), so the basin is a start's.

  - **On this arm,** 131's curriculum answer read an onset rate z of −3.6 (−3.55 to −3.76), its dwelling speed's 90th percentile where the pilot's sits (z 2.0–2.2) and its median lower (0.3–0.7, against the pilot's 1.0–1.35). Its hinge and failure-probability stages moved the onset rate to −2.6 (−2.53 to −2.74) with the 90th percentile unchanged (1.9–2.3) and the median up to 1.4–1.9. It had room on the median that the pilot's answer did not.
  - **Why the registered final stage could not:** with the onset rate's SD of 0.112 between populations, the failure probability is 9.999 at onset z −3.5 and 10.000 below about −3.55. The stage earns nothing for raising it from −4.3, and pays for any move that costs a statistic near its bar (the dwelling speed's 90th percentile, SD 0.30). 131 ran a quadratic hinge before its failure stage.

- **The hinge screen** (`probes/hinge_screen.py.txt`, its record and log).

  - **How it ran.** From the pilot's final search state (σ 0.0271), a new sep-CMA carrying its σ and steps at the pilot's answer: population 48 at 200 animals, the registered arm, 50 generations. The objective was the final stage's plus the hinge as registered above. Its generation-0 readout reproduced the pilot's final readout exactly.
  - **The rule, fixed before it ran.** Over the readouts at generations 30, 40 and 50, the screen passes when the onset rate's mean z is at least −3.0781, every other wild-type statistic is defined with mean abs(z) at most 3.0781, and each mutant passes in at least 2 of 3. If it passes, the amendment adds the hinge to the final stage. If it fails, the fix has to act in 2a.
  - **The reading: passes.** The onset rate's z rose from −4.31 at generation 0 to −3.15, −2.88 and −3.01 at generations 30, 40 and 50 (mean −3.01). Nothing else failed on the mean, and every mutant passed 3 of 3.
  - **Reported, not selected on:** its answer graded with presence on 5 fresh populations passed the onset rate in all 5 (z −2.73 to −2.97) and the wild type in 2. Chemotaxis presence failed in 3 (the ratio's lower bound −0.04, −0.09, −0.04), with the roaming bout at 4.06 in one of them. The ratio's value fell from the pilot's 0.097 to 0.048 while its z moved from −1.69 to −1.76. It ran about 4.4 h, at 315 s a generation. Its script's docstring says Logbook 131's hinge stages left the dwelling speeds unchanged; that holds for the 90th percentile only (the trace above).

- **The chemotaxis sweep** (`probes/chemotaxis_sweep.py.txt` and its record, three common populations with presence).

  - **The points:** Logbook 131's sensory gains written into the hinge screen's answer: the whole sensory block, the odour block, its lateral, derivative or concentration gains, AWC's three, AWA's three, and the odour block moved halfway.
  - **The rule:** a point restores chemotaxis when, in all three populations, chemotaxis is present, the onset rate passes, everything else passes and every mutant passes.
  - **No point restored it.** No point from the screen's answer was present in all three, and most broke the lawn's statistics. Read on those moves, the rule's third branch holds: the screen's answer does not get chemotaxis from its sensory gains alone, and the fix acts where the search still moves the network. Logbook 131's own answer, a reference in the sweep, was present in all three (the ratio 0.35). Read literally, the rule's second branch ("some point is present in all three") names it, but that branch is about moves from the screen's answer.
  - **Where it was lost:** the pilot's readouts above place it in 2b.

- **The presence screen** (`probes/presence_screen.py.txt`, its record and log).

  - **How it ran.** From the hinge screen's search state after its 50 generations (σ 0.0202), carried as before: population 48 at 200 animals, 50 generations. The objective was the hinge screen's plus the presence term as registered above, each candidate graded with presence. Its generation-0 readout reproduced the hinge screen's last readout exactly.

  - **The rule, fixed before it ran.** Over the readouts at generations 30, 40 and 50, the screen passes when all of these hold:

    - chemotaxis is present in at least 2 of the 3;
    - the onset rate's mean z is at least −3.0781;
    - every other wild-type statistic is defined, with mean abs(z) at most 3.0781;
    - local search is present in at least 2 of the 3;
    - each mutant passes in at least 2 of the 3.

    If it passes, the amendment puts the hinge in the final stage and the presence term in 2b and the final stage. If it fails, chemotaxis has to enter 2a's objective.

  - **The reading: passes.** Chemotaxis was present at every readout from generation 10. The ratio rose from 0.021 at generation 0 to 0.204, 0.212 and 0.184 at generations 30, 40 and 50, its lower bound from −0.041 to 0.139, 0.148 and 0.120, and the drift from 0.0040 to 0.0061 mm/s. The onset rate read −3.10, −2.99 and −2.90 (mean −3.00). Local search was present at every readout, and every mutant passed 3 of 3.

  - **Reported, not selected on:**

    - its answer graded with presence on 5 fresh populations passed the wild type in 4, and tph-1, pdfr-1 and tax-4 in 5 each. Chemotaxis was present in all 5 (the ratio's lower bound 0.065–0.132), and the onset rate passed in all 5 (z −2.65 to −2.93). The one failure was pre-exit enrichment at −3.10, which read −3.07 in another population;
    - pre-exit enrichment also failed at the readouts at generations 10 (−3.91) and 50 (−3.73). Its mean over 30–50 was −2.17, inside the rule. It is the noisiest statistic, with an SD of z between populations of 1.14;
    - it ran 5.6 h, at 404 s a generation over its 50 against the hinge screen's 315 s, about 28% more, the cost of grading each candidate with presence;
    - its script's docstring quotes the drift's scale as 0.000183. The term used the value computed from the fifth pilot's grade, 0.000177, as its record states.

  - The screens' candidate stores (11–13 MB each) are kept off the repository with the pilots' run states.

- **What the amendment takes from them:**

  - **the hinge goes in the final stage,** by the hinge screen's rule. It moved the onset rate where the failure probability gave nothing, as Logbook 131's hinge stage did before its failure stage;
  - **presence enters the objective,** by the presence screen's rule: once scored, presence came back by generation 10 in the pilot's basin and stayed. On the rule's means over generations 30–50 everything else held, though single readouts failed pre-exit enrichment (generations 10 and 50) and the onset rate (30);
  - **presence also enters 2b,** where the pilot lost it. That part is not screened: the screen scored presence in the final stage only. 2b starts from 2a's answer, which held chemotaxis unscored, so the term is meant to stop 2b trading it away;
  - **2b's presence term carries no failure probability of the other statistics,** as 2b never has. The term is the same in both stages;
  - **F.0b-ii's truth must clear presence's target,** since the term pulls every candidate toward it whatever the truth. That mirrors the lawn's check, added with the lawn guard. Above the target the chance part still pulls, 0.23 a bound at the target and 0.013 at 3 s;
  - **every readout grades presence,** so a run's records show where presence goes. No reading or kill rule reads it;
  - **stated, not remedied:**
    - each screen continued the fifth pilot's basin 100 generations past its final stage, which no registered run does. The sixth pilot runs the procedure from the box's centre, so its basin and answer may differ. Each described grade is one CMA seed;
    - pre-exit enrichment sits near its bar (−3.10 and −3.07 in the presence screen's grade), and the go rule reads it;
    - **the presence term can outweigh the guards.** At the floor it is 1,460 (730 a bound), above the lawn guard's 300, the saturation guard's 500 and an undefined statistic's 1,000. The fifth pilot's 2a answer read a ratio of −0.11 on its readout animals, a lower bound near −0.18, where the term is about 190 against a distance near 2. A 2b starting there would rank its candidates on presence alone until it cleared. The screen started at about 51. The term is registered as screened, uncapped;
    - presence's scales were measured at the fifth pilot's answer, where the bounds' spread may differ from another answer's;
    - the screens' readouts reused the fifth pilot's readout animals, so their readings share those animals' noise with the pilot's.

- **The amended script's checks (2026-11-19):**

  - 68 tests in the fast tier;
  - **the amended objective reproduces the presence screen's** (`probes/objective_parity.py.txt` and its record, the candidate store's SHA-256 recorded): the final stage's objective, at the fifth pilot's noise SDs, scores every one of the screen's 2,400 stored candidates within 1.4·10⁻¹⁴ of the screen's own objective. It reproduces the screen's best score in all 50 generations within 1.8·10⁻¹⁵, and its best candidates' hinge and presence terms within 4.4·10⁻¹⁶. The presence scales are the screen's to the last digit;
  - **deliberate bugs** (`probes/mutants_presence.py.txt` and its two logs):
    - the first run caught 179 of 184 applicable bugs, and 54 were not applicable (earlier code lines this amendment changed; those that still apply are ported to the new lines). One miss was the equivalent bug of earlier runs (a reading not saved, which a resumed run takes again). Four were real gaps in the hinge's tests: the procedure dropping its flag, its margin at 0, its square dropped, and an infinite value not skipped. The tests now check the registered weight, margin and targets, a statistic 2 past its target, an infinite value, and the procedure's final stage carrying the hinge in its best score;
    - on the final tests, all seven of the hinge's bugs caught, and the review's five below (`mutants_presence_review.log.txt`);
  - **the smoke on real simulation** (9.4 min) dropped its first three starts at the lawn, the descent and the bar in turn and kept the fourth. Only 2b's and the final stage's candidates were graded with presence. 2b recorded its best presence term and the final stage its hinge and presence terms, and every readout carried presence's lower bounds. The noise step, the grade, the preflight's arithmetic and the held-out reading then ran;
  - **an independent review** (2026-11-19, a reviewer without the author's reading, on snapshots of the code) found no defect in the code. It recomputed both screens' recorded hinge and presence terms from their stored candidates in every generation (within 4·10⁻¹⁶), found the presence scales bit-exact, the new seed blocks and CMA seeds clear of every earlier one, and no planning reference in package code. It found:
    - **test gaps:** the hinge's wiring into the procedure untested (fake candidates sat at the truth, their hinge 0), and the best candidate's recorded terms unchecked (every fake candidate scored the same). A fake run now puts its candidates off the lawn, and another varies their bounds and time on the lawn with their vectors; the five bugs it named are caught;
    - **misquoted or overstated figures:** the ratio's fall credited to the pilot's final stage (it is between the pilot's answer and the hinge screen's), the presence cost (28%, not 23%), Logbook 131's dwelling median (it rose; only the 90th percentile held), "everything else held" (on the rule's means; single readouts failed), the stage-1 distances (no record computed them; `probes/basin_distances.py.txt` now does), and the objective's parity stated as a test (it is a probe, now committed);
    - **the onset sweep's literal reading:** 131's answer, a reference among its points, met the rule, as in the chemotaxis sweep. Both are now disclosed;
    - **two risks not stated:** the presence term's size against the guards at a start far below the target, and the chance part's pull above it. Both are now stated;
    - wording: "past its bar" for the hinge's push, the failure probability's flat point (9.999 at −3.5), a start's cost, the drift at 2b's start, the Risks' cost bounds, and the gate unchanged while the preflight gains a check. Each is corrected.

### Slow noise's positive control

`speed_noise_control.json` (run 2026-11-10 at HEAD `52212e43`; seeds 888,000,000 + i, reference episode 888,000,200): **passes.**

Logbook 131's answer under the procedure settings on the speed-following arm, at each timescale and without the OU noise, on the same 200 animals, under the registered rule (`c2_encoder_probe.read_speed_noise_control`):

| Check | Result | Bar |
|---|---|---|
| the speed output responds: its median over one reference episode, before the soft cap | −0.92 (capped, −0.86) | within ±2 |
| the roaming 90th percentile's rise above the shortest timescale | +1.50 at 2 s, +3.45 at 8.7 s, **+4.90 at 30 s**, +4.65 at 100 s, +4.71 at 300 s (band scales) | ≥ 3.0 at some timescale |
| at the shortest timescale, every kinematics statistic against independent noise | 0.04–0.62 band scales | ≤ 2.0 |

Slow noise reaches the window statistics it was built for, on an answer whose speed output responds. Logbook 123's failure was the worm's, which did not run. The rule's responsiveness bound reads the output before the soft cap. Here both readings sit well inside it.

### Cost

At the measured rates on 16 workers (77.7 s a generation on two assays, 309.5 s on six with the mutants, readouts included):

| Stage | Estimate |
|---|---|
| a run of the procedure, its first start kept | about 29 h: stage 1 0.8 h (extrapolated from 1.6 h on 8 shared workers; the pilot measures it), 2a 6.5 h, 2b 17.2 h, the final stage 4.3 h, the noise step and the grade under 0.5 h. *(2026-11-13: the second pilot's starts 0–3 measured stage 1 at 0.9 h and 2a at 83 s a generation (80.7–84.3 s), so 2a's 300 at 6.9 h and a run at about 30 h.)* |
| each dropped start | about 2.1 h: stage 1 0.8 h, 2a's 60 generations 1.3 h. At 2 in 5 a run expects 1.5 of them (3.2 h). Five before a kept one add 10.5 h. *(2026-11-13: measured at 2.3 h.)* |
| the pilot, F.0b-ii's fit, the registered fit | one run each |
| `recovery-preflight` | 231 simulations, about 30 min |
| the leave-one-out | about 26 h |
| in all | about 114 h with every first start kept, about 127 h at 2 in 5. *(2026-11-13, at the measured rates: about 117 h with every first start kept (three runs at about 30 h, the leave-one-out at about 27 h, the preflight 0.5 h), and 2.3 h for each dropped start.)* *(2026-11-14: the third pilot measured 2a at about 87 s a generation on 16 workers, so a run at about 30.4 h, the leave-one-out at about 27.5 h, and about 119 h in all with every first start kept. A start dropped at the lawn costs about 2.3 h, at the descent about 4.4 h and at the bar about 8 h. At 2 stalls in 5, a run expects about 3 h of dropped starts.)* *(2026-11-16: the fourth pilot measured its kept start's stage 1 at 0.81 h, 2a at 6.04 h, 2b at 19.24 h (346 s a generation, against the 309.5 s above) and the final stage at 3.95 h, about 30.5 h with the noise step and the grade. Its two dropped starts cost 2.2 h at the lawn and 4.2 h at the descent, and the pilot 36.6 h in all. A 2b that starts its own search costs what a carried one did, so a run is about 30.5 h with its first start kept, the leave-one-out about 27.6 h, and the chain about 120 h in all with every first start kept, before dropped starts.)* *(2026-11-19: the fifth pilot measured its kept start's stage 1 at 0.89 h, 2a at 6.85 h, 2b at 17.87 h and the final stage at 4.32 h. Its four starts dropped at the descent cost 4.1–4.4 h each, and the pilot 47.1 h in all. Grading each candidate with presence cost the presence screen about 28% a generation (404 s against 315 s), so 2b is about 22.9 h and the final stage about 5.5 h. A run is then about 37 h with its first start kept, the leave-one-out about 34 h, and the chain about 145 h in all with every first start kept, before dropped starts. At the fifth pilot's four dropped starts, a run adds about 17 h.)* |

### The guards

- **The manifest** (`manifest.json`, bound): the SHA-256 of `c2_fit.py` and every analysis script it imports, the arm's four assay configs, `bands_v4.json`, `primitives.json`, the neuromodulator data files, the encoding table, and one digest over the package's source.
  - Every guarded stage refuses unless every digest holds and v4's bands are the validated ones.
  - Every record states HEAD and the manifest's digest.
- **`pilot`** refuses unless this Part A is committed and unchanged, with the manifest, the encoding table and the control's record bound and the control passing.
- **`recovery-preflight`** needs Part B, with `pilot.json` bound and its go rule passing. It runs once.
- **`recovery`** needs the preflight's record committed, unchanged and passing.
- **`run`, `grade`, `read` and `loo`** need Part C, with both recovery records bound and the gate passing.
- **`smoke`, `manifest` and `speed-noise-control`** are unguarded. The control and the smoke use discarded or the control's own seeds, and the smoke reads band-cohort dates only.

## Part B: the readiness pilot

**Added 2026-11-21.** The sixth readiness pilot ran at HEAD `08128bb7`, manifest `79f185d3…`, on base 1,540,000,000 with CMA seeds 9926–9929 and restarts from 9930. It ran from 2026-11-19 13:31 to 2026-11-21 10:44 AEDT. Its record is `pilot.json`, bound. The record written when its kill rule stopped it is `pilot_sixth_kill.json`. This Part changes no rule, threshold or seed.

### The go rule: passes

The pilot kept its third start, start 2. Its answer, graded with presence on the first 5 grade populations:

| | Populations passing | The rule |
|---|---|---|
| the wild type | **3 of 5** (0, 3 and 4) | at least 3 |
| tph-1 | **5 of 5** | at least 3 |
| tax-4 | **5 of 5** | at least 3 |
| pdfr-1, reported | 5 of 5 | |

Populations 1 and 2 fail on one statistic, the on-lawn onset rate. Its z is −3.16 and −3.12 against the bar of 3.07809; in the passing three it is −3.01, −3.04 and −2.78. Local search and chemotaxis are present in all five. **The go rule passes.** F.0b-ii's truth is this answer (`pilot.json`'s `answer`).

### Reported

- **The 10 more populations:** the wild type passes in 8 of 10, so 11 of 15 in all. Each mutant passes in 10 of 10, so 15 of 15.

- **Over the 15, statistic by statistic** (description):

  - The onset rate is the only statistic that fails. It fails in 4 of 15, with z from −3.12 to −3.18. Over all 15 its z runs from −2.78 to −3.18, with a mean of −3.04.
  - The chemotaxis flip fraction passes in all 15, with z from 2.82 to 2.94. That is just past the hinge's target (2.79–2.90 on the 10·asinh scale, against 2.77809).
  - No other statistic's |z| reaches 2.8.
  - D is 1.50–1.64, D_fit 1.59–1.76, and time on the small lawn 0.983–0.991.
  - Local search's lower bound is at least 2.98.
  - Chemotaxis presence's lower bounds are at least 0.0068 mm/s for the drift and 0.163 for the ratio. That is 39 and 3.7 of the presence term's scales.
  - The reversal share is 11.7–11.9%, against Roberts' 17.3%; it is reported, never graded.

- **The final stage's plateau reading:** the rule never held in its 50 generations. The final stage's median fell from 64.7 at generation 1 to 28.0 at generation 50.

- **The noise step's SDs:** 0.095 for the onset rate. The chemotaxis flip fraction's took the floor, 0.05.

- **The readouts** (description):

  - 2b's 21 readouts never passed the wild type:
    - the chemotaxis flip fraction failed in every one, with z from 3.75 to 5.39;
    - the onset rate failed in 10 of them;
    - local search was absent at generations 0–20.
  - The final stage's readouts moved the chemotaxis flip fraction from 4.04 to 2.89, and its last readout passed the wild type.
  - Every readout passed all three mutants from 2b's generation 40. Before that, tph-1 failed at 2b's generations 0–30.

- **The starts.** Each start's output spreads at stage 1's end are reported, not selected on.

  | Start | CMA seeds | Stage 1's last three readouts: on the lawn, smaller spread | Spreads at stage 1's end: speed, curvature, reversal, omega, delta | 2a's readings | Outcome |
  |---|---|---|---|---|---|
  | 0 | 9926, 9927 | 0.704, 2.89: the kill rule tripped, overridden (below) | 3.32, 2.31, 3.55, 0.15, 3.74 | lawn 0.877 | dropped at the lawn |
  | 1 | 9930, 9931 | 0.953, 3.44 | 3.36, 0.01, 3.71, 0.06, 1.77 | lawn 0.974; descent 4.96 | dropped at the descent |
  | 2 | 9932, 9933 | 0.902, 3.89 | 3.93, 3.14, 3.98, 0.25, 3.88 | lawn 0.958; descent 3.19; bar 1.69 | **kept** |

  2b and the final stage took the run's third and fourth CMA seeds, 9928 and 9929.

- **The cost per stage,** on 16 workers:

  - **The kept start:** about 37.7 h.
    - stage 1: 0.94 h;
    - 2a: 6.98 h, 83.8 s a generation;
    - 2b: 23.56 h, 424 s a generation;
    - the noise step: about 4 min;
    - the final stage: 6.14 h, 442 s a generation;
    - the grade: about 3 min.
  - **The dropped starts:**
    - start 0: 2.36 h, stage 1 0.93 h and 2a 1.43 h;
    - start 1: 4.48 h, stage 1 0.97 h and 2a 3.51 h.
  - **The pilot:** 44.6 h of computing, or 45.2 h on the clock with the 0.56 h stop at the kill.
  - **From here:** a run is about 38 h with its first start kept, as Part A estimated. Part A's 145 h in all included the pilot; the chain from here is:
    - the preflight: about 0.5 h;
    - F.0b-ii's fit: about 38 h;
    - the registered fit: about 38 h;
    - the leave-one-out: about 34 h;
    - in all, about 110 h before dropped starts. At this pilot's two dropped starts, a run adds about 7 h.

### Disclosures

- **The stage-1 kill and its override.**

  - **The kill.** At 14:27 on 2026-11-19, 0.94 h after launch, start 0's stage 1 tripped the pilot's kill rule. It was on the lawn 0.704 on the mean of readouts 130–150 (0.355, 0.919 and 0.836), against 0.75. At generations 90–120 it had read 0.85–0.91 on the lawn. Its smaller spread was 2.89, against a floor of 0.5. The pilot stopped and returned to the maintainer, as the rule says.
    - Every readout uses the same 200 animals, so the low readouts are the search's mean moving, not new animals. Start 0 later failed 2a's lawn reading too.
  - **The override.** At 15:00 the maintainer overrode the kill. The saved state's kill flag was cleared, and the pilot resumed into start 0's 2a on the same seeds. Its checkpoint keeps the kill. A later start tripping the rule would have stopped the pilot again; none did. No code, rule or seed changed.
  - **What is kept.** The record at the kill (`pilot_sixth_kill.json`) is committed. The saved state at the kill, the log up to it and the maintainer's note are kept with the pilot's state, off-repo.

- **What the override changed.** In a registered run stage 1's checkpoint is reported, not a stop. The resumed pilot therefore ran the path the registered procedure runs.

  - Start 0 was then dropped by 2a's lawn reading, a rule of every run, after 1.43 h more.
  - The kept start never tripped a kill rule, and its seeds and animals are its own.
  - Without the override the pilot would have stopped with no answer.

- **Seen before this Part:** each readout, as the pilot ran, reported to the maintainer. No rule, threshold or seed was changed after any of it.

- **Not in this Part:** the neural reading's paired test and wiring-only control. Part A allows them with Part B or Part C; they come with Part C, before `run`.

- **A pointer in Part A's disclosures:** the second pilot's start 3, whose record Part A says "is added in Part B", was recorded in the amendment of 2026-11-14 (`probes/guard_rep_3.json`, under *Disclosures*).

- **Corrected 2026-11-21, before the preflight,** after an independent check of this Part against `pilot.json`:

  - four spreads were rounded twice;
  - the stop's length, the grade's time, and presence's two minima were slightly off;
  - the kill's earlier readouts are now stated exactly;
  - the onset rate's failing range now stands apart from its range over all 15.

  No count, pass or verdict changed.
