# controller-fit Specification

## Purpose

Fitting a controller to the real wild type by the validated method, under a registered verdict, and
the mutant encoding table that every controller's predictions are graded under.

## Requirements

### Requirement: The fit and the mutant encoding table are registered before any fit

A launch record SHALL fix C.1's objective, sep-CMA settings, restarts, plateau criterion, grading population, verdict and verdict map before any fit simulation runs. An encoding table, committed with it, SHALL fix every controller's encoding of tph-1, pdfr-1 and tax-4, what each mutant is graded on, and the `unidentified` rule. An encoding SHALL be a structural knockout with no free magnitude. A mutant with no counterpart in a controller SHALL be recorded as not encodable.

#### Scenario: The fit is guarded

- **WHEN** `run` starts while the launch record is uncommitted or changed, or the encoding table differs from its version in the launch record's commit
- **THEN** it SHALL refuse to run

#### Scenario: A mutant with no counterpart

- **WHEN** a controller has no mechanism a mutant's gene acts on
- **THEN** its row SHALL read not encodable, and that controller SHALL make no prediction for that mutant

### Requirement: C.1 is fitted to the wild type by sep-CMA

On benchmark v1 (Logbook 113), C.1 SHALL be fitted by sep-CMA with 8 restarts from the box centre (σ₀ 0.2, population 16, bounds [0, 1]). The objective SHALL be the benchmark's D on the 11 wild-type statistics against the band cohorts at n = 200, for 200 fresh animals per evaluation. Each restart SHALL stop at the plateau rule or at 210 generations. A restart stopped by the cap SHALL NOT be chosen. The chosen restart SHALL be the plateaued restart with the lowest mean median objective over its last 20 generations. C.1's refit on benchmark v2 follows "C.1 is refitted on benchmark v2".

#### Scenario: No restart plateaus

- **WHEN** every restart stops at the cap
- **THEN** no verdict SHALL be read, and the run SHALL return to the maintainer

### Requirement: The wild-type verdict is the benchmark's rule

Every restart's answer SHALL be graded on five registered populations of 200 animals. On benchmark v1, a population SHALL pass when every family passes: each statistic at |z| ≤ 2.84, and local search only where its presence test holds. On benchmark v2, a population SHALL pass when every family passes: each statistic at |z| ≤ 3.0078, and local search and chemotaxis only where present. The chosen restart SHALL pass the wild type when at least 3 of its 5 populations pass. Every restart's verdict and D, the restart spread and the parameters SHALL be reported beside the verdict.

#### Scenario: C.1 fails the wild type

- **WHEN** fewer than 3 of the chosen restart's 5 populations pass
- **THEN** the failing statistics SHALL be recorded as the diagnosis, and the roadmap's pivot SHALL go to the maintainer before C.2

#### Scenario: One unlucky population does not fail a faithful controller

- **WHEN** one of five populations fails a family and the other four pass
- **THEN** the chosen restart SHALL pass the wild type

### Requirement: Mutant predictions that depend on the restart are unidentified

Among a controller's restarts whose answers pass the wild type (3 of 5 populations), a mutant prediction whose verdict differs between restarts SHALL be `unidentified`.

#### Scenario: Two passing restarts disagree

- **WHEN** one passing restart's tph-1 prediction passes and another's fails
- **THEN** the tph-1 prediction SHALL be recorded as `unidentified`

### Requirement: C.1 is refitted on benchmark v2

C.1's model SHALL be refitted to the real wild type on benchmark v2 by sep-CMA from the box centre, 8 restarts, ranking each candidate by the definedness-first score against the bands at n = 200. Every registered reading SHALL use the benchmark's own grade, graded with presence.

#### Scenario: The objective is the bands'

- **WHEN** a candidate is scored
- **THEN** its score SHALL be 1,000 × the statistics it leaves undefined + the weighted RMS of 10·asinh(z/10), for z its statistics' distances from the band centres on the band scales

### Requirement: The plateau stop waits for a defined window

A C.1b restart SHALL stop on the plateau rule only once σ ≤ σ₀ and no generation of the two windows the rule compares has a median score of 1,000 or more.

#### Scenario: An undefined median holds a restart

- **WHEN** the plateau rule and σ hold, but a median in the last 40 generations is 1,000 or more
- **THEN** the restart SHALL continue

### Requirement: Restarts are chosen on separate populations

The chosen restart SHALL be chosen on 3 selection populations per plateaued restart, graded with presence, and disjoint from the 5 registered populations its verdict is read on. A restart stopped by the cap SHALL NOT be chosen.

#### Scenario: Selection and verdict share no animals

- **WHEN** the restarts are selected and the verdict is read
- **THEN** no animal SHALL appear in both a selection population and a registered population

#### Scenario: The most passing restart is chosen

- **WHEN** two plateaued restarts pass 3 and 2 of their selection populations
- **THEN** the one passing 3 SHALL be chosen, and a tie SHALL go to the lower median D

### Requirement: A reachability pilot precedes the fits

Before Part B, one restart SHALL run 300 generations on discarded seeds to set the cap and project the cost, and its answer SHALL be graded with presence on 5 discarded populations. If it passes fewer than 3, the change SHALL return to the maintainer before any registered fit.

#### Scenario: The pilot cannot reach the wild type

- **WHEN** the pilot's answer passes 2 of its 5 discarded populations
- **THEN** no registered fit SHALL run until the maintainer decides

### Requirement: C.1 is refitted on benchmark v3 with its 25-parameter model

C.1 (v3) SHALL be C.1's 20 parameters on F.0a's box plus `v_off_food`, `sigma_off_food`, `speed_log_sd_dwell`, `speed_log_sd_roam` and `kappa_off_food`, on ranges set without search data. It SHALL be fitted to the real wild type on benchmark v3 by sep-CMA from the box centre, with 12 restarts, the definedness-first ranking score against v3's bands at n = 200, the doubly guarded plateau and a cap of 330. Selection on 5 separate populations per plateaued restart SHALL choose the restart, and the verdict SHALL be at least 3 of 5 registered populations passing with presence at |z| ≤ 3.0381.

#### Scenario: A capped restart is not chosen

- **WHEN** a restart reaches generation 330 without the guarded plateau
- **THEN** it SHALL NOT be chosen, though its registered grades SHALL be reported

#### Scenario: Seed blocks do not overlap

- **WHEN** the fit's, the answers', the selection's, the registered populations' and the recovery's animal seeds are enumerated
- **THEN** no animal seed SHALL appear in two of them

### Requirement: A reduced recovery gates the v3 fit

Before any registered v3 fit, sep-CMA SHALL recover C.1 (v3) at two known truths, the two passing probe answers. Each truth SHALL be fitted twice from the box centre on its observation, and the better restart by final objective SHALL be scored by its gain fraction, (prior D − D) / (prior D − floor). The truths' oracle SHALL reach a median gain of at least 0.85 and the box centre at most 0.25 before any recovery fit runs. The fit SHALL run only if the median gain over the two truths is at least 0.75.

#### Scenario: The recovery falls short

- **WHEN** the two truths' median gain is 0.70
- **THEN** no registered fit SHALL run until the maintainer decides

### Requirement: The probes are the v3 fit's pilot

The discarded-seed probes recorded in launch record 121 SHALL set the v3 fit's cap and stand as its reachability evidence, and no further pilot SHALL run. Every probe SHALL be disclosed in Part A with its seeds and outcome.

#### Scenario: The cap comes from the probes

- **WHEN** the cap is registered
- **THEN** it SHALL be at least 1.5 × the longest plateau generation among the 25-parameter probes

### Requirement: The 25-parameter model's mutant encodings

C.1 (v3)'s mutant encodings SHALL be C.1's registered encodings, except that tax-4 SHALL also set `kappa_off_food` to 0. The four kinematic parameters SHALL be kept by every mutant. The extension SHALL be registered in Part A, before any fit, and the original table SHALL stay byte-identical.

#### Scenario: tax-4 removes both weathervane gains

- **WHEN** C.1 (v3)'s tax-4 encoding is built
- **THEN** both `kappa_weathervane` and `kappa_off_food` SHALL be 0, and the off-food speed, curvature and spreads SHALL be unchanged

### Requirement: Slow speed noise's positive control precedes C.2's registration

Before Part A, the registered speed-noise control SHALL run under its own rule on Logbook 131's answer, under the procedure settings and on the speed-following arm. Its points SHALL be the answer at each of the control's timescales and the answer without the OU noise. If it fails, the change SHALL return to the maintainer before Part A.

#### Scenario: The control fails

- **WHEN** the roaming 90th percentile rises by less than the control's bar at every timescale
- **THEN** Part A SHALL NOT be committed until the maintainer decides

### Requirement: C.2 is fitted by a staged procedure fixed in advance

C.2 SHALL be fitted by sep-CMA (bounds [0, 1], the three guards; population 24 and 64 animals a candidate in stage 1, population 48 and 200 animals a candidate after it) in four stages, every one fixed in the launch record before any scored run:

- **stage 1:** the 77 parameters of the wiring, sensory, readout and noise blocks and the pool centres, from the graded box's centre at σ₀ 0.2, on D_fit's families, for 150 generations;
- **2a:** all 137, from stage 1's answer with the speed noise's timescale set to 1 s, at σ₀ 0.1, on D_fit's families, for 300 generations;
- **2b:** all six families with tph-1, pdfr-1 and tax-4 at half the objective, from a new isotropic search at 2a's answer at σ₀ 0.05, for 200 generations;
- **the final stage:** 2b's objective plus each statistic's chance of failing a population, from 2b's search state, for 50 generations.

Each stage SHALL run its full generation count. The plateau rule SHALL be read on the final stage and reported, not used as a stop. 2b SHALL start a new isotropic strategy at 2a's answer, on the run's third CMA seed. The final stage SHALL start a new strategy at 2b's answer, with 2b's σ and per-parameter steps. The answer SHALL be the final stage's mean, clipped to the box.

#### Scenario: 2b restarts the search at 2a's answer

- **WHEN** 2b starts, whichever start of the first half was kept
- **THEN** its strategy SHALL be isotropic at σ₀ 0.05 on the run's third CMA seed, its mean SHALL be 2a's answer, and no stage SHALL take 2a's search state

#### Scenario: The final stage keeps 2b's search steps

- **WHEN** the final stage starts from 2b's strategy with σ and per-parameter steps s_j
- **THEN** its initial σ SHALL be 2b's σ, its initial per-parameter steps SHALL be s_j, and its mean SHALL be 2b's answer

#### Scenario: Stage 1 fits only its subset

- **WHEN** stage 1 writes a candidate
- **THEN** every parameter outside its 77 SHALL be 0.5

#### Scenario: A plateau does not stop a stage

- **WHEN** the final stage's medians plateau at generation 40
- **THEN** it SHALL run to generation 50, and its plateau generation SHALL be reported

### Requirement: The first half is started until a start holds the lawn and reaches 2a's bar

The first half of the procedure (stage 1 and 2a) SHALL be started up to six times in sequence. Each start SHALL begin from the graded box's centre with its own CMA seeds and its own candidate animals, and SHALL be read on the run's readout animals. The rule SHALL be the same in every run of the procedure.

2a SHALL read each start three times, each reading a mean over three of 2a's readouts, and SHALL drop the start at the first reading it fails:

- **the lawn, at generation 60:** time on the lawn over the readouts at generations 40, 50 and 60 at least 0.95. An undefined time on the lawn SHALL count as 0;
- **the descent, at generation 150:** D_fit over the readouts at generations 130, 140 and 150 at most 4.25;
- **the bar, at generation 300:** D_fit over the readouts at generations 280, 290 and 300 at most 2.5.

An undefined D_fit SHALL fail its reading.

- **The first start that passes all three readings** SHALL be kept, and the procedure SHALL continue from its 2a.
- **A run in which every start is dropped** SHALL have no answer, and no later stage SHALL run.
- **Every start** SHALL be recorded with its readouts, its histories, every reading it took and the reason it was dropped.
- **A run whose readings have no readout, are out of order, or fall past 2a's end** SHALL be refused.

#### Scenario: A start that does not hold the lawn is dropped

- **WHEN** a start's 2a readouts at generations 40, 50 and 60 average 0.90 on the lawn
- **THEN** it SHALL stop at generation 60, and the next start SHALL begin from the box's centre on its own CMA seeds and animals

#### Scenario: A start that misses 2a's descent or bar is dropped

- **WHEN** a start holds the lawn and its 2a readouts at generations 130, 140 and 150 average D_fit 5.0, or average 3.0 there and 3.0 at generations 280, 290 and 300
- **THEN** it SHALL stop at generation 150, or at generation 300, with the reading it failed recorded, and the next start SHALL begin from the box's centre on its own CMA seeds and animals

#### Scenario: No start holds the lawn

- **WHEN** every start is dropped, at any of 2a's readings
- **THEN** no later stage SHALL run, and the run SHALL have no answer

#### Scenario: The lawn-holding reading is a mean

- **WHEN** a start's 2a readouts at generations 40, 50 and 60 read 0.90, 0.94 and 0.99 on the lawn
- **THEN** it SHALL be dropped, its mean being 0.943

#### Scenario: The descent reading is a mean

- **WHEN** a start's 2a readouts at generations 130, 140 and 150 read D_fit 4.6, 4.5 and 3.9
- **THEN** it SHALL be dropped, its mean being 4.33

#### Scenario: An interrupted drop resumes to the same answer

- **WHEN** a run stops after the generation of any of 2a's readings is saved and before its start is dropped, and is resumed
- **THEN** its starts and its answer SHALL equal an uninterrupted run's

### Requirement: C.2's objective weighs the mutants at half and scores each statistic's chance of failing

The objective SHALL be the definedness-first ranking score against the truth on the stage's families, with these additions:

- where a stage fits n_m mutants, each mutant's term (the RMS of 10·asinh(z/10) over the statistics the encoding table grades it on) SHALL weigh n_f / n_m, so the mutants together weigh as much as the n_f families;
- 500 SHALL be added when over half the motor neurons sit at abs(h) > 0.99 in over half the steps;
- 300·(1 − s/0.5) SHALL be added when the smaller of the speed and reversal outputs' 5th-to-95th-percentile spreads, s, is below 0.5;
- in every stage, 300·(0.95 − t)/0.95 SHALL be added when the time on the small lawn, t, is below 0.95. An undefined time on the lawn SHALL count as 0;
- in the final stage, 10·Σ Φ((abs(z) − bar) / SD) SHALL be added over every defined wild-type and table-graded mutant statistic, with the bars read from the bands and SD from the noise step;
- in the final stage, 5·Σ max(0, abs(10·asinh(z/10)) − target)² SHALL be added over the same statistics, the target 0.3 inside each pass rule;
- in 2b and the final stage, for each of chemotaxis presence's two lower bounds b (the drift and the log quintile ratio, as the registered grade computes them), with s its registered SD between populations, 5·max(0, 2 − b/s)² + 10·Φ(−b/s) SHALL be added, b/s floored at −10 and an undefined bound at the floor. Those stages' candidates SHALL be graded with presence.

The three guards SHALL read the wild type's simulation only. Each generation's share of candidates paying the lawn guard SHALL be recorded, and in the stages that score them its best candidate's hinge and presence terms.

The noise step SHALL grade 2b's answer on 30 populations of 200 without presence, and take each statistic's SD between them, floored at 0.05. A statistic defined in fewer than 2 populations SHALL take the floor.

#### Scenario: The mutants weigh half

- **WHEN** every family's mean square is 1 and every mutant's term is 3, with six families and three mutants
- **THEN** the score's distance SHALL be sqrt((6 + 2 × 27) / 12) = sqrt(5)

#### Scenario: A statistic at its bar

- **WHEN** a wild-type statistic's abs(z) equals the bands' wild-type bar in the final stage
- **THEN** its failure term SHALL be 10 × 0.5 = 5

#### Scenario: The lawn guard

- **WHEN** a candidate's wild type is on the small lawn 47.5% of the time, in any stage
- **THEN** its lawn guard SHALL be 150, and at 95% or more on the lawn SHALL be 0

#### Scenario: tph-1's leaving is not scored

- **WHEN** a candidate's tph-1 exits difference is far from its band
- **THEN** its score SHALL be unchanged, and the difference SHALL be recorded

#### Scenario: A statistic undefined in most noise populations

- **WHEN** a statistic is defined in only one of the 30 noise populations
- **THEN** its SD SHALL be 0.05

#### Scenario: The hinge

- **WHEN** a wild-type statistic's compressed abs(z) is 1 past its target in the final stage
- **THEN** its hinge term SHALL be 5, and inside the target SHALL be 0

#### Scenario: Presence is scored

- **WHEN** a candidate in 2b or the final stage has a lower bound at 0
- **THEN** that bound's presence term SHALL be 5 × 2² + 10 × 0.5 = 25, and a missing bound SHALL score at the floor

### Requirement: Readouts record every statistic, and runs resume safely

Every 10 generations, and at each stage's generation 0, the stage's mean SHALL be graded on the same 200 readout animals, on all six families and the three mutants, the wild type with presence. Every statistic's value and z SHALL be recorded, with each mutant's reading. The strategy and history SHALL be saved atomically after every generation. A resumed run SHALL refuse a saved state whose recorded settings differ from its own. A stored simulation record SHALL be refused at another vector. Each start's first-half records SHALL sit in that start's own store.

#### Scenario: A resume with changed settings

- **WHEN** a run is resumed with a stage's generation count or start (σ₀, or a carried search state), the lawn guard's bar or weight, or any of 2a's readings' readouts or bars changed
- **THEN** it SHALL refuse and run nothing

### Requirement: C.2's registered fit runs on the speed-following arm

Every stage of C.2's pilot, recovery, registered fit and leave-one-out SHALL simulate on the grading arm `connectome_c2_following`. Its assay configs SHALL be generated from the encoded arm's, with `continuous.on_food_reversal_speed: follows_run_speed` as their only difference.

#### Scenario: The arm differs only in the reversal speed

- **WHEN** each `encoded_following` assay config is loaded beside its `encoded` counterpart
- **THEN** the two SHALL be equal except for `environment.continuous.on_food_reversal_speed`

### Requirement: A readiness pilot precedes C.2's recovery and fit

The registered procedure SHALL run once on discarded seeds before any registered stage, configured exactly as the fit. Its go rule, fixed in Part A, is that its answer, graded with presence on 5 discarded populations, passes the wild type in at least 3, and tph-1 and tax-4 each in at least 3.

The pilot SHALL stop on any kill rule:

- a start's stage 1 unresponsive, or holding the lawn less than 75% of the time, on the mean of its last three readouts;
- no start of its six kept by 2a's readings.

In the registered runs stage 1's checkpoint SHALL be reported, not a stop. 2a's bar SHALL be a reading of every run's starts, not a kill rule.

#### Scenario: The pilot fails its go rule

- **WHEN** the pilot's answer passes tax-4 in 2 of 5 populations
- **THEN** no registered stage SHALL run until Part A is amended

#### Scenario: A registered run misses a checkpoint

- **WHEN** a registered run's stage 1 ends with its reversal output's spread at 0.2
- **THEN** it SHALL continue to its answer, and the checkpoint SHALL be reported

#### Scenario: A start's 2a ends above the bar

- **WHEN** a start's 2a ends with D_fit 2.8, in the pilot or a registered run
- **THEN** the start SHALL be dropped and the next one begun, and no kill rule SHALL stop the run

### Requirement: C.2's verdict

The answer SHALL be graded on 5 registered populations of 200, every family with presence, and each mutant on the small lawn on its population's own animals, on the statistics the encoding table grades it on. C.2 SHALL pass when the wild type passes in at least 3 of the 5, and tph-1 and tax-4 each pass in at least 3 of the 5. pdfr-1 SHALL be graded and reported, outside the verdict. The verdict SHALL be reported as in-sample.

Reported beside the verdict, and never read in its place:

- the answer on 10 more populations;
- the held-out reading;
- every readout;
- the final stage's plateau reading;
- the reversal share;
- the cost;
- the restart spread: the median capped D between the pilot's and the registered answer's populations g < 15, on W's fitted families and on all six;
- the neural reading: the answer's in-silico single-neuron stimulation responses against Creamer et al.'s measured ones, with a paired test against the unfitted network and a wiring-only control, on rules fixed in Part A.

A pass with a poor held-out reading, or with the leave-one-out's tax-4 failing, SHALL be read by the outcomes the launch record names in advance. If the fit runs under the maintainer's decision after a failed recovery gate, the verdict and its rule SHALL be unchanged, a pass SHALL be read as in-sample only, and a fail SHALL be read against the failed recovery, not attributed to the model.

#### Scenario: A fitted mutant fails the verdict

- **WHEN** the wild type passes in 4 of 5 populations and tax-4 in 2 of 5
- **THEN** C.2 SHALL fail, with tax-4 named in the diagnosis

#### Scenario: A registered fit with no answer fails

- **WHEN** every start of the registered fit is dropped
- **THEN** its verdict SHALL be fail, with the reason recorded

#### Scenario: pdfr-1 does not decide

- **WHEN** the wild type, tph-1 and tax-4 each pass in at least 3 of 5 and pdfr-1 in 1 of 5
- **THEN** C.2 SHALL pass, and pdfr-1's 1 of 5 SHALL be reported

### Requirement: The fitted worm is read on held-out recording dates

For each of 1,000 splits of a held-out date set into halves (seed 2034), the reading SHALL compute three kinds of distance, on the small lawn, kinematics and activity, with PD1074's band scales at the half's size rounded to 25:

- between the two real halves;
- between each of the answer's 15 graded populations and each half;
- between each of 15 populations of 200 animals drawn from PD1074's band cohorts and each half.

The date sets SHALL be N2's 26 unread held-out dates, and PD1074's 24 held-out dates beside them. The reading SHALL report each kind's median and 95th percentile, as description. It SHALL use no new simulation. C.1's refit before C.4 SHALL be read by the same function.

#### Scenario: The smoke reads no held-out track

- **WHEN** the smoke runs the reading
- **THEN** it SHALL read band-cohort date sets only

#### Scenario: The real halves reproduce W's minimum

- **WHEN** the reading's real-against-real distances are computed on N2's unread held-out dates
- **THEN** their 95th percentile SHALL equal `benchmark_v4_controls.w_minimum_v4`'s fitted minimum

### Requirement: A tax-4 leave-one-out, as description

After the registered fit, the same procedure SHALL run on its own seeds with tax-4 out of the objective, unless the fit ran under the maintainer's decision after a failed recovery gate, in which case it SHALL NOT run. tph-1 and pdfr-1 SHALL then be together half of it, and the failure probability SHALL omit tax-4's statistics. Its answer's tax-4 SHALL be graded on 15 populations and reported beside the registered fit's. It SHALL gate nothing.

#### Scenario: tax-4 is never scored

- **WHEN** a leave-one-out candidate is scored
- **THEN** its score SHALL be unchanged by its tax-4 reading

### Requirement: C.2's registration is in three parts

Launch record 133 SHALL be committed in three dated parts:

- **Part A, before the pilot:** every rule, seed, gate and verdict, disclosing Logbook 131's probes;
- **Part B, after the pilot:** its result;
- **Part C, after the recovery:** its result.

Part B SHALL change no rule. Part C SHALL change none either, with one exception: after a failed recovery gate, the maintainer's decision, recorded in Part C under its own heading, MAY let the registered fit run once with the gate failed, drop the leave-one-out and narrow the verdict map's branches. It SHALL change no seed, threshold, procedure or grade, and not the verdict's rule. Part A SHALL record SHA-256 digests of `c2_fit.py` and the scripts it imports, the arm's assay configs, `bands_v4.json`, `primitives.json`, the neuromodulator data files and the encoding table. Every stage SHALL check them at start and at each resume, refuse on a mismatch, and require v4's bands to be valid. The guards are:

- `pilot` SHALL refuse unless Part A is committed and unchanged;
- `recovery-preflight` SHALL refuse unless Part B is committed with the pilot's record bound and its go rule passing;
- `recovery` SHALL refuse unless the preflight's record is committed, unchanged and passing;
- `run`, `grade`, `read` and `loo` SHALL refuse unless Part C is committed with both recovery records bound and the gate passing;
- with the gate failed, `run`, `grade` and `read` SHALL proceed only if Part C carries the maintainer's decision's heading as a line of its own, `loo` SHALL refuse, and `run` SHALL refuse once the fit's record exists;
- `read` SHALL record the recovery's gate beside the verdict.

#### Scenario: The fit refuses before the recovery is recorded

- **WHEN** `run` is invoked while Part C is absent or the recovery record differs from its committed version
- **THEN** it SHALL refuse and run nothing

#### Scenario: The maintainer's decision lets the fit run with the gate failed

- **WHEN** the recovery's gate failed and Part C carries the decision's heading
- **THEN** `run`, `grade` and `read` SHALL proceed, and `loo` SHALL refuse

#### Scenario: The fit runs once under the decision

- **WHEN** `run` is invoked under the decision and the fit's record exists
- **THEN** it SHALL refuse, and `grade` and `read` SHALL still proceed

#### Scenario: The decision counts only in Part C

- **WHEN** the recovery's gate failed and the decision's heading appears only outside Part C
- **THEN** `run`, `grade`, `read` and `loo` SHALL refuse

#### Scenario: A changed config stops a run

- **WHEN** an assay config in the manifest differs from its digest at a resume
- **THEN** the stage SHALL refuse

#### Scenario: A failed pilot blocks the recovery

- **WHEN** `recovery-preflight` is invoked and the bound pilot record's go rule failed
- **THEN** it SHALL refuse

### Requirement: C.2's seed blocks do not overlap

Each run of the procedure SHALL take its animals from its own block of 40,000,000 seeds: stage 1's and 2a's blocks, a sub-block each for 2b and the final stage, the readout animals, the noise populations and the grade. Each start's stage 1 SHALL sit in its own slice of stage 1's block, and every generation of each start's 2a in its own slice of 2a's block. The smoke SHALL have its own layout inside 99,880,000–99,889,999. The pilot, the recovery fit, the truth's observations, the prior draws, the box centre, the registered fit, the leave-one-out, the control, the stand-in preflight and the smoke SHALL be disjoint from one another and from every earlier block in the seed table.

#### Scenario: Each start has its own animals

- **WHEN** two starts' candidates for stage 1 and for every generation of 2a are enumerated
- **THEN** no animal seed SHALL be shared between them, or with any other use in the run's block

#### Scenario: The longest stage stays inside its sub-block

- **WHEN** 2a's last generation's last member's animal seeds are enumerated, for every start
- **THEN** they SHALL lie below the next start's slice, and the last start's below the next run's block

### Requirement: C.1 is refitted on benchmark v4 from a warm start

C.1 (v4) SHALL be C.1 (v3)'s 25 parameters on their registered boxes plus `r_dwell_on_food` on `r_dwell`'s prior (log, 0.001–0.5 per s). It SHALL be fitted by sep-CMA from C.1 (v3)'s registered answer, its 25 unit coordinates read from Logbook 121's fit record and coordinate 25 at 0.5, at σ₀ 0.1, population 16 and bounds [0, 1]. The states SHALL NOT be re-ordered. A restart SHALL stop at the first generation of at least 40 at which the plateau rule holds, σ is at most 0.1, no median in the last 40 generations is 1,000 or more, and every median in the last 40 generations is below 100; otherwise it SHALL stop at the cap.

#### Scenario: The start reproduces C.1 (v3)'s answer

- **WHEN** the registered start is built
- **THEN** its first 25 coordinates SHALL equal C.1 (v3)'s chosen answer and its last SHALL be 0.5

#### Scenario: A restart above the tightened level does not stop

- **WHEN** a restart's medians are flat and its σ is at most 0.1, but one of its last 40 medians is 100 or more
- **THEN** it SHALL NOT stop

#### Scenario: The sigma guard reads the registered sigma0

- **WHEN** a restart's medians are flat and below 100, but its σ is 0.15
- **THEN** it SHALL NOT stop

### Requirement: C.1's refit runs on the speed-following arm

Every stage of C.1's v4 pilot, recovery, registered fit and leave-one-out SHALL simulate on the grading arm `mlp_following`. Its assay configs SHALL be generated from the `internal` arm's, with `continuous.on_food_reversal_speed: follows_run_speed` as their only difference. Every other generated assay config SHALL be unchanged.

#### Scenario: The arm differs only in the reversal speed

- **WHEN** each `internal_following` assay config is loaded beside its `internal` counterpart
- **THEN** the two SHALL be equal except for `environment.continuous.on_food_reversal_speed`

#### Scenario: The other arms' configs are unchanged

- **WHEN** every arm's assay configs are re-derived after the change
- **THEN** every file other than the six new ones SHALL be byte-identical to the committed file

### Requirement: C.1's v4 objective weighs its encodable mutants at half

Each evaluation SHALL simulate the wild type on v4's assays and then tph-1 and tax-4 on the small lawn on the same animals, under C.1 (v3)'s registered encodings. Every mutant SHALL keep `r_dwell_on_food`. The score SHALL be the definedness-first ranking score against v4's band centres at n = 200, with each mutant's RMS of compressed capped z over the encoding table's graded statistics combined by C.2's formula, so that the fitted mutants together carry half the weight. An undefined mutant difference SHALL count at the z cap.

#### Scenario: The formula is C.2's on two mutants

- **WHEN** a record is scored by C.1's v4 objective and by C.2's objective restricted to tph-1 and tax-4
- **THEN** the two scores SHALL be equal

#### Scenario: tax-4 keeps the on-food dwelling rate

- **WHEN** C.1 (v4)'s tax-4 parameters are built
- **THEN** both weathervane gains and both klinokinesis gains SHALL be 0, and `r_dwell_on_food` SHALL be unchanged

### Requirement: A pilot with a go rule precedes C.1's v4 recovery and fit

One restart from the registered start, configured as a registered restart, SHALL run on discarded seeds with a provisional cap of 400, and its answer SHALL be graded with presence on 5 discarded populations. On discarded seeds, the pilot SHALL also run the recovery preflight's separation, oracle and stability checks on its own answer, with the preflight's own function. Nothing further SHALL register unless the pilot stops on the plateau, its wild type, tph-1 and tax-4 each pass in at least 3 of the 5, the checks pass, and the projected registered compute, 15 restarts at its stop generation and mean seconds per generation, is at most 48 h. The registered cap SHALL be max(300, ⌈1.5 × the pilot's stop generation⌉), rounded up to a multiple of 10, and at most 500.

#### Scenario: The go rule fails

- **WHEN** the pilot's tph-1 passes in 2 of its 5 populations
- **THEN** no recovery or fit SHALL register, and the refit SHALL return to the maintainer

#### Scenario: The pilot does not stop

- **WHEN** the pilot reaches generation 400 without stopping
- **THEN** no recovery or fit SHALL register

#### Scenario: The projected cost is too high

- **WHEN** the pilot stops at generation 300 at 40 s a generation
- **THEN** the projection SHALL be 50 h, and no recovery or fit SHALL register

#### Scenario: The cap from a short pilot

- **WHEN** the pilot stops at generation 104
- **THEN** the registered cap SHALL be 300

### Requirement: C.1 (v4)'s verdict

Twelve restarts SHALL run from the registered start (CMA seeds 9451–9462). Every plateaued restart SHALL be graded with presence on 5 selection populations. The restart with the most selection populations in which the wild type, tph-1 and tax-4 all pass SHALL be chosen, ties going to the lower median score over its 10 re-simulations and then the lower restart. A capped restart SHALL NOT be chosen. Every restart SHALL be graded on 15 registered populations. C.1 (v4) SHALL pass when, on the chosen restart's first 5 registered populations, the wild type passes in at least 3 and tph-1 and tax-4 each pass in at least 3. If no restart plateaus, no verdict SHALL be read. Each mutant's C.4 row SHALL follow the encoding table's `unidentified` rule across every restart whose first 5 registered populations pass the wild type in at least 3, reported beside the verdict.

#### Scenario: A mutant fails the verdict

- **WHEN** the chosen restart passes the wild type in 4 of 5 and tax-4 in 2 of 5
- **THEN** C.1 (v4) SHALL fail

#### Scenario: A capped restart is not chosen

- **WHEN** a restart reaches the cap without stopping
- **THEN** it SHALL NOT be chosen, though its registered grades SHALL be reported

#### Scenario: Selection counts the full criterion

- **WHEN** restart A passes the wild type in 5 selection populations with tph-1 failing in 3 of them, and restart B passes all three in 3
- **THEN** restart B SHALL be chosen

#### Scenario: Restarts disagree on a mutant

- **WHEN** three restarts pass the wild type at 3 of 5, two of them passing tph-1 at 3 of 5 and one failing it
- **THEN** tph-1's row SHALL be `unidentified`, whatever the chosen restart's verdict

### Requirement: C.1's v4 reads beside its verdict

On the chosen restart's 15 registered populations, the held-out reading SHALL apply C.2's rule to N2's 26 and PD1074's 24 held-out recording dates, authorised by D37 as amended and launch record 136, as description. Only `read` SHALL use a held-out row: a stage that loads Scheer's tracks file, which holds every cohort, SHALL select band-cohort rows before passing any row to a computation. After the reading, one tax-4 leave-one-out restart SHALL run from the registered start with tax-4 out of the objective, and its answer SHALL be graded on 15 populations of its own. Its tax-4 SHALL be reported as "prediction passed" when it passes in at least 3 of its first 5 populations and "prediction failed" otherwise, both always shown, with a cap or a wild-type failure stated. Neither reading SHALL change the verdict.

#### Scenario: The leave-one-out gates nothing

- **WHEN** C.1 (v4) passes and the leave-one-out's tax-4 fails in 3 of its first 5 populations
- **THEN** C.1 (v4) SHALL still pass, and C.4's C.1 tax-4 row SHALL show "prediction failed" beside the fitted reading

### Requirement: C.1's v4 registration is in three parts

Launch record 136 SHALL have Part A before the pilot, Part B after the pilot and before the recovery's preflight, and Part C after the recovery and before the fit. Parts B and C SHALL change no rule. `pilot` SHALL refuse unless Part A is committed with its bound records unchanged; `recovery-preflight` and `recovery` SHALL also need Part B with the pilot's go met; `run`, `select`, `grade`, `read` and `loo` SHALL also need Part C with the recovery's gate passing. Every stage SHALL refuse if the package sources, the dependency lock and project files, or the scripts on the fit's computational path differ from Part A's commit or hold an untracked file. Part A's commit SHALL be the last commit that changed Part A's own text, so that Part B's and Part C's commits leave it in place and a dated amendment of Part A re-pins the code at its own commit. A decision to run despite a failed recovery gate SHALL be an amendment of Part A carrying the heading `### Decision: run despite the recovery's gate` on a line of its own; a mention of it inside a sentence SHALL NOT count. No stage SHALL overwrite its own record.

#### Scenario: The fit refuses before the recovery passes

- **WHEN** `run` is invoked with Part C committed, the recovery's gain at 0.70 and no decision heading in Part A
- **THEN** it SHALL refuse

#### Scenario: The code changes after Part A

- **WHEN** a file under `packages/quantum-nematode/quantumnematode` differs from Part A's commit
- **THEN** every registered stage SHALL refuse

#### Scenario: Part B does not move the pin

- **WHEN** Part B is committed with the pilot's record, and Part A's text is unchanged
- **THEN** Part A's commit SHALL stay the one that last changed Part A

#### Scenario: The maintainer decides to run despite the gate

- **WHEN** the recovery's gate failed and an amended Part A carries the decision heading on a line of its own
- **THEN** `run` SHALL open once Part C is committed, and the heading outside Part A or inside a sentence SHALL NOT open it

### Requirement: C.1's v4 seed blocks do not overlap

The animal seeds of the registered fit and its answers, the registered, selection and leave-one-out populations, the recovery's fits, truth, prior draws and start re-simulations, the leave-one-out's fit, the pilot, the separation probe and the smoke SHALL be pairwise disjoint. Each fit's answers SHALL sit above every generation's index under a cap of 500.

#### Scenario: Every block is enumerated

- **WHEN** every block's animal seeds are enumerated at the largest cap
- **THEN** no animal seed SHALL appear in two of them

### Requirement: C.1 (v3)'s predicted mutants are graded once under the encoding table

C.4 SHALL grade C.1 (v3)'s tph-1 and tax-4 under the registered encoding table, with tax-4 also removing the off-food weathervane gain.

- **The restarts:** C.1 (v3)'s restarts 0, 4 and 6, each on Logbook 121's registered populations (525,000,000 + 200·g + i, g < 5), on `benchmark_grade.ARMS["mlp"]`.
- **The simulation:** first the wild type alone on the registered populations. Then, only if it reproduces, the wild type on every v4 assay and each mutant on the small lawn, on the same animals, whose registered wild type SHALL equal the first pass's.
- **The grading:** each mutant's differences from its own wild type, graded on v4's mutant bands at n = 200.
- **A restart's verdict:** a population passes a mutant when all of the mutant's graded statistics pass in that population, and a restart passes it when at least 3 of its 5 populations do.
- **The prediction:** the encoding table's `unidentified` rule across restarts 0, 4 and 6, reported with C.1 (v3)'s v4 wild-type failure beside it.
- **pdfr-1** SHALL be reported as not encodable.
- **The reproduction check:** no prediction SHALL be read unless, for every restart and population, the v3 grade's D equals the committed `population_D` within 10⁻⁹ and its wild-type pass equals the committed `population_passes`, and restart 6's v4 D equals Logbook 124's.

#### Scenario: The wild type does not reproduce

- **WHEN** one restart's v3 D on one population differs from the committed value by more than 10⁻⁹
- **THEN** no mutant SHALL be simulated, the grade record SHALL hold the mismatch and no prediction, and `table` SHALL refuse

#### Scenario: Restarts disagree on a mutant

- **WHEN** restarts 0 and 4 pass tph-1 in 3 of 5 and restart 6 in 2 of 5
- **THEN** C.1's tph-1 prediction SHALL be `unidentified`

#### Scenario: A mutant passes in three populations

- **WHEN** a restart's tax-4 passes both graded statistics together in 3 of its 5 populations
- **THEN** that restart SHALL pass tax-4, and a restart whose two statistics pass in different populations, each in 3, SHALL pass only where both pass together

### Requirement: C.4's fidelity table is assembled from committed records

`table` SHALL write the fidelity table as JSON and Markdown from the committed records named in launch record 137 and from the grade record, recomputing no value a record holds. Its rows SHALL be:

- C.1 (v3);
- C.2's registered answer;
- C.3, as not run;
- B v4's eight negatives;
- the real-worm controls with W's minimum;
- C.1's v4 pilot answer, as description.

Each controller's row SHALL state:

- its body, fitted families and recovery control;
- its v4 wild-type passes by population and family, with D and D_fit;
- each mutant's kind (fitted, predicted or not encodable), its passes and its verdict;
- its held-out reading, or "not read".

C.2 against C.1 SHALL be reported as description with its conditions, and no verdict on which is closer.

#### Scenario: C.1's held-out cell

- **WHEN** the table is assembled
- **THEN** C.1's held-out reading SHALL be "not read", and no stage SHALL load Scheer's tracks

#### Scenario: A bound record changed

- **WHEN** a record the launch record binds differs from its committed SHA-256
- **THEN** `grade` and `table` SHALL refuse

### Requirement: C.4's stages are guarded and their seeds disjoint

`grade` SHALL refuse unless launch record 137's Part A is committed with its bound records unchanged and the script and the package sources match Part A's commit. `grade` and `table` SHALL each write their record once. The description populations (555,000,000 + 200·g + i, g < 10) and the smoke (99,870,000 + 200·g + i) SHALL be disjoint from every registered block of C.1 (v3), C.1's v4 refit and C.2.

#### Scenario: The grade runs before Part A

- **WHEN** `grade` is invoked and launch record 137 is not committed
- **THEN** it SHALL refuse

#### Scenario: The grade record exists

- **WHEN** `grade` is invoked and its record exists
- **THEN** it SHALL refuse

#### Scenario: Every block is enumerated

- **WHEN** the registered, description and smoke blocks are enumerated beside the other fits' blocks
- **THEN** no animal seed SHALL appear in two of them
