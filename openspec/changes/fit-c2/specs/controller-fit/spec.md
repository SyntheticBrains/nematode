## ADDED Requirements

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

A pass with a poor held-out reading, or with the leave-one-out's tax-4 failing, SHALL be read by the outcomes the launch record names in advance.

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

After the registered fit, the same procedure SHALL run on its own seeds with tax-4 out of the objective. tph-1 and pdfr-1 SHALL then be together half of it, and the failure probability SHALL omit tax-4's statistics. Its answer's tax-4 SHALL be graded on 15 populations and reported beside the registered fit's. It SHALL gate nothing.

#### Scenario: tax-4 is never scored

- **WHEN** a leave-one-out candidate is scored
- **THEN** its score SHALL be unchanged by its tax-4 reading

### Requirement: C.2's registration is in three parts

Launch record 133 SHALL be committed in three dated parts:

- **Part A, before the pilot:** every rule, seed, gate and verdict, disclosing Logbook 131's probes;
- **Part B, after the pilot:** its result;
- **Part C, after the recovery:** its result.

Parts B and C SHALL change no rule. Part A SHALL record SHA-256 digests of `c2_fit.py` and the scripts it imports, the arm's assay configs, `bands_v4.json`, `primitives.json`, the neuromodulator data files and the encoding table. Every stage SHALL check them at start and at each resume, refuse on a mismatch, and require v4's bands to be valid. The guards are:

- `pilot` SHALL refuse unless Part A is committed and unchanged;
- `recovery-preflight` SHALL refuse unless Part B is committed with the pilot's record bound and its go rule passing;
- `recovery` SHALL refuse unless the preflight's record is committed, unchanged and passing;
- `run`, `grade`, `read` and `loo` SHALL refuse unless Part C is committed with both recovery records bound and the gate passing.

#### Scenario: The fit refuses before the recovery is recorded

- **WHEN** `run` is invoked while Part C is absent or the recovery record differs from its committed version
- **THEN** it SHALL refuse and run nothing

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
