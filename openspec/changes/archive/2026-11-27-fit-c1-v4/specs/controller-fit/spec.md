## ADDED Requirements

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

On the chosen restart's 15 registered populations, the held-out reading SHALL apply C.2's rule to N2's 26 and PD1074's 24 held-out recording dates, authorised by D37 as amended and launch record 136, as description. Only `read` SHALL load a held-out row. After the reading, one tax-4 leave-one-out restart SHALL run from the registered start with tax-4 out of the objective, and its answer SHALL be graded on 15 populations of its own. Its tax-4 SHALL be reported as "prediction passed" when it passes in at least 3 of its first 5 populations and "prediction failed" otherwise, both always shown, with a cap or a wild-type failure stated. Neither reading SHALL change the verdict.

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
