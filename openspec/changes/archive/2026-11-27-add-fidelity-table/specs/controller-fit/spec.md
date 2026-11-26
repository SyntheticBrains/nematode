## ADDED Requirements

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
