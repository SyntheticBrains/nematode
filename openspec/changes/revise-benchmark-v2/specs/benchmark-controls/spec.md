## ADDED Requirements

### Requirement: B v2 is valid only if a faithful point worm can pass and the known exploits fail

B v2 SHALL be VALID only when all of the following hold:

- the measurement-model checks pass: the losses and the jitter on Scheer's rig, and the round trip on Chen's;
- the reachability oracle passes;
- the fresh N2 control passes, with at most 1 of its 4 units failing;
- the chemotaxis leave-one-recording-out validation passes within its registered allowance, a recording's unit failing where its chemotaxis is not present;
- B.4's six negatives each fail the wild type, and its four open-loop negatives (`hmm`, `poisson`, `random_walk`, `correlated_walk`), which ignore the odour, each fail the chemotaxis family;
- the blind worm fitted against v2's D fails the wild type at 3 of 5;
- C.1's v1 chosen answer fails the wild type at 3 of 5.

Otherwise it SHALL be VOID.

#### Scenario: The known exploit still passes

- **WHEN** C.1's v1 answer passes v2's wild type
- **THEN** B v2 SHALL be VOID

#### Scenario: One unlucky N2 unit

- **WHEN** one of the four N2 units fails and the other three pass
- **THEN** the N2 control SHALL pass

#### Scenario: A real-rate point worm cannot reach a statistic

- **WHEN** the reachability worm, through the measurement model, fails a speed statistic in more than 10% of 40 populations, or its mean z lies outside [−1, 1]
- **THEN** the run SHALL stop before Part B

### Requirement: The new statistics are validated on unread data

The new statistics SHALL be validated on N2's held-out dates that no reading has touched, against N2 bands from N2's band dates, read once under a guard. PD1074's held-out cohorts SHALL NOT be read again.

#### Scenario: A held-out PD1074 cohort is requested

- **WHEN** any v2 stage would read a PD1074 held-out date
- **THEN** it SHALL refuse

### Requirement: W's minimum v2 is recorded on both family sets

B v2 SHALL record W's minimum on the fitted families (small lawn and kinematics, between halves of N2's unread held-out dates) and on the held-out families (local search, between halves of López-Cruz's band batches; chemotaxis, between every 3-against-3 split of Chen's recordings). W SHALL read its arms on the same family set as the minimum it cites.

#### Scenario: The families agree

- **WHEN** W registers
- **THEN** its distance SHALL be over the same family set as its minimum
