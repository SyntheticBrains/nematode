## ADDED Requirements

### Requirement: B v2 is valid only if a faithful point worm can pass and the known exploits fail

B v2 SHALL be VALID only when all of the following hold:

- the reachability oracle passes;
- both fresh N2 positive-control units pass;
- B.4's six negatives each fail the wild type;
- the blind worm fitted against v2's D fails the wild type at 3 of 5;
- C.1's v1 chosen answer fails the wild type at 3 of 5.

Otherwise it SHALL be VOID.

#### Scenario: The known exploit still passes

- **WHEN** C.1's v1 answer passes v2's wild type
- **THEN** B v2 SHALL be VOID

#### Scenario: A real-rate point worm cannot reach a statistic

- **WHEN** the matched open-loop worm, through the measurement model, fails a speed or reversal statistic in more than 10% of 40 populations, or its mean z lies outside [−1, 1]
- **THEN** the run SHALL stop before Part B

### Requirement: The new statistics are validated on unread data

The new statistics SHALL be validated on N2's held-out dates that no reading has touched, against N2 bands from N2's band dates, read once under a guard. PD1074's held-out cohorts SHALL NOT be read again.

#### Scenario: A held-out PD1074 cohort is requested

- **WHEN** any v2 stage would read a PD1074 held-out date
- **THEN** it SHALL refuse

### Requirement: W's minimum v2 is on the families W reads

W's minimum v2 SHALL be the 95th percentile of D, over the small-lawn and kinematics families, between halves of N2's unread held-out dates, and W SHALL read its arms on the same families.

#### Scenario: The families agree

- **WHEN** W registers
- **THEN** its distance SHALL be over the same families as its minimum
