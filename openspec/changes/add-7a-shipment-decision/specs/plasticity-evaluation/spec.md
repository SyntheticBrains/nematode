## ADDED Requirements

### Requirement: A shipment decision states which pre-registered branch it takes and why the others were unavailable

Where a phase or shipment has pre-registered decision branches, the record SHALL name the branch taken
and SHALL state, for each branch not taken, whether it was unavailable on the evidence or merely not
chosen. A branch that is unreachable because a gate never opened SHALL be recorded as such, naming the
gate and the results that failed it, rather than as work that was skipped.

#### Scenario: An unreachable branch is distinguished from an unchosen one

- **GIVEN** a decision with pre-registered branches
- **WHEN** the verdict is recorded
- **THEN** each branch not taken SHALL be marked unavailable-on-the-evidence or not-chosen
- **AND** an unreachable branch SHALL name the gate that did not open and the results that failed it

#### Scenario: What a shipment may not be cited as is recorded with it

- **GIVEN** a shipment carrying a positive result with stated limits
- **WHEN** the record is written
- **THEN** it SHALL state what the result may not be cited as, alongside what it establishes

### Requirement: A gate whose letter and rationale diverge is recorded rather than resolved by reading

Where a pre-registered gate's literal condition and its stated rationale come apart — the reason being
satisfied while the wording is not, or the reverse — the record SHALL state both with the evidence for
each, and SHALL NOT treat the gate as met or unmet by interpretation. Resolving the divergence SHALL be
a recorded decision, and where it is not taken the gate SHALL stand as written.

#### Scenario: A satisfied rationale does not silently open a gate

- **GIVEN** a gate whose literal condition is unmet and whose stated rationale is satisfied
- **WHEN** the dependent work is considered
- **THEN** both SHALL be recorded with their evidence
- **AND** the gate SHALL NOT be treated as met until its amendment is recorded as a decision

#### Scenario: An unresolved divergence leaves the gate standing

- **GIVEN** a divergence that is recorded but not ratified
- **WHEN** the record is written
- **THEN** the gate SHALL stand as originally written, and the record SHALL say so
