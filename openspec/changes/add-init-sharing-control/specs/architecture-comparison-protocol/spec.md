## ADDED Requirements

### Requirement: A claim that two arms share an initialisation is verified by test

Where a contrast's credibility rests on two arms starting from the same initialisation — the same drawn values, the same per-unit scale, the same policy — the record SHALL identify the quantity claimed to be shared and SHALL establish it by a test that compares the constructed arms, not by an argument about how the random stream is consumed. Where a quantity is claimed to be shared and is not asserted by such a test, the contrast SHALL be reported as matched by construction only, and the claim SHALL name what was not checked.

#### Scenario: The shared quantity is asserted against constructed arms

- **GIVEN** an initialisation-sharing mode whose purpose is to make two arms comparable
- **WHEN** the mode is added
- **THEN** a test SHALL construct both arms at one seed and assert the shared quantity directly — the identical value on every element present in both, or the identical multiset per unit, as the mode claims
- **AND** the test SHALL assert that every parameter the mode does not claim to change is bitwise identical across the arms

#### Scenario: A construction argument is not evidence of sharing

- **GIVEN** a claim that two arms share an initialisation because they consume the same random stream
- **WHEN** the claim is recorded
- **THEN** it SHALL be treated as a hypothesis until a test compares the constructed arms, since a shared stream establishes only that the same values were drawn and not where they landed
- **AND** a contrast relying on the unverified claim SHALL state that its matching is by construction

#### Scenario: More than one definition of sharing exists

- **GIVEN** a structure contrast in which no single definition of "the same initialisation" is uniquely correct, because the manipulation changes which elements exist
- **WHEN** the control is designed
- **THEN** each defensible definition SHALL be run as its own arm and named in the record
- **AND** a result SHALL NOT be reported as "under shared initialisation" without naming which definition produced it
