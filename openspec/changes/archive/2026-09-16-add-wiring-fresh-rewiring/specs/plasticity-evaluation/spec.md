## ADDED Requirements

### Requirement: A control drawn from the same seeds as its runs states that coupling

Where a contrast's control condition is generated from the same seed that generates the run it is
compared against — a rewired graph, a shuffled label, a permuted input — the record SHALL state that
the control and the run are coupled, and SHALL state which of the two a later panel varies. Where two
results share their control conditions, the record SHALL NOT describe them as independent
replications of each other.

#### Scenario: The coupling is recorded with the result

- **GIVEN** a control generated from the run seed rather than from a seed of its own
- **WHEN** the result is recorded
- **THEN** the record SHALL state that the control and the run vary together
- **AND** it SHALL state what a later panel on fresh seeds would and would not separate

#### Scenario: Results sharing controls are not independent replications

- **GIVEN** two results whose control conditions are drawn from overlapping seed sets
- **WHEN** they are cited together
- **THEN** the record SHALL state the overlap
- **AND** neither SHALL be described as an independent replication of the other

### Requirement: A replication uses the original instrument unmodified

Where a result is re-run to test whether it holds, the replication SHALL be scored by the same
analysis the original was scored by, without modification. Where the original instrument cannot score
the replication, that SHALL be recorded as a limitation of the replication rather than resolved by
writing a new instrument.

#### Scenario: The instrument is not rewritten for the replication

- **GIVEN** a committed analysis that produced a result
- **WHEN** that result is replicated
- **THEN** the replication SHALL be scored by that analysis unmodified
- **AND** any new code SHALL be confined to preparing its inputs and reporting its outputs

#### Scenario: A failure to replicate is not attributed to the instrument

- **GIVEN** a replication scored by the original instrument
- **WHEN** it does not reproduce the original result
- **THEN** the record SHALL state that the reading is unchanged and the evidence differs
- **AND** the original result SHALL be withdrawn or qualified on the record rather than defended

### Requirement: A multi-panel replication fixes the disagreement case in advance

Where a replication covers more than one panel, the registration SHALL state before it runs how a
disagreement between panels is read. A split SHALL be reported as a split, and SHALL NOT be resolved
toward whichever panel supports the original claim.

#### Scenario: A split is reported as a split

- **GIVEN** a replication of two or more panels
- **WHEN** some panels replicate and others do not
- **THEN** each panel SHALL be reported against the registered branches on its own
- **AND** the pooled reading SHALL be withheld, the split recorded as evidence about the original
  claim's scope
