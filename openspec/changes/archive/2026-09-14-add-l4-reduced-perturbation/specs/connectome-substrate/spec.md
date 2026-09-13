## ADDED Requirements

### Requirement: The perturbed unit set is declarable and travels with the run

Where a substrate applies per-unit perturbation to form a learning signal, the set of perturbed units
SHALL be declarable rather than fixed at every unit, and each run SHALL record the declared set, the
number of units in it, the number of synapses it makes adaptable, and the number of perturbation draws
per scored decision. The set SHALL be derived from the loaded substrate and checked against the
declaration's expected size and membership, and a mismatch SHALL fail the run rather than proceed.

#### Scenario: A declared set is recorded with its consequences

- **GIVEN** a substrate configured to perturb a declared subset of its units
- **WHEN** the run is recorded
- **THEN** the record SHALL state the set, its unit count, the synapses it makes adaptable, and the
  draws per scored decision
- **AND** a comparison across declared sets SHALL state those numbers for each

#### Scenario: A mask that does not mask is a failure, not a duplicate

- **GIVEN** a run declaring a restricted perturbation set
- **WHEN** the set derived from the loaded substrate does not match the declaration's expected size and
  membership
- **THEN** the run SHALL fail
- **AND** it SHALL NOT be reported as a result for the declared set, which would duplicate the
  unrestricted arm under another name

#### Scenario: A restriction that depends on another mechanism to be inert says so

- **GIVEN** a restricted perturbation set whose excluded weights are nonetheless written by an
  unconditional term of the update
- **WHEN** the restriction is declared
- **THEN** the mechanism that cancels that term SHALL be required by the configuration and stated in the
  record
- **AND** a test SHALL demonstrate both that the excluded weights do not move while it is active and that
  they do move without it, so the dependency is measured rather than assumed

### Requirement: A perturbation set restricted by causal reach states what it assumes

Where a perturbation set is derived from the graph distance between units and the readout, the record
SHALL state the distance measure, the connection types it traverses, and the connection types it
ignores. Where a connection type carrying influence is excluded from the measure, the record SHALL say
in which direction that makes the mask wrong.

#### Scenario: The excluded connection types are named

- **GIVEN** a mask derived from reachability over one connection type
- **WHEN** the substrate carries others
- **THEN** the record SHALL name the excluded types and the direction of the resulting error
- **AND** a variant including them SHALL be named as follow-up rather than assumed equivalent
