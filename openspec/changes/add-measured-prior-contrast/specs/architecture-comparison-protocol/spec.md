## ADDED Requirements

### Requirement: A measured-weight positive is read against its placement-shuffled control

Where a structure contrast is compared between a measured weight prior and a random one, the record
SHALL also compare the measured prior with the same measured values permuted among the same edges,
and SHALL report the measured prior as making the structure legible only where **both** comparisons
move the contrast in the same direction by at least the registered minimum. A measured prior changes
two things at once — the distribution of values on the edges it covers, and which synapse holds which
value — and only the permuted control separates them. Where the measured-versus-random comparison
moves and the measured-versus-permuted one does not, the result SHALL be reported as an effect of the
value distribution.

#### Scenario: Both comparisons are registered before the runs

- **GIVEN** a campaign comparing a measured weight prior with a random one on a structure contrast
- **WHEN** it is registered
- **THEN** its launch record SHALL include the permuted-placement arm on every wiring the contrast
  spans, and SHALL name both interactions and the verdict each combination of them receives

#### Scenario: A distribution effect is not reported as legibility

- **GIVEN** a measured-versus-random interaction at or beyond the registered minimum
- **WHEN** the measured-versus-permuted interaction is below that minimum, unresolved, or absent
- **THEN** the record SHALL report a value-distribution effect
- **AND** SHALL NOT state that the measured weights make the structure legible

#### Scenario: The permuted control moving alone is carried, not discarded

- **GIVEN** a measured-versus-random interaction that does not move
- **WHEN** the measured-versus-permuted interaction does
- **THEN** the record SHALL report that the permutation moved the contrast and the fitted placement
  did not, as a finding about the control, and SHALL NOT read it as legibility in either direction
