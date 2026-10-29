## ADDED Requirements

### Requirement: A fit's objective may grade without presence

Grading on B v2 MAY skip the chemotaxis and local-search presence tests when only the statistics and the distance are read, as a fit's objective does. The statistics, D and D_fit SHALL then be identical to a grading with presence. Any grading read against a registered gate SHALL include presence.

#### Scenario: Presence off changes no number

- **WHEN** one simulation is graded with presence and without
- **THEN** every statistic, D and D_fit SHALL be equal, and only the presence records and family passes SHALL differ
