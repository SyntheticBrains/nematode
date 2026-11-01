## ADDED Requirements

### Requirement: A v2 simulation builds only each family's analysis window, identically

A v2 simulation SHALL build steps only over each family's analysis window, and measure the lawn families' windows with noise, lost seconds and fragment cuts drawn over the whole episode. Every graded statistic, D and presence reading SHALL be byte-identical to building and measuring the whole episode before windowing.

#### Scenario: A windowed animal is the same

- **WHEN** an animal's window is built and measured directly, and its whole episode is built, measured and then windowed, from the same seed
- **THEN** every field of every step SHALL be equal

#### Scenario: Simulations re-simulate exactly

- **WHEN** the identity check's baseline and stored records are re-simulated after the change
- **THEN** every statistic, D and presence reading SHALL equal them exactly
