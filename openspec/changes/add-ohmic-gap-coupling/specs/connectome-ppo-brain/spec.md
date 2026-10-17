## ADDED Requirements

### Requirement: Ohmic gap-junction coupling in the settling map

The brain SHALL offer `gap_coupling`: `additive` (the default) or `ohmic`. Under `ohmic`, each settling
iteration SHALL compute `h ← tanh(M⁻¹ d)`, where `d` is that iteration's drive (the chemical input and,
where present, the modulator field's held current), `M = I + L` and `L = diag(G·1) − G` is the Laplacian
of the brain's gap matrix `G` as built (with `gap_weight_mode` and `gap_weight_scale` applied); the
additive gap term SHALL NOT be added. `M⁻¹` SHALL be computed once from the fixed gap matrix. `ohmic`
SHALL be refused when gap junctions are off, under `dynamics: leaky`, with `plastic_gaps`, and under any
learning rule other than PPO, by the configuration's validator and again when the brain is built. At the
default every output, parameter and random draw SHALL be identical to the brain before the option
existed.

#### Scenario: The operator is the gap Laplacian's implicit step

- **WHEN** a brain is built with `gap_coupling: ohmic`
- **THEN** its operator SHALL equal `(I + diag(G·1) − G)⁻¹` for its gap matrix `G`, to float32 precision

#### Scenario: A strongly coupled pair cannot latch

- **WHEN** two neurons share a gap junction of conductance 100 and neither receives drive
- **THEN** both SHALL settle to zero; and when one receives drive, both SHALL settle within 1% of each
  other, inside the drive's own bound

#### Scenario: Single and batched settling agree

- **WHEN** the same inputs are settled one at a time and as a batch under `ohmic`
- **THEN** the settled states SHALL agree to float32 tolerance

#### Scenario: The default is byte-identical

- **WHEN** `gap_coupling` is at its default
- **THEN** the gap matrix, the learnable parameters, every output and the next value of every random
  generator SHALL equal the brain's before the option existed
