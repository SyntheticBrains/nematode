## ADDED Requirements

### Requirement: Ohmic gap-junction coupling in the settling map

The brain SHALL offer `gap_coupling`: `additive` (the default) or `ohmic`. Under `ohmic`, the settling
map SHALL start from `h₀ = M⁻¹ s`, `s` the injected sensors, and each settling iteration SHALL compute
`h ← tanh(M⁻¹ d)`, where `d` is that iteration's drive (the chemical input and, where present, the
modulator field's held current), `M = I + L` and `L = diag(G·1) − G` is the Laplacian of the brain's gap
matrix `G` as built (with `gap_weight_mode` and `gap_weight_scale` applied); the additive gap term SHALL
NOT be added. `M⁻¹` SHALL be computed once from the fixed gap matrix and SHALL NOT be saved with the
weights. `ohmic` SHALL be refused when gap junctions are off, under `dynamics: leaky`, with
`plastic_gaps`, under any learning rule other than PPO, and with node perturbation, activity traces or
e-prop eligibility, by the configuration's validator and again when the brain is built. At the default
every output, parameter, state-dict key and random draw SHALL be identical to the brain before the option
existed.

#### Scenario: The operator is the gap Laplacian's implicit step

- **WHEN** a brain is built with `gap_coupling: ohmic`
- **THEN** its operator SHALL equal `(I + diag(G·1) − G)⁻¹` for its gap matrix `G`, to float32
  precision, and each of its rows SHALL sum to 1

#### Scenario: The gap coupling adds no gain

- **WHEN** two neurons share a gap junction of conductance 100, have no chemical edges, start saturated
  and receive no drive
- **THEN** under `additive` they SHALL stay saturated, and under `ohmic` they SHALL fall to zero in one
  iteration; and **WHEN** one of them receives a small drive, under `additive` both SHALL saturate and
  under `ohmic` neither SHALL exceed the drive

#### Scenario: Sensors relay across gap junctions

- **WHEN** a sensor with a gap partner is injected under `ohmic`
- **THEN** the partner's initial state SHALL be that row of `M⁻¹ s`, not zero

#### Scenario: Single and batched settling agree

- **WHEN** the same inputs are settled one at a time and as a batch under `ohmic`
- **THEN** the settled states SHALL agree to float32 tolerance

#### Scenario: The default is byte-identical

- **WHEN** `gap_coupling` is at its default
- **THEN** the gap matrix, the learnable parameters, the state-dict keys, every output and the next value
  of every random generator SHALL equal the brain's before the option existed
