## ADDED Requirements

### Requirement: Strength-normalised gap junctions

The connectome PPO brain SHALL offer `gap_normalisation`, either `degree` (the default) or `strength`.
Under `degree`, each gap pair's weight SHALL be divided by `√(d_i d_j)`, where `d` is the binary gap
degree, exactly as before the option existed. Under `strength`, each gap pair's weight `w_ij`, taken
after `gap_weight_mode`, SHALL be divided by `√(s_i s_j)`, where `s_i = Σ_j w_ij` is the neuron's gap
strength computed on the brain's own wiring. `gap_weight_scale` SHALL apply after either normalisation.

`strength` SHALL be refused with gap junctions disabled and with `plastic_gaps`, both by the validator
and at build.

#### Scenario: Strength-normalised couplings are bounded

- **WHEN** a brain is built with `gap_normalisation: strength` and count weights
- **THEN** every gap coupling SHALL be at most 1
- **AND** the gap matrix SHALL be symmetric with spectral radius 1

#### Scenario: Unit weights make the two normalisations equal

- **WHEN** a brain is built with `gap_weight_mode: unit` under each normalisation
- **THEN** the two gap matrices SHALL be identical

#### Scenario: The default is unchanged

- **WHEN** a brain is built without `gap_normalisation`
- **THEN** its outputs, weights, state-dict keys and next random draw SHALL equal the brain's before the
  option existed

#### Scenario: A combination with nothing to normalise is refused

- **WHEN** `gap_normalisation: strength` is set with gap junctions disabled, or with `plastic_gaps`
- **THEN** the configuration SHALL be refused with a message naming the conflicting option
