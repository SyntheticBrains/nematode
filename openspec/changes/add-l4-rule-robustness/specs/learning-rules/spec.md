## ADDED Requirements

### Requirement: Homeostatic incoming-norm scaling

The three-factor rule SHALL offer a default-off homeostatic mode, shared by every plastic brain
through the plasticity configuration mixin. When on, the rule SHALL capture at construction, for
every unit of every plastic tensor, the norm of that unit's incoming plastic weights over the
tensor's masked entries, and after each plastic update SHALL rescale each unit's incoming
weights so that their norm returns to that target, dividing by the current norm floored at the
configured scale floor; units with a zero target SHALL be left untouched; the magnitude clamp
SHALL be applied after the rescale. Nothing SHALL be rescaled under a freeze. The
plastic-topology seam SHALL expose `plastic_fan_in_axes`, aligned with its other lists, giving
for each plastic tensor the axis to reduce over to obtain a unit's incoming norm, and the rule
SHALL NOT name any substrate's layout. The rule SHALL report the mean relative norm deviation
before the rescale (`plasticity_norm_drift`), NaN when the mode is off. With the mode off, the
rule SHALL be bit-identical to the rule without this requirement.

#### Scenario: Incoming norms return to their targets on both substrates

- **GIVEN** homeostasis on and a rule stepped over the connectome and over the MLP with
  non-trivial traces and modulators
- **WHEN** an update has been applied
- **THEN** every unit with a non-zero target SHALL have incoming norm equal to its target within
  floating-point tolerance, on the connectome over the chemical mask and on the MLP over each
  layer's rows
- **AND** the seam SHALL report fan-in axis `0` for the chemical matrix and `1` for each MLP layer

#### Scenario: Units without inputs and the bound are respected

- **GIVEN** homeostasis on
- **WHEN** a unit has no incoming edges
- **THEN** its (zero) incoming weights SHALL be unchanged
- **AND** no weight SHALL exceed the magnitude bound after the rescale

#### Scenario: Homeostasis off is bit-identical and a freeze rescales nothing

- **GIVEN** the default configuration
- **WHEN** the rule steps in any mode
- **THEN** the weights and the existing telemetry SHALL be bit-identical to the frozen reference
- **AND** with homeostasis on under a freeze, no weight SHALL change while the norm drift is still
  reported

#### Scenario: The connectome's targets follow its initialisation scale

- **GIVEN** the connectome with homeostasis on
- **WHEN** the rule captures its targets
- **THEN** every neuron with chemical inputs SHALL have a target near 1, the incoming norm its
  `1/√k` initialisation gives by construction
