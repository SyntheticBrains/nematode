## ADDED Requirements

### Requirement: Scalar Oxygen Concentration

The environment SHALL compute scalar oxygen concentration at the agent's current position from the OxygenField, without directional information.

#### Scenario: Oxygen Concentration Query

- **WHEN** `get_oxygen_concentration(position)` is called in an aerotaxis-enabled environment
- **THEN** the result SHALL be the raw O2 percentage at that position from OxygenField.get_oxygen()
- **AND** the result SHALL be a single float in [0.0, 21.0]
- **AND** the value SHALL NOT be tanh-normalized (unlike food/predator concentrations, O2 percentage is already a meaningful absolute value)

#### Scenario: Aerotaxis Disabled

- **WHEN** `get_oxygen_concentration(position)` is called with aerotaxis disabled
- **THEN** the result SHALL be None

### Requirement: Oxygen Gradient in Separated Gradients

The environment SHALL include oxygen gradient fields in the separated gradient system when aerotaxis is enabled.

#### Scenario: Separated Gradients with Oxygen

- **WHEN** `get_separated_gradients()` is called with aerotaxis enabled in oracle mode
- **THEN** the result SHALL include `oxygen_gradient_strength` (float) and `oxygen_gradient_direction` (float)
- **AND** these SHALL be computed from `OxygenField.get_gradient_polar()` at the agent's current position

#### Scenario: Non-Oracle Aerotaxis Mode

- **WHEN** `get_separated_gradients()` is called with aerotaxis in temporal or derivative mode
- **THEN** oxygen gradient fields SHALL NOT be included in the separated gradients
- **AND** only the scalar concentration SHALL be provided via temporal sensing

## MODIFIED Requirements

### Requirement: BrainParams Temporal Sensing Fields

The BrainParams data structure SHALL include optional fields for scalar concentrations, temporal derivatives, and STAM memory state.

#### Scenario: Scalar Concentration Fields

- **WHEN** temporal or derivative sensing mode is active for a modality
- **THEN** BrainParams SHALL include `food_concentration` (float or None) for scalar food signal at agent position
- **AND** SHALL include `predator_concentration` (float or None) for scalar predator signal at agent position
- **AND** SHALL include `oxygen_concentration` (float or None) for scalar oxygen signal at agent position
- **AND** these fields SHALL default to None when oracle mode is used or the modality is disabled

#### Scenario: Temporal Derivative Fields

- **WHEN** derivative sensing mode is active
- **THEN** BrainParams SHALL include `food_dconcentration_dt` (float or None) for food temporal derivative
- **AND** SHALL include `predator_dconcentration_dt` (float or None) for predator temporal derivative
- **AND** SHALL include `temperature_ddt` (float or None) for temperature temporal derivative
- **AND** SHALL include `oxygen_dconcentration_dt` (float or None) for oxygen temporal derivative
- **AND** these fields SHALL default to None when oracle or temporal mode is used

#### Scenario: STAM State Field

- **WHEN** STAM is enabled
- **THEN** BrainParams SHALL include `stam_state` (tuple of floats or None) containing the flattened STAM memory vector
- **AND** this field SHALL default to None when STAM is disabled

#### Scenario: Backward Compatibility

- **WHEN** aerotaxis is not enabled in the environment
- **THEN** `oxygen_concentration` SHALL be None
- **AND** `oxygen_dconcentration_dt` SHALL be None
- **AND** existing brain functionality SHALL be unchanged
