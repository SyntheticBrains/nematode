## ADDED Requirements

### Requirement: Scalar Food Concentration

The environment SHALL compute scalar food concentration at a given position as the sum of exponential decay magnitudes from all active food sources, without directional information.

#### Scenario: Single Food Concentration

- **WHEN** `get_food_concentration(position)` is called with one food source at (10, 10)
- **THEN** the raw concentration at position (5, 5) SHALL be `gradient_strength * exp(-distance / gradient_decay_constant)`
- **AND** the distance SHALL be Euclidean distance between the query position and the food source
- **AND** the raw value SHALL be normalized via `tanh(raw * GRADIENT_SCALING_TANH_FACTOR)` to [0, 1], consistent with oracle gradient magnitude normalization
- **AND** the result SHALL be a single non-negative float (no direction component)

#### Scenario: Multiple Food Concentration Superposition

- **WHEN** `get_food_concentration(position)` is called with multiple food sources
- **THEN** the result SHALL be the sum of individual exponential decay magnitudes from all food sources
- **AND** the decay formula SHALL use the same `gradient_strength` and `gradient_decay_constant` parameters as the gradient computation

#### Scenario: Concentration Updates on Food Collection

- **WHEN** a food source is consumed and a new one spawns
- **THEN** subsequent calls to `get_food_concentration()` SHALL reflect the updated food positions
- **AND** the consumed food's contribution SHALL be removed
- **AND** the new food's contribution SHALL be added

#### Scenario: Consistency with Gradient Computation

- **WHEN** `get_food_concentration(position)` and `_compute_food_gradient_vector(position)` are called for the same position
- **THEN** the scalar concentration SHALL equal the sum of magnitudes of individual food gradient vectors (before vector summation)
- **AND** the same decay model (exponential with `gradient_strength` and `gradient_decay_constant`) SHALL be used

### Requirement: Scalar Predator Concentration

The environment SHALL compute scalar predator danger signal at a given position as the sum of exponential decay magnitudes from all predator positions, without directional information.

#### Scenario: Single Predator Concentration

- **WHEN** `get_predator_concentration(position)` is called with one predator at (10, 10)
- **THEN** the raw concentration at position (5, 5) SHALL be `gradient_strength * exp(-distance / gradient_decay_constant)`
- **AND** the raw value SHALL be normalized via `tanh(raw * GRADIENT_SCALING_TANH_FACTOR)` to [0, 1]
- **AND** the result SHALL use predator-specific `gradient_strength` and `gradient_decay_constant` parameters
- **AND** the result SHALL be a single non-negative float

#### Scenario: Multiple Predators Concentration

- **WHEN** `get_predator_concentration(position)` is called with multiple predators
- **THEN** the result SHALL be the sum of individual exponential decay magnitudes from all predators

#### Scenario: No Predators Configured

- **WHEN** `get_predator_concentration(position)` is called with predators disabled
- **THEN** the result SHALL be 0.0

### Requirement: BrainParams Temporal Sensing Fields

The BrainParams data structure SHALL include optional fields for scalar concentrations, temporal derivatives, and STAM memory state.

#### Scenario: Scalar Concentration Fields

- **WHEN** temporal or derivative sensing mode is active for a modality
- **THEN** BrainParams SHALL include `food_concentration` (float or None) for scalar food signal at agent position
- **AND** SHALL include `predator_concentration` (float or None) for scalar predator signal at agent position
- **AND** these fields SHALL default to None when oracle mode is used

#### Scenario: Temporal Derivative Fields

- **WHEN** derivative sensing mode is active
- **THEN** BrainParams SHALL include `food_dconcentration_dt` (float or None) for food temporal derivative
- **AND** SHALL include `predator_dconcentration_dt` (float or None) for predator temporal derivative
- **AND** SHALL include `temperature_ddt` (float or None) for temperature temporal derivative
- **AND** these fields SHALL default to None when oracle or temporal mode is used

#### Scenario: STAM State Field

- **WHEN** STAM is enabled
- **THEN** BrainParams SHALL include `stam_state` (tuple of floats or None) containing the flattened STAM memory vector
- **AND** this field SHALL default to None when STAM is disabled

#### Scenario: Backward Compatibility

- **WHEN** no sensing configuration is provided (all defaults)
- **THEN** all new BrainParams fields SHALL be None
- **AND** all existing fields (gradient_strength, gradient_direction, food_gradient\_*, predator_gradient\_*, temperature_gradient\_\*, etc.) SHALL be populated as before
- **AND** no existing brain architecture SHALL be affected
