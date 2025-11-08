## ADDED Requirements

### Requirement: Dynamic Environment Configuration Schema
The configuration system SHALL support a complete schema for dynamic foraging environment parameters.

#### Scenario: Environment Type Selection
- **GIVEN** a YAML configuration file with environment section
- **WHEN** the configuration is loaded
- **THEN** the system SHALL parse `environment_type` field
- **AND** SHALL accept values: "static" (default), "dynamic"
- **AND** SHALL select appropriate environment class based on type

#### Scenario: Dynamic Environment Parameters
- **GIVEN** a configuration with `environment_type: "dynamic"`
- **WHEN** the configuration is parsed
- **THEN** the system SHALL parse `grid_size` as tuple (width, height)
- **AND** SHALL parse `num_initial_foods` (integer, default: grid_area / 50)
- **AND** SHALL parse `max_active_foods` (integer, default: num_initial_foods × 1.5)
- **AND** SHALL parse `min_food_distance` (integer, default: max(5, min(width, height) / 10))
- **AND** SHALL parse `food_spawn_interval` (integer, default: 0 for immediate)
- **AND** SHALL parse `viewport_size` as tuple (width, height, default: (11, 11))
- **AND** SHALL parse `agent_exclusion_radius` (integer, default: 10)

#### Scenario: Satiety Configuration Schema
- **GIVEN** a configuration with dynamic environment
- **WHEN** satiety parameters are specified
- **THEN** the system SHALL parse `initial_satiety` (float, default: 200.0)
- **AND** SHALL parse `satiety_decay_rate` (float, default: 1.0)
- **AND** SHALL parse `satiety_gain_per_food` (float as fraction, default: 0.2)
- **AND** SHALL validate satiety_decay_rate > 0
- **AND** SHALL validate satiety_gain_per_food between 0.0 and 1.0

#### Scenario: Gradient Configuration
- **GIVEN** a configuration with environment gradient settings
- **WHEN** the configuration is parsed
- **THEN** the system SHALL parse `gradient_decay_constant` (float, default: 10.0)
- **AND** SHALL parse `gradient_strength` (float, default: 1.0)
- **AND** SHALL parse `gradient_scaling` (enum: "exponential" or "tanh", default: "exponential")

#### Scenario: Exploration Bonus Configuration
- **GIVEN** a configuration with reward settings
- **WHEN** exploration parameters are specified
- **THEN** the system SHALL parse `exploration_bonus` (float, default: 0.05)
- **AND** SHALL allow 0.0 to disable exploration rewards
- **AND** SHALL validate exploration_bonus >= 0.0

#### Scenario: Configuration File Examples
- **GIVEN** example configuration files in `configs/examples/`
- **WHEN** users need preset dynamic environments
- **THEN** `dynamic_small.yml` SHALL provide small curriculum configuration
- **AND** `dynamic_medium.yml` SHALL provide medium curriculum configuration
- **AND** `dynamic_large.yml` SHALL provide large curriculum configuration
- **AND** each SHALL include commented parameter explanations

### Requirement: Configuration Validation for Dynamic Environments
The configuration system SHALL validate dynamic environment parameters for logical consistency and computational feasibility.

#### Scenario: Food Count Validation
- **GIVEN** a dynamic environment configuration
- **WHEN** validation is performed
- **THEN** `num_initial_foods` SHALL be positive and > 0
- **AND** `max_active_foods` SHALL be >= `num_initial_foods`
- **AND** if `num_initial_foods` is too large for grid with `min_food_distance`, SHALL emit warning

#### Scenario: Grid Size Validation
- **GIVEN** a dynamic environment configuration
- **WHEN** grid size is validated
- **THEN** both width and height SHALL be >= 10
- **AND** both SHALL be <= 200 (performance limit)
- **AND** if grid size > 100×100, SHALL log performance warning

#### Scenario: Satiety Balance Validation
- **GIVEN** a dynamic environment configuration
- **WHEN** satiety parameters are validated
- **THEN** the system SHALL check that `initial_satiety / satiety_decay_rate` provides reasonable episode length
- **AND** SHALL warn if satiety allows fewer than 50 steps
- **AND** SHALL warn if food consumption cannot sustain foraging (gain < expected consumption rate)

#### Scenario: Viewport Size Validation
- **GIVEN** a dynamic environment configuration with viewport
- **WHEN** validation occurs
- **THEN** viewport width and height SHALL be odd numbers (for centered agent)
- **AND** SHALL be at least 3×3
- **AND** SHALL not exceed grid size
- **AND** if even number provided, SHALL auto-adjust to next odd number and warn
