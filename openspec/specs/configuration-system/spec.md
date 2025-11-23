# configuration-system Specification

## Purpose
TBD - created by archiving change add-spiking-neural-network-brain. Update Purpose after archive.
## Requirements
### Requirement: Spiking Brain Configuration Schema
The configuration system SHALL support a complete schema for spiking neural network parameters.

#### Scenario: YAML Configuration Parsing
**Given** a YAML configuration file with spiking brain section  
**When** the configuration is loaded  
**Then** the system SHALL parse neuron model parameters  
**And** SHALL parse plasticity rule parameters  
**And** SHALL parse network topology parameters  
**And** SHALL validate all parameter ranges and constraints  

#### Scenario: Default Parameter Application
**Given** a spiking brain configuration with missing optional parameters  
**When** the configuration is processed  
**Then** the system SHALL apply sensible defaults  
**And** SHALL ensure all required parameters are present  
**And** SHALL log applied defaults for user awareness

### Requirement: Parameter Validation
The configuration system SHALL validate spiking neural network parameters for biological and computational feasibility.

#### Scenario: Neuron Parameter Validation
**Given** LIF neuron parameters in configuration  
**When** validation is performed  
**Then** tau_m SHALL be positive (> 0)  
**And** v_threshold SHALL be greater than v_reset  
**And** simulation time_step SHALL be appropriate for tau_m  

#### Scenario: STDP Parameter Validation
**Given** STDP plasticity parameters in configuration  
**When** validation is performed  
**Then** tau_plus and tau_minus SHALL be positive  
**And** learning_rate SHALL be in reasonable range (0.0001 - 0.1)  
**And** A_plus and A_minus SHALL be positive

### Requirement: Configuration Examples
The system SHALL provide example configurations for common spiking brain use cases.

#### Scenario: Small Network Configuration
**Given** a need for basic spiking brain testing  
**When** loading spiking_small.yml configuration  
**Then** the system SHALL configure a minimal viable spiking network  
**And** SHALL use parameters suitable for fast convergence  

#### Scenario: Medium Network Configuration
**Given** a need for standard experimental setup  
**When** loading spiking_simple_medium.yml configuration  
**Then** the system SHALL configure a balanced network  
**And** SHALL use parameters suitable for robust learning

### Requirement: Brain Type Enumeration Extension
The brain type validation SHALL include "spiking" as a valid option.

#### Scenario: Brain Type Validation
**Given** configuration specifies brain type
**When** validation occurs
**Then** "spiking" SHALL be accepted as valid
**Along with** existing "modular", "qmodular", "mlp", "qmlp" types

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
- **AND** SHALL parse `foods_on_grid` (integer, default: grid_area / 50)
- **AND** SHALL parse `target_foods_to_collect` (integer, default: foods_on_grid × 1.5)
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
- **THEN** `foods_on_grid` SHALL be positive and > 0
- **AND** `target_foods_to_collect` SHALL be >= `foods_on_grid`
- **AND** if `foods_on_grid` is too large for grid with `min_food_distance`, SHALL emit warning
- **AND** if `agent_exclusion_radius` exceeds `min_food_distance`, SHALL warn that exclusion zones may prevent food placement

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
