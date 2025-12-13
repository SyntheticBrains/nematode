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
**When** loading spiking_static_medium.yml configuration  
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

### Requirement: Predator Configuration Schema
The system SHALL support comprehensive configuration of predator behavior, appearance, and mechanics within dynamic environments.

#### Scenario: Basic Predator Configuration
- **GIVEN** a YAML configuration file with predator settings
- **WHEN** the configuration is loaded
- **THEN** the system SHALL accept the following predator parameters under `environment.dynamic.predators`:
  - `enabled` (boolean, default false)
  - `count` (integer, default 2)
  - `speed` (float, default 1.0)
  - `movement_pattern` (string, default "random")
  - `detection_radius` (integer, default 8)
  - `kill_radius` (integer, default 0)
- **AND** all parameters SHALL have sensible defaults allowing minimal configuration

#### Scenario: Predator Gradient Configuration
- **GIVEN** a configuration specifying predator gradient parameters
- **WHEN** the configuration is loaded
- **THEN** the system SHALL accept under `environment.dynamic.predators`:
  - `gradient_decay_constant` (float, default 12.0)
  - `gradient_strength` (float, default 1.0)
- **AND** these MAY differ from food gradient parameters in `environment.dynamic.foraging`
- **AND** SHALL be used to compute predator repulsion gradients

#### Scenario: Predator Penalty Configuration
- **GIVEN** a configuration with predator reward penalties
- **WHEN** the configuration is loaded
- **THEN** the system SHALL accept under `reward`:
  - `penalty_predator_proximity` (float, default 0.1) - reward penalty per step within detection radius
- **AND** penalty value interpretation: a `penalty_predator_proximity: 0.1` configuration means -0.1 reward per step (positive value subtracted to create penalty)
- **AND** penalty value of 0.0 SHALL disable proximity penalties
- **AND** the penalty SHALL use positive values that are subtracted from reward (consistent with other penalty values)

#### Scenario: Minimal Predator Enablement
- **GIVEN** a configuration with only `predators.enabled: true`
- **WHEN** the configuration is loaded
- **THEN** all other predator parameters SHALL use default values
- **AND** the simulation SHALL run with 2 predators at speed 1.0, detection radius 8, kill radius 0

### Requirement: Restructured Dynamic Environment Configuration
The system SHALL organize dynamic environment settings into logical subsections for foraging and predators to improve clarity and maintainability.

#### Scenario: Foraging Subsection Configuration
- **GIVEN** a configuration using the new structure
- **WHEN** foraging parameters are specified
- **THEN** the system SHALL accept under `environment.dynamic.foraging`:
  - `foods_on_grid` (integer) (previously named `num_initial_foods`)
  - `target_foods_to_collect` (integer) (previously named `max_active_foods`)
  - `min_food_distance` (integer)
  - `agent_exclusion_radius` (integer)
  - `gradient_decay_constant` (float)
  - `gradient_strength` (float)
- **AND** these SHALL be nested under `foraging` subsection, not at `dynamic` root level

#### Scenario: Grid and Viewport at Dynamic Root
- **GIVEN** a configuration using the new structure
- **WHEN** environment structure is specified
- **THEN** the following SHALL remain at `environment.dynamic` root level:
  - `grid_size` (integer or tuple)
  - `viewport_size` (tuple)
- **AND** these SHALL not be nested under `foraging` or `predators`
- **AND** they SHALL apply to the entire environment regardless of feature enablement

#### Scenario: Complete Restructured Configuration Example
- **GIVEN** a full dynamic environment configuration
- **WHEN** all sections are specified
- **THEN** the structure SHALL be:
```yaml
environment:
  type: dynamic
  dynamic:
    grid_size: 100
    viewport_size: [11, 11]

    foraging:
      foods_on_grid: 50
      target_foods_to_collect: 50
      min_food_distance: 10
      agent_exclusion_radius: 15
      gradient_decay_constant: 12.0
      gradient_strength: 1.0

    predators:
      enabled: true
      count: 3
      speed: 1.0
      movement_pattern: "random"
      detection_radius: 8
      kill_radius: 0
      gradient_decay_constant: 12.0
      gradient_strength: 1.0

reward:
  penalty_predator_proximity: 0.1
```

### Requirement: Backward Compatibility with Legacy Configuration
The system SHALL automatically migrate legacy flat configuration structure to new nested structure with deprecation warnings.

#### Scenario: Legacy Flat Configuration Migration
- **GIVEN** an existing configuration with flat structure:
```yaml
environment:
  type: dynamic
  dynamic:
    grid_size: 50
    foods_on_grid: 20
    target_foods_to_collect: 30
    gradient_decay_constant: 12.0
```
- **WHEN** the configuration is loaded
- **THEN** the system SHALL automatically migrate to nested structure
- **AND** food-related parameters SHALL be moved under `foraging` subsection
- **AND** a deprecation warning SHALL be logged
- **AND** the simulation SHALL run correctly with migrated configuration

#### Scenario: Migration Warning Message
- **GIVEN** a legacy flat configuration is loaded
- **WHEN** migration is performed
- **THEN** a warning message SHALL be logged stating:
  - "Flat dynamic environment configuration is deprecated"
  - "Please restructure configuration with 'foraging' subsection"
  - Specific parameters that were migrated
- **AND** the warning SHALL include example of new structure

#### Scenario: New Configuration No Migration
- **GIVEN** a configuration already using nested `foraging` subsection
- **WHEN** the configuration is loaded
- **THEN** no migration SHALL be performed
- **AND** no deprecation warning SHALL be logged
- **AND** configuration SHALL be used as-is

#### Scenario: Mixed Configuration Handling
- **GIVEN** a configuration with some parameters in `foraging` subsection and some at root level
- **WHEN** the configuration is loaded
- **THEN** the system SHALL prioritize `foraging` subsection values
- **AND** root-level values SHALL be used only if not present in `foraging`
- **AND** a warning SHALL be logged about the inconsistent structure

### Requirement: Predator Movement Pattern Validation
The system SHALL validate predator movement pattern configuration and provide clear errors for invalid values.

#### Scenario: Valid Movement Pattern
- **GIVEN** a configuration with `movement_pattern: "random"`
- **WHEN** the configuration is validated
- **THEN** validation SHALL pass
- **AND** the predator SHALL use random movement behavior

#### Scenario: Invalid Movement Pattern
- **GIVEN** a configuration with `movement_pattern: "invalid_pattern"`
- **WHEN** the configuration is validated
- **THEN** validation SHALL fail with clear error message
- **AND** error SHALL list valid options: "random"
- **AND** error SHALL indicate future options (commented): "patrol", "pursue"

#### Scenario: Future Movement Pattern Placeholder
- **GIVEN** a configuration with `movement_pattern: "pursue"`
- **WHEN** the configuration is validated
- **THEN** validation SHALL fail
- **AND** error SHALL indicate "pursue pattern not yet implemented"
- **AND** error SHALL suggest using "random" for current version

### Requirement: Configuration Examples and Templates
The system SHALL provide example configuration files demonstrating predator-enabled setups for different difficulty levels.

#### Scenario: Predator-Enabled Small Environment Example
- **GIVEN** an example configuration file `configs/examples/mlp_predators_small.yml`
- **WHEN** the file is read
- **THEN** it SHALL demonstrate:
  - 20×20 grid with predators enabled
  - 2 predators for introductory difficulty
  - All predator parameters explicitly shown with comments
  - Foraging parameters in nested subsection
- **AND** the configuration SHALL be immediately runnable

#### Scenario: Predator-Enabled Large Environment Example
- **GIVEN** an example configuration file `configs/examples/modular_predators_large.yml`
- **WHEN** the file is read
- **THEN** it SHALL demonstrate:
  - 100×100 grid with 5 predators for advanced difficulty
  - ModularBrain configuration with predator support
  - Commented explanations of predator mechanics
  - Both foraging and predator subsections fully configured

#### Scenario: Configuration Documentation
- **GIVEN** the example configuration files
- **WHEN** a user reads the files
- **THEN** each predator parameter SHALL have inline comment explaining:
  - Parameter purpose and effect
  - Valid value range
  - Recommended values for different difficulty levels
  - How parameter affects learning difficulty
