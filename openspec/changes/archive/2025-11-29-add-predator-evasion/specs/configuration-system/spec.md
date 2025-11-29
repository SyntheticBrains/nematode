# configuration-system Delta

## ADDED Requirements

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
- **GIVEN** an example configuration file `configs/examples/mlp_dynamic_small_predators.yml`
- **WHEN** the file is read
- **THEN** it SHALL demonstrate:
  - 20×20 grid with predators enabled
  - 2 predators for introductory difficulty
  - All predator parameters explicitly shown with comments
  - Foraging parameters in nested subsection
- **AND** the configuration SHALL be immediately runnable

#### Scenario: Predator-Enabled Large Environment Example
- **GIVEN** an example configuration file `configs/examples/modular_dynamic_large_predators.yml`
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
