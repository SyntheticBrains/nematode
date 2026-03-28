## ADDED Requirements

### Requirement: Sensing Configuration Schema

The configuration system SHALL support a sensing configuration section for selecting sensing modes and STAM parameters.

#### Scenario: Sensing Mode Configuration

- **WHEN** a YAML configuration includes a `sensing` section under `environment`
- **THEN** the system SHALL accept `chemotaxis_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
- **AND** SHALL accept `thermotaxis_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
- **AND** SHALL accept `nociception_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
- **AND** each mode SHALL be independently configurable

#### Scenario: STAM Configuration

- **WHEN** a YAML configuration includes STAM parameters under `environment.sensing`
- **THEN** the system SHALL accept `stam_enabled` (boolean, default: false)
- **AND** SHALL accept `stam_buffer_size` (integer, default: 30, must be > 0)
- **AND** SHALL accept `stam_decay_rate` (float, default: 0.1, must be > 0)

#### Scenario: Sensing Configuration Absent

- **WHEN** no `sensing` section is provided in the environment configuration
- **THEN** all sensing modes SHALL default to "oracle"
- **AND** STAM SHALL be disabled
- **AND** the system SHALL behave identically to pre-temporal-sensing versions

#### Scenario: Invalid Sensing Mode

- **WHEN** a sensing mode is set to an unrecognised value (e.g., `chemotaxis_mode: "invalid"`)
- **THEN** configuration validation SHALL fail with a clear error message
- **AND** the error SHALL list the valid options: "oracle", "temporal", "derivative"

### Requirement: Sensing Mode Validation

The configuration system SHALL validate sensing mode and STAM parameter combinations.

#### Scenario: Temporal Mode Without STAM Warning

- **WHEN** `chemotaxis_mode: temporal` is set without `stam_enabled: true`
- **THEN** the system SHALL log a warning that Mode A (temporal) without STAM may result in very limited sensory information
- **AND** the configuration SHALL still be accepted (STAM is recommended but not required)

#### Scenario: Derivative Mode Auto-Enables STAM

- **WHEN** any modality is set to `derivative` mode and `stam_enabled` is not explicitly set to `true`
- **THEN** the system SHALL auto-enable STAM with default parameters (`buffer_size: 30`, `decay_rate: 0.1`)
- **AND** SHALL log an info message indicating that STAM was auto-enabled because derivative mode requires temporal history
- **AND** explicitly-set STAM parameters SHALL be preserved if provided

#### Scenario: STAM Buffer Size Validation

- **WHEN** `stam_buffer_size` is set to 0 or a negative value
- **THEN** configuration validation SHALL fail
- **AND** the error SHALL indicate that buffer size must be a positive integer

#### Scenario: STAM Decay Rate Validation

- **WHEN** `stam_decay_rate` is set to 0 or a negative value
- **THEN** configuration validation SHALL fail
- **AND** the error SHALL indicate that decay rate must be a positive float

### Requirement: Temporal Sensing Example Configurations

The system SHALL provide example configurations demonstrating temporal sensing modes.

#### Scenario: Temporal Foraging Example

- **WHEN** example config `mlpppo_foraging_small_temporal.yml` is loaded
- **THEN** it SHALL configure Mode A (temporal) chemotaxis with STAM enabled
- **AND** SHALL use MLPPPO brain architecture
- **AND** SHALL use existing small environment parameters (20×20 grid, 5 foods, 500 steps)
- **AND** SHALL include inline comments explaining the sensing mode and STAM parameters

#### Scenario: Derivative Thermotaxis With Foraging Example

- **WHEN** example config `mlpppo_thermotaxis_foraging_small_temporal.yml` is loaded
- **THEN** it SHALL configure Mode B (derivative) thermotaxis with STAM enabled
- **AND** chemotaxis SHALL also be configured in temporal or derivative mode
- **AND** SHALL use existing small thermotaxis foraging environment parameters

#### Scenario: Temporal Pursuit Predators Example

- **WHEN** example config `mlpppo_pursuit_predators_small_temporal.yml` is loaded
- **THEN** it SHALL configure Mode A (temporal) nociception and chemotaxis with STAM enabled
- **AND** SHALL use pursuit predators (`movement_pattern: pursuit`)
- **AND** SHALL use existing small pursuit predator environment parameters

#### Scenario: Temporal Thermotaxis With Pursuit Predators Example

- **WHEN** example config `mlpppo_thermotaxis_pursuit_predators_small_temporal.yml` is loaded
- **THEN** it SHALL configure temporal sensing for all three modalities (chemotaxis, thermotaxis, nociception) with STAM enabled
- **AND** SHALL use pursuit predators (`movement_pattern: pursuit`)
- **AND** SHALL use existing small thermotaxis pursuit predator environment parameters

### Requirement: Complete Temporal Sensing Configuration Example

The configuration system SHALL support the following YAML structure for temporal sensing.

#### Scenario: Full Configuration Structure

- **WHEN** a complete temporal sensing configuration is provided
- **THEN** the system SHALL accept the following structure:

```yaml
environment:
  grid_size: 20
  viewport_size: [11, 11]
  sensing:
    chemotaxis_mode: temporal
    thermotaxis_mode: derivative
    nociception_mode: oracle
    stam_enabled: true
    stam_buffer_size: 30
    stam_decay_rate: 0.1
  foraging:
    foods_on_grid: 5
    target_foods_to_collect: 10
```

- **AND** the sensing section SHALL be parsed before brain construction
- **AND** sensory module translation SHALL be applied based on the sensing modes
