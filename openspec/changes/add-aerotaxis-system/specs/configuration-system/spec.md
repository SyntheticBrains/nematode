## ADDED Requirements

### Requirement: Aerotaxis Configuration Schema

The configuration system SHALL support an aerotaxis configuration section for defining oxygen field parameters, zone thresholds, and reward/penalty values.

#### Scenario: Aerotaxis Configuration Section

- **WHEN** a YAML configuration includes an `aerotaxis` section under `environment`
- **THEN** the system SHALL accept the following fields:
  - `enabled: bool` (default: false)
  - `base_oxygen: float` (default: 10.0, O2 percentage at grid center)
  - `gradient_direction: float` (default: 0.0, radians)
  - `gradient_strength: float` (default: 0.1, O2 % per cell)
  - `high_oxygen_spots: list[list[float]]` (optional, [x, y, intensity] tuples)
  - `low_oxygen_spots: list[list[float]]` (optional, [x, y, intensity] tuples)
  - `spot_decay_constant: float` (default: 5.0)
  - `comfort_reward: float` (default: 0.0)
  - `danger_penalty: float` (default: -0.5)
  - `danger_hp_damage: float` (default: 0.5)
  - `lethal_hp_damage: float` (default: 6.0)
  - `reward_discomfort_food: float` (default: 0.0, bonus for collecting food in danger zones)
  - `lethal_hypoxia_upper: float` (default: 2.0)
  - `danger_hypoxia_upper: float` (default: 5.0)
  - `comfort_lower: float` (default: 5.0)
  - `comfort_upper: float` (default: 12.0)
  - `danger_hyperoxia_upper: float` (default: 17.0)

#### Scenario: Aerotaxis Not Configured

- **WHEN** no `aerotaxis` section is present in the environment configuration
- **THEN** aerotaxis SHALL be disabled
- **AND** the system SHALL behave identically to pre-aerotaxis versions

#### Scenario: Aerotaxis With Thermotaxis

- **WHEN** both `thermotaxis` and `aerotaxis` sections are present and enabled
- **THEN** both SHALL be parsed and initialized independently
- **AND** both OxygenField and TemperatureField SHALL coexist in the environment

### Requirement: Aerotaxis Example Configurations

The system SHALL provide example configurations demonstrating aerotaxis scenarios.

#### Scenario: Oxygen Foraging Medium Oracle Example

- **WHEN** example config `configs/scenarios/oxygen_foraging/mlpppo_medium_oracle.yml` is loaded
- **THEN** it SHALL configure aerotaxis with a medium (50×50) grid
- **AND** SHALL use oracle sensing mode for food chemotaxis and aerotaxis
- **AND** SHALL include inline comments explaining oxygen field parameters

#### Scenario: Oxygen Thermal Foraging Large Oracle Example

- **WHEN** example config `configs/scenarios/oxygen_thermal_foraging/mlpppo_large_oracle.yml` is loaded
- **THEN** it SHALL configure both aerotaxis and thermotaxis on a large (100×100) grid
- **AND** temperature and oxygen gradient directions SHALL be approximately orthogonal
- **AND** sensory modules SHALL include `food_chemotaxis`, `thermotaxis`, and `aerotaxis`

## MODIFIED Requirements

### Requirement: Sensing Configuration Schema

The configuration system SHALL support a sensing configuration section for selecting sensing modes and STAM parameters.

#### Scenario: Sensing Mode Configuration

- **WHEN** a YAML configuration includes a `sensing` section under `environment`
- **THEN** the system SHALL accept `chemotaxis_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
- **AND** SHALL accept `thermotaxis_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
- **AND** SHALL accept `nociception_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
- **AND** SHALL accept `aerotaxis_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
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

- **WHEN** a sensing mode is set to an unrecognised value (e.g., `aerotaxis_mode: "invalid"`)
- **THEN** configuration validation SHALL fail with a clear error message
- **AND** the error SHALL list the valid options: "oracle", "temporal", "derivative"

### Requirement: Sensing Mode Validation

The configuration system SHALL validate sensing mode and STAM parameter combinations.

#### Scenario: Temporal Mode Without STAM Warning

- **WHEN** any modality including aerotaxis is set to `temporal` mode without `stam_enabled: true`
- **THEN** the system SHALL log a warning that Mode A (temporal) without STAM may result in very limited sensory information
- **AND** the configuration SHALL still be accepted (STAM is recommended but not required)

#### Scenario: Derivative Mode Auto-Enables STAM

- **WHEN** any modality including aerotaxis is set to `derivative` mode and `stam_enabled` is not explicitly set to `true`
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
