## ADDED Requirements

### Requirement: ThermotaxisConfig Class

The configuration system SHALL provide a ThermotaxisConfig class for YAML configuration.

#### Scenario: ThermotaxisConfig Definition

- **GIVEN** the config_loader.py module
- **WHEN** thermotaxis configuration is needed
- **THEN** ThermotaxisConfig class SHALL be defined as a Pydantic BaseModel
- **AND** it SHALL support all ThermotaxisParams fields with matching defaults

#### Scenario: ThermotaxisConfig Fields

- **GIVEN** a ThermotaxisConfig instance
- **THEN** the following fields SHALL be available with defaults:
  - `enabled: bool = False`
  - `cultivation_temperature: float = 20.0`
  - `base_temperature: float = 20.0`
  - `gradient_direction: float = 0.0`
  - `gradient_strength: float = 0.5`
  - `comfort_delta: float = 5.0`
  - `discomfort_delta: float = 10.0`
  - `danger_delta: float = 15.0`
  - `comfort_reward: float = 0.01`
  - `discomfort_penalty: float = -0.02`
  - `danger_penalty: float = -0.05`
  - `danger_hp_damage: float = 5.0`
  - `lethal_hp_damage: float = 20.0`

#### Scenario: ThermotaxisConfig to_params Method

- **GIVEN** a ThermotaxisConfig instance
- **WHEN** `to_params()` is called
- **THEN** a ThermotaxisParams instance SHALL be returned
- **AND** all field values SHALL be transferred correctly

### Requirement: DynamicEnvironmentConfig Integration

The DynamicEnvironmentConfig SHALL include thermotaxis configuration.

#### Scenario: Thermotaxis Field in DynamicEnvironmentConfig

- **GIVEN** a DynamicEnvironmentConfig instance
- **THEN** `thermotaxis: ThermotaxisConfig | None` field SHALL be available
- **AND** default SHALL be None (thermotaxis disabled)

#### Scenario: get_thermotaxis_config Method

- **GIVEN** a DynamicEnvironmentConfig with thermotaxis=None
- **WHEN** `get_thermotaxis_config()` is called
- **THEN** a default ThermotaxisConfig SHALL be returned (disabled)
- **GIVEN** a DynamicEnvironmentConfig with thermotaxis configured
- **WHEN** `get_thermotaxis_config()` is called
- **THEN** the configured ThermotaxisConfig SHALL be returned

### Requirement: YAML Configuration Support

The system SHALL support thermotaxis configuration via YAML files.

#### Scenario: Thermotaxis YAML Section

- **GIVEN** a simulation YAML config file
- **WHEN** thermotaxis settings are specified
- **THEN** the following structure SHALL be supported:

```yaml
environment:
  type: dynamic
  dynamic:
    thermotaxis:
      enabled: true
      cultivation_temperature: 20.0
      base_temperature: 20.0
      gradient_direction: 0.0
      gradient_strength: 0.5
      comfort_delta: 5.0
      discomfort_delta: 10.0
      danger_delta: 15.0
      comfort_reward: 0.05
      discomfort_penalty: -0.1
      danger_penalty: -0.3
      danger_hp_damage: 2.0
      lethal_hp_damage: 10.0
```

#### Scenario: Thermotaxis Disabled by Default

- **GIVEN** a YAML config without thermotaxis section
- **WHEN** configuration is loaded
- **THEN** thermotaxis SHALL be disabled
- **AND** no temperature field SHALL be created
