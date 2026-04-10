## ADDED Requirements

### Requirement: Pheromone Configuration Schema

The configuration system SHALL support pheromone parameters in the environment config.

#### Scenario: Pheromone Config YAML

- **GIVEN** a YAML configuration with:

```yaml
environment:
  pheromones:
    enabled: true
    food_marking:
      emission_strength: 1.0
      spatial_decay_constant: 8.0
      temporal_half_life: 50
      max_sources: 100
    alarm:
      emission_strength: 2.0
      spatial_decay_constant: 5.0
      temporal_half_life: 20
      max_sources: 50
```

- **WHEN** the configuration is loaded
- **THEN** `PheromoneConfig.enabled` SHALL be True
- **AND** `PheromoneConfig.food_marking.emission_strength` SHALL be 1.0
- **AND** `PheromoneConfig.alarm.temporal_half_life` SHALL be 20

#### Scenario: Pheromone Disabled (Default)

- **GIVEN** a YAML configuration without pheromones section
- **WHEN** the configuration is loaded
- **THEN** pheromones SHALL be disabled
- **AND** no PheromoneField instances SHALL be created

#### Scenario: Parameter Validation

- **GIVEN** pheromone config with `emission_strength: -1.0`
- **WHEN** validation is performed
- **THEN** a ValidationError SHALL be raised

## MODIFIED Requirements

### Requirement: Extended SensingConfig for Pheromone Modes

The SensingConfig SHALL support per-modality sensing modes for pheromone channels.

#### Scenario: Pheromone Sensing Modes

- **GIVEN** a SensingConfig
- **THEN** `pheromone_food_mode: SensingMode` SHALL default to ORACLE
- **AND** `pheromone_alarm_mode: SensingMode` SHALL default to ORACLE

#### Scenario: Sensing Mode Translation

- **GIVEN** `pheromone_food_mode: temporal` in config
- **WHEN** `apply_sensing_mode()` translates sensory modules
- **THEN** `pheromone_food` SHALL be replaced with `pheromone_food_temporal`

### Requirement: Multi-Agent CSV Export

The CSV export system SHALL support per-agent results for multi-agent sessions.

#### Scenario: Per-Agent Results CSV

- **GIVEN** a completed multi-agent session
- **WHEN** results are exported to `simulation_results.csv`
- **THEN** each row SHALL include an `agent_id` column
- **AND** there SHALL be one row per agent per episode

#### Scenario: Aggregate Summary CSV

- **GIVEN** a completed multi-agent session
- **WHEN** results are exported
- **THEN** a `multi_agent_summary.csv` SHALL be created
- **AND** each row SHALL contain: run, total_food, competition_events, proximity_events, alive_at_end, mean_success, gini

#### Scenario: Single-Agent Backward Compatibility

- **GIVEN** a single-agent session
- **WHEN** results are exported
- **THEN** `agent_id` column SHALL contain "default"
- **AND** no `multi_agent_summary.csv` SHALL be created
