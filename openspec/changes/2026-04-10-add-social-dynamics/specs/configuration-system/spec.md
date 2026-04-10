## ADDED Requirements

### Requirement: Social Feeding Configuration

The configuration system SHALL support social feeding parameters.

#### Scenario: Social Feeding Config Block

- **GIVEN** a YAML config with `environment.social_feeding` block
- **WHEN** the config is loaded
- **THEN** `SocialFeedingConfig` SHALL be parsed with validated fields
- **AND** `to_params()` SHALL produce a `SocialFeedingParams` dataclass

#### Scenario: Social Feeding Defaults

- **GIVEN** a YAML config without `environment.social_feeding` block
- **WHEN** the config is loaded
- **THEN** social feeding SHALL be disabled by default

#### Scenario: Social Feeding Validation

- **GIVEN** `decay_reduction` \<= 0 or `detection_radius` \<= 0
- **WHEN** config validation runs
- **THEN** a validation error SHALL be raised

### Requirement: Aggregation Pheromone Configuration

The pheromone config block SHALL support aggregation pheromone parameters.

#### Scenario: Aggregation Config Block

- **GIVEN** a YAML config with `environment.pheromones.aggregation` block
- **WHEN** the config is loaded
- **THEN** aggregation pheromone type config SHALL be parsed
- **AND** `PheromoneParams` SHALL include aggregation field defaults

#### Scenario: Aggregation Not Configured

- **GIVEN** a YAML config with `environment.pheromones` but no `aggregation` block
- **WHEN** the config is loaded
- **THEN** aggregation pheromone field SHALL not be created
- **AND** STAM SHALL remain at 6-channel mode (not 7)

### Requirement: Aggregation Sensing Mode

SensingConfig SHALL support aggregation pheromone sensing mode.

#### Scenario: Aggregation Sensing Mode Translation

- **GIVEN** `pheromone_aggregation_mode: temporal` in sensing config
- **WHEN** `apply_sensing_mode()` processes sensory modules
- **THEN** `pheromone_aggregation` SHALL be translated to `pheromone_aggregation_temporal`

#### Scenario: Aggregation Sensing Validation

- **GIVEN** `pheromone_aggregation_mode: derivative` or `temporal`
- **WHEN** `validate_sensing_config()` runs
- **THEN** STAM module SHALL be required in sensory_modules (same rule as other temporal/derivative modes)

### Requirement: Per-Agent Social Phenotype

AgentConfig SHALL support social phenotype specification.

#### Scenario: Heterogeneous Phenotypes

- **GIVEN** a multi-agent config with `agents` list
- **AND** each agent specifies `social_phenotype: social` or `social_phenotype: solitary`
- **WHEN** config is loaded
- **THEN** per-agent phenotypes SHALL be available to MultiAgentSimulation

#### Scenario: Homogeneous Default

- **GIVEN** a multi-agent config with `count: 5` (no agents list)
- **WHEN** config is loaded
- **THEN** all agents SHALL default to `social_phenotype: social`
