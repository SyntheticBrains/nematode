## ADDED Requirements

### Requirement: BrainParams Aggregation Pheromone Fields

BrainParams SHALL include fields for aggregation pheromone sensing data.

#### Scenario: Oracle Mode Fields

- **GIVEN** aggregation pheromone sensing in oracle mode
- **WHEN** BrainParams is constructed
- **THEN** `pheromone_aggregation_gradient_strength` SHALL contain the gradient magnitude [0, 1]
- **AND** `pheromone_aggregation_gradient_direction` SHALL contain direction in radians

#### Scenario: Temporal Mode Fields

- **GIVEN** aggregation pheromone sensing in temporal mode
- **WHEN** BrainParams is constructed
- **THEN** `pheromone_aggregation_concentration` SHALL contain scalar concentration [0, 1]
- **AND** `pheromone_aggregation_dconcentration_dt` SHALL contain temporal derivative

#### Scenario: Aggregation Fields Default None

- **GIVEN** pheromones disabled or aggregation not configured
- **WHEN** BrainParams is constructed
- **THEN** all 4 aggregation pheromone fields SHALL be None

### Requirement: Aggregation Pheromone Sensing Modules

Two new sensing modules SHALL be available for aggregation pheromone perception.

#### Scenario: PHEROMONE_AGGREGATION Module (Oracle)

- **GIVEN** PHEROMONE_AGGREGATION module in sensory_modules list
- **WHEN** features are extracted from BrainParams
- **THEN** classical output SHALL be [strength, relative_angle] (classical_dim=2)
- **AND** relative_angle SHALL be computed relative to agent heading

#### Scenario: PHEROMONE_AGGREGATION_TEMPORAL Module

- **GIVEN** PHEROMONE_AGGREGATION_TEMPORAL module in sensory_modules list
- **WHEN** features are extracted from BrainParams
- **THEN** classical output SHALL be [concentration, tanh(dC/dt * derivative_scale)] (classical_dim=2)

### Requirement: STAM 7-Channel Mode

The STAM buffer SHALL support 7 channels when aggregation pheromone is enabled.

#### Scenario: 7-Channel Configuration

- **GIVEN** pheromones enabled with aggregation configured
- **WHEN** STAMBuffer is created with num_channels=7
- **THEN** MEMORY_DIM SHALL be 17 (2*7 + 3)
- **AND** channel 6 SHALL be pheromone_aggregation

#### Scenario: Backward Compatibility

- **GIVEN** pheromones disabled
- **WHEN** STAMBuffer is created
- **THEN** num_channels SHALL be 4 and MEMORY_DIM SHALL be 11

#### Scenario: 6-Channel Backward Compatibility

- **GIVEN** pheromones enabled without aggregation
- **WHEN** STAMBuffer is created
- **THEN** num_channels SHALL be 6 and MEMORY_DIM SHALL be 15
