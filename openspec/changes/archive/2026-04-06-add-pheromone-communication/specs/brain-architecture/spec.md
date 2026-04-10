## MODIFIED Requirements

### Requirement: Extended BrainParams for Pheromone Sensing

The BrainParams class SHALL include optional fields for pheromone concentration and gradient information.

#### Scenario: Pheromone Oracle Fields

- **GIVEN** a brain receiving sensory input in a pheromone-enabled environment
- **WHEN** BrainParams is populated in oracle mode
- **THEN** the following fields SHALL be available:
  - `pheromone_food_gradient_strength: float | None` — magnitude of food-marking pheromone gradient
  - `pheromone_food_gradient_direction: float | None` — direction of food-marking pheromone gradient (radians)
  - `pheromone_alarm_gradient_strength: float | None` — magnitude of alarm pheromone gradient
  - `pheromone_alarm_gradient_direction: float | None` — direction of alarm pheromone gradient (radians)

#### Scenario: Pheromone Temporal Fields

- **GIVEN** pheromone sensing in temporal or derivative mode
- **WHEN** BrainParams is populated
- **THEN** the following fields SHALL be available:
  - `pheromone_food_concentration: float | None` — scalar food pheromone at agent position
  - `pheromone_alarm_concentration: float | None` — scalar alarm pheromone at agent position
  - `pheromone_food_dconcentration_dt: float | None` — temporal derivative of food pheromone
  - `pheromone_alarm_dconcentration_dt: float | None` — temporal derivative of alarm pheromone

#### Scenario: Backward Compatibility

- **GIVEN** a config without pheromones enabled
- **WHEN** BrainParams is populated
- **THEN** all pheromone fields SHALL be None
- **AND** existing brain functionality SHALL be unchanged

## ADDED Requirements

### Requirement: Pheromone Sensory Modules

The system SHALL provide sensory modules for pheromone detection following the oracle/temporal pattern.

#### Scenario: Food-Marking Pheromone Oracle Module

- **GIVEN** `PHEROMONE_FOOD` module registered in SENSORY_MODULES
- **WHEN** features are extracted with pheromone gradient data
- **THEN** strength SHALL encode gradient magnitude [0, 1]
- **AND** angle SHALL encode egocentric direction to higher concentration [-1, 1]
- **AND** classical_dim SHALL be 2

#### Scenario: Alarm Pheromone Oracle Module

- **GIVEN** `PHEROMONE_ALARM` module registered in SENSORY_MODULES
- **THEN** it SHALL follow the same pattern as PHEROMONE_FOOD
- **AND** strength encodes alarm gradient magnitude, angle encodes direction toward alarm source

#### Scenario: Pheromone Temporal Modules

- **GIVEN** `PHEROMONE_FOOD_TEMPORAL` and `PHEROMONE_ALARM_TEMPORAL` modules
- **WHEN** features are extracted with temporal pheromone data
- **THEN** strength SHALL encode scalar concentration [0, 1]
- **AND** angle SHALL encode tanh(dC/dt * derivative_scale) [-1, 1]
- **AND** classical_dim SHALL be 2 for each

### Requirement: STAM Pheromone Channel Extension

The STAM buffer SHALL support pheromone concentration channels when pheromones are enabled.

#### Scenario: 6-Channel Mode (Pheromones Enabled)

- **GIVEN** STAMBuffer with num_channels=6
- **WHEN** `get_memory_state()` is called
- **THEN** output SHALL be 15-dimensional: 6 weighted means + 6 derivatives + 2 position deltas + 1 action entropy
- **AND** channels 4 and 5 SHALL correspond to pheromone_food and pheromone_alarm

#### Scenario: 4-Channel Mode (Backward Compatible)

- **GIVEN** STAMBuffer with num_channels=4 (pheromones disabled)
- **WHEN** `get_memory_state()` is called
- **THEN** output SHALL be 11-dimensional (unchanged from current behavior)
