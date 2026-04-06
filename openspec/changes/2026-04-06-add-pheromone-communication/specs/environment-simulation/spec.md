## ADDED Requirements

### Requirement: Pheromone Field System

The environment SHALL support dynamic pheromone concentration fields using point-source exponential decay.

#### Scenario: Pheromone Concentration at Position

- **GIVEN** a PheromoneField with one source at (10, 10) with strength 1.0
- **WHEN** `get_concentration((15, 10), current_step)` is called
- **THEN** the result SHALL be `tanh(1.0 * exp(-5 / spatial_decay) * exp(-age * ln(2) / half_life))`
- **AND** the result SHALL be in [0.0, 1.0]

#### Scenario: Multiple Source Superposition

- **GIVEN** two pheromone sources at (5, 5) and (15, 15)
- **WHEN** concentration is queried at (10, 10)
- **THEN** the result SHALL be the tanh-normalized sum of both contributions

#### Scenario: Temporal Decay

- **GIVEN** a source emitted at step 0 with temporal_half_life=50
- **WHEN** concentration is queried at step 50
- **THEN** the temporal factor SHALL be approximately 0.5 (half-life)
- **AND** at step 250 (5 half-lives) SHALL be approximately 0.03

#### Scenario: Source Pruning

- **GIVEN** sources with ages exceeding `temporal_half_life * 5`
- **WHEN** `prune(current_step)` is called
- **THEN** expired sources SHALL be removed
- **AND** max_sources limit SHALL be enforced by removing oldest first

#### Scenario: Pheromone Gradient

- **GIVEN** a pheromone source at (10, 10)
- **WHEN** gradient is queried at (8, 10)
- **THEN** the gradient vector SHALL point toward (10, 10)
- **AND** gradient SHALL be computed via central differences

### Requirement: Pheromone Emission

The environment SHALL provide methods for agents to emit pheromones.

#### Scenario: Food-Marking Emission

- **WHEN** `emit_food_pheromone(position, current_step, emitter_id)` is called
- **THEN** a PheromoneSource with type FOOD_MARKING SHALL be added to the food pheromone field

#### Scenario: Alarm Emission

- **WHEN** `emit_alarm_pheromone(position, current_step, emitter_id)` is called
- **THEN** a PheromoneSource with type ALARM SHALL be added to the alarm pheromone field

#### Scenario: Pheromones Disabled

- **GIVEN** pheromones are not enabled in config
- **WHEN** any pheromone method is called
- **THEN** concentration methods SHALL return 0.0
- **AND** gradient methods SHALL return None
- **AND** emit methods SHALL be no-ops
