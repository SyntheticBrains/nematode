## ADDED Requirements

### Requirement: Social Feeding System

The environment SHALL support social feeding via satiety decay reduction when agents are near conspecifics.

#### Scenario: Social Decay Reduction Applied

- **GIVEN** social feeding enabled with `decay_reduction=0.7`
- **AND** an agent with social phenotype at position (10, 10)
- **AND** another agent within `social_detection_radius` (Manhattan distance)
- **WHEN** satiety decay is applied for the step
- **THEN** the satiety decay rate SHALL be multiplied by 0.7

#### Scenario: No Reduction When Alone

- **GIVEN** social feeding enabled
- **AND** an agent with no other agents within `social_detection_radius`
- **WHEN** satiety decay is applied for the step
- **THEN** the decay rate SHALL be multiplied by 1.0 (unchanged)

#### Scenario: Solitary Phenotype

- **GIVEN** social feeding enabled with `solitary_decay=1.2`
- **AND** an agent with solitary phenotype near other agents
- **WHEN** satiety decay is applied
- **THEN** the decay rate SHALL be multiplied by 1.2 (crowding penalty)

#### Scenario: Social Feeding Disabled

- **GIVEN** social feeding not enabled in config (default)
- **WHEN** satiety decay is applied
- **THEN** the decay rate SHALL be 1.0 regardless of nearby agents

### Requirement: Aggregation Pheromone Field

The environment SHALL support aggregation pheromones using the existing PheromoneField point-source decay model.

#### Scenario: Aggregation Field Creation

- **GIVEN** pheromones enabled with aggregation config present
- **WHEN** the environment is initialized
- **THEN** a `pheromone_field_aggregation` PheromoneField SHALL be created
- **AND** it SHALL use the aggregation-specific decay parameters

#### Scenario: Aggregation Pheromone Emission

- **WHEN** `emit_aggregation_pheromone(position, current_step, emitter_id)` is called
- **THEN** a PheromoneSource with type AGGREGATION SHALL be added to the aggregation field

#### Scenario: Aggregation Concentration

- **GIVEN** an aggregation field with active sources
- **WHEN** `get_pheromone_aggregation_concentration(position, current_step)` is called
- **THEN** concentration SHALL be computed using superposition of exponential decay
- **AND** the result SHALL be tanh-normalized to [0, 1]

#### Scenario: Aggregation Gradient

- **GIVEN** aggregation sources concentrated in one area
- **WHEN** gradient is queried at a nearby position
- **THEN** the gradient SHALL point toward the source concentration
- **AND** gradient SHALL be computed via central differences

#### Scenario: Aggregation Field Pruning

- **GIVEN** aggregation sources with short temporal_half_life (10 steps)
- **WHEN** `update_pheromone_fields(current_step)` is called
- **THEN** expired aggregation sources SHALL be pruned
- **AND** max_sources limit SHALL be enforced

#### Scenario: No Aggregation Field When Not Configured

- **GIVEN** pheromones enabled but no aggregation config
- **WHEN** environment is initialized
- **THEN** `pheromone_field_aggregation` SHALL be None
- **AND** aggregation methods SHALL return 0.0 / None

## MODIFIED Requirements

### Requirement: Satiety Decay Accepts Multiplier

The `SatietyManager.decay_satiety()` method SHALL accept an optional multiplier parameter.

#### Scenario: Default Multiplier

- **WHEN** `decay_satiety()` is called without arguments
- **THEN** the multiplier SHALL default to 1.0
- **AND** behavior SHALL be identical to current implementation

#### Scenario: Reduced Decay

- **WHEN** `decay_satiety(multiplier=0.7)` is called
- **THEN** the decay amount SHALL be 70% of the normal rate

### Requirement: Pheromone Field Update Includes Aggregation

- **GIVEN** aggregation pheromone field exists
- **WHEN** `update_pheromone_fields(current_step)` is called
- **THEN** the aggregation field SHALL be pruned alongside food and alarm fields
