## MODIFIED Requirements

### Requirement: Social Feeding in Multi-Agent Step Loop

The MultiAgentSimulation SHALL apply social feeding decay reduction during the effects phase.

#### Scenario: Social Decay Reduction in Effects Phase

- **GIVEN** social feeding enabled and an agent with social phenotype
- **AND** the agent has nearby_agents_count > 0
- **WHEN** the effects phase (section 5) processes satiety decay
- **THEN** `decay_satiety(multiplier=decay_reduction)` SHALL be called
- **AND** social_feeding_events counter SHALL be incremented

#### Scenario: Mixed Phenotype Population

- **GIVEN** 3 social and 2 solitary agents in the same environment
- **WHEN** a social agent is near a solitary agent
- **THEN** the social agent SHALL receive decay_reduction multiplier
- **AND** the solitary agent SHALL receive solitary_decay multiplier

### Requirement: Continuous Aggregation Pheromone Emission

The MultiAgentSimulation SHALL emit aggregation pheromones for all alive agents each step.

#### Scenario: Per-Step Emission

- **GIVEN** aggregation pheromone enabled
- **WHEN** the step loop processes movement (section 2)
- **THEN** each alive agent SHALL emit an aggregation pheromone at their current position
- **AND** emission_step SHALL match the current step

#### Scenario: Aggregation Emission Disabled

- **GIVEN** pheromones disabled or no aggregation config
- **WHEN** the step loop runs
- **THEN** no aggregation pheromone emission SHALL occur

## ADDED Requirements

### Requirement: Collective Behavior Metrics

MultiAgentEpisodeResult SHALL include collective behavior metrics.

#### Scenario: Social Feeding Events

- **GIVEN** a multi-agent episode with social feeding enabled
- **WHEN** the episode completes
- **THEN** `social_feeding_events` SHALL equal the count of step-agent pairs where decay reduction was applied

#### Scenario: Aggregation Index

- **GIVEN** a multi-agent episode
- **WHEN** the episode completes
- **THEN** `aggregation_index` SHALL be the mean normalized inverse pairwise distance averaged over all steps
- **AND** SHALL be 0.0 when agents are maximally dispersed
- **AND** SHALL approach 1.0 when agents are co-located

#### Scenario: Aggregation Index with Single Agent

- **GIVEN** only one agent alive during a step
- **THEN** that step SHALL contribute 0.0 to the aggregation index

#### Scenario: Alarm Evasion Events

- **GIVEN** an agent with alarm pheromone concentration > threshold at step T
- **AND** the agent moves at step T+1
- **WHEN** the agent's distance from the alarm gradient source increases
- **THEN** alarm_evasion_events SHALL be incremented

#### Scenario: Food Sharing Events

- **GIVEN** agent A emits a food-marking pheromone at position P at step T
- **AND** agent B (B != A) moves within detection radius of P within the next N steps
- **WHEN** the episode completes
- **THEN** food_sharing_events SHALL include this event

### Requirement: CSV Export of Collective Metrics

The multi_agent_summary.csv SHALL include collective metric columns.

#### Scenario: Summary CSV Columns

- **GIVEN** a multi-agent session with collective metrics
- **WHEN** multi_agent_summary.csv is written
- **THEN** columns SHALL include `social_feeding_events`, `aggregation_index`, `alarm_evasion_events`, `food_sharing_events`
