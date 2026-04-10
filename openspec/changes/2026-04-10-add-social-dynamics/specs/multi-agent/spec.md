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

- **GIVEN** an agent with alarm pheromone concentration > ALARM_EVASION_THRESHOLD (0.1) at step T
- **WHEN** alarm concentration drops to \<= ALARM_EVASION_THRESHOLD at step T+1
- **THEN** alarm_evasion_events SHALL be incremented
- **AND** repeated crossings below threshold SHALL each count as separate events

#### Scenario: Food Sharing Events

- **GIVEN** agent A emits a food-marking pheromone at position P at step T
- **AND** agent B (B != A) moves within `social_detection_radius` of P within FOOD_SHARING_LOOKBACK_STEPS (20) steps
- **WHEN** the proximity is detected
- **THEN** food_sharing_events SHALL be incremented
- **AND** the emission SHALL be removed from the tracking buffer (no double-counting)

### Requirement: CSV Export of Collective Metrics

The multi_agent_summary.csv SHALL include collective metric columns.

#### Scenario: Summary CSV Columns

- **GIVEN** a multi-agent session with collective metrics
- **WHEN** multi_agent_summary.csv is written
- **THEN** columns SHALL include `social_feeding_events`, `aggregation_index`, `alarm_evasion_events`, `food_sharing_events`
