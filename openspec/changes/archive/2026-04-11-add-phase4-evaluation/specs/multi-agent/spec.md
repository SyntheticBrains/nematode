## ADDED Requirements

### Requirement: Territorial Index Metric

MultiAgentEpisodeResult SHALL include a territorial index measuring spatial foraging specialization.

#### Scenario: Territorial Index Computation

- **GIVEN** a completed multi-agent episode where agents collected food
- **WHEN** results are computed
- **THEN** `territorial_index` SHALL be the Gini coefficient of per-agent foraging spreads
- **AND** each agent's spread is the mean Manhattan distance of their food collection positions from their centroid
- **AND** value SHALL be in [0, 1] where 0 = equal foraging patterns, 1 = maximal specialization

#### Scenario: Insufficient Data

- **GIVEN** fewer than 2 agents collected food during the episode
- **WHEN** territorial_index is computed
- **THEN** it SHALL be 0.0

#### Scenario: Single Food Per Agent

- **GIVEN** an agent collected exactly 1 food item
- **THEN** that agent's foraging spread SHALL be 0.0 (single point)

### Requirement: Alarm Response Rate Metric

MultiAgentEpisodeResult SHALL include an alarm response rate measuring causal reaction to alarm pheromone emissions.

#### Scenario: Alarm Response Rate Computation

- **GIVEN** an alarm pheromone emitted at step T by agent A
- **AND** agent B within `social_detection_radius` at step T
- **WHEN** agent B changes movement direction within ALARM_RESPONSE_WINDOW (5) steps
- **THEN** this SHALL count as a positive response

#### Scenario: Rate Calculation

- **GIVEN** N alarm response opportunities and M positive responses
- **THEN** alarm_response_rate SHALL be M / N
- **AND** SHALL be 0.0 when N = 0 (no alarm emissions or no nearby agents)

#### Scenario: Self-Exclusion

- **GIVEN** agent A emits an alarm pheromone
- **THEN** agent A SHALL NOT be counted as a response opportunity

### Requirement: CSV Export of New Metrics

#### Scenario: Summary CSV Columns

- **WHEN** multi_agent_summary.csv is written
- **THEN** columns SHALL include `territorial_index` and `alarm_response_rate`
