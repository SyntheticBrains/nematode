## ADDED Requirements

### Requirement: Per-Predator Metrics in MultiAgentEpisodeResult

`MultiAgentEpisodeResult` SHALL expose per-predator metric dicts keyed by synthesised `predator_id` so future predator-fitness functions can read them directly. The dicts mirror the existing per-agent metric pattern (`per_agent_food`, `per_agent_reward`, `per_agent_satiety`).

#### Scenario: Per-Predator Distance Traveled

- **GIVEN** a `MultiAgentEpisodeResult`
- **WHEN** the episode ends
- **THEN** it SHALL include `per_predator_distance_traveled: dict[str, int]`
- **AND** each value SHALL be the sum of cardinal-direction position changes by that predator over the episode (each step contributes 0 or 1)
- **AND** stationary predators SHALL have value 0
- **AND** wall-blocked steps (where `_apply_action` clamped the position to its current value) SHALL contribute 0

#### Scenario: Per-Predator Prey Proximity Steps

- **GIVEN** a `MultiAgentEpisodeResult`
- **WHEN** the episode ends
- **THEN** it SHALL include `per_predator_prey_proximity_steps: dict[str, int]`
- **AND** for each predator, the value SHALL be the count of simulation steps where at least one alive agent was within `predator.detection_radius` (Manhattan distance)
- **AND** the count SHALL increment by 1 per step regardless of how many alive agents are in range (NOT scaled by prey count)
- **AND** stationary predators SHALL still accumulate this counter

#### Scenario: Per-Predator Kills

- **GIVEN** a `MultiAgentEpisodeResult`
- **WHEN** the episode ends
- **THEN** it SHALL include `per_predator_kills: dict[str, int]`
- **AND** each value SHALL be the count of step-events where this predator's damage application brought an agent's HP to 0
- **AND** the sum of all values SHALL equal the total agent deaths attributable to predator damage in the episode (no double-counting, no missing kills)

### Requirement: Multi-Predator Kill Attribution Rule

When `apply_predator_damage_for(aid)` brings an agent's HP to 0 and multiple predators have the agent inside their `damage_radius` simultaneously, the system SHALL credit the kill to exactly one predator using the closest-by-Manhattan rule with lexicographic tie-break on `predator_id`.

#### Scenario: Single Predator Damage

- **GIVEN** an agent at HP equal to `predator_damage` and one predator covering the agent in its damage radius
- **WHEN** `apply_predator_damage_for` brings HP to 0
- **THEN** `per_predator_kills[predator_id]` SHALL increment by 1
- **AND** all other predators' `per_predator_kills` entries SHALL remain unchanged

#### Scenario: Multi-Predator Damage with Distinct Distances

- **GIVEN** an agent at HP equal to `predator_damage`
- **AND** predator_0 at Manhattan distance 1, predator_1 at Manhattan distance 2, both with `damage_radius: 2`
- **WHEN** `apply_predator_damage_for` brings HP to 0
- **THEN** `per_predator_kills["predator_0"]` SHALL increment by 1 (closest)
- **AND** `per_predator_kills["predator_1"]` SHALL remain unchanged

#### Scenario: Multi-Predator Damage with Equal Distances (Tie-Break)

- **GIVEN** an agent at HP equal to `predator_damage`
- **AND** predator_0 and predator_1 both at Manhattan distance 1 from the agent, both with `damage_radius: 1`
- **WHEN** `apply_predator_damage_for` brings HP to 0
- **THEN** `per_predator_kills["predator_0"]` SHALL increment by 1 (lex tie-break, `"predator_0" < "predator_1"`)
- **AND** `per_predator_kills["predator_1"]` SHALL remain unchanged

## MODIFIED Requirements

### Requirement: Multi-Agent Metrics and Results

The system SHALL expose per-agent, aggregate, collective-behavior, and per-predator metrics on `MultiAgentEpisodeResult` so downstream evaluation, fitness functions, and CSV export consumers can read them uniformly.

#### Scenario: Per-Agent

- Each agent has independent EpisodeResult with termination_reason, path, food_history

#### Scenario: Aggregate

- `total_food_collected`, `per_agent_food`, `food_competition_events`, `proximity_events`
- `agents_alive_at_end`, `mean_agent_success`, `food_gini_coefficient`

#### Scenario: Collective Behavior Metrics

- `social_feeding_events`: count of step-agent pairs where decay reduction applied
- `aggregation_index`: mean normalized inverse pairwise distance (0=dispersed, 1=clustered)
- `alarm_evasion_events`: zone exits (concentration drops below ALARM_EVASION_THRESHOLD=0.1)
- `food_sharing_events`: non-emitter agent approaching food-marking source within FOOD_SHARING_LOOKBACK_STEPS=20

#### Scenario: Per-Predator Metrics

- `per_predator_kills`: dict keyed by `predator_id`, count of step-events where this predator's damage brought an agent's HP to 0 (multi-predator tie-break: closest-by-Manhattan, lex on `predator_id`)
- `per_predator_prey_proximity_steps`: dict keyed by `predator_id`, count of simulation steps where ≥1 alive agent was within this predator's `detection_radius` (NOT scaled by prey count)
- `per_predator_distance_traveled`: dict keyed by `predator_id`, sum of cardinal-direction position changes over the episode (stationary predators: 0; wall-blocked steps: 0)
