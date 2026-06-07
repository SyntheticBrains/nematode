# multi-agent Specification

## Purpose

This specification defines the multi-agent infrastructure for the Quantum Nematode project, enabling 2-10 independent agents to operate simultaneously in a shared environment. Multi-agent scenarios create exponential state spaces through agent-agent interactions, identified as the primary pathway for quantum advantage investigation (Phase 4). This spec covers agent orchestration, food competition, termination policies, pheromone emission integration, social feeding, collective behavior metrics, and aggregate results.

## Requirements

### Requirement: Multi-Agent Simulation Orchestration

The system SHALL provide a `MultiAgentSimulation` orchestrator that coordinates multiple independent agents in a shared environment using synchronous simultaneous stepping.

#### Scenario: Synchronous Step Loop

- All alive agents perceive, decide, and move simultaneously
- Food competition resolved after all movements
- Predators update once per step (not per agent)
- Effects (temperature, oxygen, satiety) applied per agent after predators
- Agents processed in sorted agent_id order (deterministic)

#### Scenario: Episode Lifecycle

- `prepare_episode()` called per brain at episode start
- STAM buffers and satiety reset per agent
- `learn(episode_done=True)` and `post_process_episode()` called at episode end

### Requirement: Food Competition Resolution

The system SHALL resolve contested food (multiple agents on the same food position) according to the configured competition policy, while agents at different food positions consume independently.

#### Scenario: FIRST_ARRIVAL Policy

- Lexicographically first agent_id wins contested food

#### Scenario: RANDOM Policy

- Winner selected via environment's seeded RNG

#### Scenario: Different Foods

- Agents at different food positions consume independently (no competition event)

### Requirement: Agent Termination Policy

The system SHALL apply the configured termination policy (Freeze, Remove, or End All) when an agent terminates, determining whether the agent remains on the grid, is removed, or ends the episode for all agents.

#### Scenario: Freeze (Default)

- Agent's `alive` set to False; remains on grid; no actions in subsequent steps
- Frozen agents still sensed via proximity; episode continues for alive agents

#### Scenario: Remove

- Agent removed from environment's agents dict; no longer sensed

#### Scenario: End All

- Any single agent termination ends the episode for all agents

### Requirement: Social Proximity Detection

The system SHALL count nearby agents per agent within `social_detection_radius` (Manhattan distance), excluding self and the "default" placeholder, counting frozen agents but not removed agents.

#### Scenario: Proximity Counting

- Nearby agents counted per agent within `social_detection_radius` (Manhattan distance)
- Self excluded; "default" placeholder excluded
- Frozen agents counted; removed agents not counted

### Requirement: Pheromone Emission in Step Loop

The system SHALL emit pheromones during the step loop — food-marking on consumption, alarm on predator damage, and continuous aggregation — and SHALL update the pheromone fields once per step, with all emission and update behavior a no-op when the corresponding feature is disabled.

#### Scenario: Food-Marking on Consumption

- FOOD_MARKING source added at consumed food position with consuming agent's emitter_id

#### Scenario: Alarm on Predator Damage

- ALARM source added at agent position when `apply_predator_damage_for()` returns damage > 0

#### Scenario: Continuous Aggregation Emission

- Each alive agent emits AGGREGATION pheromone at current position every step (after movement)
- No-op when aggregation not configured

#### Scenario: Field Update

- `update_pheromone_fields(current_step)` called once per step; no-op when pheromones disabled

### Requirement: Social Feeding in Step Loop

The system SHALL apply social feeding satiety-decay reduction during the step loop when social feeding is enabled and an agent has nearby agents, using the phenotype-appropriate multiplier and incrementing the `social_feeding_events` counter when applied.

#### Scenario: Decay Reduction

- When social feeding enabled and nearby_agents_count > 0:
  - Social phenotype: `decay_satiety(multiplier=decay_reduction)`
  - Solitary phenotype: `decay_satiety(multiplier=solitary_decay)`
- `social_feeding_events` counter incremented when multiplier applied

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
- **AND** each value SHALL be the count of step-events where this predator was the **attributed** killer per the "Multi-Predator Kill Attribution Rule" requirement (closest covering predator, lex tie-break on `predator_id`; defensive global-closest fallback when no predator covers)
- **AND** the sum of all values SHALL equal the total agent deaths attributable to predator damage in the episode (no double-counting, no missing kills — exactly one predator credited per kill event)

### Requirement: Multi-Predator Kill Attribution Rule

When `apply_predator_damage_for(aid)` brings an agent's HP to 0, the simulation (NOT the env) SHALL credit the kill to exactly one predator using a two-phase rule:

1. **Phase 1 — covering predators (primary)**: among predators whose Manhattan distance to the agent is `≤ predator.damage_radius`, select the closest by Manhattan distance. Tie-break on lexicographic `predator_id` (so `predator_0` beats `predator_1`).
2. **Phase 2 — defensive fallback**: if no covering predator is found (residual-HP edge case where the predator that originally caused the damage has since moved out of range), select the predator with smallest Manhattan distance among ALL predators (no `damage_radius` constraint), with the same lex tie-break. The simulation SHALL emit a debug-level log message (i.e. `logger.debug(...)`) so the fallback case is visible in forensic logs without raising the level on routine episodes.

The env-side `apply_predator_damage_for` applies a fixed damage tick without identifying the responsible predator; per-predator attribution is therefore performed by `MultiAgentSimulation._attribute_kill_to_predator(agent_position)` after each damage call by iterating the predator list and applying the rule above.

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

### Requirement: CSV Export of Collective Metrics

The system SHALL export collective metrics to `multi_agent_summary.csv` with the specified columns.

#### Scenario: Summary CSV Columns

- `multi_agent_summary.csv` columns include: run, total_food, competition_events, proximity_events, alive_at_end, mean_success, gini, social_feeding_events, aggregation_index, alarm_evasion_events, food_sharing_events

### Requirement: Agent Spawn Placement

The system SHALL place agents using Poisson disk sampling with `min_agent_distance` separation, never spawning on food or predator positions, and SHALL fall back to random valid positions on dense grids.

#### Scenario: Placement Strategy

- Poisson disk sampling with `min_agent_distance` separation
- No agent spawns on food or predator positions
- Fallback to random valid positions on dense grids

### Requirement: Grid Size Validation

The system SHALL enforce a minimum grid size of `max(5, ceil(5 * sqrt(num_agents)))`.

#### Scenario: Minimum Grid Size

- Minimum: `max(5, ceil(5 * sqrt(num_agents)))`

### Requirement: Reproducible Seeding

The system SHALL derive deterministic per-agent sub-seeds via `blake2b(session_seed + agent_id)` so that runs are reproducible, including that the FIRST_ARRIVAL policy awards the same agent the same contested food across reruns.

#### Scenario: Deterministic Seeding

- Per-agent sub-seeds: `blake2b(session_seed + agent_id)` deterministic
- FIRST_ARRIVAL policy: same agent wins same contested food across reruns

### Requirement: Territorial Index Metric

MultiAgentEpisodeResult SHALL include a territorial index measuring spatial foraging specialization.

#### Scenario: Territorial Index Computation

- Gini coefficient of per-agent foraging spreads (mean Manhattan distance of food collection positions from centroid)
- Value in \[0, 1\]: 0 = equal foraging patterns, 1 = maximal specialization
- 0.0 when fewer than 2 agents collected food
- Single food per agent: spread = 0.0

### Requirement: Alarm Response Rate Metric

MultiAgentEpisodeResult SHALL include an alarm response rate measuring causal reaction to alarm pheromone emissions.

#### Scenario: Alarm Response Rate Computation

- When alarm emitted at step T by agent A, count nearby agents (within `social_detection_radius`) that change direction within ALARM_RESPONSE_WINDOW (5) steps
- Rate = direction changes / opportunities; 0.0 when no opportunities
- Emitting agent excluded from response opportunities

### Requirement: CSV Export of Evaluation Metrics

The system SHALL include the territorial-index, alarm-response-rate, and per-simulation evaluation columns in the exported CSV files.

#### Scenario: Evaluation CSV Columns

- `multi_agent_summary.csv` SHALL include `territorial_index` and `alarm_response_rate` columns
- `simulation_results.csv` SHALL include `config_name`, `total_reward`, `success`, `satiety_remaining`, `foods_available` columns
