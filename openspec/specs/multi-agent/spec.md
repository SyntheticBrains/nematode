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

#### Scenario: FIRST_ARRIVAL Policy

- Lexicographically first agent_id wins contested food

#### Scenario: RANDOM Policy

- Winner selected via environment's seeded RNG

#### Scenario: Different Foods

- Agents at different food positions consume independently (no competition event)

### Requirement: Agent Termination Policy

#### Scenario: Freeze (Default)

- Agent's `alive` set to False; remains on grid; no actions in subsequent steps
- Frozen agents still sensed via proximity; episode continues for alive agents

#### Scenario: Remove

- Agent removed from environment's agents dict; no longer sensed

#### Scenario: End All

- Any single agent termination ends the episode for all agents

### Requirement: Social Proximity Detection

- Nearby agents counted per agent within `social_detection_radius` (Manhattan distance)
- Self excluded; "default" placeholder excluded
- Frozen agents counted; removed agents not counted

### Requirement: Pheromone Emission in Step Loop

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

#### Scenario: Decay Reduction

- When social feeding enabled and nearby_agents_count > 0:
  - Social phenotype: `decay_satiety(multiplier=decay_reduction)`
  - Solitary phenotype: `decay_satiety(multiplier=solitary_decay)`
- `social_feeding_events` counter incremented when multiplier applied

### Requirement: Multi-Agent Metrics and Results

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

### Requirement: CSV Export of Collective Metrics

- `multi_agent_summary.csv` columns include: run, total_food, competition_events, proximity_events, alive_at_end, mean_success, gini, social_feeding_events, aggregation_index, alarm_evasion_events, food_sharing_events

### Requirement: Agent Spawn Placement

- Poisson disk sampling with `min_agent_distance` separation
- No agent spawns on food or predator positions
- Fallback to random valid positions on dense grids

### Requirement: Grid Size Validation

- Minimum: `max(5, ceil(5 * sqrt(num_agents)))`

### Requirement: Reproducible Seeding

- Per-agent sub-seeds: `blake2b(session_seed + agent_id)` deterministic
- FIRST_ARRIVAL policy: same agent wins same contested food across reruns
