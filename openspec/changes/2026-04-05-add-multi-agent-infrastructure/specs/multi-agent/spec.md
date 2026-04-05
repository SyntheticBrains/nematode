# multi-agent Specification (NEW)

## Purpose

This specification defines the multi-agent infrastructure for the Quantum Nematode project, enabling 2-10 independent agents to operate simultaneously in a shared environment. Multi-agent scenarios create exponential state spaces through agent-agent interactions, identified as the primary pathway for quantum advantage investigation (Phase 4). This spec covers agent orchestration, food competition, termination policies, and aggregate metrics -- but not pheromone communication or social behaviors (Phase 4 Deliverable 2).

## ADDED Requirements

### Requirement: Multi-Agent Simulation Orchestration

The system SHALL provide a `MultiAgentSimulation` orchestrator that coordinates multiple independent agents in a shared environment using synchronous simultaneous stepping.

#### Scenario: Synchronous Step Loop

- **GIVEN** a MultiAgentSimulation with N agents in a shared environment
- **WHEN** a simulation step is executed
- **THEN** all alive agents SHALL perceive their environment state simultaneously
- **AND** all alive agents SHALL select actions via their independent brains
- **AND** all movements SHALL be applied before any consequences are resolved
- **AND** food competition SHALL be resolved after all movements
- **AND** predators SHALL update once per step (not per agent)
- **AND** effects (temperature, oxygen, satiety) SHALL be applied per agent after predators

#### Scenario: Agent Processing Order

- **GIVEN** multiple agents with different agent_ids
- **WHEN** agents are processed within a step
- **THEN** agents SHALL be processed in sorted agent_id order
- **AND** this ordering SHALL be deterministic across runs with the same seed

#### Scenario: Episode Lifecycle

- **GIVEN** a multi-agent episode
- **WHEN** `run_episode()` is called
- **THEN** each agent's brain SHALL have `prepare_episode()` called at episode start
- **AND** each agent's STAM buffer SHALL be reset at episode start
- **AND** each agent's satiety SHALL be reset at episode start
- **AND** each agent's brain SHALL have `learn(episode_done=True)` called at episode end
- **AND** each agent's brain SHALL have `post_process_episode()` called at episode end

### Requirement: Food Competition Resolution

The system SHALL resolve conflicts when multiple agents attempt to consume the same food item in the same timestep.

#### Scenario: Two Agents at Same Food (FIRST_ARRIVAL Policy)

- **GIVEN** agents "agent_0" and "agent_1" both at a food position after movement
- **WHEN** food competition is resolved with FIRST_ARRIVAL policy
- **THEN** "agent_0" SHALL consume the food (lexicographically first)
- **AND** "agent_1" SHALL receive no food that step
- **AND** the food SHALL be removed from the environment exactly once

#### Scenario: Two Agents at Same Food (RANDOM Policy)

- **GIVEN** agents "agent_0" and "agent_1" both at a food position after movement
- **WHEN** food competition is resolved with RANDOM policy
- **THEN** exactly one agent SHALL be selected using the environment's seeded RNG
- **AND** the other agent SHALL receive no food that step

#### Scenario: Agents at Different Foods

- **GIVEN** agent_0 at food position A and agent_1 at food position B
- **WHEN** food competition is resolved
- **THEN** both agents SHALL independently consume their respective food
- **AND** no competition event SHALL be recorded

#### Scenario: Food Competition Tracking

- **GIVEN** a multi-agent episode
- **WHEN** food competition events occur
- **THEN** the total count of competition events SHALL be tracked in MultiAgentEpisodeResult
- **AND** a competition event SHALL be counted each time 2+ agents occupy the same food cell

### Requirement: Agent Termination Policy

The system SHALL support configurable termination policies for when individual agents die during a multi-agent episode.

#### Scenario: Freeze Policy (Default)

- **GIVEN** a multi-agent simulation with termination_policy="freeze"
- **WHEN** an agent terminates (starved, health depleted, predator death)
- **THEN** the agent's `alive` state SHALL be set to False
- **AND** the agent SHALL remain on the grid at its current position
- **AND** the agent SHALL not move or take actions in subsequent steps
- **AND** other agents SHALL still be able to sense the frozen agent via proximity
- **AND** the episode SHALL continue for remaining alive agents

#### Scenario: Remove Policy

- **GIVEN** a multi-agent simulation with termination_policy="remove"
- **WHEN** an agent terminates
- **THEN** the agent SHALL be removed from the environment's agents dict
- **AND** other agents SHALL no longer sense the removed agent

#### Scenario: End All Policy

- **GIVEN** a multi-agent simulation with termination_policy="end_all"
- **WHEN** any single agent terminates
- **THEN** all agents SHALL be immediately terminated
- **AND** the episode SHALL end

#### Scenario: Episode Completion

- **GIVEN** a multi-agent episode with freeze or remove policy
- **WHEN** all agents have terminated OR max_steps is reached
- **THEN** the episode SHALL end
- **AND** MultiAgentEpisodeResult SHALL be returned with per-agent results

### Requirement: Social Proximity Detection

The system SHALL compute the number of nearby agents for each agent as a minimal social signal.

#### Scenario: Nearby Agent Counting

- **GIVEN** agent_0 at position (10, 10) and agent_1 at position (13, 10) with social_detection_radius=5
- **WHEN** nearby agent count is computed for agent_0
- **THEN** the count SHALL be 1 (Manhattan distance 3 \<= 5)

#### Scenario: No Nearby Agents

- **GIVEN** agent_0 at (10, 10) and agent_1 at (50, 50) with social_detection_radius=5
- **WHEN** nearby agent count is computed for agent_0
- **THEN** the count SHALL be 0

#### Scenario: Terminated Agents in Freeze Mode

- **GIVEN** a frozen (terminated) agent within detection radius
- **WHEN** nearby agent count is computed for an alive agent
- **THEN** the frozen agent SHALL be counted in the proximity total

#### Scenario: Self-Exclusion

- **WHEN** nearby agent count is computed for agent_0
- **THEN** agent_0 SHALL NOT count itself in the result

### Requirement: Multi-Agent Metrics and Results

The system SHALL provide per-agent and aggregate metrics for multi-agent episodes.

#### Scenario: Per-Agent Metrics

- **GIVEN** a completed multi-agent episode
- **WHEN** results are computed
- **THEN** each agent SHALL have its own EpisodeResult with termination_reason, path, and food_history
- **AND** each agent SHALL have independent success/failure determination

#### Scenario: Aggregate Metrics

- **GIVEN** a completed multi-agent episode with 5 agents
- **WHEN** MultiAgentEpisodeResult is computed
- **THEN** it SHALL include `total_food_collected` (sum across all agents)
- **AND** `per_agent_food: dict[str, int]` (food per agent)
- **AND** `food_competition_events: int` (contested food count)
- **AND** `proximity_events: int` (steps where 2+ agents within detection radius)
- **AND** `agents_alive_at_end: int`
- **AND** `mean_agent_success: float` (fraction of agents that succeeded)
- **AND** `food_gini_coefficient: float` (0=equal distribution, 1=monopoly)

### Requirement: Agent Spawn Placement

The system SHALL place multiple agents with minimum separation using Poisson disk sampling.

#### Scenario: Minimum Agent Distance

- **GIVEN** 5 agents with min_agent_distance=5
- **WHEN** agents are spawned
- **THEN** all agent pairs SHALL have Manhattan distance >= 5

#### Scenario: Avoid Food and Predator Positions

- **WHEN** agent spawn positions are computed
- **THEN** no agent SHALL spawn on an existing food position
- **AND** no agent SHALL spawn on an existing predator position

#### Scenario: Fallback on Dense Grids

- **GIVEN** a grid too dense for Poisson disk sampling to find valid positions
- **WHEN** sampling exhausts maximum attempts
- **THEN** the system SHALL fall back to random valid positions (no overlap with food/predators)

### Requirement: Grid Size Validation

The system SHALL validate that the grid is large enough for the number of agents.

#### Scenario: Grid Too Small

- **GIVEN** grid_size=10 and num_agents=10
- **WHEN** validation is performed
- **THEN** the system SHALL raise ValueError
- **AND** the error message SHALL include the minimum required grid size

#### Scenario: Minimum Size Formula

- **GIVEN** num_agents agents
- **WHEN** grid size is validated
- **THEN** the minimum grid size SHALL be `max(5, ceil(5 * sqrt(num_agents)))`

### Requirement: Reproducible Multi-Agent Seeding

The system SHALL provide deterministic, reproducible multi-agent simulations.

#### Scenario: Per-Agent Sub-Seeds

- **GIVEN** a session seed and multiple agents
- **WHEN** agents are initialized
- **THEN** each agent SHALL receive a deterministic sub-seed derived from `hash((session_seed, agent_id)) % 2^32`
- **AND** results SHALL be identical across runs with the same session seed

#### Scenario: Food Competition Determinism

- **GIVEN** FIRST_ARRIVAL food competition policy
- **WHEN** the same scenario is run twice with the same seed
- **THEN** the same agent SHALL win each contested food in both runs
