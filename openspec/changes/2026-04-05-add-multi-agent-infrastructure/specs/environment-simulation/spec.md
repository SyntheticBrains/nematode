## ADDED Requirements

### Requirement: AgentState Dataclass

The environment SHALL manage per-agent mutable state via an `AgentState` dataclass, decoupling agent state from the environment singleton.

#### Scenario: AgentState Fields

- **GIVEN** an AgentState instance
- **THEN** it SHALL contain the following fields:
  - `agent_id: str` -- unique identifier
  - `position: tuple[int, int]` -- current grid position
  - `body: list[tuple[int, int]]` -- body segment positions
  - `direction: Direction` -- current facing direction
  - `hp: float` -- current health points
  - `visited_cells: set[tuple[int, int]]` -- cells visited this episode
  - `wall_collision_occurred: bool` -- whether wall was hit this step (default False)
  - `alive: bool` -- whether agent is still active (default True)
  - `steps_in_comfort_zone: int` -- thermotaxis comfort zone step counter (default 0)
  - `total_thermotaxis_steps: int` -- total steps with thermotaxis active (default 0)
  - `steps_in_oxygen_comfort_zone: int` -- aerotaxis comfort zone step counter (default 0)
  - `total_aerotaxis_steps: int` -- total steps with aerotaxis active (default 0)

#### Scenario: Default Agent Backward Compatibility

- **GIVEN** an environment created without multi-agent configuration
- **WHEN** the environment is initialized
- **THEN** a single AgentState with agent_id="default" SHALL be created
- **AND** `self.agent_pos`, `self.body`, `self.current_direction` SHALL be properties delegating to the "default" AgentState
- **AND** all existing code using these properties SHALL work unchanged

### Requirement: Agent Registry

The environment SHALL maintain a registry of all agents.

#### Scenario: Agent Collection

- **GIVEN** a DynamicForagingEnvironment
- **THEN** `self.agents` SHALL be a `dict[str, AgentState]` keyed by agent_id

#### Scenario: Add Agent

- **GIVEN** a DynamicForagingEnvironment
- **WHEN** `add_agent(agent_id, position, max_body_length)` is called
- **THEN** a new AgentState SHALL be created with the specified position
- **AND** the agent SHALL have full HP, Direction.UP, empty body, alive=True
- **AND** the agent SHALL be added to `self.agents`

#### Scenario: Query Agents

- **WHEN** `get_agent_ids()` is called
- **THEN** it SHALL return a sorted list of all agent_id strings
- **WHEN** `get_agent_state(agent_id)` is called
- **THEN** it SHALL return the AgentState for that agent_id
- **AND** SHALL raise KeyError if agent_id is not found

### Requirement: Per-Agent Comfort Zone Tracking

Environment comfort zone tracking counters SHALL be per-agent, not global.

#### Scenario: Per-Agent Comfort Scores

- **GIVEN** a multi-agent environment with thermotaxis and aerotaxis enabled
- **WHEN** `apply_temperature_effects_for(agent_id)` is called
- **THEN** it SHALL update that agent's `steps_in_comfort_zone` and `total_thermotaxis_steps` counters
- **AND** SHALL NOT affect other agents' comfort tracking counters

#### Scenario: Backward Compatible Comfort Properties

- **GIVEN** a single-agent environment
- **WHEN** `self.steps_in_comfort_zone` is accessed
- **THEN** it SHALL delegate to `self.agents["default"].steps_in_comfort_zone`
- **AND** `get_temperature_comfort_score()` and `get_oxygen_comfort_score()` SHALL read from the default agent

### Requirement: Position-Parameterized Methods

The environment SHALL provide agent-specific variants of all agent-implicit methods.

#### Scenario: Movement by Agent ID

- **GIVEN** an environment with agents "agent_0" and "agent_1"
- **WHEN** `move_agent_for("agent_0", Action.FORWARD)` is called
- **THEN** only agent_0's position, body, and direction SHALL be updated
- **AND** agent_1's state SHALL be unchanged

#### Scenario: Goal Check by Agent ID

- **GIVEN** agent_0 at a food position and agent_1 not at food
- **WHEN** `reached_goal_for("agent_0")` is called
- **THEN** it SHALL return True
- **WHEN** `reached_goal_for("agent_1")` is called
- **THEN** it SHALL return False

#### Scenario: Gradient Queries by Agent ID

- **WHEN** `get_separated_gradients_for(agent_id)` is called
- **THEN** gradients SHALL be computed relative to the specified agent's position
- **AND** `get_food_concentration_for(agent_id)` SHALL return food concentration at that agent's position
- **AND** `get_predator_concentration_for(agent_id)` SHALL return predator concentration at that agent's position

#### Scenario: Boundary and Danger by Agent ID

- **WHEN** `is_agent_at_boundary_for(agent_id)` is called
- **THEN** it SHALL check the specified agent's position against grid boundaries
- **AND** `is_agent_in_danger_for(agent_id)` SHALL check that agent's position against predator detection radii
- **AND** `apply_predator_damage_for(agent_id)` SHALL apply damage to that agent's HP

#### Scenario: Temperature and Oxygen by Agent ID

- **WHEN** `get_temperature_for(agent_id)` is called
- **THEN** it SHALL return the temperature at the specified agent's position
- **AND** `get_oxygen_for(agent_id)` SHALL return oxygen at that agent's position
- **AND** `apply_temperature_effects_for(agent_id)` SHALL apply effects to that agent's HP
- **AND** `apply_oxygen_effects_for(agent_id)` SHALL apply effects to that agent's HP

#### Scenario: Food Consumption by Agent ID

- **WHEN** `consume_food_for(agent_id)` is called
- **AND** the specified agent is at a food position
- **THEN** the food SHALL be consumed and a new food SHALL spawn
- **AND** the consumed food position SHALL be returned

#### Scenario: Backward Compatible Originals

- **WHEN** the original `move_agent(action)` is called
- **THEN** it SHALL delegate to `move_agent_for("default", action)`
- **AND** the same delegation pattern SHALL apply to all other original methods

## MODIFIED Requirements

### Requirement: Predator Multi-Target Pursuit

Pursuit predators SHALL chase the nearest agent when multiple agents exist.

#### Scenario: Single Agent (Backward Compatible)

- **GIVEN** a single "default" agent
- **WHEN** `update_predators()` is called
- **THEN** pursuit predators SHALL chase the default agent's position
- **AND** behavior SHALL be identical to current single-agent code

#### Scenario: Multiple Agents

- **GIVEN** agents at positions (10, 10), (30, 30), and (50, 50), and a pursuit predator at (12, 12)
- **WHEN** `update_predators()` is called
- **THEN** the predator SHALL chase the nearest agent at (10, 10)
- **AND** distance SHALL be computed using Manhattan distance

#### Scenario: Only Alive Agents Targeted

- **GIVEN** the nearest agent is terminated (alive=False)
- **WHEN** predator targeting is computed
- **THEN** the predator SHALL chase the nearest ALIVE agent
- **AND** terminated agents SHALL be excluded from targeting
