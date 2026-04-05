## Why

Multi-agent interaction is the primary pathway identified for quantum advantage investigation. Every prior phase has shown classical approaches matching quantum at current complexity levels (2-9D observations, 4 discrete actions, ~10K effective states). Multi-agent scenarios create exponential state spaces (state x number of agents), partial observability (each agent has local view), and strategic interactions -- all identified as quantum advantage thresholds in the Phase 3 quantum checkpoint.

The current architecture is fundamentally single-agent: `BaseEnvironment` stores one `agent_pos`, one `body`, one `current_direction`; `DynamicForagingEnvironment` adds one `agent_hp`, one `visited_cells`; all 20+ sensing methods implicitly use `self.agent_pos`. This change adds the infrastructure for 2-10 independent agents in the same environment -- the critical prerequisite for pheromone communication, social dynamics, and cooperative/competitive behaviors in later deliverables.

C. elegans exhibits well-documented social behaviors: aggregation on bacterial lawns, social feeding rate modulation, npr-1 variation determining solitary vs. social phenotypes, and alarm pheromone signaling. This infrastructure enables their simulation.

## What Changes

### 1. AgentState Dataclass

Extract per-agent mutable state from `BaseEnvironment` / `DynamicForagingEnvironment` into an `AgentState` dataclass holding: `agent_id`, `position`, `body`, `direction`, `hp`, `visited_cells`, `wall_collision_occurred`, `alive`, and per-agent comfort zone tracking counters (`steps_in_comfort_zone`, `total_thermotaxis_steps`, `steps_in_oxygen_comfort_zone`, `total_aerotaxis_steps`). The environment maintains `agents: dict[str, AgentState]` keyed by agent ID. Existing `self.agent_pos`, `self.body`, `self.current_direction`, `self.agent_hp`, and comfort counters become properties delegating to `agents["default"]` for backward compatibility.

### 2. Position-Parameterized Environment Methods

Add explicit `*_for(agent_id)` variants of all agent-implicit methods: `move_agent_for()`, `reached_goal_for()`, `get_separated_gradients_for()`, `get_food_concentration_for()`, `get_predator_concentration_for()`, `is_agent_at_boundary_for()`, `is_agent_in_damage_radius_for()`, `apply_predator_damage_for()`, `get_temperature_for()`, `get_oxygen_for()`, etc. These directly index the `agents` dict -- no implicit "active agent" context switching. Original methods remain as aliases delegating to the `"default"` agent.

### 3. Multi-Agent Predator Targeting

Pursuit predators chase the nearest alive agent (Manhattan distance) rather than a single hardcoded agent position. `Predator.update_position()` gains an `agent_positions` parameter accepting a list of positions. `DynamicForagingEnvironment.update_predators()` passes all alive agent positions (terminated/frozen agents are excluded from targeting). Single-agent behavior is identical when only one agent exists.

### 4. Social Proximity Observation

Add `nearby_agents_count: int | None` field to `BrainParams` representing how many other agents are within a configurable `social_detection_radius`. New `SOCIAL_PROXIMITY` sensory module with `classical_dim=1` returning normalized count (`min(count, 10) / 10.0`). This is the minimal social signal -- richer signals (pheromone gradients, agent identity) belong to the pheromone deliverable.

### 5. Food Competition Resolution

`FoodCompetitionPolicy` enum with `FIRST_ARRIVAL` (deterministic, sorted by agent_id) and `RANDOM` policies. When multiple agents occupy the same food cell after movement, the policy determines who consumes it. Food is consumed once per timestep -- no splitting.

### 6. MultiAgentSimulation Orchestrator

New `MultiAgentSimulation` class in `quantumnematode/agent/multi_agent.py` that owns a single `DynamicForagingEnvironment` and a list of `QuantumNematodeAgent` instances. Runs the synchronous step loop:

1. For each agent: build BrainParams, run brain, collect action
2. For each agent: apply movement via `env.move_agent_for(agent_id, action)`
3. Resolve food competition (agents at same food cell)
4. Update predators once (chase nearest agent)
5. For each agent: check predator damage, apply temperature/oxygen effects, decay satiety
6. Track per-agent + aggregate metrics

Terminated agents freeze in place (stop acting, remain on grid). Episode ends when all agents terminate or max_steps reached. Termination policy is configurable: `freeze`, `remove`, or `end_all`.

### 7. Multi-Agent Configuration

Optional `multi_agent` section in `SimulationConfig`:

- Shorthand for homogeneous populations: `count: N` uses top-level `brain` config for all agents
- Explicit heterogeneous: `agents` list with per-agent `id` and `brain` config
- Shared: all agents share `environment`, `reward`, `satiety`, `sensing`, `max_steps` config
- Per-agent optional `weights_path` for loading pre-trained models (curriculum training support)
- `food_competition`, `social_detection_radius`, `termination_policy`, `min_agent_distance` settings
- Grid size validation: `grid_size >= 5 * sqrt(num_agents)`
- Agent spawn placement via Poisson disk sampling with `min_agent_distance`

When `multi_agent` is absent or `enabled: false`, the simulation runs in single-agent mode identically to today.

### 8. Per-Agent Metrics and Exports

Each agent maintains its own `EpisodeTracker` and `MetricsTracker`. New `MultiAgentEpisodeResult` aggregates: total food collected, per-agent food share, food competition events, proximity events, per-agent termination results. CSV exports gain `agent_id` column in `simulation_results.csv` plus a `multi_agent_summary.csv` with aggregate stats. Model persistence uses `final_{agent_id}.pt` naming.

## Capabilities

**New**: `multi-agent` -- multi-agent orchestration, food competition, agent state management, agent-agent observation, aggregate metrics, per-agent model persistence, curriculum weight loading.

**Modified**: `environment-simulation` (AgentState extraction, position-parameterized methods, agent spawn placement), `brain-architecture` (nearby_agents_count in BrainParams, SOCIAL_PROXIMITY module), `configuration-system` (multi-agent config section).

## Impact

**Core code:**

- `quantumnematode/env/env.py` -- AgentState dataclass, `agents` dict, `*_for(agent_id)` methods, backward-compatible property aliases, agent spawn placement
- `quantumnematode/agent/multi_agent.py` -- NEW: MultiAgentSimulation, MultiAgentEpisodeResult, FoodCompetitionPolicy
- `quantumnematode/brain/arch/_brain.py` -- nearby_agents_count field in BrainParams
- `quantumnematode/brain/modules.py` -- SOCIAL_PROXIMITY sensory module
- `quantumnematode/utils/config_loader.py` -- AgentConfig, MultiAgentConfig, multi_agent in SimulationConfig
- `quantumnematode/utils/brain_factory.py` -- Weight loading per agent_id
- `quantumnematode/agent/agent.py` -- Accept agent_id parameter
- `scripts/run_simulation.py` -- Multi-agent branch, per-agent CSV, aggregate CSV

**Configs:**

- `configs/scenarios/multi_agent_foraging/` -- NEW: 2, 5, 10 agent scenario configs

**Tests:**

- `tests/.../agent/test_multi_agent.py` -- NEW: multi-agent orchestration tests
- `tests/.../env/test_env.py` -- AgentState and spawn placement tests

## Breaking Changes

None. All existing single-agent configs work unchanged. The `agent_pos`, `body`, `current_direction`, `agent_hp` properties delegate to the default agent. All existing tests pass without modification.

## Backward Compatibility

When no `multi_agent` section is present in config (or `enabled: false`), the simulation runs in single-agent mode identically to today. The environment creates a single `"default"` AgentState. All property accessors, existing methods, and the `StandardEpisodeRunner` operate through the default agent path. No existing code paths are modified -- only new code paths are added for multi-agent mode.

## Dependencies

None beyond existing Pydantic, NumPy.
