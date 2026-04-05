# Tasks: Multi-Agent Infrastructure

## Phase 1: AgentState Extraction

**Dependencies**: None
**Parallelizable**: No (foundational change)

- [ ] 1.1 Create `AgentState` dataclass in `quantumnematode/env/env.py`

  - Fields: `agent_id: str`, `position: tuple[int, int]`, `body: list[tuple[int, int]]`, `direction: Direction`, `hp: float`, `visited_cells: set[tuple[int, int]]`, `wall_collision_occurred: bool = False`, `alive: bool = True`
  - Validation: AgentState is a plain dataclass (not Pydantic) to match existing Predator pattern

- [ ] 1.2 Add `agents: dict[str, AgentState]` to `BaseEnvironment.__init__()`

  - Create single `"default"` AgentState from existing `start_pos`, `max_body_length` params
  - Initialize with `direction=Direction.UP`, `visited_cells={start_pos}`, `alive=True`

- [ ] 1.3 Convert `agent_pos` to a property delegating to `self.agents["default"].position`

  - Both getter and setter
  - Same for `body` and `current_direction`

- [ ] 1.4 Convert `agent_hp` in `DynamicForagingEnvironment` to a property delegating to `self.agents["default"].hp`

  - Same for `visited_cells` and `wall_collision_occurred`
  - Ensure `__init__` sets via the AgentState, not directly

- [ ] 1.5 Run full test suite (`uv run pytest -m "not nightly"`) -- all existing tests must pass unchanged

  - This validates backward compatibility of the property delegation

## Phase 2: Position-Parameterized Environment Methods

**Dependencies**: Phase 1
**Parallelizable**: Yes (each method is independent)

- [ ] 2.1 Refactor `move_agent()` internals to accept position/body/direction from an AgentState

  - Extract core logic into `_apply_movement(agent_state: AgentState, action: Action)`
  - `move_agent()` calls `_apply_movement(self.agents["default"], action)`
  - Add `move_agent_for(agent_id: str, action: Action)` calling `_apply_movement(self.agents[agent_id], action)`

- [ ] 2.2 Add `reached_goal_for(agent_id: str) -> bool`

  - Check if `agents[agent_id].position` is at any food position
  - Original `reached_goal()` delegates to `reached_goal_for("default")`

- [ ] 2.3 Refactor gradient computation to accept position parameter

  - `_compute_food_gradient(position) -> GradientPolar` (extracted from `get_state`)
  - `_compute_predator_gradient(position) -> GradientPolar`
  - `get_separated_gradients_for(agent_id, ...)`, `get_food_concentration_for(agent_id)`, `get_predator_concentration_for(agent_id)`
  - Originals delegate to `*_for("default")`

- [ ] 2.4 Add boundary and danger checks for specific agents

  - `is_agent_at_boundary_for(agent_id)`, `is_agent_in_danger_for(agent_id)`, `is_agent_in_damage_radius_for(agent_id)`
  - `apply_predator_damage_for(agent_id) -> float`
  - Originals delegate to `*_for("default")`

- [ ] 2.5 Add temperature and oxygen accessors for specific agents

  - `get_temperature_for(agent_id)`, `get_temperature_gradient_for(agent_id)`, `get_temperature_zone_for(agent_id)`
  - `get_oxygen_for(agent_id)`, `get_oxygen_gradient_for(agent_id)`, `get_oxygen_zone_for(agent_id)`
  - `apply_temperature_effects_for(agent_id)`, `apply_oxygen_effects_for(agent_id)`
  - Originals delegate to `*_for("default")`

- [ ] 2.6 Add `consume_food_for(agent_id) -> tuple[int, int] | None`

  - Consume food at specified agent's position
  - Original delegates to `consume_food_for("default")`

- [ ] 2.7 Add `add_agent(agent_id: str, position: tuple[int, int] | None, max_body_length: int)` method

  - Spawns agent at given position (or random valid position)
  - Initializes AgentState with full HP, empty body, Direction.UP
  - Add `get_agent_ids() -> list[str]` and `get_agent_state(agent_id) -> AgentState`

- [ ] 2.8 Run full test suite -- all existing tests must pass unchanged

## Phase 3: Multi-Agent Predator Targeting

**Dependencies**: Phase 1
**Parallelizable**: Yes (with Phase 2)

- [ ] 3.1 Add `agent_positions: list[tuple[int, int]] | None = None` parameter to `Predator.update_position()`

  - When provided, pursuit predators chase nearest position (Manhattan distance)
  - When only `agent_pos` provided, behavior identical to current code
  - Both parameters optional; `agent_positions` takes precedence when both provided

- [ ] 3.2 Update `DynamicForagingEnvironment.update_predators()` to pass all alive agent positions

  - Collect positions from `self.agents.values()` where `alive=True`
  - Pass as `agent_positions` list
  - Single-agent backward compat: when only `"default"` agent exists, behavior is identical

- [ ] 3.3 Add tests for multi-target pursuit

  - Predator chases nearest of 3 agents
  - Predator switches target when agents move
  - Stationary predators unaffected by multi-agent

## Phase 4: BrainParams Social Field and Sensory Module

**Dependencies**: None (independent of env changes)
**Parallelizable**: Yes (with Phases 1-3)

- [ ] 4.1 Add `nearby_agents_count: int | None = None` field to `BrainParams` in `brain/arch/_brain.py`

  - Default None (no social signal in single-agent mode)
  - No impact on existing code (field is optional with None default)

- [ ] 4.2 Add `SOCIAL_PROXIMITY = "social_proximity"` to `ModuleName` enum in `brain/modules.py`

- [ ] 4.3 Implement `_social_proximity_core(params: BrainParams) -> CoreFeatures`

  - `strength = min(count, 10) / 10.0` (normalized agent count)
  - `angle = 0.0`, `binary = 0.0`
  - Register in `SENSORY_MODULES` dict with `classical_dim=1`

- [ ] 4.4 Add tests for social_proximity module

  - 0 nearby agents -> strength=0.0
  - 5 nearby agents -> strength=0.5
  - 10+ nearby agents -> strength=1.0 (clamped)
  - None nearby_agents_count -> strength=0.0

## Phase 5: Food Competition Resolution

**Dependencies**: Phase 2 (needs `consume_food_for`)
**Parallelizable**: No

- [ ] 5.1 Create `FoodCompetitionPolicy` enum in `quantumnematode/agent/multi_agent.py`

  - Values: `FIRST_ARRIVAL`, `RANDOM`

- [ ] 5.2 Implement `resolve_food_competition()`

  - Input: `contested: dict[GridPosition, list[str]]` (food position -> agent_ids present)
  - Input: `policy: FoodCompetitionPolicy`, `rng: np.random.Generator`
  - Output: `dict[str, GridPosition | None]` (agent_id -> food they get, or None)
  - `FIRST_ARRIVAL`: winner is first in sorted agent_ids
  - `RANDOM`: winner is random choice from contestants

- [ ] 5.3 Add tests for both policies

  - 2 agents at same food: one wins, one gets None
  - 3 agents at same food: only one wins
  - 2 agents at different foods: both win
  - `FIRST_ARRIVAL` deterministic ordering
  - `RANDOM` uses provided rng

## Phase 6: Agent Spawn Placement

**Dependencies**: Phase 2 (needs `add_agent`)
**Parallelizable**: Yes (with Phase 5)

- [ ] 6.1 Implement `_initialize_agent_positions()` in `DynamicForagingEnvironment`

  - Poisson disk sampling with `min_agent_distance` separation
  - Avoid food and predator positions
  - Fallback to random valid positions if sampling exhausts attempts
  - Return list of positions

- [ ] 6.2 Add grid size validation

  - `grid_size >= 5 * sqrt(num_agents)` check
  - Clear error message with minimum size recommendation

- [ ] 6.3 Add tests for spawn placement

  - Agents spawn with minimum separation
  - Agents don't spawn on food or predators
  - Validation rejects too-small grids

## Phase 7: MultiAgentSimulation Orchestrator

**Dependencies**: Phases 1-6
**Parallelizable**: No (integrates all prior work)

- [ ] 7.1 Create `quantumnematode/agent/multi_agent.py` with core classes

  - `FoodCompetitionPolicy` (from Phase 5)
  - `MultiAgentEpisodeResult` dataclass
  - `MultiAgentSimulation` class

- [ ] 7.2 Implement `MultiAgentSimulation.__init__()`

  - Accepts: `env: DynamicForagingEnvironment`, `agents: list[QuantumNematodeAgent]`, `food_policy: FoodCompetitionPolicy`, `social_detection_radius: int`, `termination_policy: str`
  - Validates: all agents have unique agent_ids, agent_ids match env's agents dict

- [ ] 7.3 Implement `_compute_nearby_agents_count(agent_id: str) -> int`

  - Count other alive agents within `social_detection_radius` (Manhattan distance)
  - O(n) per agent call

- [ ] 7.4 Implement `step()` method with synchronous step order

  - 1: Perception + decision for each alive agent
  - 2: Movement for each alive agent
  - 3: Food competition resolution
  - 4: Update predators once
  - 5: Per-agent effects (damage, temperature, oxygen, satiety)
  - 6: Termination checks -- set `alive=False` on terminated agents
  - Return per-agent step results

- [ ] 7.5 Implement `run_episode()` method

  - Calls `step()` in loop until all agents terminated or max_steps
  - Handles episode preparation (brain.prepare_episode, stam.reset, satiety reset per agent)
  - Handles episode cleanup (brain.learn final, brain.post_process_episode per agent)
  - Returns `MultiAgentEpisodeResult`

- [ ] 7.6 Implement `_handle_agent_termination(agent_id, reason)`

  - Set `AgentState.alive = False`
  - Record termination in per-agent tracker
  - `freeze` policy: agent remains in agents dict, position fixed
  - `remove` policy: remove from agents dict
  - `end_all` policy: mark all agents as terminated

- [ ] 7.7 Implement aggregate metrics computation

  - `total_food_collected`, `per_agent_food`, `food_competition_events`, `proximity_events`
  - `agents_alive_at_end`, `mean_agent_success`
  - `food_gini_coefficient` (distribution equality)

- [ ] 7.8 Add comprehensive tests for MultiAgentSimulation

  - 2-agent episode runs to completion
  - 5-agent episode runs to completion
  - Food competition tracked correctly
  - Terminated agents freeze in place
  - Proximity events counted
  - All termination policies work

## Phase 8: QuantumNematodeAgent Multi-Agent Awareness

**Dependencies**: Phase 2
**Parallelizable**: Yes (with Phases 5-7)

- [ ] 8.1 Add optional `agent_id: str = "default"` parameter to `QuantumNematodeAgent.__init__()`

  - Store as `self.agent_id`
  - No behavior change when `"default"`

- [ ] 8.2 Add `nearby_agents_count` injection point in `_create_brain_params()`

  - Add optional `nearby_agents_count: int | None = None` parameter
  - Set in BrainParams when provided (by orchestrator)
  - None when not provided (single-agent mode)

- [ ] 8.3 Verify per-agent component isolation

  - Each agent has independent `_episode_tracker`, `_satiety_manager`, `_metrics_tracker`, `_reward_calculator`, `_food_handler`, `_stam`
  - These are already per-agent by construction -- just verify with test

## Phase 9: Configuration Layer

**Dependencies**: Phase 7 (needs MultiAgentSimulation)
**Parallelizable**: No

- [ ] 9.1 Create `AgentConfig` Pydantic model in `config_loader.py`

  - `id: str`
  - `brain: BrainContainerConfig`
  - `weights_path: str | None = None`

- [ ] 9.2 Create `MultiAgentConfig` Pydantic model

  - `enabled: bool = False`
  - `count: int | None = None` (shorthand)
  - `agents: list[AgentConfig] | None = None` (explicit)
  - `food_competition: str = "first_arrival"`
  - `social_detection_radius: int = 5`
  - `termination_policy: str = "freeze"`
  - `min_agent_distance: int = 5`
  - Validator: exactly one of `count` or `agents` when `enabled=True`

- [ ] 9.3 Add `multi_agent: MultiAgentConfig | None = None` to `SimulationConfig`

- [ ] 9.4 Implement `configure_multi_agent()` factory function

  - When `count` set: create N agents with top-level `brain` config, auto-generate agent_ids (`agent_0`, `agent_1`, ...)
  - When `agents` set: create agents with per-agent brain configs
  - Each agent gets deterministic sub-seed from session seed + agent_id
  - Load weights from `weights_path` if specified

- [ ] 9.5 Add tests for config loading

  - Shorthand form (count=5)
  - Explicit form (3 different brains)
  - Validation: enabled without count or agents -> error
  - Validation: both count and agents -> error
  - weights_path specified but file missing -> clear error

## Phase 10: Model Persistence

**Dependencies**: Phase 8
**Parallelizable**: Yes (with Phase 9)

- [ ] 10.1 Update weight save path to include agent_id

  - `final.pt` when `agent_id == "default"` (backward compat)
  - `final_{agent_id}.pt` otherwise
  - Update `brain_factory.py` or relevant save/load utilities

- [ ] 10.2 Add weight loading support per agent

  - Load from `weights_path` config if specified
  - Apply after brain construction, before training

- [ ] 10.3 Add tests for multi-agent weight save/load

  - Save 3 agents -> 3 separate .pt files
  - Load pre-trained weights into multi-agent config
  - Single-agent save -> `final.pt` unchanged

## Phase 11: Simulation Script Integration

**Dependencies**: Phases 7, 9
**Parallelizable**: No

- [ ] 11.1 Add multi-agent branch in `scripts/run_simulation.py`

  - Detect `multi_agent.enabled` after config loading
  - Single-agent path: unchanged
  - Multi-agent path: create env, call `configure_multi_agent()`, create `MultiAgentSimulation`, run episodes

- [ ] 11.2 Multi-agent episode loop

  - For each run: `multi_sim.run_episode()` -> `MultiAgentEpisodeResult`
  - Per-agent and aggregate logging
  - Reset between episodes

- [ ] 11.3 CSV export with agent_id column

  - `simulation_results.csv`: one row per agent per episode
  - `multi_agent_summary.csv`: one row per episode with aggregates

- [ ] 11.4 Multi-agent console output

  - Per-agent success rates, food collected
  - Aggregate stats: total food, competition events, Gini coefficient
  - Summary table at end of session

## Phase 12: Scenario Configurations

**Dependencies**: Phase 9
**Parallelizable**: Yes (with Phase 11)

- [ ] 12.1 Create `configs/scenarios/multi_agent_foraging/` directory

- [ ] 12.2 Create `mlpppo_medium_2agents_oracle.yml`

  - 2 agents, MLP PPO, 50x50, oracle sensing
  - 10 food items, 20 target foods
  - Baseline: simplest multi-agent scenario

- [ ] 12.3 Create `mlpppo_medium_5agents_oracle.yml`

  - 5 agents, MLP PPO, 50x50, oracle sensing
  - 15 food items, 30 target foods
  - Tests scaling and food competition

- [ ] 12.4 Create `mlpppo_large_10agents_oracle.yml`

  - 10 agents, MLP PPO, 100x100, oracle sensing
  - 30 food items, 60 target foods
  - Tests maximum agent count

- [ ] 12.5 Create `mixed_brains_medium_3agents_oracle.yml`

  - 3 agents: mlpppo + lstmppo + qrh
  - 50x50, oracle sensing, 10 food items
  - Tests heterogeneous brain architectures

## Phase 13: Verification and Documentation

**Dependencies**: All phases
**Parallelizable**: Partially

- [ ] 13.1 Run `uv run pytest -m "not nightly"` -- ALL tests pass (existing + new)

- [ ] 13.2 Run `uv run pre-commit run -a` -- all linting/formatting passes

- [ ] 13.3 Backward compatibility smoke test

  - Run existing `configs/scenarios/foraging/mlpppo_small_oracle.yml` with 10 runs
  - Verify identical behavior (same success rate, same metrics structure)

- [ ] 13.4 Multi-agent smoke test: 2 agents

  - Run `mlpppo_medium_2agents_oracle.yml` with 100 episodes
  - Verify: completes without errors, per-agent CSV generated, aggregate CSV generated

- [ ] 13.5 Multi-agent stability test: 5 agents

  - Run `mlpppo_medium_5agents_oracle.yml` with 500 episodes
  - Verify: all agents produce valid results, no crashes

- [ ] 13.6 Multi-agent scaling test: 10 agents

  - Run `mlpppo_large_10agents_oracle.yml` with 100 episodes
  - Verify: performance linear in agent count (measure wall-clock per step)

- [ ] 13.7 Mixed brain test

  - Run `mixed_brains_medium_3agents_oracle.yml` with 100 episodes
  - Verify: all brain types produce valid results in multi-agent context

- [ ] 13.8 Food competition correctness

  - Log competition events across multi-agent runs
  - Verify: food is never double-consumed, competition counts are accurate

- [ ] 13.9 Model persistence test

  - Run 3-agent session, verify `final_agent_0.pt`, `final_agent_1.pt`, `final_agent_2.pt` created
  - Run single-agent session, verify `final.pt` still created (backward compat)

- [ ] 13.10 Update `docs/roadmap.md`

  - Mark Phase 4 Deliverable 1 status as in-progress
  - Add multi-agent infrastructure to Current State section when complete

______________________________________________________________________

## Summary

| Phase | Tasks | Dependencies | Parallelizable |
|-------|-------|-------------|----------------|
| 1. AgentState Extraction | 5 | None | No |
| 2. Position-Parameterized Methods | 8 | Phase 1 | Yes |
| 3. Predator Multi-Target | 3 | Phase 1 | Yes (with 2) |
| 4. Social Proximity Module | 4 | None | Yes (with 1-3) |
| 5. Food Competition | 3 | Phase 2 | No |
| 6. Agent Spawn Placement | 3 | Phase 2 | Yes (with 5) |
| 7. Multi-Agent Orchestrator | 8 | Phases 1-6 | No |
| 8. Agent Multi-Agent Awareness | 3 | Phase 2 | Yes (with 5-7) |
| 9. Configuration Layer | 5 | Phase 7 | No |
| 10. Model Persistence | 3 | Phase 8 | Yes (with 9) |
| 11. Script Integration | 4 | Phases 7, 9 | No |
| 12. Scenario Configs | 5 | Phase 9 | Yes (with 11) |
| 13. Verification | 10 | All | Partially |

**Total: 64 tasks across 13 phases**
