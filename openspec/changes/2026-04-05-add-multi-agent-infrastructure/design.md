## Overview

The simulation currently assumes a single agent throughout the entire stack: environment state (`agent_pos`, `body`, `current_direction`, `agent_hp`), environment methods (20+ references to `self.agent_pos`), the runner's step loop, food handling, reward calculation, and metrics tracking. Adding multi-agent support requires decoupling per-agent state from the environment singleton while preserving the single-agent API.

The predator system provides a useful pattern: predators are independent entities with positions managed in a list (`self.predators: list[Predator]`), each updated per step via `update_predators()`. Agents follow a similar entity-in-environment pattern.

## Goals / Non-Goals

**Goals:**

- 2-10 independent agents running stably in the same environment with independent brains
- Simultaneous (synchronous) stepping -- all agents act per timestep
- Food competition resolution when multiple agents reach the same food
- Per-agent metrics + aggregate multi-agent metrics
- O(n) performance in agent count
- Full backward compatibility with existing single-agent configs and tests
- Curriculum support via optional pre-trained weight loading

**Non-Goals:**

- Pheromone communication (Phase 4 Deliverable 2)
- Agent-agent communication channels beyond proximity count
- Heterogeneous environments per agent (all share the same fields)
- Turn-based stepping
- Agent reproduction or death/respawn (terminated agents freeze)
- Multi-agent rendering (headless only for this change)
- Collaborative/team reward structures (per-agent independent scoring)
- ManyworldsEpisodeRunner support (quantum branching is single-agent only; multi-agent uses MultiAgentSimulation exclusively)

**Constraints:**

- Multi-agent mode always passes the shared `DynamicForagingEnvironment` explicitly to each `QuantumNematodeAgent`. Agents MUST NOT self-create environments (`env` parameter is required, not optional, in multi-agent context).
- Weight save uses `save_weights()` from `brain/weights.py`; weight load uses `load_weights()` from the same module. The call sites are in `scripts/run_simulation.py` (save at line ~845) and `utils/brain_factory.py` / `run_simulation.py` (load).
- CSV export logic lives in `quantumnematode/report/csv_export.py`, called from `run_simulation.py`.

## Component Hierarchy

```text
SimulationConfig
  |-- multi_agent: MultiAgentConfig
  |     |-- agents: list[AgentConfig]    (or count: int for homogeneous)
  |     |-- food_competition: str
  |     |-- social_detection_radius: int
  |     |-- termination_policy: str
  |     |-- min_agent_distance: int
  |
  v
run_simulation.py
  |-- (multi_agent.enabled?) --yes--> MultiAgentSimulation
  |                                     |-- env: DynamicForagingEnvironment
  |                                     |     |-- agents: dict[str, AgentState]
  |                                     |     |-- foods, predators, fields (shared)
  |                                     |
  |                                     |-- agents: list[QuantumNematodeAgent]
  |                                     |     |-- each has own brain, tracker, satiety, stam
  |                                     |
  |                                     |-- step loop (synchronous)
  |                                     |-- food competition resolver
  |                                     |-- aggregate metrics
  |
  |-- (multi_agent.enabled?) --no---> QuantumNematodeAgent (unchanged)
                                        |-- StandardEpisodeRunner (unchanged)
```

## Design Decisions

### Decision 1: AgentState Extraction

Extract per-agent mutable state into a dataclass within `env.py`:

```python
@dataclass
class AgentState:
    agent_id: str
    position: tuple[int, int]
    body: list[tuple[int, int]]
    direction: Direction
    hp: float
    visited_cells: set[tuple[int, int]]
    wall_collision_occurred: bool = False
    alive: bool = True  # False when terminated (frozen in place)
    # Per-agent comfort tracking (moved from env globals)
    steps_in_comfort_zone: int = 0
    total_thermotaxis_steps: int = 0
    steps_in_oxygen_comfort_zone: int = 0
    total_aerotaxis_steps: int = 0
```

`BaseEnvironment.__init__` creates `self.agents: dict[str, AgentState]` with a single `"default"` entry initialized from existing `start_pos`, `max_body_length` params. Existing properties become:

```python
@property
def agent_pos(self) -> tuple[int, int]:
    return self.agents["default"].position

@agent_pos.setter
def agent_pos(self, value: tuple[int, int]) -> None:
    self.agents["default"].position = value
```

Same pattern for `body`, `current_direction`, `agent_hp`, `visited_cells`, `wall_collision_occurred`.

**Why not context switching**: An "active agent" pattern (`set_active_agent(id)` then call methods) is fragile -- if any code path forgets to restore context, bugs are silent and non-deterministic. Explicit `agent_id` parameters are self-documenting and safe.

### Decision 2: Explicit `*_for(agent_id)` Methods

Each agent-implicit method gets a `*_for` variant that directly indexes the agents dict. The implementation strategy depends on whether the existing method already accepts a position parameter:

**Methods that already accept `position` parameter** (minimal change): `get_food_concentration()`, `get_predator_concentration()`, `get_temperature()`, `get_temperature_gradient()`, `get_oxygen()`, `get_oxygen_gradient()`, `get_temperature_zone()`, `get_oxygen_zone()`. Also `get_separated_gradients()` where position is a required parameter. For these, the `*_for` variant simply resolves the agent_id to a position and calls the existing method:

```python
def get_food_concentration_for(self, agent_id: str) -> float:
    return self.get_food_concentration(position=self.agents[agent_id].position)
```

**Methods that need actual refactoring** (use `self.agent_pos` or `self.agent_hp` internally): `move_agent()`, `reached_goal()`, `consume_food()`, `is_agent_in_danger()`, `is_agent_in_damage_radius()`, `is_agent_at_boundary()`, `apply_predator_damage()`, `apply_temperature_effects()`, `apply_oxygen_effects()`. For these, internal logic is extracted to operate on an AgentState:

```python
def move_agent_for(self, agent_id: str, action: Action) -> None:
    agent_state = self.agents[agent_id]
    # Core logic extracted to _apply_movement(agent_state, action)

def move_agent(self, action: Action) -> None:
    self.move_agent_for("default", action)

def reached_goal_for(self, agent_id: str) -> bool:
    return self.agents[agent_id].position in self._food_positions_set
```

**`apply_temperature_effects_for()` / `apply_oxygen_effects_for()`**: These methods also update comfort zone tracking counters (`steps_in_comfort_zone`, `total_thermotaxis_steps`, etc.) and write to `agent_hp`. In multi-agent mode, these counters are per-agent (stored in `AgentState`), and HP is written to the specified agent's state.

This avoids code duplication while preserving backward compatibility.

### Decision 3: Synchronous Simultaneous Stepping

All agents choose actions simultaneously, then all actions are applied, then consequences resolve:

```text
MultiAgentSimulation.step():
  +--------------------------+
  | 1. PERCEPTION            |  For each agent (sorted by agent_id):
  |    Build BrainParams     |    - env.*_for(agent_id) to read state
  |    Run brain             |    - brain.run_brain(params) -> action
  |    Collect action        |    - Skip frozen (terminated) agents
  +--------------------------+
              |
  +--------------------------+
  | 2. MOVEMENT              |  For each agent:
  |    Apply all actions     |    - env.move_agent_for(agent_id, action)
  |    simultaneously        |    - Agents pass through each other
  +--------------------------+
              |
  +--------------------------+
  | 3. FOOD RESOLUTION       |  Collect agents at food positions
  |    Competition policy    |  Apply FIRST_ARRIVAL or RANDOM
  |    Consume food          |  Winner gets food; others get nothing
  +--------------------------+
              |
  +--------------------------+
  | 4. PREDATORS             |  env.update_predators() once
  |    Chase nearest agent   |  Per-agent damage checks
  +--------------------------+
              |
  +--------------------------+
  | 5. EFFECTS               |  For each agent:
  |    Temperature/oxygen    |    - HP damage from zones
  |    Satiety decay         |    - Starvation check
  |    Termination check     |    - Freeze if terminated
  +--------------------------+
              |
  +--------------------------+
  | 6. METRICS               |  Per-agent tracking
  |    Track step data       |  Aggregate multi-agent stats
  +--------------------------+
```

This is O(n) in agents for all steps except predator pursuit (O(n\*p) where p = predator count -- negligible for practical sizes).

### Decision 4: Food Competition

```python
class FoodCompetitionPolicy(StrEnum):
    FIRST_ARRIVAL = "first_arrival"  # Deterministic: sorted by agent_id
    RANDOM = "random"                # Random winner among contestants
```

When multiple agents occupy the same food cell after step 2:

- Build `contested: dict[GridPosition, list[str]]` mapping food positions to agent_ids present
- For each contested food, select winner per policy
- Winner's `FoodConsumptionHandler` processes food (satiety restore, healing, metrics)
- Others at same cell get nothing that step

Existing `consume_food()` is binary (food exists or doesn't) -- no fractional splitting needed.

### Decision 5: Pass-Through Agent Collisions

Agents pass through each other. Each agent's own body still blocks itself (existing self-collision check). This matches biology: C. elegans on bacterial lawns move freely through each other. No collision detection between agents.

### Decision 6: Predator Multi-Target

`Predator.update_position()` signature changes:

```python
def update_position(
    self,
    grid_size: int,
    rng: np.random.Generator,
    agent_pos: tuple[int, int] | None = None,        # Backward compat
    agent_positions: list[tuple[int, int]] | None = None,  # Multi-agent
) -> None:
```

When `agent_positions` is provided, pursuit predators chase the nearest position (Manhattan distance). When only `agent_pos` is provided, behavior is identical to current single-agent code. `DynamicForagingEnvironment.update_predators()` passes all alive agent positions.

### Decision 7: Terminated Agent Freeze

When an agent terminates (STARVED, HEALTH_DEPLETED, COMPLETED_ALL_FOOD, or MAX_STEPS):

- `AgentState.alive` set to `False`
- Agent stops being included in perception/action loop (step 1)
- Agent remains on grid -- other agents can still sense it via proximity
- Agent's position is fixed (no movement)
- Episode continues for remaining alive agents
- Episode ends when ALL agents are terminated or max_steps reached

Configurable via `termination_policy`:

- `freeze` (default): As described above
- `remove`: Terminated agent removed from agents dict entirely
- `end_all`: Episode ends immediately when any agent terminates

### Decision 8: Social Proximity Module

```python
class ModuleName(StrEnum):
    ...
    SOCIAL_PROXIMITY = "social_proximity"

def _social_proximity_core(params: BrainParams) -> CoreFeatures:
    count = params.nearby_agents_count or 0
    normalized = min(count, 10) / 10.0
    return CoreFeatures(strength=normalized, angle=0.0, binary=0.0)
```

Registered with `classical_dim=1` (strength only). Quantum transform maps normalized count to a single qubit rotation. The orchestrator computes `nearby_agents_count` as the number of other agents (both alive AND frozen) within `social_detection_radius` (Manhattan distance). Frozen agents are physically present on the grid and detectable. This differs from predator targeting (Decision 6), which only considers alive agents -- predators chase prey, not corpses.

### Decision 9: Agent Spawn Placement

Reuse the Poisson disk sampling pattern from `_initialize_foods()`:

```python
def _initialize_agent_positions(
    self,
    num_agents: int,
    min_agent_distance: int,
) -> list[tuple[int, int]]:
    """Spawn agents with minimum separation, avoiding food and predator positions."""
```

Agents spawn with `min_agent_distance` separation from each other, food sources, and predators. Default `min_agent_distance: 5`. Falls back to random valid positions if Poisson sampling exhausts attempts (large agent count on small grid).

### Decision 10: Reproducibility

- Agents processed in sorted `agent_id` order (deterministic iteration of dict)
- Each agent gets a deterministic sub-seed: `agent_seed = hash((session_seed, agent_id)) % (2**32)`
- Food competition resolution is deterministic for `FIRST_ARRIVAL` policy
- `RANDOM` policy uses the environment's seeded RNG

### Decision 11: Model Persistence

Weight save path changes from `final.pt` to `final_{agent_id}.pt`:

```python
# In weight save logic
def get_weight_path(base_dir: str, agent_id: str = "default") -> str:
    if agent_id == "default":
        return os.path.join(base_dir, "final.pt")  # Backward compat
    return os.path.join(base_dir, f"final_{agent_id}.pt")
```

Per-agent `weights_path` in config enables curriculum loading:

```yaml
multi_agent:
  agents:
    - id: agent_0
      brain: { name: mlpppo, config: { ... } }
      weights_path: "results/foraging_pretrained/final.pt"  # Load pre-trained
    - id: agent_1
      brain: { name: lstmppo, config: { ... } }
      # No weights_path -- train from scratch
```

### Decision 12: Grid Size Validation

```python
import math

def validate_multi_agent_grid(grid_size: int, num_agents: int) -> None:
    min_size = max(MIN_GRID_SIZE, int(5 * math.sqrt(num_agents)))
    if grid_size < min_size:
        raise ValueError(
            f"Grid size {grid_size} too small for {num_agents} agents. "
            f"Minimum: {min_size}."
        )
```

Practical guidance (documented, not enforced):

- Small (20x20): 2-3 agents
- Medium (50x50): 5 agents
- Large (100x100): 10 agents
- Food: 2-3 items per agent recommended

## Configuration Schema

### Shorthand (homogeneous population)

```yaml
brain:
  name: mlpppo
  config:
    sensory_modules: [food_chemotaxis, nociception, social_proximity]
    actor_hidden_dim: 64
    # ...

multi_agent:
  enabled: true
  count: 5
  food_competition: first_arrival
  social_detection_radius: 5
  termination_policy: freeze
  min_agent_distance: 5

environment:
  grid_size: 50
  foraging:
    foods_on_grid: 15  # ~3 per agent
    target_foods_to_collect: 30
  # ...
```

### Explicit (heterogeneous population)

```yaml
multi_agent:
  enabled: true
  food_competition: first_arrival
  social_detection_radius: 5
  termination_policy: freeze
  min_agent_distance: 5
  agents:
    - id: agent_0
      brain:
        name: mlpppo
        config:
          sensory_modules: [food_chemotaxis, nociception, social_proximity]
          actor_hidden_dim: 64
      weights_path: "results/pretrained/final.pt"  # Optional
    - id: agent_1
      brain:
        name: lstmppo
        config:
          sensory_modules: [food_chemotaxis, nociception, social_proximity]
          hidden_dim: 64
    - id: agent_2
      brain:
        name: qrh
        config:
          sensory_modules: [food_chemotaxis, nociception, social_proximity]
          num_qubits: 4
```

### Pydantic Models

```python
class AgentConfig(BaseModel):
    id: str
    brain: BrainContainerConfig
    weights_path: str | None = None

class MultiAgentConfig(BaseModel):
    enabled: bool = False
    count: int | None = None  # Shorthand: N agents with top-level brain
    agents: list[AgentConfig] | None = None  # Explicit: per-agent configs
    food_competition: str = "first_arrival"
    social_detection_radius: int = 5
    termination_policy: str = "freeze"
    min_agent_distance: int = 5
```

Validation: exactly one of `count` or `agents` must be set when `enabled=True`.

## Per-Agent Metrics and CSV Export

### Per-Run CSV (`simulation_results.csv`)

Add `agent_id` column. Each row is one agent's result for one episode:

```csv
run,agent_id,seed,success,steps,total_reward,termination_reason,foods_collected,...
0,agent_0,12345,True,247,15.3,completed_all_food,10,...
0,agent_1,12346,False,500,8.1,max_steps,6,...
1,agent_0,12347,True,189,17.1,completed_all_food,10,...
```

### Aggregate CSV (`multi_agent_summary.csv`)

```csv
run,total_food,food_competition_events,proximity_events,agents_alive_at_end,mean_success,...
0,16,3,47,2,0.5,...
1,20,5,52,1,0.5,...
```

### MultiAgentEpisodeResult

```python
@dataclass
class MultiAgentEpisodeResult:
    agent_results: dict[str, EpisodeResult]
    total_food_collected: int
    per_agent_food: dict[str, int]
    food_competition_events: int
    proximity_events: int
    agents_alive_at_end: int
    mean_agent_success: float
    food_gini_coefficient: float  # Distribution equality (0=equal, 1=monopoly)
```

## Risks / Trade-offs

- **Position-parameterized methods add API surface** -- Mitigated by keeping original methods as thin wrappers. Internal logic refactored to accept position, avoiding duplication.

- **Predator chase with multiple agents is more complex** -- Mitigated by simple nearest-agent heuristic. O(n\*p) per step is negligible for practical sizes (10 agents * 10 predators = 100 distance computations).

- **No rendering** -- Acceptable for infrastructure phase. Headless execution with CSV/metrics is sufficient. Rendering is a follow-up.

- **Sequential brain execution** -- Each agent runs its own brain instance sequentially. For 10 agents with quantum brains, that's 10x circuit evaluations per step. Linear and acceptable, but users should be aware of wall-clock implications.

- **Food competition ordering bias** -- `FIRST_ARRIVAL` deterministically favors lower agent_ids. Mitigated by offering `RANDOM` policy and documenting the bias. For architecture comparison, the bias is consistent across architectures.
