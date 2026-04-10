## Overview

The pheromone communication system adds biologically grounded chemical signaling to the multi-agent environment. Agents emit pheromones on events (finding food, taking predator damage), the chemicals diffuse and decay over time, and other agents sense pheromone concentrations at their position using chemosensory neurons. This replaces the oracle social proximity module with real chemical communication following the same oracle/temporal/derivative sensing pattern established for chemotaxis, thermotaxis, and aerotaxis.

## Goals / Non-Goals

**Goals:**

- Two pheromone types: food-marking (attractant) and alarm (repellent)
- Point-source decay field matching food gradient math
- Oracle and temporal sensing modes for each type
- STAM integration for temporal pheromone sensing
- Event-driven emission (food consumption → food-marking, predator damage → alarm)
- Full backward compatibility when pheromones disabled
- CSV export with agent_id for multi-agent sessions (deferred item cleanup)

**Non-Goals:**

- Aggregation pheromones (constant emission — better paired with social feeding in a later deliverable)
- Dense grid diffusion (point-source decay is sufficient and consistent with existing patterns)
- Pheromone-based agent identity (agents can't distinguish who emitted)
- Social feeding mechanics (feeding rate enhancement near others)
- Competitive foraging / territorial behavior

## Design Decisions

### Decision 1: Point-Source Decay Model

Pheromones are modeled as point sources with exponential spatial and temporal decay:

```python
@dataclass
class PheromoneSource:
    position: tuple[int, int]
    pheromone_type: PheromoneType
    strength: float
    emission_step: int
    emitter_id: str

class PheromoneField:
    def get_concentration(self, position: tuple[int, int], current_step: int) -> float:
        total = 0.0
        for source in self._sources:
            distance = manhattan_dist(position, source.position)
            age = current_step - source.emission_step
            contribution = (
                source.strength
                * exp(-distance / self.spatial_decay_constant)
                * exp(-age * ln(2) / self.temporal_half_life)
            )
            total += contribution
        return tanh(total * PHEROMONE_SCALING_FACTOR)
```

**Why point-source decay**: Same O(S) math as food gradients. Consistent with existing patterns. Biologically adequate for steady-state gradient approximation. Pheromone diffusion on C. elegans timescales (minutes) is fast enough that the gradient approximation is reasonable.

**Why Manhattan distance**: Consistent with grid-world movement (agents move in cardinal directions). Could use Euclidean for smoother gradients but Manhattan is simpler and matches the movement model.

### Decision 2: Separate Fields Per Pheromone Type

Each pheromone type has its own `PheromoneField` instance:

```python
self.pheromone_field_food: PheromoneField | None = None
self.pheromone_field_alarm: PheromoneField | None = None
```

**Why separate**: Independent sensing modes (food pheromone in temporal mode, alarm in oracle mode). Independent decay parameters. Independent source counts. Matches how different pheromone receptor pathways work in C. elegans (ASK neurons respond to different ascarosides than ADL neurons).

### Decision 3: Event-Driven Emission

Pheromones are emitted on specific events, not continuously:

- **Food-marking**: Emitted when an agent successfully consumes food, at the food's position. This models C. elegans depositing trail pheromones near bacterial lawns.
- **Alarm**: Emitted when an agent takes predator damage (HP decreases), at the agent's current position. This models injured C. elegans releasing repellent chemicals.

Emission happens in the MultiAgentSimulation step loop:

1. After food consumption in `_resolve_food_step()` — deposit food-marking pheromone
2. After predator damage in the predator phase — deposit alarm pheromone

### Decision 4: STAM Dynamic Channel Extension

```python
class STAMBuffer:
    def __init__(self, buffer_size=30, decay_rate=0.1, num_channels=4):
        # num_channels=4: food, temp, predator, oxygen (no pheromones)
        # num_channels=6: + pheromone_food, pheromone_alarm
        self.num_channels = num_channels
        self.MEMORY_DIM = compute_memory_dim(num_channels)  # 2*N + 3

def compute_memory_dim(num_channels: int) -> int:
    """weighted_means(N) + derivatives(N) + pos_deltas(2) + action_entropy(1)"""
    return num_channels * 2 + 3
```

**Why dynamic**: When pheromones are disabled, STAM produces 11-dim output (backward compatible). When enabled, 15-dim. The STAMSensoryModule's `classical_dim` is set from the STAMBuffer instance, and brain `input_dim` is derived dynamically from the module list. No hardcoded dimensions need updating.

**Channel indices** (when pheromones enabled):

- 0: food, 1: temperature, 2: predator, 3: oxygen, 4: pheromone_food, 5: pheromone_alarm

### Decision 5: Pheromone Sensing Modules

Follow the exact pattern of food_chemotaxis / food_chemotaxis_temporal:

```python
# Oracle: gradient strength + direction (toward pheromone concentration)
def _pheromone_food_core(params: BrainParams) -> CoreFeatures:
    strength = params.pheromone_food_gradient_strength or 0.0
    angle = _compute_relative_angle(
        params.pheromone_food_gradient_direction,
        params.agent_direction,
    )
    return CoreFeatures(strength=strength, angle=angle)

# Temporal: scalar concentration + dC/dt
def _pheromone_food_temporal_core(params: BrainParams) -> CoreFeatures:
    strength = params.pheromone_food_concentration or 0.0
    raw_deriv = params.pheromone_food_dconcentration_dt or 0.0
    angle = float(np.tanh(raw_deriv * params.derivative_scale))
    return CoreFeatures(strength=strength, angle=angle)
```

### Decision 6: Source Pruning Strategy

Each step, `PheromoneField.prune(current_step)` removes sources where:

- `age > max_age` (derived from `temporal_half_life * 5` — ~3% remaining strength)
- Total sources exceed `max_sources` (oldest removed first)

This bounds memory usage: with `max_sources=100` and typical emission rates (~1 per food consumption), the source list stays small.

### Decision 7: Pheromone as "Productive Region" Signal

Food-marking pheromones mark where food **was found**, not where food currently is. After consumption, food respawns at a random valid position. The pheromone trail signals "this region is productive" — useful because:

- Food density varies spatially (Poisson disk minimum distances create clusters)
- Multiple food items in the same region create overlapping pheromone signals
- Agents following pheromone trails reach areas with higher food encounter probability
- This matches real C. elegans behavior: pheromone trails guide worms to bacterial lawns (areas), not individual bacteria

## Configuration Schema

### PheromoneTypeConfig

```python
@dataclass
class PheromoneTypeConfig:
    emission_strength: float = 1.0
    spatial_decay_constant: float = 8.0
    temporal_half_life: float = 50.0  # steps
    max_sources: int = 100

@dataclass
class PheromoneParams:
    enabled: bool = False
    food_marking: PheromoneTypeConfig = field(default_factory=PheromoneTypeConfig)
    alarm: PheromoneTypeConfig = field(
        default_factory=lambda: PheromoneTypeConfig(
            emission_strength=2.0,
            spatial_decay_constant=5.0,
            temporal_half_life=20.0,
            max_sources=50,
        )
    )
```

### SensingConfig Extension

```python
class SensingConfig(BaseModel):
    # ... existing fields ...
    pheromone_food_mode: SensingMode = SensingMode.ORACLE
    pheromone_alarm_mode: SensingMode = SensingMode.ORACLE
```

## CSV Export Design (Deferred Items)

### Per-Agent Results CSV

Add `agent_id` as the second column in `simulation_results.csv`:

```csv
run,agent_id,steps,total_reward,...
0,agent_0,201,15.3,...
0,agent_1,241,18.1,...
```

For single-agent mode, `agent_id` is `"default"` (backward compatible — existing parsers can ignore the column).

### Multi-Agent Summary CSV

New `multi_agent_summary.csv` written alongside `simulation_results.csv`:

```csv
run,total_food,food_competition_events,proximity_events,agents_alive_at_end,mean_success,gini
0,5,1,23,0,0.0,0.50
```

### Session Summary Table

Print at end of multi-agent session:

```text
=== Session Summary ===
Episodes: 500
Mean food/episode: 3.2
Mean competition events: 0.8
Mean Gini: 0.45
Per-agent success: agent_0=12%, agent_1=8%, agent_2=15%
```

## Implementation Notes

- **current_step parameter**: All pheromone concentration/gradient methods on the environment take `current_step` for temporal decay computation. Callers in `_create_brain_params` and `_compute_temporal_data` pass `self._episode_tracker.steps`. The `*_for(agent_id)` methods propagate `current_step` through. Default `current_step=0` in method signatures is a convenience for tests only — production calls must always pass the actual step.

## Risks / Trade-offs

- **Source list growth**: Bounded by `max_sources` and pruning. Worst case ~100 sources per type = 200 distance computations per concentration query. Negligible.

- **STAM breaking change**: MEMORY_DIM 11→15 when pheromones enabled. Trained models with 4-channel STAM cannot be loaded into 6-channel mode. This is acceptable — pheromone-enabled configs are new and won't have pre-trained weights.

- **Pheromone signal noise**: With many agents emitting food-marking pheromones, the field becomes saturated and gradients flatten. Bounded by `max_sources` pruning (oldest removed first). Alarm pheromones are rarer (only on damage events).

- **No gradient for temporal mode**: Temporal sensing provides scalar concentration + dC/dt but no directional information. Agents must infer direction from their own movement + temporal changes (same as biologically honest food sensing). This is intentionally harder than oracle mode.
