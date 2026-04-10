## Overview

This deliverable adds social dynamics to the multi-agent environment: social feeding (satiety decay reduction near conspecifics), aggregation pheromones (continuous ascaroside-like emission), and collective behavior metrics. These create measurable aggregation pressure — the prerequisite for documenting emergent social behaviors (Phase 4 exit criterion).

## Goals / Non-Goals

**Goals:**

- Social feeding via satiety decay reduction when near conspecifics
- Aggregation pheromone as 3rd PheromoneType with continuous emission
- Per-agent npr-1 phenotype (social vs solitary)
- Four collective behavior metrics in MultiAgentEpisodeResult and CSV export
- Oracle and temporal sensing modes for aggregation pheromone
- STAM 7-channel mode for temporal aggregation sensing
- Full backward compatibility when features disabled

**Non-Goals:**

- Territorial behavior or zone defense (Phase 4 Deliverable 4)
- Game-theoretic analysis / Nash equilibria (Phase 4 Deliverable 4)
- Adjacent-cell feeding (pharyngeal pumping rate model — future enhancement)
- Pheromone-based agent identity discrimination
- Aggregation pheromone replacing SOCIAL_PROXIMITY module (complementary, not replacement)

## Design Decisions

### Decision 1: Satiety Decay Reduction (Not Food Value Bonus)

Real C. elegans social feeding works by increased pharyngeal pumping and reduced locomotion on bacterial lawns near conspecifics. The food doesn't become more nutritious — the worm conserves energy. We model this as reduced satiety decay rate, not increased satiety gain.

```python
@dataclass
class SocialFeedingParams:
    enabled: bool = False
    decay_reduction: float = 0.7      # multiplier on satiety decay when near others
    solitary_decay: float = 1.0       # multiplier for solitary phenotype
```

Detection radius is shared with the multi-agent config's `social_detection_radius` parameter (default 5), which also controls SOCIAL_PROXIMITY sensing and nearby-agent counting.

**Integration point**: In `MultiAgentSimulation.run_episode()`, section 5 (EFFECTS), before `decay_satiety()`:

```python
# Compute decay multiplier based on social feeding
decay_mult = 1.0
if self.env.social_feeding.enabled:
    nearby = nearby_per_agent.get(aid, 0)
    if nearby > 0:
        phenotype = self._agent_phenotypes.get(aid, "social")
        if phenotype == "social":
            decay_mult = self.env.social_feeding.decay_reduction
        else:
            decay_mult = self.env.social_feeding.solitary_decay
agent._satiety_manager.decay_satiety(multiplier=decay_mult)
```

**Why decay, not gain**: The agent must discover through learning that proximity to others extends survival. The reward function already rewards food and penalizes starvation — improved satiety propagates through existing reward channels. A direct reward bonus would be double-counting.

### Decision 2: Aggregation Pheromone as 3rd PheromoneType

```python
class PheromoneType(StrEnum):
    FOOD_MARKING = "food_marking"
    ALARM = "alarm"
    AGGREGATION = "aggregation"
```

The existing `PheromoneField` class handles all the physics (spatial decay, temporal decay, gradient, pruning). The only behavioral difference is emission pattern: continuous (every agent every step) vs event-driven (food-marking on consumption, alarm on damage).

**Default parameters** tuned for "current presence" signal:

```python
PheromoneTypeConfig(
    emission_strength=0.5,        # weaker than food-marking (1.0) or alarm (2.0)
    spatial_decay_constant=10.0,  # wider spread than alarm (5.0)
    temporal_half_life=10.0,      # much shorter than food-marking (50.0)
    max_sources=200,              # higher cap due to continuous emission
)
```

Short half-life (10 steps, max_age=50) means the field reflects where agents are *now*. With 5 agents emitting every step and max_sources=200, we retain ~40 steps of history per agent before pruning.

**Performance**: With max_sources=200, `get_concentration()` is O(200) per query. Gradient uses 5 queries (center + 4 neighbors) = O(1000). For 10 agents: O(10,000) per step. Acceptable for grid-world simulation.

### Decision 3: STAM 7-Channel Mode

```python
# stam.py
CHANNELS_BASE = 4        # food, temperature, predator, oxygen
CHANNELS_PHEROMONE = 6   # + pheromone_food, pheromone_alarm
CHANNELS_PHEROMONE_FULL = 7  # + pheromone_aggregation

IDX_PHEROMONE_AGGREGATION = 6
```

Channel count determined from config:

- 4: pheromones disabled
- 6: pheromones enabled (food-marking + alarm)
- 7: pheromones enabled AND aggregation type configured

`compute_memory_dim(7) = 2*7 + 3 = 17`. The STAM constructor already takes `num_channels` — passing 7 requires no architectural change.

### Decision 4: Collective Behavior Metrics

Four metrics designed to detect Phase 4 exit criteria:

**1. social_feeding_events** (int): Count of step-agent pairs where social decay reduction was applied. Simple counter incremented in the effects phase.

**2. aggregation_index** (float): Mean normalized inverse pairwise distance.

```python
def _compute_aggregation_index(positions: list[tuple[int, int]], grid_size: int) -> float:
    """0 = maximally dispersed, 1 = all agents at same position."""
    if len(positions) < 2:
        return 0.0
    max_dist = 2 * (grid_size - 1)  # max Manhattan distance on grid
    total_proximity = 0.0
    n_pairs = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
            total_proximity += 1.0 - (dist / max_dist)
            n_pairs += 1
    return total_proximity / n_pairs
```

Computed per step, averaged over episode. O(N^2) per step where N = agents — acceptable for N \<= 10.

**3. alarm_evasion_events** (int): Counts zone exits from alarm pheromone regions. For each agent, if alarm concentration was above `ALARM_EVASION_THRESHOLD` (0.1) on the previous step and drops to or below the threshold on the current step, this counts as an evasion event. Requires tracking previous-step alarm concentrations per agent. Zone-exit semantics prevent double-counting from oscillation near the threshold boundary.

**4. food_sharing_events** (int): When a food-marking pheromone is emitted, track the emission position. Over the next `FOOD_SHARING_LOOKBACK_STEPS` (20) steps, if a *different* agent moves to within `social_detection_radius` of that position, count as a food-sharing event. Entry removed from buffer after detection to prevent double-counting.

### Decision 5: Per-Agent Phenotype

`AgentConfig` in config_loader.py gets `social_phenotype: Literal["social", "solitary"] = "social"`. Stored as a dict `_agent_phenotypes: dict[str, str]` on `MultiAgentSimulation`.

For homogeneous configs (count-based), all agents default to "social". For heterogeneous configs (agents list), each agent can specify its phenotype. This enables mixed-population experiments studying npr-1 variation — e.g., 3 social + 2 solitary agents competing for food.

## Configuration Schema

### SocialFeedingConfig

```yaml
environment:
  social_feeding:
    enabled: true
    decay_reduction: 0.7      # 30% slower energy burn near conspecifics
    solitary_decay: 1.0        # no change for solitary phenotype (>1.0 for crowding penalty)
```

Detection radius is shared with `multi_agent.social_detection_radius` (default 5).

### Aggregation Pheromone Config

Added to existing pheromone config block:

```yaml
environment:
  pheromones:
    enabled: true
    food_marking: { ... }  # existing
    alarm: { ... }         # existing
    aggregation:
      emission_strength: 0.5
      spatial_decay_constant: 10.0
      temporal_half_life: 10
      max_sources: 200
```

### Sensing Mode

```yaml
sensing:
  pheromone_aggregation_mode: oracle  # or temporal, derivative
```

### Per-Agent Phenotype

```yaml
multi_agent:
  agents:
    - brain_type: mlpppo
      social_phenotype: social
    - brain_type: mlpppo
      social_phenotype: solitary
```

## Risks / Trade-offs

- **Aggregation pheromone saturation**: With N agents emitting every step, the field can saturate in dense areas, flattening gradients. Mitigated by low emission_strength (0.5) and short half-life (10 steps). Empirical tuning may be needed.

- **STAM MEMORY_DIM 17**: Larger input dimension for classical brains. Existing 6-channel models incompatible with 7-channel mode — acceptable since aggregation-enabled configs are new.

- **Decay reduction strength**: 0.7x may be too strong or too weak. Making it configurable covers this; evaluation will determine optimal values.

- **Aggregation index O(N^2)**: Quadratic in agent count per step. For N=10: 45 pairs per step. For N=100 (hypothetical): 4950 pairs. Current system supports max 10 agents, so this is fine.

- **Food sharing detection**: Requires tracking recent emission positions and computing distances to all agents. Buffer bounded by configurable lookback window (default 20 steps * max food events).
