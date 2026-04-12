## Why

Food-marking pheromones deposit chemical signals at food consumption positions, intended to guide other agents to productive foraging areas. However, food currently respawns at uniformly random positions after consumption (`spawn_food()` only checks distance constraints, no spatial bias). This makes food-marking pheromone trails point to stale locations where food no longer exists — the signal is actively misleading rather than helpful.

This was identified during Phase 4 evaluation (Campaigns B2+D): alarm and aggregation pheromones have valid spatial signals (predator locations are persistent, agent locations are real-time), but food-marking pheromones cannot be meaningfully evaluated without food spatial persistence. Food-marking pheromone modules were excluded from Phase 4 pheromone evaluation as a workaround.

## What Changes

### 1. Food Hotspots (Patch-Based Spawning)

Add configurable food hotspot regions to `ForagingParams`, modeled on the existing `hot_spots`/`cold_spots` pattern in thermotaxis and `high_oxygen_spots`/`low_oxygen_spots` in aerotaxis:

- `food_hotspots: list[FoodHotspot] | None` — list of (x, y, weight) centers defining spawn probability patches. Weight controls relative spawn density.
- `food_hotspot_bias: float = 0.0` — probability (0.0-1.0) that a given food spawn targets a hotspot rather than uniform random. 0.0 = disabled (backward compatible).
- `food_hotspot_decay: float = 8.0` — exponential decay constant controlling how quickly spawn probability drops with distance from hotspot center.

Both initial placement (`_initialize_foods()`) and respawning (`spawn_food()`) use the same bias logic, matching the existing `safe_zone_food_bias` pattern. Food-marking pheromone trails then correctly signal "this is a productive region."

### 2. Static Food Mode (No Respawn)

Add `no_respawn: bool = False` to `ForagingParams`. When True, consumed food is not replaced — agents deplete patches. This changes task dynamics fundamentally (finite food supply) and enables scenarios where pheromone trails guide agents to remaining clusters.

### 3. Satiety-Dependent Foraging (Anti-Monopolist)

Food hotspots create a "local monopolist" problem: the first agent to discover a patch eats all the food before pheromone-guided agents can arrive. This is both unrealistic (real C. elegans exhibit npr-1-mediated dwelling/satiation behavior on bacterial lawns) and prevents meaningful pheromone evaluation.

Add `satiety_food_threshold: float | None = None` to `ForagingParams`. When set, agents cannot consume food when their satiety exceeds this fraction of max satiety (e.g., 0.8 = can't eat above 80% full). This is an environment constraint, not a reward change — the agent physically cannot pick up food when sated, forcing it to move away and leave food for pheromone-following agents.

Biologically grounded: C. elegans on bacterial lawns reduce pharyngeal pumping when well-fed and transition from dwelling to roaming behavior.

### 4. Configuration

```yaml
environment:
  foraging:
    foods_on_grid: 15
    food_hotspots:
      - [12, 12, 1.0]    # patch near center-left
      - [38, 38, 1.0]    # patch near center-right
      - [12, 38, 0.5]    # weaker patch
    food_hotspot_bias: 0.8
    food_hotspot_decay: 8.0
    satiety_food_threshold: 0.8  # can't eat above 80% satiety
    no_respawn: false
```

## Capabilities

**Modified**: `environment-simulation` (food spawning with spatial bias, static food mode, satiety-gated consumption), `configuration-system` (ForagingConfig hotspot and satiety threshold fields).

## Impact

**Core code:**

- `quantumnematode/dtypes.py` — `FoodHotspot` type alias
- `quantumnematode/env/env.py` — Extend `ForagingParams`, add `_sample_hotspot_candidate()`, modify `_initialize_foods()`, `spawn_food()`, and `consume_food_for()`
- `quantumnematode/agent/multi_agent.py` — Update `_resolve_food_step()` to re-offer food when sated winner can't eat
- `quantumnematode/agent/reward_calculator.py` — Suppress goal bonus when agent is on food but can't eat due to satiety gate
- `quantumnematode/utils/config_loader.py` — Extend `ForagingConfig` with hotspot YAML fields
- `quantumnematode/env/__init__.py` — Export `FoodHotspot`

**Configs:**

- `configs/scenarios/multi_agent_foraging/` — Small (20×20) config for mechanics verification and logbook 011 baseline comparison; medium (50×50) config for primary pheromone evaluation

**Tests:**

- New tests for hotspot spawning, no-respawn mode, satiety-gated consumption, multi-agent food re-offer, reward suppression, YAML loading

## Breaking Changes

None. All defaults match current behavior: `food_hotspots=None`, `food_hotspot_bias=0.0`, `no_respawn=False`, `satiety_food_threshold=None` (disabled).

## Backward Compatibility

Existing configs produce identical behavior. No food parameters change defaults.

## Dependencies

None. Uses existing NumPy RNG (`self.rng`) for sampling.
