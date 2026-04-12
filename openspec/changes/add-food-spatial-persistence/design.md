## Context

Food currently respawns at uniformly random positions via `spawn_food()` (env.py), constrained only by `min_food_distance` and `agent_exclusion_radius`. The existing `safe_zone_food_bias` provides probabilistic bias toward thermotaxis comfort zones but has no concept of persistent food regions. The pattern for spatial features is well established: `ThermotaxisParams.hot_spots` and `AerotaxisParams.high_oxygen_spots` both use `list[tuple[int, int, float]]` with exponential decay.

Food-marking pheromones are emitted at consumption positions in multi-agent mode (`multi_agent.py:895`), creating chemical trails. Without spatial persistence, these trails point to locations where food will never reappear.

## Goals

- Make food-marking pheromone trails meaningful by ensuring food spawns in persistent regions
- Follow established spatial feature patterns (thermotaxis/aerotaxis hotspots)
- Maintain full backward compatibility with existing configs
- Add static food mode for depletion-based scenarios

## Non-Goals

- Visualization of food hotspot regions in pygame (future overlay)
- Automatic hotspot derivation from initial food placement
- Reward shaping for satiety-dependent behavior (the mechanism is environmental, not reward-based)

## Decisions

### Decision 1: Probabilistic bias, not hard constraint

Food hotspot bias is a probability (0.0-1.0) that a given spawn attempt targets a hotspot, not a hard constraint that all food must be near hotspots. This matches the `safe_zone_food_bias` pattern and allows mixing hotspot-biased and uniform spawning.

**Rationale:** Hard constraints make it difficult to place `foods_on_grid` items when hotspots are small or close together. Probabilistic bias degrades gracefully.

### Decision 2: Exponential decay sampling

Hotspot sampling uses exponential decay: `P(distance) ∝ exp(-distance / decay_constant)`. The candidate position is drawn from a 2D distribution centered on a randomly selected hotspot (weighted by hotspot weight). Grid bounds are enforced by clamping.

**Rationale:** Matches the conceptual model of bacterial lawns — high density at center, tapering off. Same math as temperature/oxygen fields.

### Decision 3: Hotspot selection weighted by weight field

When multiple hotspots exist, the sampling first selects which hotspot to target, with probability proportional to the weight values. Then samples a position near that hotspot.

**Rationale:** Allows configuring asymmetric patches — a weight=2.0 hotspot gets twice as many food spawns as a weight=1.0 hotspot.

### Decision 4: Both initialization and respawn use same bias

`_initialize_foods()` and `spawn_food()` both apply the hotspot bias. Initial food placement creates the patches; respawning maintains them.

**Rationale:** Consistency. If initial food is in patches but respawns are uniform, patches would dissolve over time.

### Decision 5: Static food mode is a simple boolean

`no_respawn: bool = False` makes `spawn_food()` return False immediately. No new task dynamics — just food depletion.

**Rationale:** Minimal implementation. Changes episode dynamics fundamentally (finite food supply, shorter episodes) but requires no reward or brain changes.

### Decision 6: Satiety-gated food collection as environment constraint

Satiety-dependent foraging is implemented as a simple environment gate: `consume_food_for()` checks agent satiety against `satiety_food_threshold` and refuses collection if satiety exceeds the threshold. This is not a reward signal — the agent physically cannot pick up food when sated.

**Rationale:** Prevents the local monopolist problem where one agent depletes a hotspot before pheromone-guided agents arrive. Biologically grounded in C. elegans npr-1-mediated dwelling behavior (sated worms reduce pharyngeal pumping). As an environment constraint rather than reward shaping, it doesn't require brain changes — agents learn to move away when they can't eat.

**Alternative rejected:** Reward-based approaches (movement penalties when sated, no-reward-when-full) require tuning reward weights and may interfere with existing reward signals. The environmental gate is simpler and more predictable.

### Decision 7: Hotspot bias composes with safe_zone_food_bias

Both biases are independent probabilistic filters applied during candidate generation. Hotspot bias selects the sampling distribution (hotspot vs uniform). Safe zone bias is applied afterward as a rejection filter. No special interaction handling needed.

**Rationale:** Orthogonal concerns. Hotspots control where food patches are; safe zone bias controls whether food avoids dangerous temperature zones.

## Risks / Trade-offs

1. **Dense hotspots with strict constraints**: If hotspots are small and `min_food_distance` is large, the retry loop may exhaust `MAX_POISSON_ATTEMPTS`. Mitigated by the existing retry mechanism and logging.

2. **Hotspot centers outside grid**: Invalid hotspot coordinates could cause edge effects. Mitigated by clamping in `_sample_hotspot_candidate()` and validation in ForagingConfig.

3. **Food-marking pheromone evaluation still requires multi-agent testing**: This change enables meaningful evaluation but doesn't perform it. Follow-up evaluation campaigns needed.
