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
- Temperature/oxygen zone integration with food hotspots (food hotspots in safe zones is achievable via existing `safe_zone_food_bias` — no new code needed)
- Full pheromone evaluation campaign (follow-up work after this change lands)
- New brain architectures or sensory modules (BrainParams already has satiety field; agents learn to move away when sated)

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

### Decision 6: Satiety gate in caller layer, not environment

Satiety lives on `QuantumNematodeAgent._satiety_manager`, not on `AgentState` in the environment. Therefore the satiety gate is checked in the two callers of `consume_food_for()`: `FoodConsumptionHandler.check_and_consume_food()` (single-agent) and `_resolve_food_step()` (multi-agent). Both have access to the agent object and thus satiety. The environment's `consume_food_for()` remains a pure grid/food operation.

**Rationale:** Keeps the environment layer as a pure grid manager. Satiety is agent-layer state — the environment doesn't know about it and shouldn't need to.

**Alternative rejected:** Adding satiety to `AgentState` or passing it to `consume_food_for()` would leak agent-layer concerns into the environment.

### Decision 7: Pre-filter sated agents from food competition

In `_resolve_food_step()`, sated agents are excluded from the `contested` map before competition resolution. They don't enter the competition at all — food at their position is available to hungry agents.

**Rationale:** Simpler than re-offering after a sated winner is selected. Avoids retry loops. A sated agent standing on food simply doesn't compete, leaving it for hungry agents.

### Decision 8: Suppress goal bonus via `can_eat` parameter

Add `can_eat: bool = True` parameter to `calculate_reward()`. Callers pass `False` when the agent is on food but can't eat due to satiety gate. The goal bonus at `reward_calculator.py:159` is gated on `can_eat`. Defaults to `True` for backward compatibility — existing callers are unaffected.

**Rationale:** Clean interface. The reward calculator doesn't need to know about satiety directly — it just needs to know whether the agent can consume this step.

### Decision 9: Hotspot bias composes with safe_zone_food_bias

Both biases are independent probabilistic filters applied during candidate generation. Hotspot bias selects the sampling distribution (hotspot vs uniform). Safe zone bias is applied afterward as a rejection filter. No special interaction handling needed.

**Rationale:** Orthogonal concerns. Hotspots control where food patches are; safe zone bias controls whether food avoids dangerous temperature zones. For environments with thermotaxis, setting `safe_zone_food_bias=0.7` alongside `food_hotspot_bias=0.8` would naturally place food patches in safe temperature zones — biologically realistic (bacterial lawns grow in favorable conditions).

### Decision 10: Both small (20×20) and medium (50×50) evaluation configs

Small configs verify mechanics work and enable comparison against logbook 011 baselines (all run on 20×20). Medium configs provide proper spatial separation for hotspots — on 20×20, hotspot decay radii overlap too much for pheromone trails to add value over random exploration. The medium (50×50) config is the primary evaluation environment.

**Rationale:** Logbook 011 found that pheromones were redundant on 20×20 because "collective exploration advantage" dominated — the grid was too small for spatial information to matter. Medium grids create meaningful trail distances.

### Decision 11: No changes to success criteria

Success remains `foods_collected >= target_foods_to_collect`. Satiety gating makes the task harder (agents can't eat continuously) but doesn't change what success means. Key evaluation metrics are food-sharing events, per-agent food Gini coefficient, and time-to-target — all already tracked.

**Rationale:** Changing success criteria would invalidate comparisons with prior logbooks. The interesting question is whether pheromone-guided agents reach target faster, not whether the target itself changes.

### Decision 12: No brain or sensory module changes needed

BrainParams already has a `satiety` field populated at agent.py:722. Agents already receive their satiety level and can learn behavioral responses to the satiety gate. No new sensory modules are required — satiety is homeostatic state, not a sensory modality.

**Rationale:** The satiety gate is an environment constraint that agents discover through experience (attempting to eat and failing). The existing learning loop handles this.

## Risks / Trade-offs

1. **Dense hotspots with strict constraints**: If hotspots are small and `min_food_distance` is large, the retry loop may exhaust `MAX_POISSON_ATTEMPTS`. Mitigated by the existing retry mechanism and logging.

2. **Hotspot centers outside grid**: Invalid hotspot coordinates could cause edge effects. Mitigated by clamping in `_sample_hotspot_candidate()` and validation in ForagingConfig.

3. **Food-marking pheromone evaluation still requires multi-agent testing**: This change enables meaningful evaluation but doesn't perform it. Follow-up evaluation campaign needed with temporal sensing mode (not oracle — oracle makes pheromones redundant per logbook 011 finding).

4. **Satiety gate parameter sensitivity**: Too low a threshold blocks eating too early (agents starve); too high has no effect. Pilot evaluation recommended: test 0.6, 0.8, 0.9 thresholds with short runs before full campaign.
