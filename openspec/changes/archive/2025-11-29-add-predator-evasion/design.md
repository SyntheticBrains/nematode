# Design: Predator Evasion System

## Context

This change introduces predator evasion behavior to the nematode simulation, enabling research into multi-objective reinforcement learning where agents must balance conflicting goals (foraging vs survival). The design is grounded in C. elegans neurobiology research showing chemosensory detection, gradient-based navigation, and escape responses.

**Key biological findings informing the design:**

- Primary detection mechanism: chemosensory (chemical gradient detection at distance)
- Secondary detection: mechanosensory (touch-triggered escape)
- Behavioral response: backing away from threats, suppression of foraging during danger
- Learning: both innate avoidance and learned threat association

**Stakeholders:**

- Quantum ML researchers exploring multi-objective optimization
- Computational neuroscientists modeling realistic C. elegans behavior
- RL practitioners studying approach-avoidance conflict resolution

## Goals / Non-Goals

### Goals

- Biologically realistic predator avoidance based on C. elegans research
- Minimal disruption to existing brain architectures (no changes to ModularBrain or MLPBrain required)
- Unified gradient system that naturally integrates food attraction and predator repulsion
- Clear learning signal through instant death on collision
- Backward compatibility with all existing configurations and simulations
- Support for single predator type (active random walker) initially

### Non-Goals

- Health/damage system (future enhancement)
- Multiple predator types (fungi traps, toxic patches - future enhancement)
- Pursuit/patrol behaviors (future enhancement)
- Predator-predator interactions
- Multi-agent scenarios (multiple nematodes)

## Decisions

### Decision 1: Unified Gradient System

**Choice:** Predators contribute negative values to the same gradient field as food sources (superposition).

**Rationale:**

- **Biological accuracy**: C. elegans integrates multiple chemical cues through the same chemosensory neurons
- **Emergent complexity**: Agent must learn to balance attraction/repulsion in real-time
- **Implementation simplicity**: Reuse existing gradient calculation infrastructure
- **Brain compatibility**: No changes needed to ModularBrain (Chemotaxis module) or MLPBrain (2D observation space)

**Alternatives considered:**

- Separate gradient fields: Would require expanding MLPBrain to 4D observations and adding new module to ModularBrain; less biologically realistic but easier initial learning
- Binary detection: Would lose distance information; too simplistic for realistic behavior

**Implementation:**

```python
# Food sources contribute positive gradient
food_gradient_vector = sum([
    food.gradient_strength * exp(-distance / food.decay) * unit_vector
    for food in foods
])

# Predators contribute negative gradient
predator_gradient_vector = sum([
    -predator.gradient_strength * exp(-distance / predator.decay) * unit_vector
    for predator in predators
])

# Combined observation
total_gradient = food_gradient_vector + predator_gradient_vector
```

### Decision 2: Instant Death on Collision

**Choice:** Predator collision causes immediate episode termination with `TerminationReason.PREDATOR`. Kill radius will default to zero.

**Rationale:**

- **Clear learning signal**: Binary outcome (survived/died) is easier for RL agents to learn from
- **Biological realism**: Nematode-trapping fungi and predatory mites often result in complete capture
- **Simpler metrics**: Success rate is unambiguous
- **Matches scope**: Aligns with "get started" approach

**Alternatives considered:**

- Health point system: More gradual feedback but adds complexity; deferred to future enhancement
- Satiety penalty: Would conflate predator damage with starvation; unclear signal

### Decision 3: Configuration Restructuring

**Choice:** Nest existing foraging parameters under `dynamic.foraging` subsection, add `dynamic.predators` subsection.

**Rationale:**

- **Organization**: Clear separation between foraging mechanics and predator mechanics
- **Scalability**: Easy to add future subsections (e.g., `obstacles`, `terrain`)
- **Readability**: Configuration intent is immediately clear

**Structure:**

```yaml
reward:
  reward_goal: 2.0
  reward_distance_scale: 0.5
  penalty_step: 0.005
  penalty_predator_death: 10.0      # Positive value, negated when applied
  penalty_predator_proximity: 0.1   # Positive value, negated when applied
  penalty_starvation: 10.0

environment:
  type: dynamic
  dynamic:
    grid_size: 100
    viewport_size: [11, 11]

    foraging:
      foods_on_grid: 50
      target_foods_to_collect: 50
      min_food_distance: 10
      agent_exclusion_radius: 15
      gradient_decay_constant: 12.0
      gradient_strength: 1.0

    predators:
      enabled: true
      count: 3
      speed: 1.0
      movement_pattern: "random"
      detection_radius: 8
      kill_radius: 0
      gradient_decay_constant: 12.0
      gradient_strength: 1.0
```

**Note:** Predator-related penalties are configured in the `reward` section (not in predator config) for consistency with all other penalties. All penalty values are stored as positive numbers and negated when applied to rewards in the implementation.

**Migration:** All example configurations have been migrated to the nested structure.

### Decision 4: Active Random Walker Predator

**Choice:** Predators move randomly at configurable speed (default 1.0x agent speed).

**Rationale:**

- **Simplicity**: Random walk is easy to implement and debug
- **Biological precedent**: Many predatory mites exhibit random search patterns
- **Baseline**: Establishes baseline difficulty before adding pursuit behaviors
- **Future extensible**: `movement_pattern` field allows future addition of "patrol" or "pursue"

**Alternatives considered:**

- Stationary traps (fungi): Too easy; agent can memorize positions
- Active pursuit: Too hard initially; requires sophisticated sensing; deferred to future

### Decision 5: Proximity Penalty

**Choice:** Small penalty (0.1 default, stored as positive value, applied as negative) per step when agent is within predator detection radius.

**Rationale:**

- **Biological realism**: Mimics stress response when predator is nearby
- **Learning signal**: Encourages proactive avoidance, not just reactive escape
- **Configurable**: Can be tuned or disabled based on research needs
- **Consistent pattern**: All penalties stored as positive values in config, negated when applied to reward

**Implementation:**

- Configured in `reward.penalty_predator_proximity` (positive value, e.g., 0.1)
- Applied in reward calculator as `-penalty_predator_proximity` when any predator is within `detection_radius`
- Death penalty similarly configured as `reward.penalty_predator_death` (positive value, e.g., 10.0)
- Applied in episode runner as `-reward_config.penalty_predator_death` upon collision
- Both penalties cap at their configured values (do not stack per predator)

### Decision 6: Predator Type - Predatory Mite

**Choice:** Model predatory mites (e.g., *Hypoaspis* species) as the initial predator type.

**Rationale:**

- **Research precedent**: Well-documented C. elegans predator in literature
- **Visual distinction**: Spider emoji (🕷️) clearly different from nematode (🪱)
- **Size/speed**: Justifies similar movement speed to nematode

**ASCII representation:** `#` symbol (chosen for visual density/threat appearance)

## Risks / Trade-offs

### Risk 1: Learning Difficulty

**Risk:** Unified gradient system may be too challenging for agents to learn initially (conflicting signals).

**Mitigation:**

- Start with low predator count (1-2) and large detection radius (8 cells)
- Proximity penalty provides early warning signal
- Curriculum learning: train on foraging-only first, then enable predators
- If learning fails, can add separate gradient observation in future iteration

**Monitoring:** Track convergence metrics in early experiments; if agents fail to learn survival behavior after 10K episodes, revisit gradient design.

### Risk 2: Configuration Migration Complexity

**Risk:** Restructuring configuration from flat to nested may break existing user workflows.

**Mitigation:**

- Implement automatic migration with clear deprecation warnings
- Maintain backward compatibility for at least 3 releases
- Provide migration script for batch config updates
- Document migration path clearly in CHANGELOG

### Risk 3: Computational Overhead

**Risk:** Predator gradient calculations may slow down simulation steps.

**Mitigation:**

- Reuse existing gradient infrastructure (already optimized)
- Predator gradients calculated same way as food gradients
- Limit predator count in initial configurations (≤5)
- Performance target: <100ms per step maintained for 100×100 grids

**Monitoring:** Add timing instrumentation to gradient calculation; fail build if step time exceeds threshold.

## Migration Plan

### Phase 1: Code Implementation (Week 1-2)

1. Add `Predator` class to `env.py`
2. Extend `DynamicForagingEnvironment` with predator support
3. Update configuration schema with automatic migration
4. Add predator metrics to tracking system
5. Implement proximity penalty in reward calculator

### Phase 2: Testing & Validation (Week 2-3)

1. Unit tests for predator mechanics
2. Integration tests for gradient superposition
3. Backward compatibility tests (existing configs)
4. Performance benchmarks (ensure <100ms step time)

### Phase 3: Configuration & Examples (Week 3)

1. Create example configs with predators enabled
2. Add predator benchmark categories
3. Update documentation with usage examples
4. Add migration warnings to config loader

### Phase 4: Visualization & Reporting (Week 3-4)

1. Add predator rendering (emoji/ASCII themes)
2. Implement detection radius visualization
3. Add danger status display
4. Create predator-specific plots (encounters, evasions)

### Rollback Plan

If critical issues discovered:

1. Set `predators.enabled: false` as hardcoded default
2. Add feature flag to disable predator code path entirely
3. Revert configuration restructuring if migration fails
4. Fall back to original flat configuration schema

## Resolved Implementation Decisions

The following questions were resolved during implementation and testing:

1. **Predator gradient strength:**

   - **Final choice:** Same magnitude as food attraction (1.0 default)
   - **Rationale:** Biological realism - C. elegans integrates chemical cues with similar sensitivity. Allows balanced multi-objective learning where neither signal dominates.
   - **Configurability:** Can be tuned per-environment via `predators.gradient_strength`

2. **Kill radius vs detection radius:**

   - **Final choice:** `kill_radius=0` (exact overlap required), `detection_radius=8`
   - **Rationale:** Provides clear separation between danger detection (chemical gradient at distance) and actual collision. Prevents "near miss" kills that would confuse learning signal.
   - **Observed behavior:** Works well in practice - agents learn to maintain distance from predators

3. **Proximity penalty stacking:**

   - **Final choice:** Penalty caps at configured value (does not stack per predator)
   - **Implementation:** `if any(distance <= detection_radius): penalty = -config.penalty_predator_proximity`
   - **Rationale:** Simplifies learning signal - "danger present" vs "danger absent" binary state. Prevents overwhelming negative rewards when multiple predators cluster.

4. **Predator spawn exclusion:**

   - **Final choice:** Predators spawn outside detection radius of agent (using Manhattan distance)
   - **Implementation:** `_initialize_predators` attempts up to 100 tries to find positions where `distance_to_agent > detection_radius`
   - **Rationale:** Prevents immediate danger at episode start, giving agent time to orient. Still more challenging than food's agent_exclusion_radius since predators are mobile.
   - **Observed behavior:** Provides fair starting conditions while maintaining difficulty as predators move randomly toward agent

5. **Viewport rendering priority:**

   - **Final choice:** Viewport remains fixed at configured size, predators outside viewport not shown visually
   - **Implementation:** Gradient detection works regardless of viewport (agent observes via chemosensory input, not visual)
   - **Rationale:** Matches biological reality - C. elegans detects chemical gradients beyond visual range
