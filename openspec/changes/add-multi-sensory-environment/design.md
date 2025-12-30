# Design: Add Multi-Sensory Environment Foundation

## Context

Phase 1 of the Quantum Nematode roadmap introduces multi-modal sensory capabilities. This requires foundational infrastructure changes that all sensory modalities (thermotaxis, aerotaxis, mechanosensation) will build upon.

**Stakeholders**: Research team, future Phase 1+ development
**Constraints**: Must maintain backward compatibility with Phase 0 benchmarks

## Goals / Non-Goals

**Goals:**
- Establish extensible infrastructure for multi-sensory simulation
- Enable HP-based survival mechanics as alternative to instant death
- Support diverse predator behaviors for richer ecological complexity
- Unify feature extraction for quantum and classical brain architectures
- Maintain full backward compatibility

**Non-Goals:**
- Implement specific sensory modalities (thermotaxis is separate proposal)
- Change existing benchmark results
- Optimize for performance (simplicity first)

## Decisions

### Decision 1: Lazy Gradient Computation (Not NxN Storage)

**What**: Compute sensory gradients on-demand at agent's position rather than pre-computing and storing NxN grids.

**Why**:
- Memory efficient: O(1) storage vs O(N²) per modality
- Consistent with existing food/predator gradient pattern
- Scales to larger grids without memory issues
- Easy to add new modalities

**Alternatives Considered**:
- Pre-computed grids: Faster lookup but O(N²×M) memory for M modalities
- Caching: Added complexity without significant benefit for current grid sizes

### Decision 2: Feature Flags for All New Features

**What**: All new capabilities (health system, predator types, thermotaxis) are disabled by default and enabled via explicit config flags.

**Why**:
- Backward compatibility: existing configs work unchanged
- Gradual adoption: can enable features incrementally
- Clear testing: can isolate feature effects

**Implementation**:
```yaml
health_system:
  enabled: false  # Opt-in
thermotaxis:
  enabled: false  # Opt-in
```

### Decision 3: Unified Feature Extraction Layer

**What**: Create `brain/features.py` with shared feature extraction used by both quantum (ModularBrain) and classical (PPOBrain) architectures.

**Why**:
- Ensures both architectures receive identical sensory information
- Enables fair comparison between quantum and classical approaches
- Reduces code duplication
- Simplifies adding new sensory modalities

**Architecture**:
```
BrainParams
     │
     v
extract_sensory_features() ─────────────────────┐
     │                                          │
     v                                          v
ModularBrain                              PPOBrain
(converts to RX/RY/RZ rotations)    (concatenates to input vector)
```

### Decision 4: Scientific Module Naming with Neuron References

**What**: Rename feature extraction functions to scientific names (chemotaxis, thermotaxis, nociception, etc.) with C. elegans neuron references in docstrings.

**Why**:
- Aligns with biological literature
- Makes research publications clearer
- Documents which neural circuits are being modeled

**Mapping**:
| Current | New | Neurons |
|---------|-----|---------|
| appetitive_features | food_chemotaxis_features | AWC, AWA |
| aversive_features | nociception_features | ASH, ADL |
| thermotaxis_features | thermotaxis_features | AFD |
| oxygen_features | aerotaxis_features | URX, BAG |
| touch_features | mechanosensation_features | ALM, PLM, AVM |

### Decision 5: Temperature Zone Integration with Health

**What**: Temperature affects both rewards (behavioral learning signal) AND health (survival pressure).

| Zone | Temperature | Reward Effect | Health Effect |
|------|-------------|---------------|---------------|
| Comfort | 15-25°C | Small reward | None |
| Discomfort | 10-15°C, 25-30°C | Penalty | None |
| Danger | <10°C, >30°C | Larger penalty | HP damage |
| Lethal | <5°C, >35°C | Largest penalty | Rapid HP drain |

**Why**:
- Biologically grounded: extreme temperatures are harmful to C. elegans
- Creates meaningful trade-offs: risky paths may have better food access
- Health system unifies all survival threats (predators, temperature, oxygen)

### Decision 6: Mixed Predator Types per Environment

**What**: Support multiple predator types in same environment (e.g., 2 stationary + 1 pursuit).

**Why**:
- More realistic ecological complexity
- Richer learning challenges
- Enables studying different avoidance strategies

**Configuration**:
```yaml
predators:
  enabled: true
  types:
    - type: stationary
      count: 2
      damage_radius: 2.0
    - type: pursuit
      count: 1
      speed: 0.5
      detection_radius: 5.0
```

### Decision 7: Food Spawning Bias Toward Safe Zones

**What**: When thermotaxis enabled, 80% of food spawns in safe temperature zones, 20% anywhere.

**Why**:
- Ensures learnable tasks (food accessible without extreme risk)
- Creates risk/reward trade-offs (some food in dangerous zones)
- Configurable for different task difficulties

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Complexity creep | Feature flags allow incremental adoption |
| Performance impact | Lazy computation, profile if needed |
| Breaking existing benchmarks | All features opt-in, extensive testing |
| Module renaming breaks configs | Update ModuleName enum, support aliases |

## Migration Plan

1. Add new BrainParams fields (non-breaking)
2. Add health system (opt-in)
3. Add predator types (opt-in)
4. Add mechanosensation (opt-in)
5. Create unified feature extraction
6. Rename modules (with backward-compatible aliases)
7. Update example configs
8. Document changes

**Rollback**: All features are opt-in; disable via config if issues arise.

### Decision 8: HP and Satiety Coexistence

**What**: HP (health points) and satiety are independent systems that coexist. Food restores both.

**Why**:
- **Satiety** models time-based hunger pressure (metabolic needs)
- **HP** models threat-based damage (injury from predators, temperature extremes)
- These are biologically distinct: C. elegans can be both hungry AND injured
- Enables rich multi-objective scenarios (manage hunger, avoid damage, seek food)

**Behavior**:
| System | Decreases From | Increases From | Termination |
|--------|---------------|----------------|-------------|
| Satiety | Time decay (every step) | Eating food | STARVATION |
| HP | Predator contact, temperature extremes | Eating food, configurable healing | HEALTH_DEPLETED |

**Configuration Example**:
```yaml
# Both systems enabled
satiety:
  enabled: true
  decay_rate: 0.5
  starvation_threshold: 0.0

health_system:
  enabled: true
  max_hp: 100
  predator_damage: 10
  food_healing: 5
```

## Open Questions

1. Should health regenerate over time (without food)?
   - Current decision: No, only food heals
   - Can revisit based on gameplay testing

2. Should pursuit predators have stamina/cooldown?
   - Current decision: No, always pursue when in range
   - Can add if learning becomes too difficult
