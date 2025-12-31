# Change: Add Multi-Sensory Environment Foundation

## Why

Phase 1 of the Quantum Nematode roadmap requires extending the simulation with multi-modal sensory capabilities and enhanced threat behaviors. The current `DynamicForagingEnvironment` only supports chemotaxis (food gradients) and basic random-walk predators. Real C. elegans use multiple sensory modalities simultaneously (temperature, oxygen, touch) and face diverse predator types.

This proposal establishes the foundational infrastructure that all Phase 1 sensory modalities depend on:

1. Extended `BrainParams` with fields for new sensory inputs
2. Health system (HP-based damage model) as an alternative to instant death
3. Enhanced predator behaviors (stationary toxic zones, active pursuit)
4. Mechanosensation (boundary and predator contact detection)
5. Unified feature extraction layer for both quantum and classical brains
6. Extended reward system with configurable multi-objective weights

## What Changes

### 1. BrainParams Extensions

Add new optional fields to `BrainParams` for Phase 1 sensory modalities:

- Temperature sensing: `temperature`, `temperature_gradient_strength`, `temperature_gradient_direction`, `cultivation_temperature`
- Health tracking: `health`, `max_health`
- Mechanosensation: `boundary_contact`, `predator_contact`

All fields are Optional with None defaults for backward compatibility.

### 2. Health System (Opt-in)

Add HP-based damage tracking as an alternative to instant-death predator encounters:

- Configurable `max_hp` (default 100)
- Predator contact deals damage (configurable `predator_damage`)
- Temperature extremes deal damage (configurable `hp_damage_temperature_*`)
- Food provides healing (configurable `food_healing`)
- Episode terminates when HP reaches 0 (`TerminationReason.HEALTH_DEPLETED`)
- Enabled via `health.enabled: true` in config

**Relationship with Satiety System**: HP and satiety are independent systems that coexist:

- **Satiety**: Time-based hunger pressure (decays constantly, starvation terminates episode)
- **HP**: Threat-based damage tracking (only decreases from damage events like predator contact or temperature)
- **Food**: Restores BOTH satiety AND HP when consumed
- Both can be enabled simultaneously for complex multi-objective survival scenarios

### 3. Enhanced Predator Behaviors

Extend `Predator` class to support behavior types:

- **Stationary**: Fixed position, larger damage radius (toxic zones, nematode-trapping fungi)
- **Pursuit**: Actively tracks agent within detection radius
- **Random** (existing): Random walk movement

Support mixed types per environment (e.g., 2 stationary + 1 pursuit).

### 4. Mechanosensation

Add touch/collision detection:

- `boundary_contact`: Agent is touching grid boundary
- `predator_contact`: Agent is within predator kill radius
- New `mechanosensation_features()` module for brain integration

### 5. Unified Feature Extraction

Create shared feature extraction layer (`brain/features.py`):

- `extract_sensory_features(params: BrainParams) -> dict[str, np.ndarray]`
- Used by ModularBrain (converts to RX/RY/RZ rotations)
- Used by PPOBrain (concatenates to input vector)

Rename existing modules to scientific names with neuron references:

- `appetitive_features` -> `food_chemotaxis_features` (AWC, AWA neurons)
- `aversive_features` -> `nociception_features` (ASH, ADL neurons)

### 6. Multi-Objective Rewards

Extend `RewardConfig` with configurable weights for:

- Temperature comfort/discomfort/danger penalties
- Health gain/damage rewards
- Boundary collision penalties
- All weights configurable via YAML

### 7. Evaluation Extensions

- Add `TerminationReason.HEALTH_DEPLETED`
- Add optional per-objective scores to `SimulationResult`
- Extend composite score calculation for multi-objective tasks

### 8. Hierarchical Benchmark Naming

Introduce hierarchical benchmark categories to scale with increasing modality combinations:

```text
basic/                              # Single objective (foraging only)
├── foraging_small/
├── foraging_medium/
└── foraging_large/

survival/                           # Food + predators
├── foraging_predator_small/        # food + random predators
├── foraging_predator_pursuit_small/# food + pursuit predators
└── foraging_predator_mixed_small/  # food + mixed predator types

thermotaxis/                        # Temperature-aware tasks (added by add-thermotaxis-system)
multisensory/                       # Multiple modalities combined (future)
ablation/                           # Controlled studies (added by add-ablation-toolkit)
```

**Naming Convention**: Task names are always explicit about what's included:

- `foraging` = food collection goal

- `predator` = predators enabled

- `thermo` = thermotaxis enabled (short form in combinations)

- Modifiers follow the base: `foraging_predator_small`, not `predator_small`

- Migrate existing benchmarks to `basic/` and `survival/` categories

- Support hierarchical paths in benchmark categorization logic

- Maintain backward compatibility with flat category names during transition

### 9. Food Spawning in Safe Zones

When thermotaxis is enabled, bias food spawning toward safe temperature zones:

- 80% of food in safe zones (comfort + discomfort temperature)
- 20% anywhere (risk/reward trade-off)
- Configurable via `food_safe_zone_ratio`

## Impact

**Affected Specs:**

- `environment-simulation`: MODIFIED - Add health system, predator types, mechanosensation
- `brain-architecture`: MODIFIED - Add BrainParams extensions, feature extraction
- `benchmark-management`: MODIFIED - Add multi-objective scoring

**Affected Code:**

- `packages/quantum-nematode/quantumnematode/env/env.py` - Health system, predator types
- `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py` - BrainParams extensions
- `packages/quantum-nematode/quantumnematode/brain/modules.py` - Module renaming, mechanosensation
- `packages/quantum-nematode/quantumnematode/brain/features.py` - NEW: Unified feature extraction
- `packages/quantum-nematode/quantumnematode/agent/agent.py` - RewardConfig extensions
- `packages/quantum-nematode/quantumnematode/benchmark/convergence.py` - Multi-objective scoring
- `packages/quantum-nematode/quantumnematode/experiment/metadata.py` - SimulationResult extensions

**Breaking Changes:**

- None. All changes are additive with feature flags.

**Dependencies:**

- This proposal is the foundation for `add-thermotaxis-system`
- Can be developed in parallel with `add-ablation-toolkit`

**Backward Compatibility:**

- All new BrainParams fields are Optional with None defaults
- Health system is opt-in (disabled by default)
- Enhanced predators are opt-in (default to existing random walk)
- Existing configs work unchanged
