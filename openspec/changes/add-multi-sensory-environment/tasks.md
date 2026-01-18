# Tasks: Add Multi-Sensory Environment Foundation

## Overview

Implementation tasks for Phase 1 foundational infrastructure, organized by component.

______________________________________________________________________

## 1. BrainParams Extensions

### 1.1 Add Sensory Fields

- [x] Add `temperature: float | None` to BrainParams
- [x] Add `temperature_gradient_strength: float | None` to BrainParams
- [x] Add `temperature_gradient_direction: float | None` to BrainParams
- [x] Add `cultivation_temperature: float | None` to BrainParams
- [x] Add `health: float | None` to BrainParams
- [x] Add `max_health: float | None` to BrainParams
- [x] Add `boundary_contact: bool | None` to BrainParams
- [x] Add `predator_contact: bool | None` to BrainParams
- [x] Update docstrings with field descriptions

**Validation**: Existing tests pass, new fields default to None ✅

______________________________________________________________________

## 2. Health System

> **Note**: HP and satiety are independent systems that coexist. HP tracks threat-based damage (predators, temperature), while satiety tracks time-based hunger. Food restores BOTH.

### 2.1 Environment Health Tracking

- [x] Add `health_enabled: bool` to DynamicForagingEnvironment
- [x] Add `agent_hp: float` and `max_hp: float` to environment state
- [x] Add `HealthConfig` dataclass for configuration
- [x] Implement HP initialization on episode reset
- [x] Ensure HP system operates independently from existing satiety system

### 2.2 Damage and Healing

- [x] Implement predator damage on contact (configurable `predator_damage`)
- [x] Implement food healing (configurable `food_healing`)
- [x] Ensure food consumption restores both HP AND satiety when both systems enabled
- [x] Add temperature damage (for thermotaxis integration) ✅ *(implemented in add-thermotaxis-system)*
- [x] Cap HP at max_hp, floor at 0

### 2.3 Termination

- [x] Add `TerminationReason.HEALTH_DEPLETED` enum value
- [x] Implement HP depletion check in step function
- [x] Return appropriate termination when HP reaches 0
- [x] Document distinction from STARVATION termination (satiety system)

### 2.4 Configuration

- [x] Add `health` section to environment YAML schema
- [x] Add config loader support for health system
- [x] Create example config with health system enabled
- [x] Create example config with BOTH health system and satiety enabled

**Validation**: Agent can survive predator contact, die from accumulated damage. Food restores both HP and satiety.

### 2.5 Health System Observability

> **Note**: When adding new tracking systems (like health), ensure full observability coverage across console output, session-level aggregates, per-run exports, and visualizations.

#### Console Output

- [x] Display per-run health value in simulation summary (e.g., `Health: 85.0`)
- [x] Display "Failed runs - Health Depleted" count in session summary
- [x] Format health values to 1 decimal place for readability
- [x] Track `total_health_depleted` counter at session level in run_simulation.py

#### Session-Level Reporting

- [x] Add `total_health_depleted` to PerformanceMetrics
- [x] Include health depletion percentage in session summary output
- [x] Generate session-level health progression plot (multi-run overlay)

#### Per-Run Exports

- [x] Export `health_history.csv` per run (step-by-step HP values)
- [x] Include health metrics in `foraging_summary.csv` (final HP, died_to_health_depletion)
- [x] Generate per-run health progression plot (`health_progression.png`)

#### Data Capture

- [x] Track health at each step via `EpisodeTracker.track_health()`
- [x] Capture final 0 HP value before early return on health depletion
- [x] Include `health_history` in EpisodeTrackingData and SimulationResult
- [x] Track `died_to_health_depletion` boolean in SimulationResult

**Validation**: Console shows health per run and session totals, exports include health history CSV and plots at both session and per-run level.

______________________________________________________________________

## 3. Enhanced Predator Behaviors

### 3.1 Predator Type Refactor

- [x] Add `PredatorType` enum (RANDOM, STATIONARY, PURSUIT)
- [x] Add `predator_type` field to Predator class
- [x] Refactor `Predator.move()` to dispatch by type

### 3.2 Stationary Predator

- [x] Implement stationary behavior (no movement)
- [x] Add configurable `damage_radius` for toxic zones
- [x] Stationary predators affect larger area than random predators

### 3.3 Pursuit Predator

- [x] Implement pursuit behavior (move toward agent)
- [x] Add `detection_radius` for pursuit activation
- [x] Pursuit only activates when agent within detection radius
- [x] Outside detection radius, behave as random

### 3.4 Mixed Types Configuration

- [x] Update predator config schema to support type selection (movement_pattern: random|stationary|pursuit)

> **Note**: Full mixed-type support (multiple types per environment) deferred to future work.
> Current implementation supports one type per environment via PredatorParams.predator_type.

- [x] Implement mixed-type spawning (types list in config) - defer

- [x] Update gradient computation for different predator types - defer, we need to implement mixed-type predator spawning first

**Validation**: Pursuit predators track agent, stationary predators don't move ✅

______________________________________________________________________

## 4. Mechanosensation

### 4.1 Boundary Detection

- [x] Detect when agent is at grid boundary (x=0, x=max, y=0, y=max)
- [x] Set `boundary_contact` in BrainParams
- [x] Add boundary collision penalty to reward calculator

### 4.2 Predator Contact Detection

- [x] Detect when agent is within predator kill radius
- [x] Set `predator_contact` in BrainParams
- [x] Distinguish contact from proximity (already tracked)

### 4.3 Feature Extraction Module

- [x] Create `mechanosensation_features()` in modules.py
- [x] Encode boundary contact as RX rotation
- [x] Encode predator contact as RY rotation
- [x] Add ALM, PLM, AVM neuron references to docstring

**Validation**: BrainParams correctly reports contact states ✅

______________________________________________________________________

## 5. Unified Feature Extraction

### 5.1 Unified SensoryModule Architecture

> **Note**: The original `brain/features.py` was consolidated into `brain/modules.py` during refactoring.

- [x] Create `SensoryModule` dataclass with unified interface:
  - `extract(params) -> CoreFeatures` - architecture-agnostic extraction
  - `to_quantum(params) -> np.ndarray` - returns [rx, ry, rz] gate angles
  - `to_classical(params) -> np.ndarray` - returns [strength, angle] semantic values
  - `to_quantum_dict(params) -> dict` - convenience method for ModularBrain
- [x] Create `CoreFeatures` dataclass with semantic ranges:
  - `strength: float` in [0, 1] where 0 = no signal
  - `angle: float` in [-1, 1] where 0 = aligned with agent
  - `binary: float` for on/off signals
- [x] Build `SENSORY_MODULES` registry as single source of truth
- [x] Add `extract_classical_features()` for PPOBrain
- [x] Add `get_classical_feature_dimension()` utility
- [x] Delete `brain/features.py` (consolidated into modules.py)

### 5.2 Module Renaming and Registry Consolidation

- [x] Rename `appetitive_features` to `food_chemotaxis_features`
- [x] Rename `aversive_features` to `nociception_features`
- [x] Add neuron references to all `SensoryModule` descriptions:
  - chemotaxis: ASE neurons ✅
  - food_chemotaxis: AWC, AWA neurons ✅
  - nociception: ASH, ADL neurons ✅
  - thermotaxis: AFD neurons ✅
  - aerotaxis: URX, BAG neurons ✅
  - mechanosensation: ALM, PLM, AVM neurons ✅
  - proprioception: DVA, PVD neurons ✅
  - vision: ASJ, ASI neurons ✅
  - action: interneurons ✅
- [x] Remove old registries (`MODULE_FEATURE_EXTRACTORS`, `CORE_FEATURE_EXTRACTORS`)
- [x] Remove standalone quantum transform functions (consolidated into `SensoryModule` methods)
- [x] Update `ModuleName` enum with all 9 modules
- [x] Add backward compatibility aliases:
  - `appetitive` → `food_chemotaxis`
  - `aversive` → `nociception`
  - `oxygen` → `aerotaxis`

### 5.3 Integration with Brains

- [x] ModularBrain uses `SENSORY_MODULES[module].to_quantum_dict(params)`
- [x] QModularBrain uses `SENSORY_MODULES[module].to_quantum_dict(params)`
- [x] PPOBrain uses unified extraction via `extract_classical_features()`:
  - Added `sensory_modules` config option to `PPOBrainConfig`
  - Each module contributes 2 features [strength, angle] with semantic ranges
  - Auto-computes `input_dim` from modules (2 features per module)
  - Legacy mode (default) still uses 2-feature preprocessing
- [x] Ensure backward compatibility with existing configs (legacy module names still work)

### 5.4 PPO Training Improvements for Multi-Feature Learning

> **Context**: With unified sensory modules (4 features: food_chemotaxis + nociception), PPO required additional tuning to match legacy 2-feature performance. The network must independently learn food=good and predator=bad correlations.

- [x] Fixed predator gradient direction semantics (point TOWARD danger, not away)
- [x] Added reward shaping for multi-feature credit assignment:
  - `reward_distance_scale: 0.3` (reduced from 0.5)
  - `penalty_predator_proximity: 0.3` (increased from 0.1)
  - `penalty_health_damage: 1.5` (increased from 0.5)
- [x] Added learning rate scheduling to PPOBrainConfig:
  - `lr_warmup_episodes: int` - episodes to warm up LR
  - `lr_warmup_start: float | None` - starting LR (default 10% of base)
  - `lr_decay_episodes: int | None` - episodes to decay LR after warmup
  - `lr_decay_end: float | None` - final LR (default 10% of base)
- [x] Added LR scheduling tests to test_ppo.py (8 tests)

**Results**: Unified 4-feature mode achieves 73% late-stage success (within 2% of legacy's 75%)

**Validation**: Unified SensoryModule architecture complete ✅

- Single source of truth: `SENSORY_MODULES` registry
- All modules usable by both quantum and classical brains
- Clear interface: `module.to_quantum(params)` vs `module.to_classical(params)`
- Scientific documentation lives with module definitions

______________________________________________________________________

## 6. Multi-Objective Rewards

### 6.1 RewardConfig Extensions

- [x] Add `reward_temperature_comfort: float` ✅ *(implemented in add-thermotaxis-system)*
- [x] Add `penalty_temperature_discomfort: float` ✅ *(implemented in add-thermotaxis-system)*
- [x] Add `penalty_temperature_danger: float` ✅ *(implemented in add-thermotaxis-system)*
- [x] Add `hp_damage_temperature_danger: float` ✅ *(implemented in add-thermotaxis-system)*
- [x] Add `hp_damage_temperature_lethal: float` ✅ *(implemented in add-thermotaxis-system)*
- [x] Add `reward_health_gain: float`
- [x] Add `penalty_health_damage: float`
- [x] Add `penalty_boundary_collision: float`

### 6.2 RewardCalculator Updates

- [x] Add temperature comfort/discomfort reward calculation ✅ *(implemented in add-thermotaxis-system)*
- [x] Add health-based reward calculation
- [x] Add boundary collision penalty
- [x] Ensure rewards are only applied when features are enabled

**Validation**: Rewards correctly computed for multi-objective scenarios

______________________________________________________________________

## 7. Evaluation Extensions

### 7.1 SimulationResult Extensions

- [x] Add `temperature_comfort_score: float | None` ✅ *(implemented in add-thermotaxis-system via get_temperature_comfort_score())*
- [x] Add `survival_score: float | None` ✅
- [x] Decision: Use continuous `temperature_comfort_score` instead of binary `thermotaxis_success`

### 7.2 Composite Score Update

- [x] Extend `calculate_composite_score()` to handle multi-objective metrics ✅
- [x] When multi-objective enabled: reweight base components with hierarchical scoring ✅
  - Survival acts as gate: if survival_score < 0.1, score capped at 30% of primary
  - With thermotaxis: 50% success, 15% survival, 10% comfort, 15% efficiency, 10% learning
  - Without thermotaxis: 50% success, 20% survival, 20% efficiency, 10% learning
- [x] Document updated scoring formula ✅ *(in calculate_composite_score docstring)*

### 7.3 Experiment Tracking

- [x] Track per-objective scores during episode ✅
- [x] Include multi-objective metrics in experiment JSON ✅
  - Added `survival_score` and `temperature_comfort_score` to PerRunResult
  - Added avg/post-convergence fields to ResultsMetadata
  - Added avg fields to ConvergenceMetrics

### 7.4 Baseline Configuration

- [x] Create pure isothermal thermotaxis config (`ppo_thermotaxis_isothermal_small.yml`) ✅
  - Uniform temperature (gradient_strength=0) for baseline comparison

**Validation**: Multi-objective scores computed and tracked correctly ✅

______________________________________________________________________

## 8. Food Spawning in Safe Zones

> **Status**: Implemented in `feature/add-thermotaxis-benchmarking` branch.

### 8.1 Safe Zone Spawning

- [x] Add `safe_zone_food_bias: float` config parameter (default 0.0) to `ForagingParams`
  - Parameter name changed from `food_safe_zone_ratio` to `safe_zone_food_bias` for clarity
  - Located in `ForagingParams` dataclass alongside other foraging config
- [x] Implement `_is_safe_temperature_zone(position)` helper method
  - Returns `True` for COMFORT, DISCOMFORT_COLD, DISCOMFORT_HOT zones (no HP damage)
  - Returns `True` if thermotaxis disabled (all positions are "safe")
- [x] Bias food spawning toward safe temperature zones when thermotaxis enabled
  - Modified `_initialize_foods()` and `spawn_food()` to respect bias
  - With probability `bias`, only accepts positions in safe zones
- [x] Fall back to random spawning when thermotaxis disabled
  - `_is_safe_temperature_zone()` returns `True` for all positions when thermotaxis disabled

**Validation**: Food distribution biased toward safe zones ✅

- Config: `safe_zone_food_bias: 0.8` in `ppo_thermotaxis_foraging_large.yml`
- Expected: ~80% of food in COMFORT/DISCOMFORT zones, ~20% anywhere

______________________________________________________________________

## 9. Visualization

> **Status**: Temperature zone and toxic zone visualization implemented in `add-thermotaxis-system`.

### 9.1 Temperature Zone Coloring

- [x] Extend Rich theme to support background colors ✅ *(implemented in add-thermotaxis-system)*
- [x] Implement temperature zone color mapping ✅ *(implemented in add-thermotaxis-system)*:
  - Lethal cold (\<5°C): blue
  - Danger cold (5-10°C): cyan
  - Discomfort cold (10-15°C): light cyan
  - Comfort (15-25°C): default (no background)
  - Discomfort hot (25-30°C): light goldenrod
  - Danger hot (30-35°C): orange
  - Lethal hot (>35°C): red
- [x] Implement priority system: Agent > Predators > Food > Temperature ✅ *(implemented in add-thermotaxis-system)*
  - Foreground styles (entity colors) combined with background styles (zone colors)
  - Toxic zones (stationary predator damage radius) have highest background priority

### 9.2 Toxic Zone Coloring

- [x] Add toxic zone background for stationary predator damage_radius ✅ *(implemented in add-thermotaxis-system)*
  - Purple (`on medium_purple`) background
  - Priority over temperature zones
- [x] Add predator foreground styles ✅ *(implemented in add-thermotaxis-system)*
  - Random: magenta, Stationary: dark_magenta, Pursuit: yellow

### 9.3 Debug Logging

- [ ] Add environment snapshot logging at episode start
- [ ] Include temperature samples at key positions *(deferred)*
- [ ] Log health system state changes

**Validation**: 9 tests in test_env.py::TestZoneVisualization ✅

______________________________________________________________________

## 10. Hierarchical Benchmark Naming

### 10.1 Category Infrastructure

- [ ] Define benchmark category hierarchy:

  ```text
  basic/          # Single objective (foraging only)
  survival/       # Food + predators
  thermotaxis/    # Temperature-aware (reserved for add-thermotaxis-system)
  multisensory/   # Multiple modalities (future)
  ablation/       # Controlled studies (reserved for add-ablation-toolkit)
  ```

- [ ] Update benchmark categorization logic to support hierarchical paths

- [ ] Implement path pattern: `{category}/{task}_{size}/{brain_type}`

### 10.2 Naming Convention

- [ ] Document naming convention: task names always explicit about what's included
  - `foraging` = food collection goal
  - `predator` = predators enabled
  - `thermo` = thermotaxis enabled (short form)
  - Modifiers follow the base: `foraging_predator_small`, not `predator_small`

### 10.3 Migration

- [ ] Migrate `foraging_small` → `basic/foraging_small`
- [ ] Migrate `foraging_medium` → `basic/foraging_medium`
- [ ] Migrate `predator_small` → `survival/foraging_predator_small`
- [ ] Add backward compatibility mapping for flat category names
- [ ] Update leaderboard to display hierarchical categories

### 10.4 Validation

- [ ] Ensure existing benchmark results remain valid after migration
- [ ] Test category detection for new benchmark runs

**Validation**: New benchmarks use hierarchical paths, old results still accessible

______________________________________________________________________

## 11. Configuration and Documentation

### 11.1 Example Configs

- [x] Create `configs/examples/ppo_thermotaxis_foraging_small.yml` ✅ *(implemented in add-thermotaxis-system)*
- [x] Create `configs/examples/ppo_thermotaxis_stationary_predators_small.yml` ✅ *(implemented in add-thermotaxis-system)*
- [x] Create `configs/examples/ppo_thermotaxis_pursuit_predators_small.yml` ✅ *(implemented in add-thermotaxis-system)*
- [x] Create `configs/examples/ppo_health_predators_small.yml`
- [x] Create `configs/examples/ppo_health_satiety_predators_small.yml`
- [x] Create `configs/examples/ppo_pursuit_predators_small.yml`
- [x] Create `configs/examples/ppo_stationary_predators_small.yml`

### 11.2 Documentation

- [ ] Update environment documentation with new features
- [ ] Document health system configuration
- [ ] Document predator type configuration
- [ ] Document multi-objective reward configuration
- [ ] Document hierarchical benchmark category structure

______________________________________________________________________

## Dependencies

```text
1. BrainParams Extensions ──┐
                            │
2. Health System ───────────┼──► 6. Multi-Objective Rewards
                            │
3. Enhanced Predators ──────┤
                            │
4. Mechanosensation ────────┼──► 5. Unified Feature Extraction
                            │
                            └──► 7. Evaluation Extensions
                                        │
                                        v
                            8. Food Spawning + 9. Visualization
                                        │
                                        v
                            10. Hierarchical Benchmarks + 11. Config/Docs
```

Work streams 1-4 can proceed in parallel. Work stream 5 depends on 4. Work streams 6-7 depend on 1-4. Work streams 8-9 depend on all above. Work stream 10 can proceed independently but should be done before 11.
