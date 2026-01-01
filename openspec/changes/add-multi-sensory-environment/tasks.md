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
- [ ] Add temperature damage (for thermotaxis integration) *(deferred to add-thermotaxis-system)*
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

### 5.1 Create Feature Extraction Layer

- [x] Create `brain/features.py` module
- [x] Implement `extract_sensory_features(params: BrainParams) -> dict[str, np.ndarray]`
- [x] Return feature vectors for each sensory modality
- [x] Add `extract_flat_features()` for classical networks (PPO)
- [x] Add `get_feature_dimension()` utility

### 5.2 Module Renaming

- [x] Rename `appetitive_features` to `food_chemotaxis_features`
- [x] Rename `aversive_features` to `nociception_features`
- [x] Add neuron references to all module docstrings:
  - chemotaxis: ASE neurons ✅
  - food_chemotaxis: AWC, AWA neurons ✅
  - nociception: ASH, ADL neurons ✅
  - thermotaxis: AFD neurons ✅
  - aerotaxis: URX, BAG neurons ✅
  - mechanosensation: ALM, PLM, AVM neurons ✅
- [x] Update MODULE_FEATURE_EXTRACTORS dict
- [x] Update ModuleName enum
- [x] Add backward compatibility aliases (appetitive→food_chemotaxis, aversive→nociception, oxygen→aerotaxis)

### 5.3 Integration with Brains

- [x] ModularBrain already uses unified extraction via `extract_features_for_module()`
- [ ] Update PPOBrain to optionally use unified extraction (concatenate to input) *(optional enhancement)*
- [x] Ensure backward compatibility with existing configs (legacy module names still work)

**Validation**: Feature extraction layer complete with backward compatibility ✅

______________________________________________________________________

## 6. Multi-Objective Rewards

### 6.1 RewardConfig Extensions

- [ ] Add `reward_temperature_comfort: float` *(deferred to add-thermotaxis-system)*
- [ ] Add `penalty_temperature_discomfort: float` *(deferred to add-thermotaxis-system)*
- [ ] Add `penalty_temperature_danger: float` *(deferred to add-thermotaxis-system)*
- [ ] Add `hp_damage_temperature_danger: float` *(deferred to add-thermotaxis-system)*
- [ ] Add `hp_damage_temperature_lethal: float` *(deferred to add-thermotaxis-system)*
- [x] Add `reward_health_gain: float`
- [x] Add `penalty_health_damage: float`
- [x] Add `penalty_boundary_collision: float`

### 6.2 RewardCalculator Updates

- [ ] Add temperature comfort/discomfort reward calculation *(deferred to add-thermotaxis-system)*
- [x] Add health-based reward calculation
- [x] Add boundary collision penalty
- [x] Ensure rewards are only applied when features are enabled

**Validation**: Rewards correctly computed for multi-objective scenarios

______________________________________________________________________

## 7. Evaluation Extensions

### 7.1 SimulationResult Extensions

- [ ] Add `temperature_comfort_score: float | None` *(deferred to add-thermotaxis-system)*
- [ ] Add `survival_score: float | None`
- [ ] Add `thermotaxis_success: bool | None` *(deferred to add-thermotaxis-system)*

### 7.2 Composite Score Update

- [ ] Extend `calculate_composite_score()` to handle multi-objective metrics
- [ ] When multi-objective enabled: reweight base components
- [ ] Document updated scoring formula

### 7.3 Experiment Tracking

- [ ] Track per-objective scores during episode
- [ ] Include multi-objective metrics in experiment JSON

**Validation**: Multi-objective scores computed and tracked correctly

______________________________________________________________________

## 8. Food Spawning in Safe Zones

> **Note**: This work stream is deferred to `add-thermotaxis-system` as it depends on temperature zones.

### 8.1 Safe Zone Spawning

- [ ] Add `food_safe_zone_ratio: float` config parameter (default 0.8) *(deferred to add-thermotaxis-system)*
- [ ] Implement `sample_safe_position()` method *(deferred to add-thermotaxis-system)*
- [ ] Bias food spawning toward safe temperature zones when thermotaxis enabled *(deferred to add-thermotaxis-system)*
- [ ] Fall back to random spawning when thermotaxis disabled *(deferred to add-thermotaxis-system)*

**Validation**: Food distribution biased toward safe zones

______________________________________________________________________

## 9. Visualization

> **Note**: Temperature zone visualization is deferred to `add-thermotaxis-system`.

### 9.1 Temperature Zone Coloring

- [ ] Extend Rich theme to support background colors *(deferred to add-thermotaxis-system)*
- [ ] Implement temperature zone color mapping *(deferred to add-thermotaxis-system)*:
  - Lethal cold (\<5°C): blue
  - Danger cold (5-10°C): cyan
  - Discomfort cold (10-15°C): light cyan
  - Comfort (15-25°C): white (default)
  - Discomfort hot (25-30°C): light yellow
  - Danger hot (30-35°C): yellow
  - Lethal hot (>35°C): red
- [ ] Implement priority system: Agent > Predators > Food > Temperature *(deferred to add-thermotaxis-system)*

### 9.2 Debug Logging

- [ ] Add environment snapshot logging at episode start
- [ ] Include temperature samples at key positions *(deferred to add-thermotaxis-system)*
- [ ] Log health system state changes

**Validation**: Temperature zones visible in Rich theme output

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

- [ ] Create `configs/examples/ppo_thermotaxis_foraging_small.yml` *(deferred to add-thermotaxis-system)*
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
