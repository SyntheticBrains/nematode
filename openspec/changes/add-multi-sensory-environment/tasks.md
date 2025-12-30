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

- [x] Add `health_system_enabled: bool` to DynamicForagingEnvironment
- [x] Add `agent_hp: float` and `max_hp: float` to environment state
- [x] Add `HealthSystemConfig` dataclass for configuration
- [x] Implement HP initialization on episode reset
- [x] Ensure HP system operates independently from existing satiety system

### 2.2 Damage and Healing

- [x] Implement predator damage on contact (configurable `predator_damage`)
- [x] Implement food healing (configurable `food_healing`)
- [ ] Ensure food consumption restores both HP AND satiety when both systems enabled
- [ ] Add temperature damage (for thermotaxis integration)
- [x] Cap HP at max_hp, floor at 0

### 2.3 Termination

- [x] Add `TerminationReason.HEALTH_DEPLETED` enum value
- [x] Implement HP depletion check in step function
- [ ] Return appropriate termination when HP reaches 0
- [x] Document distinction from STARVATION termination (satiety system)

### 2.4 Configuration

- [ ] Add `health_system` section to environment YAML schema
- [ ] Add config loader support for health system
- [ ] Create example config with health system enabled
- [ ] Create example config with BOTH health system and satiety enabled

**Validation**: Agent can survive predator contact, die from accumulated damage. Food restores both HP and satiety.

______________________________________________________________________

## 3. Enhanced Predator Behaviors

### 3.1 Predator Type Refactor

- [ ] Add `PredatorType` enum (RANDOM, STATIONARY, PURSUIT)
- [ ] Add `predator_type` field to Predator class
- [ ] Refactor `Predator.move()` to dispatch by type

### 3.2 Stationary Predator

- [ ] Implement stationary behavior (no movement)
- [ ] Add configurable `damage_radius` for toxic zones
- [ ] Stationary predators affect larger area than random predators

### 3.3 Pursuit Predator

- [ ] Implement pursuit behavior (move toward agent)
- [ ] Add `detection_radius` for pursuit activation
- [ ] Pursuit only activates when agent within detection radius
- [ ] Outside detection radius, behave as random

### 3.4 Mixed Types Configuration

- [ ] Update predator config schema to support type list:

  ```yaml
  types:
    - type: stationary
      count: 2
    - type: pursuit
      count: 1
  ```

- [ ] Implement mixed-type spawning

- [ ] Update gradient computation for different predator types

**Validation**: Pursuit predators track agent, stationary predators don't move

______________________________________________________________________

## 4. Mechanosensation

### 4.1 Boundary Detection

- [ ] Detect when agent is at grid boundary (x=0, x=max, y=0, y=max)
- [ ] Set `boundary_contact` in BrainParams
- [ ] Add boundary collision penalty to reward calculator

### 4.2 Predator Contact Detection

- [ ] Detect when agent is within predator kill radius
- [ ] Set `predator_contact` in BrainParams
- [ ] Distinguish contact from proximity (already tracked)

### 4.3 Feature Extraction Module

- [ ] Create `mechanosensation_features()` in modules.py
- [ ] Encode boundary contact as RX rotation
- [ ] Encode predator contact as RY rotation
- [ ] Add ALM, PLM, AVM neuron references to docstring

**Validation**: BrainParams correctly reports contact states

______________________________________________________________________

## 5. Unified Feature Extraction

### 5.1 Create Feature Extraction Layer

- [ ] Create `brain/features.py` module
- [ ] Implement `extract_sensory_features(params: BrainParams) -> dict[str, np.ndarray]`
- [ ] Return feature vectors for each sensory modality

### 5.2 Module Renaming

- [ ] Rename `appetitive_features` to `food_chemotaxis_features`
- [ ] Rename `aversive_features` to `nociception_features`
- [ ] Add neuron references to all module docstrings:
  - chemotaxis: ASE neurons
  - food_chemotaxis: AWC, AWA neurons
  - nociception: ASH, ADL neurons
  - thermotaxis: AFD neurons
  - aerotaxis: URX, BAG neurons
  - mechanosensation: ALM, PLM, AVM neurons
- [ ] Update MODULE_FEATURE_EXTRACTORS dict
- [ ] Update ModuleName enum

### 5.3 Integration with Brains

- [ ] Update ModularBrain to use unified extraction (convert to RX/RY/RZ)
- [ ] Update PPOBrain to use unified extraction (concatenate to input)
- [ ] Ensure backward compatibility with existing configs

**Validation**: Both brain types can consume new sensory features

______________________________________________________________________

## 6. Multi-Objective Rewards

### 6.1 RewardConfig Extensions

- [ ] Add `reward_temperature_comfort: float`
- [ ] Add `penalty_temperature_discomfort: float`
- [ ] Add `penalty_temperature_danger: float`
- [ ] Add `hp_damage_temperature_danger: float`
- [ ] Add `hp_damage_temperature_lethal: float`
- [ ] Add `reward_health_gain: float`
- [ ] Add `penalty_health_damage: float`
- [ ] Add `penalty_boundary_collision: float`

### 6.2 RewardCalculator Updates

- [ ] Add temperature comfort/discomfort reward calculation
- [ ] Add health-based reward calculation
- [ ] Add boundary collision penalty
- [ ] Ensure rewards are only applied when features are enabled

**Validation**: Rewards correctly computed for multi-objective scenarios

______________________________________________________________________

## 7. Evaluation Extensions

### 7.1 SimulationResult Extensions

- [ ] Add `temperature_comfort_score: float | None`
- [ ] Add `survival_score: float | None`
- [ ] Add `thermotaxis_success: bool | None`

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

### 8.1 Safe Zone Spawning

- [ ] Add `food_safe_zone_ratio: float` config parameter (default 0.8)
- [ ] Implement `sample_safe_position()` method
- [ ] Bias food spawning toward safe temperature zones when thermotaxis enabled
- [ ] Fall back to random spawning when thermotaxis disabled

**Validation**: Food distribution biased toward safe zones

______________________________________________________________________

## 9. Visualization

### 9.1 Temperature Zone Coloring

- [ ] Extend Rich theme to support background colors
- [ ] Implement temperature zone color mapping:
  - Lethal cold (\<5°C): blue
  - Danger cold (5-10°C): cyan
  - Discomfort cold (10-15°C): light cyan
  - Comfort (15-25°C): white (default)
  - Discomfort hot (25-30°C): light yellow
  - Danger hot (30-35°C): yellow
  - Lethal hot (>35°C): red
- [ ] Implement priority system: Agent > Predators > Food > Temperature

### 9.2 Debug Logging

- [ ] Add environment snapshot logging at episode start
- [ ] Include temperature samples at key positions
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

- [ ] Create `configs/examples/thermotaxis_foraging_small.yml`
- [ ] Create `configs/examples/health_system_demo.yml`
- [ ] Create `configs/examples/pursuit_predators.yml`

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
