# Tasks: Add Thermotaxis System

## Overview

Implementation tasks for thermotaxis sensory system, building on the foundation from `add-multi-sensory-environment`.

**Prerequisites:** Complete `add-multi-sensory-environment` first (BrainParams extensions, health system, reward config).

---

## 1. TemperatureField Class

### 1.1 Create Temperature Module
- [ ] Create `packages/quantum-nematode/quantumnematode/env/temperature.py`
- [ ] Define `TemperatureField` dataclass with fields:
  - `grid_size: int`
  - `base_temperature: float` (default 20.0, cultivation temperature)
  - `gradient_direction: float` (radians, 0 = temp increases to right)
  - `gradient_strength: float` (degrees per cell, default 0.5)
  - `hot_spots: list[tuple[int, int, float]]` (x, y, intensity)
  - `cold_spots: list[tuple[int, int, float]]` (x, y, intensity)

### 1.2 Temperature Computation
- [ ] Implement `get_temperature(position: tuple[int, int]) -> float`
  - Linear gradient component based on direction and strength
  - Add hot spot contributions (exponential falloff)
  - Add cold spot contributions (exponential falloff)
- [ ] Implement `get_gradient(position: tuple[int, int]) -> tuple[float, float]`
  - Central difference approximation
  - Return (dx, dy) gradient vector

### 1.3 Temperature Zone Classification
- [ ] Implement `get_zone(temperature: float) -> TemperatureZone`
- [ ] Define `TemperatureZone` enum: LETHAL_COLD, DANGER_COLD, DISCOMFORT_COLD, COMFORT, DISCOMFORT_HOT, DANGER_HOT, LETHAL_HOT
- [ ] Use configurable thresholds from environment config

**Validation**: Unit tests for temperature calculation and zone classification

---

## 2. Environment Integration

### 2.1 Temperature Field in Environment
- [ ] Add `thermotaxis_enabled: bool` to DynamicForagingEnvironment
- [ ] Add `temperature_field: TemperatureField | None` attribute
- [ ] Initialize temperature field from config when enabled
- [ ] Add `ThermotaxisConfig` dataclass for YAML configuration

### 2.2 Temperature in Step Execution
- [ ] Compute temperature at agent position each step
- [ ] Populate `BrainParams.temperature` with current temperature
- [ ] Populate `BrainParams.temperature_gradient_strength`
- [ ] Populate `BrainParams.temperature_gradient_direction`
- [ ] Populate `BrainParams.cultivation_temperature`

### 2.3 Temperature Zone Effects
- [ ] Apply comfort reward when in comfort zone
- [ ] Apply discomfort penalty when in discomfort zone
- [ ] Apply danger penalty + HP damage when in danger zone
- [ ] Apply lethal HP damage when in lethal zone
- [ ] Track temperature_comfort_score (time in comfort / total time)

### 2.4 Temperature Termination
- [ ] Check for HP depletion from temperature damage
- [ ] Log temperature as cause when terminating from temperature damage

**Validation**: Integration test with agent navigating temperature gradient

---

## 3. Feature Extraction

### 3.1 Thermotaxis Features Implementation

> **Note**: Uses spatial gradient sensing (matches chemotaxis pattern). Temporal sensing (biologically accurate) to be added when memory systems are implemented per roadmap.

- [ ] Implement `thermotaxis_features(params: BrainParams) -> dict[RotationAxis, float]`
- [ ] RX: Temperature deviation from cultivation temperature
  - `temp_deviation = (current_temp - cultivation_temp) / 15.0` (normalized)
  - Scale to [-π/2, π/2]
- [ ] RY: Relative angle to temperature gradient (spatial)
  - Egocentric (relative to agent facing direction)
  - Scale to [-π/2, π/2]
- [ ] RZ: Temperature gradient strength
  - `tanh(gradient_magnitude) * π/2`
- [ ] Handle None values gracefully (return zeros)
- [ ] Add docstring note documenting spatial vs temporal sensing trade-off

### 3.2 Module Registration
- [ ] Add `ModuleName.THERMOTAXIS` to enum (if not already present)
- [ ] Register `thermotaxis_features` in `MODULE_FEATURE_EXTRACTORS`
- [ ] Add AFD neuron reference to docstring

### 3.3 Unified Feature Extraction Integration
- [ ] Add thermotaxis to `extract_sensory_features()` in features.py
- [ ] Ensure PPOBrain receives thermotaxis features in input vector
- [ ] Ensure ModularBrain can map thermotaxis module to qubits

**Validation**: Unit tests for feature extraction with various temperature values

---

## 4. Benchmarks and Validation

### 4.1 Thermotaxis Benchmark Configs
- [ ] Create `configs/examples/thermotaxis_foraging_small.yml`
  - 20x20 grid, food collection goal
  - Thermotaxis enabled with linear gradient
  - Health system enabled
  - Success: collect all food + stay >60% in comfort zone
- [ ] Create `configs/examples/thermotaxis_foraging_medium.yml`
- [ ] Create `configs/examples/thermotaxis_isothermal_small.yml`
  - Pure thermotaxis task (no food collection goal)
  - Success: stay >80% in comfort zone for episode duration

### 4.2 Benchmark Categories (Hierarchical Naming)

Adopt hierarchical benchmark naming convention for scalability:
```
thermotaxis/                    # Temperature-aware tasks
├── isothermal_small/           # Pure temp comfort (no food goal)
│   ├── quantum/
│   └── classical/
├── foraging_small/             # Food + temp constraint
├── foraging_predator_small/    # Food + temp + predators
└── foraging_medium/
```

- [ ] Add `thermotaxis/` top-level benchmark category
- [ ] Define category paths: `thermotaxis/isothermal_small`, `thermotaxis/foraging_small`
- [ ] Update benchmark categorization logic to support hierarchical paths
- [ ] Ensure backward compatibility with existing `basic/` and `survival/` categories

### 4.3 Biological Validation
- [ ] Research C. elegans thermotaxis literature for validation targets
- [ ] Document expected isothermal tracking behavior
- [ ] Create validation metrics comparing agent to biological data

**Validation**: Run benchmarks with PPO and ModularBrain

---

## 5. Visualization

### 5.1 Temperature Background Colors
- [ ] Extend Rich theme renderer to query temperature at each cell
- [ ] Apply background color based on temperature zone
- [ ] Ensure entity rendering (agent, food, predator) takes priority over background

### 5.2 Debug Logging
- [ ] Log temperature field parameters at episode start
- [ ] Log temperature samples at corners and center
- [ ] Log agent's current temperature in step output

**Validation**: Visual inspection of temperature rendering in Rich theme

---

## 6. Documentation

### 6.1 Configuration Documentation
- [ ] Document thermotaxis configuration options
- [ ] Provide example configurations with explanations
- [ ] Document temperature zone thresholds

### 6.2 Architecture Documentation
- [ ] Document thermotaxis_features() computation
- [ ] Document AFD neuron modeling approach
- [ ] Add to optimization methods documentation

---

## Dependencies

```
add-multi-sensory-environment (MUST be complete)
         │
         v
1. TemperatureField Class
         │
         v
2. Environment Integration ────► 3. Feature Extraction
         │                              │
         v                              v
4. Benchmarks ◄────────────────────────┘
         │
         v
5. Visualization + 6. Documentation
```

All tasks in this proposal depend on `add-multi-sensory-environment` being complete first.

---

## Success Criteria

- [ ] TemperatureField computes correct temperatures and gradients
- [ ] Agent receives temperature in BrainParams when thermotaxis enabled
- [ ] thermotaxis_features() produces valid rotation values
- [ ] PPO can learn to navigate to comfort zone (>60% time)
- [ ] ModularBrain with thermotaxis module achieves comparable performance
- [ ] Combined chemotaxis+thermotaxis task is learnable
- [ ] Temperature zones visible in Rich theme rendering
