# Tasks: Add Thermotaxis System

## Overview

Implementation tasks for thermotaxis sensory system, building on the foundation from `add-multi-sensory-environment`.

**Prerequisites:** Complete `add-multi-sensory-environment` first (BrainParams extensions, health system, reward config).

______________________________________________________________________

## 1. TemperatureField Class

### 1.1 Create Temperature Module

- [x] Create `packages/quantum-nematode/quantumnematode/env/temperature.py`
- [x] Define `TemperatureField` dataclass with fields:
  - `grid_size: int`
  - `base_temperature: float` (default 20.0, cultivation temperature)
  - `gradient_direction: float` (radians, 0 = temp increases to right)
  - `gradient_strength: float` (degrees per cell, default 0.5)
  - `hot_spots: list[TemperatureSpot]` (x, y, intensity) - using typed alias
  - `cold_spots: list[TemperatureSpot]` (x, y, intensity) - using typed alias

### 1.2 Temperature Computation

- [x] Implement `get_temperature(position: GridPosition) -> float`
  - Linear gradient component based on direction and strength
  - Add hot spot contributions (exponential falloff)
  - Add cold spot contributions (exponential falloff)
- [x] Implement `get_gradient(position: GridPosition) -> GradientVector`
  - Central difference approximation
  - Return (dx, dy) gradient vector
- [x] Implement `get_gradient_polar(position: GridPosition) -> GradientPolar`
  - Returns (magnitude, direction) in polar coordinates

### 1.3 Temperature Zone Classification

- [x] Implement `get_zone(temperature: float) -> TemperatureZone`
- [x] Define `TemperatureZone` enum: LETHAL_COLD, DANGER_COLD, DISCOMFORT_COLD, COMFORT, DISCOMFORT_HOT, DANGER_HOT, LETHAL_HOT
- [x] Use configurable thresholds via `TemperatureZoneThresholds` dataclass

**Validation**: Unit tests for temperature calculation and zone classification ✅ (12 tests in test_temperature.py)

______________________________________________________________________

## 2. Environment Integration

### 2.1 Temperature Field in Environment

- [x] Add `thermotaxis: ThermotaxisParams` to DynamicForagingEnvironment
- [x] Add `temperature_field: TemperatureField | None` attribute
- [x] Initialize temperature field from config when enabled
- [x] Add `ThermotaxisParams` dataclass for YAML configuration

### 2.2 Temperature in Step Execution

- [x] Compute temperature at agent position each step (`get_temperature()`)
- [x] Populate `BrainParams.temperature` with current temperature
- [x] Populate `BrainParams.temperature_gradient_strength`
- [x] Populate `BrainParams.temperature_gradient_direction`
- [x] Populate `BrainParams.cultivation_temperature`

### 2.3 Temperature Zone Effects

- [x] Apply comfort reward when in comfort zone
- [x] Apply discomfort penalty when in discomfort zone
- [x] Apply danger penalty + HP damage when in danger zone
- [x] Apply lethal HP damage when in lethal zone
- [x] Track temperature_comfort_score (time in comfort / total time)
- [x] Implement `apply_temperature_effects()` method
- [x] Implement `get_temperature_comfort_score()` method
- [x] Implement `reset_thermotaxis()` method

### 2.4 Temperature Termination

- [x] Check for HP depletion from temperature damage (via existing health system)
- ~~Log temperature as cause when terminating from temperature damage~~ *(not needed - uses existing HEALTH_DEPLETED termination)*

### 2.5 Configuration Loader Support

- [ ] Add `ThermotaxisConfig` class to `config_loader.py`
- [ ] Add `thermotaxis` field to `DynamicEnvironmentConfig`
- [ ] Implement `to_params()` method for converting to `ThermotaxisParams`
- [ ] Add thermotaxis section to YAML schema documentation

**Validation**: Integration test with agent navigating temperature gradient ✅ (10 tests in test_env.py::TestThermotaxisIntegration)

______________________________________________________________________

## 3. Feature Extraction

### 3.1 Thermotaxis Features Implementation

> **Note**: Uses spatial gradient sensing (matches chemotaxis pattern). Temporal sensing (biologically accurate) to be added when memory systems are implemented per roadmap.

- [x] Implement `thermotaxis_features(params: BrainParams) -> dict[RotationAxis, float]`
- [x] RX: Temperature deviation from cultivation temperature
  - `temp_deviation = (current_temp - cultivation_temp) / 15.0` (normalized)
  - Scale to [-π/2, π/2]
- [x] RY: Relative angle to temperature gradient (spatial)
  - Egocentric (relative to agent facing direction)
  - Scale to [-π/2, π/2]
- [x] RZ: Temperature gradient strength
  - `tanh(gradient_magnitude) * π/2`
- [x] Handle None values gracefully (return zeros)
- [x] Add docstring note documenting spatial vs temporal sensing trade-off

### 3.2 Module Registration

- [x] Add `ModuleName.THERMOTAXIS` to enum (if not already present)
- [x] Register `thermotaxis_features` in `SENSORY_MODULES` registry
- [x] Add AFD neuron reference to docstring

### 3.3 Unified Feature Extraction Integration

- [x] Add thermotaxis to unified `SensoryModule` architecture in `modules.py`
- [x] Ensure PPOBrain receives thermotaxis features via `extract_classical_features()`
- [x] Ensure ModularBrain can map thermotaxis module to qubits via `to_quantum_dict()`

**Validation**: Unit tests for feature extraction with various temperature values ✅ (4 tests in test_modules.py::TestThermotaxisModule)

______________________________________________________________________

## 4. Benchmarks and Validation

### 4.1 Thermotaxis Benchmark Configs

- [x] Create `configs/examples/ppo_thermotaxis_foraging_small.yml`
  - 20x20 grid, food collection goal
  - Thermotaxis enabled with linear gradient
  - Health system enabled
  - sensory_modules: [food_chemotaxis, thermotaxis]
- [x] Create `configs/examples/ppo_thermotaxis_stationary_predators_small.yml`
  - Multi-objective: food + stationary predators + temperature
  - sensory_modules: [food_chemotaxis, nociception, mechanosensation, thermotaxis]
- [x] Create `configs/examples/ppo_thermotaxis_pursuit_predators_small.yml`
  - Multi-objective: food + pursuit predators + temperature
  - sensory_modules: [food_chemotaxis, nociception, mechanosensation, thermotaxis]
- [ ] Create `configs/examples/thermotaxis_isothermal_small.yml`
  - Pure thermotaxis task (no food collection goal)
  - Success: stay >80% in comfort zone for episode duration

### 4.2 Benchmark Categories (Hierarchical Naming)

Adopt hierarchical benchmark naming convention for scalability:

```text
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

______________________________________________________________________

## 5. Observability and Exports

### 5.1 Console Output

- [ ] Display `temperature_comfort_score` in per-run simulation summary
- [ ] Display current temperature in step debug output (when verbose)
- [ ] Display temperature zone counts in session summary

### 5.2 Experiment Tracking (JSON)

- [ ] Add `temperature_comfort_score: float | None` to experiment JSON output
- [ ] Add `final_temperature: float | None` to experiment JSON
- [ ] Add `steps_in_comfort_zone: int` and `total_thermotaxis_steps: int`

### 5.3 Per-Run CSV Exports

- [ ] Export `temperature_history.csv` (step, temperature, zone)
- [ ] Include `temperature_comfort_score` in `foraging_summary.csv`
- [ ] Include `final_temperature` and `died_to_temperature` in summary

### 5.4 Per-Run Plots

- [ ] Generate `temperature_progression.png` (temperature over time)
- [ ] Color-code background by temperature zone
- [ ] Overlay comfort zone boundaries as horizontal lines

### 5.5 Session-Level Plots

- [ ] Generate session-level temperature comfort score distribution
- [ ] Generate multi-run temperature progression overlay plot
- [ ] Include temperature stats in session summary plots

**Validation**: Console shows temperature metrics, exports include temperature history and plots

______________________________________________________________________

## 6. Visualization

### 6.1 Temperature Background Colors

- [ ] Extend Rich theme renderer to query temperature at each cell
- [ ] Apply background color based on temperature zone
- [ ] Ensure entity rendering (agent, food, predator) takes priority over background

### 6.2 Debug Logging

- [ ] Log temperature field parameters at episode start
- [ ] Log temperature samples at corners and center
- [ ] Log agent's current temperature in step output

**Validation**: Visual inspection of temperature rendering in Rich theme

______________________________________________________________________

## 7. Documentation

### 7.1 Configuration Documentation

- [ ] Document thermotaxis configuration options
- [ ] Provide example configurations with explanations
- [ ] Document temperature zone thresholds

### 7.2 Architecture Documentation

- [ ] Document thermotaxis_features() computation
- [ ] Document AFD neuron modeling approach
- [ ] Add to optimization methods documentation

______________________________________________________________________

## Dependencies

```text
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
5. Observability + 6. Visualization + 7. Documentation
```

All tasks in this proposal depend on `add-multi-sensory-environment` being complete first.

______________________________________________________________________

## Success Criteria

- [x] TemperatureField computes correct temperatures and gradients ✅ (12 tests)
- [x] Agent receives temperature in BrainParams when thermotaxis enabled ✅
- [x] thermotaxis_features() produces valid rotation values ✅ (4 tests)
- [ ] PPO can learn to navigate to comfort zone (>60% time) *(ready for evaluation)*
- [ ] ModularBrain with thermotaxis module achieves comparable performance *(ready for evaluation)*
- [ ] Combined chemotaxis+thermotaxis task is learnable *(ready for evaluation)*
- [ ] Temperature zones visible in Rich theme rendering *(deferred to Section 5)*
