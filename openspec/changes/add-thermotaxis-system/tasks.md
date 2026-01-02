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
  - `spot_decay_constant: float` (exponential falloff rate for hot/cold spots)

### 1.2 Temperature Computation

Uses shared types from `quantumnematode/dtypes.py`: `GridPosition`, `GradientPolar`, `GradientVector`, `TemperatureSpot`

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

**Validation**: Unit tests for temperature calculation and zone classification ✅ (14 tests in test_temperature.py)

______________________________________________________________________

## 2. Environment Integration

### 2.1 Temperature Field in Environment

- [x] Add `thermotaxis: ThermotaxisParams` to DynamicForagingEnvironment
- [x] Add `temperature_field: TemperatureField | None` attribute
- [x] Add `steps_in_comfort_zone: int` and `total_thermotaxis_steps: int` tracking counters
- [x] Initialize temperature field from config when enabled
- [x] Import `GridPosition` and `GradientPolar` types from dtypes for method signatures
- [x] Add `ThermotaxisParams` dataclass with fields:
  - `enabled`, `cultivation_temperature`, `base_temperature`
  - `gradient_direction`, `gradient_strength`
  - `hot_spots`, `cold_spots` (optional localized temperature sources)
  - `comfort_reward`, `discomfort_penalty`, `danger_penalty`
  - `danger_hp_damage`, `lethal_hp_damage`
  - `comfort_delta`, `discomfort_delta`, `danger_delta` (zone thresholds)

### 2.2 Temperature in Step Execution

- [x] Compute temperature at agent position each step (`get_temperature()`)
- [x] Populate `BrainParams.temperature` with current temperature
- [x] Populate `BrainParams.temperature_gradient_strength`
- [x] Populate `BrainParams.temperature_gradient_direction`
- [x] Populate `BrainParams.cultivation_temperature`
- [x] Integration in `agent.py::_create_brain_params()` to query env thermotaxis state

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

- [x] Add `ThermotaxisConfig` class to `config_loader.py`
- [x] Add `thermotaxis` field to `DynamicEnvironmentConfig`
- [x] Implement `to_params()` method for converting to `ThermotaxisParams`
- [x] Integrate temperature effects into `runners.py` step loop
- [x] Update `scripts/run_simulation.py` to load and pass thermotaxis config
- [x] Update `scripts/run_evolution.py` to load and pass thermotaxis config
- [x] Update `agent.py::reset_environment()` to preserve thermotaxis config
- [ ] Add thermotaxis section to YAML schema documentation

**Validation**: Integration tests ✅

- 10 tests in test_env.py::TestThermotaxisIntegration (environment methods)
- 3 tests in test_runners.py::TestThermotaxisIntegration (episode runner integration)

______________________________________________________________________

## 3. Feature Extraction

### 3.1 Thermotaxis Features Implementation

> **Note**: Uses spatial gradient sensing (matches chemotaxis pattern). Temporal sensing (biologically accurate) to be added when memory systems are implemented per roadmap.

- [x] Implement `_thermotaxis_core(params: BrainParams) -> CoreFeatures` in unified SensoryModule architecture
- [x] CoreFeatures mapping (uses standard transform):
  - `strength`: Temperature gradient magnitude (`tanh(gradient_strength)`) → RX
  - `angle`: Relative direction to warmer temperatures (egocentric) → RY
  - `binary`: Temperature deviation from cultivation temp (`(T - Tc) / 15.0`, clipped to [-1, 1]) → RZ
- [x] Standard transform applied:
  - RX = strength × π - π/2 (gradient strength: [-π/2, π/2])
  - RY = angle × π/2 (direction to warmer: [-π/2, π/2])
  - RZ = binary × π/2 (temperature deviation: [-π/2, π/2])
- [x] Handle None values gracefully (return zeros when thermotaxis disabled)
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

- [x] TemperatureField computes correct temperatures and gradients ✅ (14 tests in test_temperature.py)
- [x] Agent receives temperature in BrainParams when thermotaxis enabled ✅
- [x] thermotaxis_features() produces valid rotation values ✅ (4 tests in test_modules.py)
- [x] Environment integration complete ✅ (10 tests in test_env.py::TestThermotaxisIntegration)
- [x] Runners integration with temperature effects ✅ (3 tests in test_runners.py::TestThermotaxisIntegration)
- [ ] PPO can learn to navigate to comfort zone (>60% time) *(ready for evaluation)*
- [ ] ModularBrain with thermotaxis module achieves comparable performance *(ready for evaluation)*
- [ ] Combined chemotaxis+thermotaxis task is learnable *(ready for evaluation)*
- [ ] Temperature zones visible in Rich theme rendering *(deferred to Section 6)*

______________________________________________________________________

## Files Modified (Reference for Future Sensory Systems)

This section documents all files touched during thermotaxis implementation to serve as a template for adding future sensory systems like aerotaxis.

### Core Implementation Files

| File | Changes |
|------|---------|
| `quantumnematode/env/temperature.py` | **NEW** - TemperatureField class, TemperatureZone enum, TemperatureZoneThresholds |
| `quantumnematode/env/env.py` | ThermotaxisParams dataclass, temperature_field attribute, get_temperature(), get_temperature_gradient(), get_temperature_zone(), apply_temperature_effects(), get_temperature_comfort_score(), reset_thermotaxis() |
| `quantumnematode/brain/modules.py` | \_thermotaxis_core() function, SENSORY_MODULES registry entry for THERMOTAXIS |
| `quantumnematode/dtypes.py` | TemperatureSpot type alias (shared types already existed) |
| `quantumnematode/utils/config_loader.py` | ThermotaxisConfig class, to_params() method, DynamicEnvironmentConfig.thermotaxis field |
| `quantumnematode/agent/agent.py` | \_create_brain_params() thermotaxis state population, reset_environment() config preservation |
| `quantumnematode/agent/runners.py` | apply_temperature_effects() call in step loop, HP depletion check |

### Script Files

| File | Changes |
|------|---------|
| `scripts/run_simulation.py` | Load thermotaxis config, pass to environment constructor |
| `scripts/run_evolution.py` | Load thermotaxis config, pass to environment constructor |

### Test Files

| File | Tests Added |
|------|-------------|
| `tests/.../env/test_temperature.py` | **NEW** - 14 tests for TemperatureField |
| `tests/.../env/test_env.py` | TestThermotaxisIntegration class - 10 tests |
| `tests/.../brain/test_modules.py` | TestThermotaxisModule class - 4 tests |
| `tests/.../agent/test_runners.py` | TestThermotaxisIntegration class - 3 tests |

### Config Files

| File | Purpose |
|------|---------|
| `configs/examples/ppo_thermotaxis_foraging_small.yml` | Basic thermotaxis + food foraging |
| `configs/examples/ppo_thermotaxis_stationary_predators_small.yml` | Thermotaxis + stationary predators |
| `configs/examples/ppo_thermotaxis_pursuit_predators_small.yml` | Thermotaxis + pursuit predators |

### Pattern for Adding New Sensory Systems (e.g., Aerotaxis)

1. **Create field module** (like `temperature.py` → `oxygen.py`)
2. **Add params dataclass** to `env.py` (like `ThermotaxisParams` → `AerotaxisParams`)
3. **Add environment methods** for querying field state
4. **Implement core features** in `modules.py` following SensoryModule pattern
5. **Add config loader** class with `to_params()` method
6. **Update agent.py** to populate BrainParams with new sensory data
7. **Update runners.py** to apply zone effects if applicable
8. **Update scripts** to load and pass new config
9. **Add comprehensive tests** for each layer
