# Change: Add Thermotaxis System

## Why

Thermotaxis (temperature-guided navigation) is one of the best-characterized behaviors in C. elegans. Real worms navigate toward their cultivation temperature (Tc) using AFD thermosensory neurons. Implementing thermotaxis:

1. Adds the first new sensory modality beyond chemotaxis
2. Enables multi-modal sensory integration research (chemotaxis + thermotaxis)
3. Validates the Phase 1 environment infrastructure from `add-multi-sensory-environment`
4. Creates richer decision-making scenarios (food vs. temperature comfort trade-offs)
5. Provides biological validation target (isothermal tracking behavior)

This is a critical Phase 1 deliverable that tests whether our architectures can handle dual-modality sensing.

## What Changes

### 1. TemperatureField Class

Create a new class to define spatial temperature distributions:

- Linear temperature gradients (configurable direction and strength)
- Hot spots and cold spots (localized temperature perturbations)
- On-demand temperature computation at any position (O(1) storage)
- Gradient vector computation via central difference

### 2. Environment Integration

Integrate temperature sensing into DynamicForagingEnvironment:

- Add `thermotaxis_enabled` feature flag
- Add `temperature_field: TemperatureField` when enabled
- Compute temperature at agent position each step
- Populate BrainParams temperature fields
- Apply temperature zone rewards/penalties
- Apply HP damage for danger/lethal zones (requires health system)

### 3. Thermotaxis Feature Extraction

Implement `_thermotaxis_core()` in modules.py using unified SensoryModule architecture:

- CoreFeatures mapping (uses standard transform):
  - `strength`: Temperature gradient magnitude (`tanh(gradient_strength)`) → RX
  - `angle`: Relative direction to warmer temperatures (egocentric) → RY
  - `binary`: Temperature deviation from cultivation temp (`(T - Tc) / 15.0`) → RZ
- Standard transform applied (consistent with other sensory modules):
  - RX = strength × π - π/2 (gradient strength: [-π/2, π/2])
  - RY = angle × π/2 (direction to warmer: [-π/2, π/2])
  - RZ = binary × π/2 (temperature deviation: [-π/2, π/2])
- AFD neuron-inspired computation
- Seamlessly integrates with PPOBrain via `extract_classical_features()` and ModularBrain via `to_quantum_dict()`

**Note on Biological Accuracy**: Real C. elegans thermotaxis uses temporal comparison (sensing temperature change over time as the worm moves) rather than direct spatial gradient sensing. The spatial gradient approach used here is computationally equivalent for stateless brains and matches the existing chemotaxis pattern. When memory systems are added to the architecture (roadmap item), temporal sensing features should be implemented as a more biologically accurate alternative.

### 4. Thermotaxis Benchmarks

Create benchmark configurations and validation:

- Foraging + thermotaxis constraint benchmarks (primary)
- Pure isothermal tracking benchmarks (for ablation studies)
- Success criteria: >60% time in comfort zone AND survival
- Biological validation targets based on C. elegans literature

## Impact

**Affected Specs:**

- `environment-simulation`: ADDED - TemperatureField, temperature zone mechanics, ThermotaxisParams
- `brain-architecture`: MODIFIED - \_thermotaxis_core() in SensoryModule architecture
- `configuration-system`: ADDED - ThermotaxisConfig class

**Affected Code:**

Core implementation:

- `quantumnematode/env/temperature.py` - NEW: TemperatureField, TemperatureZone, TemperatureZoneThresholds
- `quantumnematode/env/env.py` - ThermotaxisParams, temperature methods, apply_temperature_effects()
- `quantumnematode/brain/modules.py` - \_thermotaxis_core(), SENSORY_MODULES[THERMOTAXIS]
- `quantumnematode/dtypes.py` - TemperatureSpot type alias
- `quantumnematode/utils/config_loader.py` - ThermotaxisConfig class

Agent integration:

- `quantumnematode/agent/agent.py` - BrainParams population, reset_environment() preservation
- `quantumnematode/agent/runners.py` - apply_temperature_effects() in step loop

Scripts:

- `scripts/run_simulation.py` - thermotaxis config loading
- `scripts/run_evolution.py` - thermotaxis config loading

Configs:

- `configs/examples/ppo_thermotaxis_foraging_small.yml` - NEW
- `configs/examples/ppo_thermotaxis_stationary_predators_small.yml` - NEW
- `configs/examples/ppo_thermotaxis_pursuit_predators_small.yml` - NEW

**Dependencies:**

- **REQUIRES** `add-multi-sensory-environment` to be implemented first
- Specifically requires: BrainParams temperature fields, health system, reward extensions

**Breaking Changes:**

- None. Thermotaxis is opt-in via `thermotaxis.enabled: true`

**Backward Compatibility:**

- Existing configs work unchanged (thermotaxis disabled by default)
- Existing brains receive None for temperature fields when disabled
