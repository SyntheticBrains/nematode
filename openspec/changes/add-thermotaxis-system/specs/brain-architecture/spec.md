## MODIFIED Requirements

### Requirement: Thermotaxis Sensory Module

The brain architecture SHALL provide a thermotaxis sensory module following the unified SensoryModule pattern.

#### Scenario: Thermotaxis Module Registration

- **GIVEN** the SENSORY_MODULES registry in modules.py
- **WHEN** thermotaxis features are requested
- **THEN** `ModuleName.THERMOTAXIS` SHALL be registered in the registry
- **AND** the module SHALL use `transform_type="standard"`
- **AND** the module SHALL NOT be marked as placeholder (`is_placeholder=False`)

#### Scenario: Thermotaxis Core Features Extraction

- **GIVEN** BrainParams with thermotaxis enabled (temperature fields populated)
- **WHEN** `_thermotaxis_core(params)` is called
- **THEN** CoreFeatures SHALL be returned with:
  - `strength`: Temperature gradient magnitude via `tanh(temperature_gradient_strength)`
  - `angle`: Relative direction to warmer temperature (egocentric, normalized to [-1, 1])
  - `binary`: Temperature deviation from cultivation temp via `(T - Tc) / 15.0` clipped to [-1, 1]

#### Scenario: Thermotaxis Disabled Handling

- **GIVEN** BrainParams with `temperature=None` (thermotaxis disabled)
- **WHEN** `_thermotaxis_core(params)` is called
- **THEN** CoreFeatures SHALL return zeros: `CoreFeatures(strength=0.0, angle=0.0, binary=0.0)`

### Requirement: Thermotaxis Quantum Transform

The thermotaxis module SHALL use the standard transform for quantum gate angles.

#### Scenario: Standard Transform Application

- **GIVEN** CoreFeatures with `strength=0.5`, `angle=0.3`, `binary=-0.2`
- **WHEN** `to_quantum(params)` is called on the thermotaxis module
- **THEN** quantum angles SHALL be computed as:
  - RX = strength * pi - pi/2 = 0.5 * pi - pi/2 = 0.0 radians
  - RY = angle * pi/2 = 0.3 * pi/2 = 0.471 radians
  - RZ = binary * pi/2 = -0.2 * pi/2 = -0.314 radians

#### Scenario: Temperature Deviation Encoding

- **GIVEN** an agent at cultivation temperature (T = Tc = 20.0)
- **WHEN** thermotaxis features are extracted
- **THEN** `binary` SHALL equal 0.0 (no deviation)
- **AND** RZ SHALL equal 0.0 radians
- **GIVEN** an agent 15°C hotter than Tc (T = 35.0, Tc = 20.0)
- **WHEN** thermotaxis features are extracted
- **THEN** `binary` SHALL equal 1.0 (max hot deviation)
- **AND** RZ SHALL equal pi/2 radians
- **GIVEN** an agent 15°C colder than Tc (T = 5.0, Tc = 20.0)
- **WHEN** thermotaxis features are extracted
- **THEN** `binary` SHALL equal -1.0 (max cold deviation)
- **AND** RZ SHALL equal -pi/2 radians

### Requirement: Thermotaxis Classical Features

The thermotaxis module SHALL support classical feature extraction for PPOBrain.

#### Scenario: Classical Feature Extraction

- **GIVEN** a PPOBrain configured with `sensory_modules: [thermotaxis]`
- **WHEN** `extract_classical_features(params, [ModuleName.THERMOTAXIS])` is called
- **THEN** a 2-element array SHALL be returned: `[strength, angle]`
- **AND** values SHALL preserve semantic meaning:
  - strength in [0, 1] (gradient magnitude)
  - angle in [-1, 1] (direction to warmer)

### Requirement: Thermotaxis Module Documentation

The thermotaxis module SHALL include biological documentation.

#### Scenario: AFD Neuron Reference

- **GIVEN** the thermotaxis module docstring
- **THEN** it SHALL reference AFD neurons as the biological basis
- **AND** it SHALL note that spatial gradient sensing is used (vs temporal)
- **AND** it SHALL document the semantic meaning of each CoreFeatures field
