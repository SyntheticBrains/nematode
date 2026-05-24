# brain-architecture Specification

## Purpose

Extend the brain architecture system to support the CRH (Classical Reservoir Hybrid) brain type with ReservoirHybridBase base class, the QLIF-LSTM (Quantum LIF Long Short-Term Memory) brain type, and the QEF (Quantum Entangled Features) brain type.

## Requirements

### Requirement: Brain Type Registry

The `BrainType` enum SHALL remain the canonical typed dispatch key and SHALL include every supported brain architecture as enum members. Its membership SHALL be kept consistent with the brain plugin registry via a startup-time consistency check.

#### Scenario: CRH Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.CRH` SHALL exist with value `"crh"` in the BrainType enum
- **AND** CRH SHALL be included in `CLASSICAL_BRAIN_TYPES`
- **AND** `"crh"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: CRH Brain Factory

- **WHEN** `instantiate_brain("crh", config)` is called
- **THEN** the registry SHALL return a `CRHBrain` instance
- **AND** SHALL accept `CRHBrainConfig` for configuration

#### Scenario: CRH Config Loading

- **WHEN** a YAML config specifies `brain.name: crh`
- **THEN** the config loader SHALL parse brain config using `CRHBrainConfig`
- **AND** SHALL support all CRH-specific fields plus inherited ReservoirHybridBase fields

#### Scenario: QLIF-LSTM Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.QLIF_LSTM` SHALL exist with value `"qliflstm"` in the BrainType enum
- **AND** QLIF_LSTM SHALL be included in `QUANTUM_BRAIN_TYPES`
- **AND** `"qliflstm"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: QLIF-LSTM Config Loading

- **WHEN** a YAML config specifies `brain.name: qliflstm`
- **THEN** the config loader SHALL parse brain config using `QLIFLSTMBrainConfig`
- **AND** SHALL support all QLIF-LSTM-specific fields

#### Scenario: QLIF-LSTM Brain Factory Instantiation

- **WHEN** the brain factory receives a `QLIFLSTMBrainConfig`
- **THEN** the factory SHALL instantiate a `QLIFLSTMBrain` with the provided configuration

#### Scenario: QEF Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.QEF` SHALL exist with value `"qef"` in the BrainType enum
- **AND** QEF SHALL be included in `QUANTUM_BRAIN_TYPES`
- **AND** `"qef"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: QEF Brain Factory

- **WHEN** `instantiate_brain("qef", config)` is called
- **THEN** the registry SHALL return a `QEFBrain` instance
- **AND** SHALL accept `QEFBrainConfig` for configuration
- **AND** SHALL raise ValueError if config is not QEFBrainConfig

#### Scenario: QEF Config Loading

- **WHEN** a YAML config specifies `brain.name: qef`
- **THEN** the config loader SHALL parse brain config using `QEFBrainConfig`
- **AND** SHALL support all QEF-specific fields plus inherited ReservoirHybridBaseConfig fields

#### Scenario: ConnectomePPO Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.CONNECTOMEPPO` SHALL exist with value `"connectomeppo"` in the BrainType enum
- **AND** CONNECTOMEPPO SHALL be included in `CLASSICAL_BRAIN_TYPES`
- **AND** `"connectomeppo"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: ConnectomePPO Brain Factory

- **WHEN** `instantiate_brain("connectomeppo", config)` is called
- **THEN** the registry SHALL return a `ConnectomePPOBrain` instance
- **AND** SHALL accept `ConnectomePPOBrainConfig` for configuration

#### Scenario: ConnectomePPO Config Loading

- **WHEN** a YAML config specifies `brain.name: connectomeppo`
- **THEN** the config loader SHALL parse brain config using `ConnectomePPOBrainConfig`
- **AND** SHALL support all ConnectomePPO-specific fields

#### Scenario: Registry-enum consistency check at import time

- **WHEN** `quantumnematode.brain.arch` finishes importing
- **THEN** `BrainType` SHALL be a `StrEnum`
- **AND** `{bt.value for bt in BrainType}` SHALL equal the set of names registered via `@register_brain(...)`
- **AND** any mismatch SHALL raise an exception that identifies which names are present on each side

### Requirement: Module Exports

The `quantumnematode.brain.arch` package SHALL re-export every supported `Brain` class together with its matching config class.

#### Scenario: CRH Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `CRHBrain` and `CRHBrainConfig` SHALL be importable
- **AND** `ReservoirHybridBase` and `ReservoirHybridBaseConfig` SHALL be importable

#### Scenario: QLIF-LSTM Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `QLIFLSTMBrain` and `QLIFLSTMBrainConfig` SHALL be importable
- **AND** SHALL be included in the `__all__` export list

#### Scenario: QEF Brain Exports

- **WHEN** importing from `quantumnematode.brain.arch`
- **THEN** `QEFBrain` and `QEFBrainConfig` SHALL be importable
- **AND** SHALL be included in the `__all__` export list

### Requirement: Extended BrainParams for Multi-Sensory Input

The BrainParams class SHALL include additional optional fields for Phase 1 sensory modalities, enabling brains to process temperature, health, mechanosensory, and oxygen information.

#### Scenario: Temperature Sensing Fields

- **GIVEN** a brain receiving sensory input in a thermotaxis-enabled environment
- **WHEN** BrainParams is populated
- **THEN** the following fields SHALL be available:
  - `temperature: float | None` - Current temperature at agent position (degrees Celsius)
  - `temperature_gradient_strength: float | None` - Magnitude of local temperature gradient
  - `temperature_gradient_direction: float | None` - Direction of increasing temperature (radians)
  - `cultivation_temperature: float | None` - Agent's preferred temperature (Tc)
- **AND** all fields SHALL default to None when thermotaxis is disabled

#### Scenario: Oxygen Sensing Fields

- **GIVEN** a brain receiving sensory input in an aerotaxis-enabled environment
- **WHEN** BrainParams is populated
- **THEN** the following fields SHALL be available:
  - `oxygen_concentration: float | None` - Current O2 percentage at agent position (0.0-21.0%)
  - `oxygen_gradient_strength: float | None` - Magnitude of local oxygen gradient (oracle mode only)
  - `oxygen_gradient_direction: float | None` - Direction of increasing oxygen (radians, oracle mode only)
  - `oxygen_dconcentration_dt: float | None` - Temporal derivative of oxygen concentration (derivative mode only)
- **AND** all fields SHALL default to None when aerotaxis is disabled

#### Scenario: Health System Fields

- **GIVEN** a brain receiving sensory input in a health-system-enabled environment
- **WHEN** BrainParams is populated
- **THEN** the following fields SHALL be available:
  - `health: float | None` - Current health points
  - `max_health: float | None` - Maximum health points
- **AND** all fields SHALL default to None when health system is disabled

#### Scenario: Mechanosensation Fields

- **GIVEN** a brain receiving sensory input
- **WHEN** BrainParams is populated
- **THEN** the following fields SHALL be available:
  - `boundary_contact: bool | None` - Whether agent is touching grid boundary
  - `predator_contact: bool | None` - Whether agent is in physical contact with predator
- **AND** all fields SHALL default to None when not applicable

#### Scenario: Backward Compatibility

- **GIVEN** an existing brain configuration from Phase 0
- **WHEN** the brain receives BrainParams
- **THEN** all new fields SHALL be None
- **AND** existing brain functionality SHALL be unchanged
- **AND** no code modifications SHALL be required for existing brains

### Requirement: Unified Feature Extraction Layer

The system SHALL provide a unified feature extraction layer that converts BrainParams into sensory feature vectors, shared by both quantum and classical brain architectures.

#### Scenario: Unified Extraction Function

- **GIVEN** a populated BrainParams instance
- **WHEN** `extract_sensory_features(params)` is called
- **THEN** the function SHALL return a dictionary of numpy arrays:
  - `chemotaxis: np.ndarray` - Food gradient features
  - `thermotaxis: np.ndarray` - Temperature gradient features (if enabled)
  - `nociception: np.ndarray` - Predator avoidance features
  - `mechanosensation: np.ndarray` - Touch/contact features (if enabled)
  - `aerotaxis: np.ndarray` - Oxygen gradient features (if enabled)
- **AND** each array SHALL contain normalized feature values

#### Scenario: ModularBrain Feature Consumption

- **GIVEN** a ModularBrain instance receiving features from unified extraction
- **WHEN** the brain processes the features
- **THEN** features SHALL be converted to RX/RY/RZ quantum gate rotations
- **AND** conversion SHALL use the existing module feature extractor pattern
- **AND** new sensory modules SHALL map to specified qubits

#### Scenario: PPOBrain Feature Consumption

- **GIVEN** a PPOBrain instance receiving features from unified extraction
- **WHEN** the brain processes the features
- **THEN** features SHALL be concatenated into a single input vector
- **AND** the input dimension SHALL match the total feature count
- **AND** the actor-critic networks SHALL process the combined vector

### Requirement: Scientific Module Naming Convention

Feature extraction modules SHALL use scientific names for sensory modalities with C. elegans neuron references in documentation.

#### Scenario: Module Function Naming

- **GIVEN** the feature extraction modules in modules.py
- **WHEN** module functions are defined
- **THEN** function names SHALL follow scientific convention:
  - `chemotaxis_features` - Chemical gradient sensing (ASE neurons)
  - `food_chemotaxis_features` - Food-seeking chemotaxis (AWC, AWA neurons)
  - `nociception_features` - Noxious stimulus avoidance (ASH, ADL neurons)
  - `thermotaxis_features` - Temperature gradient sensing (AFD neurons)
  - `aerotaxis_features` - Oxygen gradient sensing (URX, BAG neurons)
  - `mechanosensation_features` - Touch/contact sensing (ALM, PLM, AVM neurons)

#### Scenario: Module Docstring Neuron References

- **GIVEN** a feature extraction module function
- **WHEN** the function docstring is read
- **THEN** the docstring SHALL include:
  - Scientific name of the sensory modality
  - Primary C. elegans neurons involved
  - Brief description of the biological behavior being modeled

#### Scenario: Module Renaming Backward Compatibility

- **GIVEN** existing code using old module names (appetitive_features, aversive_features)
- **WHEN** the code runs after renaming
- **THEN** the ModuleName enum SHALL support both old and new names
- **AND** deprecation warnings MAY be logged for old names
- **AND** existing configurations SHALL continue to work

### Requirement: Mechanosensation Feature Extraction

The system SHALL provide a mechanosensation feature extraction module that encodes touch and contact information for brain processing.

#### Scenario: Mechanosensation Feature Extraction

- **GIVEN** BrainParams with `boundary_contact: True` and `predator_contact: False`
- **WHEN** `mechanosensation_features(params)` is called
- **THEN** the function SHALL return rotation values:
  - RX: Boundary contact encoding (scaled to gate range)
  - RY: Predator contact encoding (scaled to gate range)
  - RZ: Combined contact urgency (if any contact)
- **AND** values SHALL be in range [-π/2, π/2]

#### Scenario: No Contact Feature Values

- **GIVEN** BrainParams with `boundary_contact: False` and `predator_contact: False`
- **WHEN** `mechanosensation_features(params)` is called
- **THEN** all rotation values SHALL be 0.0
- **AND** this indicates no touch sensation

### Requirement: Extended Reward Configuration

The RewardConfig class SHALL include configurable weights for multi-objective rewards including temperature comfort, health changes, and mechanosensation penalties.

#### Scenario: Temperature Reward Configuration

- **GIVEN** a RewardConfig for a thermotaxis-enabled environment
- **WHEN** reward parameters are specified
- **THEN** the following fields SHALL be configurable:
  - `reward_temperature_comfort: float` - Reward per step in comfort zone (default 0.05)
  - `penalty_temperature_discomfort: float` - Penalty per step in discomfort zone (default 0.1)
  - `penalty_temperature_danger: float` - Penalty per step in danger zone (default 0.3)
  - `temperature_comfort_min: float` - Lower bound of comfort zone (default 15.0)
  - `temperature_comfort_max: float` - Upper bound of comfort zone (default 25.0)
  - `temperature_danger_min: float` - Lower bound of danger zone (default 10.0)
  - `temperature_danger_max: float` - Upper bound of danger zone (default 30.0)

#### Scenario: Health Reward Configuration

- **GIVEN** a RewardConfig for a health-system-enabled environment
- **WHEN** reward parameters are specified
- **THEN** the following fields SHALL be configurable:
  - `reward_health_gain: float` - Reward per HP gained (default 0.1)
  - `penalty_health_damage: float` - Penalty per HP lost (default 0.2)
  - `hp_damage_temperature_danger: float` - HP lost per step in danger zone (default 2.0)
  - `hp_damage_temperature_lethal: float` - HP lost per step in lethal zone (default 10.0)

#### Scenario: Mechanosensation Penalty Configuration

- **GIVEN** a RewardConfig
- **WHEN** reward parameters are specified
- **THEN** the following field SHALL be configurable:
  - `penalty_boundary_collision: float` - Penalty for hitting grid boundary (code default 0.0 for backward compatibility; example configs use 0.02)

### Requirement: Multi-Objective Evaluation Metrics

The SimulationResult SHALL include optional per-objective scores for multi-sensory tasks.

#### Scenario: Temperature Comfort Score

- **GIVEN** an episode in a thermotaxis-enabled environment
- **WHEN** episode metrics are computed
- **THEN** `temperature_comfort_score: float | None` SHALL be calculated
- **AND** score SHALL equal (steps in comfort zone) / (total steps)
- **AND** score SHALL range from 0.0 to 1.0

#### Scenario: Oxygen Comfort Score

- **GIVEN** an episode in an aerotaxis-enabled environment
- **WHEN** episode metrics are computed
- **THEN** `oxygen_comfort_score: float | None` SHALL be calculated
- **AND** score SHALL equal (steps in oxygen comfort zone) / (total aerotaxis steps)
- **AND** score SHALL range from 0.0 to 1.0

#### Scenario: Survival Score

- **GIVEN** an episode in a health-system-enabled environment
- **WHEN** episode metrics are computed
- **THEN** `survival_score: float | None` SHALL be calculated
- **AND** score SHALL equal (final HP) / (max HP)
- **AND** score SHALL range from 0.0 to 1.0

#### Scenario: Thermotaxis Success Flag

- **GIVEN** thermotaxis enabled with success threshold of 60%
- **WHEN** episode metrics are computed
- **THEN** `thermotaxis_success: bool | None` SHALL be set
- **AND** success SHALL be True if temperature_comfort_score >= 0.6 AND agent survived
- **AND** success SHALL be False otherwise

#### Scenario: Aerotaxis Success Flag

- **GIVEN** aerotaxis enabled with success threshold of 60%
- **WHEN** episode metrics are computed
- **THEN** `aerotaxis_success: bool | None` SHALL be set
- **AND** success SHALL be True if oxygen_comfort_score >= 0.6 AND agent survived
- **AND** success SHALL be False otherwise

#### Scenario: Multi-Objective Composite Score

- **GIVEN** multi-objective metrics are available
- **WHEN** composite benchmark score is calculated
- **THEN** base Phase 0 score components SHALL be weighted at 85%
- **AND** multi-objective components SHALL be weighted at 15%
- **AND** the formula SHALL be documented for reproducibility

### Requirement: Extended BrainParams for Social Sensing

The BrainParams class SHALL include an optional field for social proximity information in multi-agent scenarios.

#### Scenario: Social Proximity Field

- **GIVEN** a brain receiving sensory input in a multi-agent environment
- **WHEN** BrainParams is populated
- **THEN** `nearby_agents_count: int | None` SHALL be available
- **AND** this field SHALL default to None in single-agent mode

### Requirement: Social Proximity Sensory Module

The system SHALL provide a sensory module for social proximity detection in multi-agent scenarios.

#### Scenario: Module Registration

- **GIVEN** the sensory module system
- **THEN** `SOCIAL_PROXIMITY` SHALL be a valid ModuleName with `classical_dim=1`

#### Scenario: Feature Extraction

- **GIVEN** BrainParams with `nearby_agents_count=N`
- **WHEN** features are extracted
- **THEN** strength SHALL be `min(N, 10) / 10.0`, clamped to [0, 1]

### Requirement: Extended BrainParams for Pheromone Sensing

The BrainParams class SHALL include optional fields for pheromone concentration and gradient information.

#### Scenario: Pheromone Oracle Fields

- **WHEN** BrainParams is populated in oracle mode
- **THEN** `pheromone_food_gradient_strength`, `pheromone_food_gradient_direction`, `pheromone_alarm_gradient_strength`, `pheromone_alarm_gradient_direction` (all `float | None`) SHALL be available

#### Scenario: Pheromone Temporal Fields

- **WHEN** BrainParams is populated in temporal/derivative mode
- **THEN** `pheromone_food_concentration`, `pheromone_alarm_concentration`, `pheromone_food_dconcentration_dt`, `pheromone_alarm_dconcentration_dt` (all `float | None`) SHALL be available

#### Scenario: Pheromone Fields Backward Compatibility

- **GIVEN** pheromones disabled
- **THEN** all pheromone BrainParams fields SHALL be None

### Requirement: Pheromone Sensory Modules

The system SHALL provide sensory modules for pheromone detection following the oracle/temporal pattern.

#### Scenario: Oracle Pheromone Modules

- `PHEROMONE_FOOD` and `PHEROMONE_ALARM` SHALL be registered with `classical_dim=2`
- Strength encodes gradient magnitude [0, 1], angle encodes egocentric direction [-1, 1]

#### Scenario: Temporal Pheromone Modules

- `PHEROMONE_FOOD_TEMPORAL` and `PHEROMONE_ALARM_TEMPORAL` SHALL be registered with `classical_dim=2`
- Strength encodes scalar concentration [0, 1], angle encodes tanh(dC/dt * derivative_scale) [-1, 1]

### Requirement: STAM Pheromone Channel Extension

The STAM buffer SHALL support variable channel counts for pheromone sensing.

#### Scenario: 4-Channel Mode (Base)

- **GIVEN** pheromones disabled
- **THEN** STAM SHALL use 4 channels and MEMORY_DIM=11

#### Scenario: 6-Channel Mode (Food-Marking + Alarm)

- **GIVEN** pheromones enabled (food-marking + alarm)
- **THEN** STAM SHALL use 6 channels and MEMORY_DIM=15
- **AND** channels 4, 5 correspond to pheromone_food, pheromone_alarm

#### Scenario: 7-Channel Mode (+ Aggregation)

- **GIVEN** pheromones enabled with aggregation configured
- **THEN** STAM SHALL use 7 channels and MEMORY_DIM=17
- **AND** channel 6 corresponds to pheromone_aggregation

### Requirement: BrainParams Aggregation Pheromone Fields

BrainParams SHALL include fields for aggregation pheromone sensing data.

#### Scenario: Oracle Mode

- `pheromone_aggregation_gradient_strength` and `pheromone_aggregation_gradient_direction` (both `float | None`)

#### Scenario: Temporal Mode

- `pheromone_aggregation_concentration` and `pheromone_aggregation_dconcentration_dt` (both `float | None`)

#### Scenario: Default None

- **GIVEN** aggregation not configured
- **THEN** all 4 aggregation fields SHALL be None

### Requirement: Aggregation Pheromone Sensing Modules

The aggregation-pheromone sensing modules SHALL be registered in both oracle and temporal variants so the brain can consume aggregation-pheromone fields through the unified sensory-module API.

#### Scenario: Oracle Module

- `PHEROMONE_AGGREGATION` SHALL be registered with `classical_dim=2` (strength + relative angle)

#### Scenario: Temporal Module

- `PHEROMONE_AGGREGATION_TEMPORAL` SHALL be registered with `classical_dim=2` (concentration + derivative)

### Requirement: Brain Plugin Registry

The system SHALL provide a decorator-registration brain plugin registry as the single source of truth for the mapping from a brain name to its config class, Brain class, and `BrainType` enum member. Adding a new architecture SHALL touch at most six files (the new brain module, its config, the `BrainType` enum addition, `__init__.py` re-exports, the test module, and an example YAML config) and SHALL introduce zero per-architecture branches in the simulation loop or training loop.

#### Scenario: Decorator registers a brain

- **WHEN** a brain module declares `@register_brain(name="<name>", config_cls=<Config>, brain_type=BrainType.<X>, families=(...))` on its Brain class at module import time
- **THEN** the registry SHALL record the `(name, config_cls, brain_cls, brain_type, families)` tuple
- **AND** the brain SHALL be retrievable via `get_registration(name)` returning the same tuple
- **AND** the brain name SHALL appear in `list_registered_brains()`

#### Scenario: Duplicate name detection

- **WHEN** a second `@register_brain(name="X", ...)` call uses a name already in the registry
- **THEN** the registry SHALL raise `ValueError` at import time with a message identifying the conflicting brain modules

#### Scenario: Unknown name lookup raises

- **WHEN** `instantiate_brain(name="not_a_brain", config=...)` is called with a name not in the registry
- **THEN** the call SHALL raise `ValueError` with a message listing the available registered names

#### Scenario: Registry instantiates any registered brain through one code path

- **WHEN** `instantiate_brain(name, config, **infra_kwargs)` is called with `name` in the registry
- **THEN** the function SHALL return a `Brain` Protocol-conforming instance of the registered Brain class
- **AND** the function SHALL apply the registered config-class type-check before instantiation
- **AND** no per-brain `if`/`elif` branches SHALL be required in the function body

#### Scenario: All architecture modules self-register at import time

- **WHEN** `quantumnematode.brain.arch` is imported
- **THEN** every architecture module under `brain/arch/` SHALL load and self-register
- **AND** `list_registered_brains()` SHALL return a set containing every member of the `BrainType` enum's string-value namespace

### Requirement: BrainType Enum and Registry Consistency

The `BrainType` enum SHALL remain the public typed dispatch key for callers outside the brain-architecture subsystem (evolution framework, predator-brain factory). Its membership SHALL be kept consistent with the registry via a startup-time consistency check.

#### Scenario: Enum and registry agree at import time

- **WHEN** `quantumnematode.brain.arch` finishes importing
- **THEN** `BrainType` SHALL be a `StrEnum` (so enum values are first-class strings)
- **AND** the set `{bt.value for bt in BrainType}` SHALL equal `list_registered_brains()`
- **AND** a mismatch SHALL raise a clear exception identifying which names are missing from which side

#### Scenario: Family type-aliases derived from registry metadata

- **WHEN** `BRAIN_TYPES` Literal, `QUANTUM_BRAIN_TYPES` set, `CLASSICAL_BRAIN_TYPES` set, and `SPIKING_BRAIN_TYPES` set are inspected
- **THEN** their contents SHALL be derived from the `families` metadata on each `@register_brain(...)` registration
- **AND** SHALL NOT require hand-maintenance separate from the per-brain decorator

### Requirement: Brain Topology Protocol

The system SHALL provide a `BrainTopology` Protocol that exposes the structural and forward-pass interface of a brain's network, factored out from learning-rule concerns (optimisers, replay buffers, value heads).

#### Scenario: Topology exposes shape attributes

- **WHEN** a `BrainTopology` implementation is inspected
- **THEN** the instance SHALL expose `n_inputs: int`, `n_outputs: int`, and `n_hidden: int` attributes

#### Scenario: Topology computes a forward pass

- **WHEN** `topology.forward(x)` is called with a Tensor of shape compatible with `n_inputs`
- **THEN** the call SHALL return a Tensor of shape compatible with `n_outputs`
- **AND** the call SHALL be free of optimiser, replay-buffer, or value-head state changes

#### Scenario: Topology applies weight mask

- **WHEN** `topology.apply_weight_mask(weights)` is called with a candidate weight tensor
- **THEN** the call SHALL return the weight tensor projected onto the topology's allowed connectivity manifold
- **AND** for dense topologies the default implementation SHALL be the identity function
- **AND** for sparse / strict-mask topologies (e.g. connectome-constrained) the call SHALL zero out weights along non-existent edges

### Requirement: Learning Rule Protocol

The system SHALL provide a `LearningRule` Protocol that encapsulates how a topology's weights are updated from experience, factored out from topology concerns.

#### Scenario: Rule owns the optimiser state

- **WHEN** a `LearningRule` implementation is constructed
- **THEN** the rule instance SHALL own its optimiser, value head (if any), replay buffer (if any), advantage estimator (if any), and gradient clipper (if any)
- **AND** these objects SHALL NOT be exposed on the `BrainTopology` interface

#### Scenario: Rule advances the topology weights

- **WHEN** `rule.step(topology, batch)` is called with an experience batch
- **THEN** the rule SHALL compute gradients with respect to `topology` parameters
- **AND** SHALL apply optimiser updates to those parameters
- **AND** SHALL apply `topology.apply_weight_mask(...)` to any updated weights that are subject to a topology mask
- **AND** SHALL return a `RuleStepReport` summarising the update (loss components, gradient norms)

#### Scenario: Rule resets per-episode state

- **WHEN** `rule.reset_episode()` is called at the start of a new episode
- **THEN** the rule SHALL clear any per-episode state (advantage estimator buffers, recurrent-state caches owned by the rule)
- **AND** SHALL NOT clear the optimiser state or the persistent replay buffer

### Requirement: External Brain Protocol Surface is Unchanged

The external `Brain` Protocol surface (`run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` plus `history_data` and `latest_data` attributes) SHALL remain unchanged. Every registered brain SHALL satisfy the existing `Brain` Protocol exactly as defined at [packages/quantum-nematode/quantumnematode/brain/arch/\_brain.py:346-368](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py).

#### Scenario: Existing Brain Protocol consumers continue to work

- **WHEN** existing simulation / training / evolution code that calls `brain.run_brain(...)` / `brain.update_memory(...)` / etc. is executed against a registry-instantiated brain
- **THEN** the brain SHALL respond to every Protocol method with identical signatures and identical observable behaviour to its pre-refactor implementation
- **AND** no Protocol method SHALL be added, removed, or renamed

### Requirement: Migration Regression Bar — MLPPPO and LSTMPPO Byte-Equivalence

The MLPPPO and LSTMPPO brains, as the Phase 6 Gate 1 G1.d MUST architectures, SHALL produce byte-identical training trajectories and parameter tensors pre-refactor and post-refactor on at least one smoke config each, with all RNG seeds pinned.

#### Scenario: MLPPPO byte-equivalence on klinotaxis smoke config

- **GIVEN** a captured fixture of MLPPPO training trajectory + final parameter tensor on a klinotaxis smoke config with seed pinned, recorded pre-refactor
- **WHEN** the same config is re-run post-refactor with the same seed
- **THEN** every per-step action probability tensor SHALL satisfy `torch.equal(post, pre)`
- **AND** every per-episode parameter tensor SHALL satisfy `torch.equal(post, pre)`

#### Scenario: LSTMPPO byte-equivalence on klinotaxis smoke config

- **GIVEN** a captured fixture of LSTMPPO training trajectory + final parameter tensor on a klinotaxis smoke config with seed pinned, recorded pre-refactor
- **WHEN** the same config is re-run post-refactor with the same seed
- **THEN** every per-step action probability tensor + LSTM hidden state SHALL satisfy `torch.equal(post, pre)`
- **AND** every per-episode parameter tensor SHALL satisfy `torch.equal(post, pre)`

### Requirement: Migration Regression Bar — Other 17 Architectures Numerical Equivalence

The remaining 17 brain architectures SHALL produce parameter tensors after a 5-step smoke training pre-refactor and post-refactor that satisfy `torch.allclose(rtol=0, atol=1e-7)`, with all RNG seeds pinned.

#### Scenario: Per-architecture numerical-equivalence smoke

- **GIVEN** a brain architecture in `{QVARCIRCUIT, QQLEARNING, MLP_REINFORCE, MLP_DQN, QRC, QRH, QEF, CRH, SPIKING_REINFORCE, QSNN_REINFORCE, QSNN_PPO, HYBRID_QUANTUM, HYBRID_CLASSICAL, HYBRID_QUANTUM_CORTEX, QLIF_LSTM, QRH_QLSTM, CRH_QLSTM}`
- **AND** a 5-step smoke training config with a pinned seed
- **WHEN** the brain is trained for 5 steps pre-refactor and post-refactor on that config
- **THEN** every Pydantic-exposed parameter tensor SHALL satisfy `torch.allclose(post, pre, rtol=0, atol=1e-7)`
- **AND** any architecture exceeding the tolerance SHALL be either fixed or have its tolerance widened with explicit written justification in the T2 logbook

#### Scenario: Quantum architectures use deterministic simulator

- **WHEN** a quantum-family architecture's regression-equivalence test is executed
- **THEN** the test SHALL use a deterministic statevector simulator (not the noisy AerSimulator)
- **AND** SHALL pin shot-RNG seeds so QPU-shot variance does not introduce drift
