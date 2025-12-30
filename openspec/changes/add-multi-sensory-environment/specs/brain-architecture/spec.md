## ADDED Requirements

### Requirement: Extended BrainParams for Multi-Sensory Input
The BrainParams class SHALL include additional optional fields for Phase 1 sensory modalities, enabling brains to process temperature, health, and mechanosensory information.

#### Scenario: Temperature Sensing Fields
- **GIVEN** a brain receiving sensory input in a thermotaxis-enabled environment
- **WHEN** BrainParams is populated
- **THEN** the following fields SHALL be available:
  - `temperature: float | None` - Current temperature at agent position (degrees Celsius)
  - `temperature_gradient_strength: float | None` - Magnitude of local temperature gradient
  - `temperature_gradient_direction: float | None` - Direction of increasing temperature (radians)
  - `cultivation_temperature: float | None` - Agent's preferred temperature (Tc)
- **AND** all fields SHALL default to None when thermotaxis is disabled

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
  - `penalty_boundary_collision: float` - Penalty for hitting grid boundary (default 0.02)

### Requirement: Multi-Objective Evaluation Metrics
The SimulationResult SHALL include optional per-objective scores for multi-sensory tasks.

#### Scenario: Temperature Comfort Score
- **GIVEN** an episode in a thermotaxis-enabled environment
- **WHEN** episode metrics are computed
- **THEN** `temperature_comfort_score: float | None` SHALL be calculated
- **AND** score SHALL equal (steps in comfort zone) / (total steps)
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

#### Scenario: Multi-Objective Composite Score
- **GIVEN** multi-objective metrics are available
- **WHEN** composite benchmark score is calculated
- **THEN** base Phase 0 score components SHALL be weighted at 85%
- **AND** multi-objective components SHALL be weighted at 15%
- **AND** the formula SHALL be documented for reproducibility
