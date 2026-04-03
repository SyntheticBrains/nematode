## MODIFIED Requirements

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
