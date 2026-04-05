## MODIFIED Requirements

### Requirement: Extended BrainParams for Social Sensing

The BrainParams class SHALL include an optional field for social proximity information in multi-agent scenarios.

#### Scenario: Social Proximity Field

- **GIVEN** a brain receiving sensory input in a multi-agent environment
- **WHEN** BrainParams is populated
- **THEN** the following field SHALL be available:
  - `nearby_agents_count: int | None` -- Number of other agents within social detection radius
- **AND** this field SHALL default to None in single-agent mode
- **AND** no existing brain functionality SHALL be affected when this field is None

#### Scenario: Single-Agent Backward Compatibility

- **GIVEN** a single-agent simulation (no multi_agent config)
- **WHEN** BrainParams is populated
- **THEN** `nearby_agents_count` SHALL be None
- **AND** all existing brain architectures SHALL function identically to before this change

## ADDED Requirements

### Requirement: Social Proximity Sensory Module

The system SHALL provide a sensory module for social proximity detection in multi-agent scenarios.

#### Scenario: Module Registration

- **GIVEN** the sensory module system
- **THEN** `SOCIAL_PROXIMITY` SHALL be a valid ModuleName
- **AND** it SHALL be registered in `SENSORY_MODULES` with `classical_dim=1`

#### Scenario: Feature Extraction with Nearby Agents

- **GIVEN** BrainParams with `nearby_agents_count=5`
- **WHEN** the social_proximity module extracts features
- **THEN** it SHALL return `CoreFeatures(strength=0.5, angle=0.0, binary=0.0)`
- **AND** strength SHALL be computed as `min(count, 10) / 10.0`

#### Scenario: No Nearby Agents

- **GIVEN** BrainParams with `nearby_agents_count=0` or `nearby_agents_count=None`
- **WHEN** the social_proximity module extracts features
- **THEN** it SHALL return `CoreFeatures(strength=0.0, angle=0.0, binary=0.0)`

#### Scenario: High Agent Count Clamping

- **GIVEN** BrainParams with `nearby_agents_count=15`
- **WHEN** the social_proximity module extracts features
- **THEN** strength SHALL be clamped to 1.0 (min(15, 10) / 10.0 = 1.0)

#### Scenario: Classical Feature Dimension

- **WHEN** `get_classical_feature_dimension` is called for a module list including SOCIAL_PROXIMITY
- **THEN** SOCIAL_PROXIMITY SHALL contribute exactly 1 dimension to the feature vector

#### Scenario: Quantum Feature Transform

- **GIVEN** BrainParams with `nearby_agents_count=5`
- **WHEN** the social_proximity module generates quantum rotation angles
- **THEN** it SHALL produce valid rotation angles within [-pi/2, pi/2]
