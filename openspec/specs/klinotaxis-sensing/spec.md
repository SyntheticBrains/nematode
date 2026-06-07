# klinotaxis-sensing Specification

## Purpose

Defines requirements for klinotaxis (head-sweep) sensing mode providing local spatial gradient information via lateral concentration sampling, enabling biologically accurate chemotaxis and pheromone trail-following.

## Requirements

### Requirement: Klinotaxis Sensing Mode

When `SensingMode.KLINOTAXIS` is selected for a modality, the system SHALL sample concentration at the agent's center, perpendicular-left, and perpendicular-right positions, compute the lateral gradient as `right - left`, compute the temporal derivative via STAM, and auto-enable STAM whenever any modality uses klinotaxis.

#### Scenario: Lateral and temporal sampling under klinotaxis

- **WHEN** `SensingMode.KLINOTAXIS` is selected for a modality

- **THEN** the system SHALL sample concentration at the agent's position (center), 1 cell perpendicular-left of heading (left), and 1 cell perpendicular-right of heading (right)

- **AND** the lateral gradient SHALL be computed as `right - left`

- **AND** the temporal derivative SHALL be computed via STAM (same as derivative mode)

- **AND** STAM SHALL be auto-enabled when any modality uses klinotaxis

### Requirement: Head-Sweep Geometry

When computing lateral offsets for a given heading direction, the system SHALL apply the heading-specific left/right offsets, reuse the last non-STAY heading for STAY, and clamp all positions to grid bounds [0, grid_size-1].

#### Scenario: Heading-specific lateral offsets

- **WHEN** computing lateral offsets for a given heading direction

- **THEN** UP heading SHALL use left=(x-1,y), right=(x+1,y)

- **AND** RIGHT heading SHALL use left=(x,y+1), right=(x,y-1)

- **AND** DOWN heading SHALL use left=(x+1,y), right=(x-1,y)

- **AND** LEFT heading SHALL use left=(x,y-1), right=(x,y+1)

- **AND** STAY heading SHALL use the last non-STAY heading direction

- **AND** positions SHALL be clamped to grid bounds [0, grid_size-1]

### Requirement: Klinotaxis Sensory Modules

Klinotaxis sensory modules SHALL produce CoreFeatures with classical_dim=3 (strength, angle, binary), and `apply_sensing_mode()` SHALL substitute the configured oracle modules with their klinotaxis variants.

#### Scenario: Feature extraction produces classical_dim=3

- **WHEN** a klinotaxis sensory module extracts features

- **THEN** it SHALL produce CoreFeatures with classical_dim=3:

  - strength: scalar concentration at agent position (same as temporal)
  - angle: `tanh(lateral_gradient * lateral_scale)` normalized to [-1, 1]
  - binary: `tanh(dC/dt * derivative_scale)` normalized to [-1, 1]

#### Scenario: Oracle module substitution to klinotaxis variants

- **WHEN** the following oracle modules are configured with klinotaxis mode

- **THEN** `apply_sensing_mode()` SHALL substitute them as follows:

  - `food_chemotaxis` → `food_chemotaxis_klinotaxis`
  - `nociception` → `nociception_klinotaxis`
  - `thermotaxis` → `thermotaxis_klinotaxis`
  - `aerotaxis` → `aerotaxis_klinotaxis`
  - `pheromone_food` → `pheromone_food_klinotaxis`
  - `pheromone_alarm` → `pheromone_alarm_klinotaxis`
  - `pheromone_aggregation` → `pheromone_aggregation_klinotaxis`

### Requirement: apply_sensing_mode Substitution

When `apply_sensing_mode()` processes a module name, it SHALL use explicit mode matching to select the klinotaxis, temporal, or original oracle module, and SHALL preserve existing TEMPORAL and DERIVATIVE mode behavior exactly.

#### Scenario: Explicit mode matching for module selection

- **WHEN** `apply_sensing_mode()` processes a module name

- **THEN** it SHALL use explicit mode matching:

  - `== KLINOTAXIS` → `*_klinotaxis` module
  - `!= ORACLE and != KLINOTAXIS` → `*_temporal` module (covers TEMPORAL + DERIVATIVE)
  - `== ORACLE` → original oracle module

- **AND** existing TEMPORAL and DERIVATIVE mode behavior SHALL be preserved exactly

### Requirement: Lateral Scale Configuration

When `lateral_scale` is configured on SensingConfig, it SHALL be a positive float (default 50.0), SHALL be passed to BrainParams for use by klinotaxis sensory modules, and SHALL be independent of `derivative_scale`.

#### Scenario: lateral_scale configuration and propagation

- **WHEN** `lateral_scale` is configured on SensingConfig

- **THEN** it SHALL be a positive float (default 50.0)

- **AND** it SHALL be passed to BrainParams for use by klinotaxis sensory modules

- **AND** it SHALL be independent of `derivative_scale`

### Requirement: BrainParams Lateral Gradient Fields

When klinotaxis mode is active for a modality, BrainParams SHALL include the corresponding lateral gradient field, and all lateral gradient fields SHALL default to None when klinotaxis is not active.

#### Scenario: Lateral gradient fields per modality

- **WHEN** klinotaxis mode is active for a modality

- **THEN** BrainParams SHALL include the corresponding lateral gradient field:

  - `food_lateral_gradient: float | None`
  - `predator_lateral_gradient: float | None`
  - `temperature_lateral_gradient: float | None`
  - `oxygen_lateral_gradient: float | None`
  - `pheromone_food_lateral_gradient: float | None`
  - `pheromone_alarm_lateral_gradient: float | None`
  - `pheromone_aggregation_lateral_gradient: float | None`

- **AND** all lateral gradient fields SHALL default to None when klinotaxis is not active
