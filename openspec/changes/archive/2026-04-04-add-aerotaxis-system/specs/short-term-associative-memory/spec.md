## MODIFIED Requirements

### Requirement: STAM Buffer Storage

The system SHALL provide an exponential-decay memory buffer that stores recent sensory readings, positions, and actions for temporal integration.

#### Scenario: Recording Sensory Data

- **WHEN** the agent completes a simulation step
- **THEN** the STAM buffer SHALL record the current scalar readings for all 4 channels (food concentration, temperature, predator concentration, oxygen concentration)
- **AND** SHALL record the step-to-step position change (dx, dy) as a proprioceptive movement signal, NOT absolute grid coordinates
- **AND** SHALL record the action taken in that step
- **AND** the most recent entry SHALL be stored at index 0

#### Scenario: Buffer Size Limit

- **WHEN** the buffer contains `buffer_size` entries (default 30) and a new entry is recorded
- **THEN** the oldest entry SHALL be discarded
- **AND** the buffer SHALL never exceed `buffer_size` entries
- **AND** the buffer SHALL use a circular buffer (deque) for O(1) append and discard

#### Scenario: Episode Reset

- **WHEN** a new episode begins
- **THEN** the STAM buffer SHALL be cleared of all entries
- **AND** subsequent calls to `get_memory_state()` SHALL return zero vectors until new data is recorded
- **AND** no state from the previous episode SHALL influence the new episode

### Requirement: Temporal Derivative Computation

The system SHALL compute temporal derivatives (rate of change) for each scalar channel using weighted finite differences.

#### Scenario: Derivative With Sufficient History

- **WHEN** the buffer contains at least 2 entries
- **THEN** the temporal derivative for each of the 4 channels SHALL be computed as `Σ(w[i] * (C[i] - C[i+1])) / Σ(w[i])` where i=0 is most recent
- **AND** channel 3 (oxygen) SHALL compute dO2/dt using the same formula as channels 0-2

#### Scenario: Derivative With Insufficient History

- **WHEN** the buffer contains fewer than 2 entries
- **THEN** the temporal derivative SHALL return 0.0 for all 4 channels
- **AND** this SHALL be biologically plausible (sensory neurons require an adaptation period)

#### Scenario: Derivative Sign Semantics

- **WHEN** the agent moves toward a food source (concentration increasing)
- **THEN** the food concentration temporal derivative SHALL be positive
- **AND** when the agent moves away (concentration decreasing), the derivative SHALL be negative
- **AND** when stationary or equidistant, the derivative SHALL be near zero
- **AND** the same sign semantics SHALL apply to oxygen (positive = increasing O2)

### Requirement: Fixed-Size Memory State Vector

The system SHALL produce a fixed-size memory state vector suitable for neural network input, regardless of buffer fill level.

#### Scenario: Full Memory State Output

- **WHEN** `get_memory_state()` is called
- **THEN** it SHALL return a numpy array of exactly 11 floats:
  - 4 weighted scalar means (food concentration, temperature, predator concentration, oxygen concentration)
  - 4 temporal derivatives (dC/dt for each channel)
  - 2 position deltas (dx, dy: deviation of most recent step-to-step movement from weighted mean of recent movements — captures change in movement pattern)
  - 1 action variety metric (entropy of recent actions — computational convenience for learning, not biologically motivated)
- **AND** the array shape SHALL be (11,) regardless of buffer fill level

#### Scenario: Empty Buffer State

- **WHEN** `get_memory_state()` is called with an empty buffer
- **THEN** all 11 values SHALL be 0.0
- **AND** the array shape SHALL still be (11,)

#### Scenario: Partially Filled Buffer

- **WHEN** the buffer has fewer than `buffer_size` entries
- **THEN** weighted means and derivatives SHALL be computed using only available entries
- **AND** zero-weighting for missing entries SHALL NOT be applied (use actual entries only)
- **AND** the output SHALL still be exactly 11 floats

#### Scenario: Disabled Channel Handling

- **WHEN** aerotaxis is not enabled in the environment
- **THEN** the oxygen channel (index 3) SHALL record 0.0 each step
- **AND** the oxygen weighted mean (index 3) and derivative (index 7) SHALL be 0.0
- **AND** the remaining 9 values SHALL be computed normally

### Requirement: STAM Sensory Module

The system SHALL provide a sensory module registry entry for STAM that integrates with the existing SensoryModule framework.

#### Scenario: Classical Feature Output

- **WHEN** the STAM module's `to_classical()` is called
- **THEN** it SHALL return the full 11-float memory state vector
- **AND** `classical_dim` SHALL be 11
- **AND** the output SHALL be compatible with `extract_classical_features()` concatenation

#### Scenario: Quantum Feature Output

- **WHEN** the STAM module's `to_quantum()` is called
- **THEN** it SHALL return a 3-element array [rx, ry, rz] compressed from the memory state
- **AND** the compression SHALL use: rx=mean of weighted scalar means (indices 0-3), ry=mean of temporal derivatives (indices 4-7), rz=action entropy (index 10)
- **AND** values SHALL be scaled to [-π/2, π/2] range

#### Scenario: STAM Module When Disabled

- **WHEN** the STAM module is included in sensory modules but STAM is not enabled (stam_state is None)
- **THEN** `to_classical()` SHALL return a zero vector of shape (11,)
- **AND** `to_quantum()` SHALL return a zero vector of shape (3,)

### Requirement: STAM Configuration

The system SHALL support configurable STAM parameters via YAML configuration.

#### Scenario: STAM Parameter Defaults

- **WHEN** `stam_enabled: true` is set without other STAM parameters
- **THEN** `buffer_size` SHALL default to 30
- **AND** `decay_rate` SHALL default to 0.1
- **AND** the number of scalar channels SHALL be 4 (food concentration, temperature, predator concentration, oxygen concentration)
- **AND** if a channel's source is disabled (e.g., thermotaxis not configured, aerotaxis not configured, or predators disabled), the channel SHALL record 0.0 and its temporal derivative SHALL be 0.0

#### Scenario: Custom STAM Parameters

- **WHEN** STAM parameters are explicitly configured
- **THEN** the system SHALL accept `stam_buffer_size` (integer, > 0)
- **AND** SHALL accept `stam_decay_rate` (float, > 0)
- **AND** custom values SHALL override defaults

#### Scenario: STAM Disabled by Default

- **WHEN** no sensing configuration is provided
- **THEN** STAM SHALL be disabled
- **AND** no STAM module SHALL be added to sensory modules
- **AND** no memory state SHALL be computed
