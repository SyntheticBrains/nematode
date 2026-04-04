# temporal-sensing Specification

## ADDED Requirements

### Requirement: Sensing Mode Selection

The system SHALL support three sensing modes for each gradient-based sensory modality (chemotaxis, thermotaxis, nociception, aerotaxis), selectable independently per modality via configuration.

#### Scenario: Oracle Mode (Default)

- **WHEN** a modality is configured with `mode: oracle` or no sensing config is provided
- **THEN** the system SHALL provide spatial gradient information (magnitude and direction) as in the existing implementation
- **AND** backward compatibility with all existing configurations SHALL be maintained

#### Scenario: Temporal Mode (Mode A)

- **WHEN** a modality is configured with `mode: temporal`
- **THEN** the system SHALL provide only the scalar value at the agent's current position (concentration, temperature, oxygen percentage)
- **AND** the system SHALL NOT provide gradient magnitude or gradient direction for that modality
- **AND** the agent SHALL rely on STAM memory buffers to infer gradient direction from movement history

#### Scenario: Derivative Mode (Mode B)

- **WHEN** a modality is configured with `mode: derivative`
- **THEN** the system SHALL provide the scalar value at the agent's current position
- **AND** the system SHALL provide the temporal derivative (rate of change) of that scalar value
- **AND** the system SHALL NOT provide gradient direction for that modality
- **AND** the temporal derivative SHALL be computed from the agent's recent sensory history

#### Scenario: Independent Per-Modality Configuration

- **WHEN** sensing modes are configured as `chemotaxis_mode: temporal`, `thermotaxis_mode: derivative`, `nociception_mode: oracle`, `aerotaxis_mode: temporal`
- **THEN** chemotaxis SHALL use temporal mode (scalar only)
- **AND** thermotaxis SHALL use derivative mode (scalar + dT/dt)
- **AND** nociception SHALL use oracle mode (spatial gradient)
- **AND** aerotaxis SHALL use temporal mode (scalar O2 only)
- **AND** all four modes SHALL operate correctly within the same simulation

### Requirement: Temporal Sensory Modules

The system SHALL provide sensory module registry entries for temporal sensing that replace oracle modules transparently across all brain architectures.

#### Scenario: Food Chemotaxis Temporal Module

- **WHEN** chemotaxis is configured in temporal or derivative mode
- **THEN** a `food_chemotaxis_temporal` module SHALL be registered in the SensoryModule registry
- **AND** the module SHALL extract strength directly from the `food_concentration` field on BrainParams (already tanh-normalized to [0, 1] by the environment method — the module SHALL NOT re-normalize)
- **AND** the module SHALL extract angle from the food temporal derivative (dC/dt), normalized via `tanh(derivative)` to [-1, 1] to match the CoreFeatures angle range, or 0 when not available
- **AND** the module SHALL produce valid quantum gate angles via `to_quantum()` and classical features via `to_classical()`

#### Scenario: Nociception Temporal Module

- **WHEN** nociception is configured in temporal or derivative mode
- **THEN** a `nociception_temporal` module SHALL be registered in the SensoryModule registry
- **AND** the module SHALL extract strength directly from the `predator_concentration` field on BrainParams (already tanh-normalized to [0, 1] by the environment method — the module SHALL NOT re-normalize)
- **AND** the module SHALL extract angle from the predator temporal derivative, normalized via `tanh(derivative)` to [-1, 1], or 0 when not available

#### Scenario: Thermotaxis Temporal Module

- **WHEN** thermotaxis is configured in temporal or derivative mode
- **THEN** a `thermotaxis_temporal` module SHALL be registered in the SensoryModule registry
- **AND** the module SHALL extract strength from temperature deviation from cultivation temperature (normalized to [-1, 1])
- **AND** the module SHALL extract angle from the temperature temporal derivative (dT/dt), normalized via `tanh(derivative)` to [-1, 1], or 0 when not available
- **AND** the module SHALL include temperature deviation as the binary field (classical_dim=3)

#### Scenario: Aerotaxis Temporal Module

- **WHEN** aerotaxis is configured in temporal or derivative mode
- **THEN** an `aerotaxis_temporal` module SHALL be registered in the SensoryModule registry
- **AND** the module SHALL extract strength from the absolute oxygen comfort deviation: `abs(clip((O2 - 8.5) / 12.5, -1, 1))`
- **AND** the module SHALL extract angle from the oxygen temporal derivative (dO2/dt), normalized via `tanh(derivative * derivative_scale)` to [-1, 1], or 0 when not available
- **AND** the module SHALL include signed oxygen comfort deviation as the binary field (classical_dim=3)

#### Scenario: Brain Architecture Transparency

- **WHEN** temporal sensing modules are used in place of oracle modules
- **THEN** all 19 brain architectures SHALL function without modification
- **AND** the `extract_classical_features()` function SHALL return correctly-shaped feature vectors
- **AND** the `to_quantum()` function SHALL return valid 3-element gate angle arrays
- **AND** the only change visible to brains SHALL be the semantic content of the features (scalar vs gradient)

### Requirement: Automatic Module Translation

The system SHALL automatically substitute temporal sensory module names for oracle module names based on the sensing configuration.

#### Scenario: Module Name Substitution

- **WHEN** a brain config specifies `sensory_modules: [food_chemotaxis, mechanosensation]` and `chemotaxis_mode: temporal`
- **THEN** the system SHALL automatically replace `food_chemotaxis` with `food_chemotaxis_temporal`
- **AND** `mechanosensation` SHALL remain unchanged (it is already biologically accurate)
- **AND** the substitution SHALL occur at config load time, before brain construction

#### Scenario: Aerotaxis Module Substitution

- **WHEN** a brain config specifies `sensory_modules: [aerotaxis]` and `aerotaxis_mode: temporal`
- **THEN** the system SHALL automatically replace `aerotaxis` with `aerotaxis_temporal`
- **AND** the substitution SHALL follow the same pattern as chemotaxis/thermotaxis/nociception

#### Scenario: STAM Module Auto-Append

- **WHEN** `stam_enabled: true` in the sensing configuration
- **THEN** the system SHALL automatically append the `stam` module to the brain's sensory modules list
- **AND** the append SHALL occur regardless of which sensing modes are selected
- **AND** the `stam` module SHALL NOT be appended if already present in the list

#### Scenario: Combined Gradient Module Substitution

- **WHEN** a brain config specifies `sensory_modules: [chemotaxis]` (combined gradient) and `chemotaxis_mode: temporal`
- **THEN** the system SHALL replace `chemotaxis` with `food_chemotaxis_temporal`
- **AND** the system SHALL add `nociception_temporal` if `nociception_mode` is not `oracle`, or `nociception` if `nociception_mode` is `oracle`, unless a nociception module is already present
- **AND** this ensures the predator signal (previously embedded in the combined gradient) is not silently dropped

### Requirement: Mechanosensation Exemption

Mechanosensation SHALL NOT be affected by sensing mode configuration, as it is already biologically accurate.

#### Scenario: Mechanosensation Unchanged

- **WHEN** any sensing mode is configured (temporal, derivative, or oracle)
- **THEN** boundary contact detection SHALL remain binary (touching/not touching)
- **AND** predator contact detection SHALL remain binary (contact/no contact)
- **AND** no temporal derivative or scalar replacement SHALL be applied to mechanosensation
