# Spec: connectome-ppo-brain

## ADDED Requirements

### Requirement: Sanity-floor arms share the substrate they bound

The panel's floors are only interpretable if they differ from the arm they bound in the dimension under test and in nothing else. Selecting a floor SHALL therefore change the learning behaviour without changing the substrate the learning acts on.

Concretely: a frozen-weights floor and an unmodulated-Hebbian floor SHALL construct the same topology parameters, including the **same motor readout**, as the plasticity arm they are compared against at the same seed. Because the readout is set to the anatomical contrast under the plasticity rule and left at its random initialisation under the gradient rule, a floor configured under the gradient rule would carry a different decoder, and any gap between the arms would confound decoding with learning.

`ConnectomePPOBrainConfig` SHALL expose the unmodulated mode through the same rule-selection field as the other rules, so an arm is a one-key change. The unmodulated selection SHALL carry the same activity-trace requirement as the modulated one, validated at load time, since it reads the same trace.

#### Scenario: The floors and the plastic arm share a substrate

- **WHEN** a plastic arm, a frozen-weights floor, and an unmodulated-Hebbian floor are constructed at the same seed
- **THEN** all three SHALL have byte-identical topology parameters at initialisation, the motor readout included

#### Scenario: The frozen floor does not learn

- **GIVEN** a frozen-weights floor
- **WHEN** an episode is run
- **THEN** the chemical weights SHALL be bit-identical to their initial values

#### Scenario: The unmodulated floor learns without reward

- **GIVEN** an unmodulated-Hebbian floor
- **WHEN** an episode is run
- **THEN** the chemical weights SHALL have changed from their initial values
- **AND** running the same episode under a different reward stream SHALL produce the same weights

#### Scenario: Unmodulated selection requires a trace

- **WHEN** a configuration selects the unmodulated mode without enabling activity traces
- **THEN** loading SHALL fail with an error naming both fields

### Requirement: Sanity-floor configurations

Configurations for both floors SHALL ship with this change, each differing from the plastic wild-type arm by the minimal key set that selects it, so the floor comparison holds every other condition constant.

#### Scenario: Each floor config is a minimal delta

- **WHEN** a floor configuration is compared with the plastic wild-type configuration
- **THEN** the only differences SHALL be the keys that select that floor
- **AND** no environment, reward, satiety, or sensing key SHALL differ

#### Scenario: Floor configs load and select their arms

- **WHEN** each floor configuration is loaded through the standard loader
- **THEN** it SHALL validate
- **AND** the resulting brain SHALL be configured as that floor
