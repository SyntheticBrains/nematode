## MODIFIED Requirements

### Requirement: Klinotaxis Sensory Modules

Klinotaxis sensory modules SHALL produce CoreFeatures with classical_dim=3 (strength, angle, binary), and `apply_sensing_mode()` SHALL substitute the configured oracle modules with their klinotaxis variants. When the adaptive chemosensory sensor (the `chemical-gradient-fidelity` capability) is enabled for a **chemosensory channel** (`food_chemotaxis`, and the pheromone / CO₂ channels where active — NOT thermotaxis, nociception, or predator mechano/chemosensation), the `strength` and/or `binary` (temporal-derivative) fields of that channel SHALL carry the adaptive transform's output per the configured channel-interaction mode; when it is disabled, the fields retain their current non-adaptive definitions, and non-chemosensory channels SHALL NOT be altered. *(As-built: the per-channel `*_klinotaxis_core` functions are pure and duplicated with no shared chokepoint, so the adaptive transform is applied **upstream, in the agent's sensory-assembly step** — it reshapes the channel's concentration / temporal-derivative inputs before they reach the cores — rather than inside the cores. The behavioural contract above is unchanged. This tranche wires the **food** channel only; pheromone / CO₂-chemo channels are a follow-up.)*

#### Scenario: Feature extraction produces classical_dim=3

- **WHEN** a klinotaxis sensory module extracts features

- **THEN** it SHALL produce CoreFeatures with classical_dim=3:

  - strength: scalar concentration at agent position (same as temporal), or, when the adaptive sensor is enabled in magnitude-contrast interaction, the adaptive contrast readout
  - angle: `tanh(lateral_gradient * lateral_scale)` normalized to [-1, 1]
  - binary: `tanh(dC/dt * derivative_scale)` normalized to [-1, 1], or, when the adaptive sensor is enabled in derivative-channel (fold-change) interaction, the background-normalized temporal-derivative readout

#### Scenario: Adaptive transform applies per configured channel interaction

- **WHEN** the adaptive chemosensory sensor is enabled for a klinotaxis chemical channel
- **THEN** the configured channel-interaction mode SHALL determine which field carries the adaptive readout (derivative-channel fold-change reshapes `binary`; magnitude-contrast supplies the `strength` contrast readout), and the non-adaptive definitions SHALL apply to any field not targeted by the configured mode

#### Scenario: Adaptive sensor disabled preserves current fields

- **WHEN** the adaptive chemosensory sensor is not enabled
- **THEN** `strength`, `angle`, and `binary` retain their current non-adaptive definitions with no change

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
