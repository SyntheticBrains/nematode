# aerotaxis Specification

## Purpose

This specification defines the aerotaxis (oxygen sensing) system for the Quantum Nematode project. It implements C. elegans oxygen-guided navigation with asymmetric zone classification (5 zones), oracle and temporal sensing modes, zone-based rewards/HP damage, visualization, and full experiment tracking pipeline.

## Requirements
### Requirement: Oxygen Field Generation

The system SHALL provide an OxygenField class that computes spatial oxygen concentration distributions using a linear gradient plus high/low oxygen spots with exponential decay, mirroring the TemperatureField architecture.

#### Scenario: Base Oxygen at Grid Center

- **WHEN** an OxygenField is created with `base_oxygen: 10.0` and `grid_size: 100`
- **THEN** `get_oxygen((50, 50))` SHALL return approximately 10.0 (center of grid)
- **AND** the base oxygen SHALL represent the ambient O2 percentage at the grid center

#### Scenario: Linear Oxygen Gradient

- **WHEN** an OxygenField has `gradient_direction: 0.0` (east) and `gradient_strength: 0.1`
- **THEN** oxygen SHALL increase by 0.1% per cell moving east from center
- **AND** oxygen SHALL decrease by 0.1% per cell moving west from center
- **AND** the gradient contribution SHALL be computed as `(rel_x * cos(direction) + rel_y * sin(direction)) * gradient_strength` where rel_x/rel_y are relative to grid center

#### Scenario: High Oxygen Spots

- **WHEN** an OxygenField has `high_oxygen_spots: [[75, 50, 8.0]]`
- **THEN** oxygen at position (75, 50) SHALL be increased by approximately 8.0% from the base+gradient value
- **AND** the effect SHALL decay exponentially with distance: `intensity * exp(-distance / spot_decay_constant)`
- **AND** spots at greater distance SHALL contribute less

#### Scenario: Low Oxygen Spots

- **WHEN** an OxygenField has `low_oxygen_spots: [[25, 25, 6.0]]`
- **THEN** oxygen at position (25, 25) SHALL be decreased by approximately 6.0% from the base+gradient value
- **AND** low oxygen spots SHALL represent bacterial consumption sinks (areas where bacteria deplete O2)

#### Scenario: Oxygen Value Clamping

- **WHEN** the computed oxygen concentration would be below 0.0 or above 21.0
- **THEN** the value SHALL be clamped to the range [0.0, 21.0]
- **AND** 21.0 represents atmospheric oxygen maximum
- **AND** 0.0 represents complete anoxia

#### Scenario: Gradient Computation via Central Difference

- **WHEN** `get_gradient(position)` is called
- **THEN** the oxygen gradient SHALL be computed using central difference: `dO2/dx ≈ (O2(x+1, y) - O2(x-1, y)) / 2`
- **AND** `get_gradient_polar(position)` SHALL return (magnitude, direction) in polar coordinates
- **AND** direction SHALL point toward increasing oxygen concentration

### Requirement: Asymmetric Oxygen Zone Classification

The system SHALL classify oxygen concentrations into zones using absolute percentage thresholds, reflecting the biological asymmetry between hypoxia and hyperoxia responses.

#### Scenario: Comfort Zone

- **WHEN** oxygen concentration is between 5.0% and 12.0% (inclusive)
- **THEN** the zone SHALL be classified as `OxygenZone.COMFORT`
- **AND** this SHALL represent the preferred range matching C. elegans biology (URX/BAG neuron quiescent)

#### Scenario: Danger Hypoxia Zone

- **WHEN** oxygen concentration is between 2.0% and 5.0% (exclusive of comfort boundary)
- **THEN** the zone SHALL be classified as `OxygenZone.DANGER_HYPOXIA`
- **AND** this SHALL represent BAG neuron activation (detecting low oxygen)

#### Scenario: Lethal Hypoxia Zone

- **WHEN** oxygen concentration is below 2.0%
- **THEN** the zone SHALL be classified as `OxygenZone.LETHAL_HYPOXIA`
- **AND** this SHALL represent anaerobic conditions incompatible with aerobic metabolism

#### Scenario: Danger Hyperoxia Zone

- **WHEN** oxygen concentration is between 12.0% and 17.0% (exclusive of comfort boundary)
- **THEN** the zone SHALL be classified as `OxygenZone.DANGER_HYPEROXIA`
- **AND** this SHALL represent URX/AQR/PQR neuron activation (detecting high oxygen)

#### Scenario: Lethal Hyperoxia Zone

- **WHEN** oxygen concentration is above 17.0%
- **THEN** the zone SHALL be classified as `OxygenZone.LETHAL_HYPEROXIA`
- **AND** this SHALL represent oxidative stress conditions

#### Scenario: Configurable Zone Thresholds

- **WHEN** custom OxygenZoneThresholds are provided (e.g., `comfort_lower: 6.0`, `comfort_upper: 10.0`)
- **THEN** zone boundaries SHALL use the custom thresholds
- **AND** default thresholds SHALL be `lethal_hypoxia_upper: 2.0`, `danger_hypoxia_upper: 5.0`, `comfort_lower: 5.0`, `comfort_upper: 12.0`, `danger_hyperoxia_upper: 17.0`

### Requirement: Oxygen Zone Effects

The environment SHALL apply zone-based rewards, penalties, and HP damage for oxygen zones, paralleling the thermotaxis zone effect system.

#### Scenario: Comfort Zone No Reward

- **WHEN** the agent is in the oxygen comfort zone (5-12% O2)
- **THEN** `apply_oxygen_effects()` SHALL return `comfort_reward` (default 0.0) as reward delta
- **AND** SHALL return 0.0 HP damage
- **AND** the step SHALL be counted toward oxygen comfort zone tracking

#### Scenario: Danger Zone Penalty and Damage

- **WHEN** the agent is in a danger zone (hypoxia or hyperoxia)
- **THEN** `apply_oxygen_effects()` SHALL return the configured `danger_penalty` (default -0.5) as reward delta
- **AND** SHALL return the configured `danger_hp_damage` (default 0.5) as HP damage
- **AND** HP SHALL be reduced by the damage amount (clamped to 0.0 minimum)

#### Scenario: Lethal Zone Maximum Damage

- **WHEN** the agent is in a lethal zone (hypoxia or hyperoxia)
- **THEN** `apply_oxygen_effects()` SHALL return the configured `danger_penalty` as reward delta
- **AND** SHALL return the configured `lethal_hp_damage` (default 6.0) as HP damage

#### Scenario: Brave Foraging Bonus in Danger Zones

- **WHEN** the agent collects food while in an oxygen danger zone (hypoxia or hyperoxia)
- **THEN** the system SHALL apply the configured `reward_discomfort_food` bonus (default 0.0)
- **AND** this encourages agents to briefly enter dangerous O2 zones for food when the reward is positive

#### Scenario: Oxygen Comfort Score Tracking

- **WHEN** an episode completes in an aerotaxis-enabled environment
- **THEN** `get_oxygen_comfort_score()` SHALL return the ratio of steps in comfort zone to total aerotaxis steps
- **AND** the score SHALL range from 0.0 to 1.0

#### Scenario: Aerotaxis Disabled

- **WHEN** aerotaxis is not enabled in the environment configuration
- **THEN** `apply_oxygen_effects()` SHALL return (0.0, 0.0)
- **AND** no oxygen-related tracking SHALL occur

### Requirement: Aerotaxis Oracle Sensory Module

The system SHALL provide an `aerotaxis` sensory module for oracle mode that extracts oxygen gradient features using the unified SensoryModule architecture.

#### Scenario: Oracle Feature Extraction

- **WHEN** the `aerotaxis` module's `_aerotaxis_core(params)` is called with oracle sensing
- **THEN** `strength` SHALL be `tanh(oxygen_gradient_strength)` normalized to [0, 1]
- **AND** `angle` SHALL be the egocentric relative angle from the agent's heading to the direction of increasing oxygen, normalized to [-1, 1]
- **AND** `binary` SHALL be the oxygen comfort deviation: `clip((O2 - 8.5) / 12.5, -1, 1)` where 8.5 is the comfort midpoint and 12.5 is the maximum realistic deviation
- **AND** `classical_dim` SHALL be 3

#### Scenario: Module Registration

- **WHEN** the sensory module registry is initialized
- **THEN** `ModuleName.AEROTAXIS = "aerotaxis"` SHALL be registered in the ModuleName enum
- **AND** `SENSORY_MODULES["aerotaxis"]` SHALL be a SensoryModule with `classical_dim=3` and `transform_type="standard"`

#### Scenario: Quantum Gate Angle Output

- **WHEN** the aerotaxis module's `to_quantum()` is called
- **THEN** the output SHALL be a 3-element array [rx, ry, rz] computed from CoreFeatures using the standard transform
- **AND** values SHALL be in [-π/2, π/2] range

#### Scenario: Aerotaxis Disabled

- **WHEN** `oxygen_concentration` is None in BrainParams
- **THEN** the module SHALL return `CoreFeatures(0.0, 0.0, 0.0)` (zero signal)

### Requirement: Aerotaxis Temporal Sensory Module

The system SHALL provide an `aerotaxis_temporal` sensory module for temporal and derivative sensing modes.

#### Scenario: Temporal Feature Extraction

- **WHEN** the `aerotaxis_temporal` module's `_aerotaxis_temporal_core(params)` is called
- **THEN** `strength` SHALL be the absolute oxygen comfort deviation: `abs(clip((O2 - 8.5) / 12.5, -1, 1))`
- **AND** `angle` SHALL be the scaled temporal derivative: `tanh(dO2/dt * derivative_scale)` where `derivative_scale` is from BrainParams
- **AND** `binary` SHALL be the signed oxygen comfort deviation: `clip((O2 - 8.5) / 12.5, -1, 1)`
- **AND** `classical_dim` SHALL be 3

#### Scenario: Module Registration

- **WHEN** the sensory module registry is initialized
- **THEN** `ModuleName.AEROTAXIS_TEMPORAL = "aerotaxis_temporal"` SHALL be registered in the ModuleName enum
- **AND** `SENSORY_MODULES["aerotaxis_temporal"]` SHALL be a SensoryModule with `classical_dim=3`

#### Scenario: Derivative Not Available

- **WHEN** `oxygen_dconcentration_dt` is None in BrainParams (temporal mode without derivative)
- **THEN** the `angle` feature SHALL be 0.0

### Requirement: Aerotaxis Visualization

The system SHALL render oxygen zones visually across all themes that support zone rendering.

#### Scenario: Pixel Theme Zone Overlays

- **WHEN** the pixel (Pygame) theme renders a frame in an aerotaxis-enabled environment
- **THEN** oxygen zone overlays SHALL be rendered as semi-transparent colored surfaces
- **AND** lethal hypoxia SHALL use dark red (180, 40, 40, 90)
- **AND** danger hypoxia SHALL use red-brown (200, 80, 60, 70)
- **AND** comfort SHALL be transparent (no overlay)
- **AND** danger hyperoxia SHALL use light cyan (80, 200, 220, 70)
- **AND** lethal hyperoxia SHALL use bright cyan (40, 180, 220, 90)

#### Scenario: Render Layer Ordering

- **WHEN** both thermotaxis and aerotaxis are enabled
- **THEN** the rendering pipeline SHALL be: background → temperature zones → oxygen zones → toxic zones → entities
- **AND** oxygen overlays SHALL alpha-blend on top of temperature overlays

#### Scenario: Status Bar Display

- **WHEN** aerotaxis is enabled and the pixel theme status bar is rendered
- **THEN** the status bar SHALL display the current oxygen percentage (e.g., "O2: 8.3%")
- **AND** SHALL display the current oxygen zone name (e.g., "COMFORT")
- **AND** the display SHALL appear alongside temperature information when both are enabled

#### Scenario: Rich/Terminal Theme Support

- **WHEN** a Rich or terminal-based theme renders in an aerotaxis-enabled environment
- **THEN** oxygen zone symbols SHALL be defined in ThemeSymbolSet
- **AND** oxygen zone background colors SHALL be defined in DarkColorRichStyleConfig

### Requirement: Aerotaxis Configuration

The system SHALL support aerotaxis configuration via YAML with opt-in enablement and full parameter control.

#### Scenario: Aerotaxis Disabled by Default

- **WHEN** no `aerotaxis` section is provided in environment configuration
- **THEN** aerotaxis SHALL be disabled
- **AND** no OxygenField SHALL be created
- **AND** all oxygen-related BrainParams fields SHALL be None
- **AND** existing configurations SHALL work unchanged

#### Scenario: Minimal Aerotaxis Configuration

- **WHEN** the YAML configuration includes:

```yaml
environment:
  aerotaxis:
    enabled: true
```

- **THEN** aerotaxis SHALL be enabled with default parameters: `base_oxygen: 10.0`, `gradient_strength: 0.1`, `gradient_direction: 0.0`
- **AND** default zone thresholds SHALL apply (comfort 5-12%, danger hypoxia 2-5%, danger hyperoxia 12-17%)

#### Scenario: Full Aerotaxis Configuration

- **WHEN** a complete aerotaxis configuration is provided:

```yaml
environment:
  aerotaxis:
    enabled: true
    base_oxygen: 10.0
    gradient_direction: 1.5708
    gradient_strength: 0.08
    high_oxygen_spots:
      - [80, 50, 6.0]
    low_oxygen_spots:
      - [30, 30, 5.0]
    spot_decay_constant: 6.0
    comfort_reward: 0.0
    danger_penalty: -0.5
    danger_hp_damage: 0.5
    lethal_hp_damage: 6.0
    reward_discomfort_food: 0.0
    lethal_hypoxia_upper: 2.0
    danger_hypoxia_upper: 5.0
    comfort_lower: 5.0
    comfort_upper: 12.0
    danger_hyperoxia_upper: 17.0
```

- **THEN** all parameters SHALL be parsed and applied to the OxygenField and zone system

#### Scenario: Combined Thermal and Oxygen Configuration

- **WHEN** both `thermotaxis` and `aerotaxis` sections are enabled
- **THEN** both fields SHALL coexist independently
- **AND** `apply_temperature_effects()` and `apply_oxygen_effects()` SHALL both be called per step
- **AND** rewards/penalties from both systems SHALL be additive
- **AND** HP damage from both systems SHALL be additive

### Requirement: Aerotaxis Data Pipeline

The system SHALL track, export, and aggregate oxygen sensing data through the full experiment tracking pipeline, parallel to the existing temperature data pipeline.

#### Scenario: Per-Step Oxygen History Recording

- **WHEN** the agent completes a step in an aerotaxis-enabled environment
- **THEN** the episode tracker SHALL record the current oxygen percentage via `track_oxygen(oxygen)` in `agent/tracker.py`
- **AND** the oxygen value SHALL be appended to `oxygen_history: list[float]` in `EpisodeTrackingData` in `report/dtypes.py`
- **AND** the runner step loop SHALL call `track_oxygen()` parallel to the existing `track_temperature()` call

#### Scenario: Per-Run Oxygen History in SimulationResult

- **WHEN** a run completes in an aerotaxis-enabled environment
- **THEN** `SimulationResult` in `report/dtypes.py` SHALL include `oxygen_history: list[float] | None` (None if aerotaxis disabled)
- **AND** SHALL include `oxygen_comfort_score: float | None` (None if aerotaxis disabled)
- **AND** `run_simulation.py` SHALL populate both fields from the episode tracker and environment

#### Scenario: Per-Run CSV and Plot Exports

- **WHEN** per-run data is exported in `report/csv_export.py`
- **THEN** the system SHALL export `oxygen_history.csv` with columns [step, oxygen] (parallel to `temperature_history.csv`)
- **AND** the foraging_summary.csv SHALL include final_oxygen, mean_oxygen, min_oxygen, max_oxygen metrics (parallel to temperature metrics)
- **AND** `report/plots.py` SHALL generate `oxygen_progression.png` showing step-by-step O2 percentage with mean line (parallel to `temperature_progression.png`)

#### Scenario: Experiment Metadata Aggregation

- **WHEN** experiment results are aggregated in `experiment/tracker.py` and `experiment/metadata.py`
- **THEN** `PerRunResult` SHALL include `oxygen_comfort_score: float | None`
- **AND** `ResultsMetadata` SHALL include `avg_oxygen_comfort_score: float | None` and `post_convergence_oxygen_comfort_score: float | None`
- **AND** aggregation SHALL average oxygen_comfort_score across runs (parallel to temperature_comfort_score)

#### Scenario: Per-Step Data Flushing

- **WHEN** per-step data is flushed to save memory after export
- **THEN** `result.oxygen_history` SHALL be set to None (parallel to temperature_history flushing)
- **AND** scalar snapshots SHALL be preserved before flushing

#### Scenario: Main Simulation Results CSV

- **WHEN** simulation results are exported to `simulation_results.csv`
- **THEN** the fieldnames SHALL include `oxygen_comfort_score`
- **AND** the value SHALL be the per-run oxygen comfort score or NaN if aerotaxis disabled
