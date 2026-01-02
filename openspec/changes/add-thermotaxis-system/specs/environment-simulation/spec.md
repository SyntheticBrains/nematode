## ADDED Requirements

### Requirement: Temperature Field Spatial Distribution

The system SHALL provide a TemperatureField class that defines spatial temperature distributions for thermotaxis simulation.

#### Scenario: Linear Temperature Gradient

- **GIVEN** a TemperatureField with `base_temperature: 20.0`, `gradient_direction: 0.0`, `gradient_strength: 0.5`
- **WHEN** temperature is queried at position (10, 5) in a 20x20 grid
- **THEN** temperature SHALL be computed as: `base + x * strength * cos(direction) + y * strength * sin(direction)`
- **AND** position (10, 5) SHALL have temperature approximately 25.0°C (20 + 10\*0.5)

#### Scenario: Temperature Hot Spot

- **GIVEN** a TemperatureField with a hot spot at (15, 15) with intensity 10.0
- **WHEN** temperature is queried at position (15, 15)
- **THEN** temperature SHALL include full hot spot contribution (+10.0°C)
- **WHEN** temperature is queried at position (18, 15) (3 cells away)
- **THEN** hot spot contribution SHALL be reduced by exponential decay

#### Scenario: Temperature Cold Spot

- **GIVEN** a TemperatureField with a cold spot at (5, 5) with intensity -10.0
- **WHEN** temperature is queried at position (5, 5)
- **THEN** temperature SHALL include full cold spot contribution (-10.0°C)
- **AND** cold spots SHALL use same exponential decay as hot spots

#### Scenario: Temperature Gradient Vector Computation

- **GIVEN** a TemperatureField with a linear gradient
- **WHEN** gradient is queried at position (10, 10)
- **THEN** gradient SHALL be computed via central difference approximation
- **AND** gradient SHALL return (dx, dy) vector pointing toward increasing temperature

### Requirement: Thermotaxis Environment Integration

The DynamicForagingEnvironment SHALL integrate temperature field sensing when thermotaxis is enabled.

#### Scenario: Thermotaxis Initialization

- **GIVEN** a configuration with `thermotaxis.enabled: true`
- **WHEN** the environment is initialized
- **THEN** a TemperatureField SHALL be created from config parameters
- **AND** the field SHALL be queryable throughout the episode

#### Scenario: Temperature Sensing Per Step

- **GIVEN** an agent at position (10, 10) in a thermotaxis-enabled environment
- **WHEN** the step is executed
- **THEN** current temperature SHALL be computed at agent position
- **AND** temperature gradient SHALL be computed at agent position
- **AND** values SHALL be populated in BrainParams

#### Scenario: Thermotaxis Disabled Default

- **GIVEN** a configuration without thermotaxis settings
- **WHEN** the environment initializes
- **THEN** thermotaxis SHALL be disabled
- **AND** BrainParams temperature fields SHALL be None
- **AND** no temperature rewards/penalties SHALL apply

### Requirement: Temperature Zone Mechanics

The system SHALL apply rewards, penalties, and health effects based on temperature zones.

#### Scenario: Comfort Zone Reward

- **GIVEN** thermotaxis enabled with comfort_delta=5.0 (comfort zone: Tc±5°C = 15-25°C for Tc=20)
- **WHEN** agent is at a position with temperature 20°C
- **THEN** a comfort reward SHALL be applied (configurable, default +0.01)
- **AND** no HP damage SHALL occur

#### Scenario: Discomfort Zone Penalty

- **GIVEN** thermotaxis enabled with discomfort_delta=10.0 (discomfort: Tc±10°C outside comfort)
- **WHEN** agent is at a position with temperature 12°C (within 10-15°C zone)
- **THEN** a discomfort penalty SHALL be applied (configurable, default -0.02)
- **AND** no HP damage SHALL occur

#### Scenario: Danger Zone Damage

- **GIVEN** thermotaxis and health system enabled with danger_delta=15.0 (danger: Tc±15°C outside discomfort)
- **WHEN** agent is at a position with temperature 8°C (within 5-10°C zone)
- **THEN** a danger penalty SHALL be applied (configurable, default -0.05)
- **AND** HP damage SHALL be applied (configurable, default -5.0 HP per step)

#### Scenario: Lethal Zone Rapid Damage

- **GIVEN** thermotaxis and health system enabled with lethal zones (beyond danger_delta)
- **WHEN** agent is at a position with temperature 3°C (beyond 5°C, i.e., < Tc-15)
- **THEN** rapid HP damage SHALL be applied (configurable, default -20.0 HP per step)
- **AND** this SHALL likely cause HP depletion within a few steps

### Requirement: Thermotaxis Success Criteria

The system SHALL track thermotaxis-specific success metrics for benchmark evaluation.

#### Scenario: Temperature Comfort Score Calculation

- **GIVEN** an episode where agent spent 75 steps in comfort zone out of 100 total steps
- **WHEN** episode metrics are computed
- **THEN** `temperature_comfort_score` SHALL equal 0.75

#### Scenario: Thermotaxis Success Threshold

- **GIVEN** a thermotaxis benchmark with success threshold 60%
- **WHEN** agent achieves temperature_comfort_score >= 0.6 AND survives
- **THEN** `thermotaxis_success` SHALL be True
- **WHEN** agent achieves temperature_comfort_score < 0.6 OR dies
- **THEN** `thermotaxis_success` SHALL be False

#### Scenario: Combined Foraging-Thermotaxis Success

- **GIVEN** a foraging+thermotaxis benchmark
- **WHEN** episode metrics are evaluated
- **THEN** success SHALL require BOTH:
  - Primary goal completion (collect target foods OR survive to max steps)
  - Thermotaxis threshold met (>60% time in comfort zone)

### Requirement: Thermotaxis Benchmark Categories

The system SHALL provide hierarchical benchmark categories for thermotaxis-enabled simulations.

#### Scenario: Thermotaxis Foraging Benchmark

- **GIVEN** a simulation with thermotaxis enabled and food collection goal
- **WHEN** benchmark category is determined
- **THEN** category SHALL follow hierarchical pattern: `thermotaxis/foraging_small/quantum` or `thermotaxis/foraging_small/classical`
- **AND** this SHALL be distinct from basic foraging benchmarks (`basic/foraging_small/`)

#### Scenario: Pure Thermotaxis Benchmark

- **GIVEN** a simulation with thermotaxis enabled and no food collection goal
- **WHEN** benchmark category is determined
- **THEN** category SHALL be `thermotaxis/isothermal_small/quantum` or `thermotaxis/isothermal_small/classical`
- **AND** success SHALL be based solely on temperature comfort score

#### Scenario: Thermotaxis with Predators Benchmark

- **GIVEN** a simulation with thermotaxis and predators enabled
- **WHEN** benchmark category is determined
- **THEN** category SHALL be `thermotaxis/foraging_predator_small/quantum` or `thermotaxis/foraging_predator_small/classical`
- **AND** this combines temperature, food, and threat avoidance objectives

### Requirement: ThermotaxisParams Configuration

The DynamicForagingEnvironment SHALL accept a ThermotaxisParams dataclass for configuration.

#### Scenario: ThermotaxisParams Fields

- **GIVEN** ThermotaxisParams configuration
- **THEN** the following fields SHALL be supported:
  - `enabled: bool` (default False) - master toggle
  - `cultivation_temperature: float` (default 20.0) - agent's preferred temperature (Tc)
  - `base_temperature: float` (default 20.0) - environment base temperature
  - `gradient_direction: float` (default 0.0) - direction of linear gradient in radians
  - `gradient_strength: float` (default 0.5) - temperature change per cell (°C/cell)
  - `hot_spots: list[tuple[int, int, float]] | None` - localized heat sources
  - `cold_spots: list[tuple[int, int, float]] | None` - localized cold sources
  - `comfort_delta: float` (default 5.0) - comfort zone boundary from Tc
  - `discomfort_delta: float` (default 10.0) - discomfort zone boundary from Tc
  - `danger_delta: float` (default 15.0) - danger zone boundary from Tc
  - `comfort_reward: float` (default 0.01) - reward per step in comfort zone
  - `discomfort_penalty: float` (default -0.02) - penalty per step in discomfort zone
  - `danger_penalty: float` (default -0.05) - penalty per step in danger zone
  - `danger_hp_damage: float` (default 5.0) - HP damage per step in danger zone
  - `lethal_hp_damage: float` (default 20.0) - HP damage per step in lethal zone

### Requirement: Environment Thermotaxis Methods

The DynamicForagingEnvironment SHALL provide methods for temperature querying and effects.

#### Scenario: Temperature Query Methods

- **GIVEN** a thermotaxis-enabled environment
- **THEN** the following methods SHALL be available:
  - `get_temperature(position: GridPosition | None) -> float | None`
  - `get_temperature_gradient(position: GridPosition | None) -> GradientPolar | None`
  - `get_temperature_zone(position: GridPosition | None) -> TemperatureZone | None`
  - `apply_temperature_effects() -> tuple[float, float]` (returns reward_delta, hp_damage)
  - `get_temperature_comfort_score() -> float` (returns steps_in_comfort / total_steps)
  - `reset_thermotaxis() -> None` (resets tracking counters)

#### Scenario: Temperature Zone Enum

- **GIVEN** the TemperatureZone enum
- **THEN** the following zones SHALL be defined:
  - `LETHAL_COLD` - temperature < Tc - danger_delta
  - `DANGER_COLD` - Tc - danger_delta \<= temperature < Tc - discomfort_delta
  - `DISCOMFORT_COLD` - Tc - discomfort_delta \<= temperature < Tc - comfort_delta
  - `COMFORT` - Tc - comfort_delta \<= temperature \<= Tc + comfort_delta
  - `DISCOMFORT_HOT` - Tc + comfort_delta < temperature \<= Tc + discomfort_delta
  - `DANGER_HOT` - Tc + discomfort_delta < temperature \<= Tc + danger_delta
  - `LETHAL_HOT` - temperature > Tc + danger_delta
