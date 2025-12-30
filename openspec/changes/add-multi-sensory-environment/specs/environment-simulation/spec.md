## ADDED Requirements

### Requirement: Health System for Multi-Objective Survival
The system SHALL provide an optional HP-based health system as an alternative to instant death from predator encounters, enabling graduated survival mechanics.

#### Scenario: Health System Initialization
- **GIVEN** a configuration with `health_system.enabled: true` and `health_system.max_hp: 100`
- **WHEN** the environment is initialized
- **THEN** the agent SHALL start with HP equal to max_hp (100)
- **AND** the health system SHALL track current HP throughout the episode

#### Scenario: Predator Damage with Health System
- **GIVEN** health system enabled with `predator_damage: 10`
- **WHEN** the agent is within predator kill radius (contact)
- **THEN** the agent SHALL lose 10 HP instead of immediate death
- **AND** the episode SHALL continue if HP > 0
- **AND** predator contact event SHALL be logged

#### Scenario: Food Healing
- **GIVEN** health system enabled with `food_healing: 5`
- **WHEN** the agent consumes food
- **THEN** the agent SHALL gain 5 HP
- **AND** HP SHALL be capped at max_hp
- **AND** healing SHALL occur in addition to satiety restoration (both systems benefit from food)

#### Scenario: HP and Satiety Coexistence
- **GIVEN** both health system and satiety system enabled
- **WHEN** the agent consumes food
- **THEN** satiety SHALL be restored (existing behavior)
- **AND** HP SHALL be restored by `food_healing` amount
- **AND** the two systems SHALL operate independently otherwise
- **AND** satiety SHALL decay each step (time-based hunger)
- **AND** HP SHALL only decrease from damage events (threat-based)

#### Scenario: Health Depletion Termination
- **GIVEN** an agent with HP approaching 0
- **WHEN** HP reaches exactly 0
- **THEN** the episode SHALL terminate immediately
- **AND** the termination reason SHALL be `TerminationReason.HEALTH_DEPLETED`
- **AND** a death penalty reward SHALL be applied

#### Scenario: Health System Disabled (Default)
- **GIVEN** a configuration without health_system or `health_system.enabled: false`
- **WHEN** predator collision occurs
- **THEN** the existing instant-death behavior SHALL apply
- **AND** backward compatibility SHALL be maintained

### Requirement: Enhanced Predator Behavior Types
The system SHALL support multiple predator behavior types including stationary toxic zones and active pursuit predators.

#### Scenario: Stationary Predator Behavior
- **GIVEN** a predator configured with `type: stationary`
- **WHEN** the simulation step executes
- **THEN** the predator SHALL NOT change position
- **AND** the predator SHALL have configurable `damage_radius` for area-of-effect
- **AND** the predator SHALL represent toxic zones or trapping fungi

#### Scenario: Pursuit Predator Behavior
- **GIVEN** a predator configured with `type: pursuit` and `detection_radius: 5`
- **WHEN** the agent is within 5 cells (Manhattan distance) of the predator
- **THEN** the predator SHALL move one cell toward the agent
- **AND** movement direction SHALL minimize distance to agent
- **WHEN** the agent is outside detection radius
- **THEN** the predator SHALL move randomly (existing behavior)

#### Scenario: Mixed Predator Types Configuration
- **GIVEN** a configuration with multiple predator types:
  ```yaml
  predators:
    types:
      - type: stationary
        count: 2
      - type: pursuit
        count: 1
  ```
- **WHEN** the environment initializes
- **THEN** exactly 2 stationary predators SHALL be spawned
- **AND** exactly 1 pursuit predator SHALL be spawned
- **AND** each predator SHALL behave according to its type

#### Scenario: Predator Type Gradient Contribution
- **GIVEN** predators of different types in the environment
- **WHEN** predator gradient is computed
- **THEN** all predator types SHALL contribute to the unified gradient field
- **AND** stationary predators with larger damage_radius MAY have higher gradient strength

### Requirement: Mechanosensation Contact Detection
The system SHALL detect physical contact events including grid boundary contact and predator physical contact for mechanosensory processing.

#### Scenario: Grid Boundary Contact Detection
- **GIVEN** an agent at position (0, 5) (at left boundary of grid)
- **WHEN** environment state is queried
- **THEN** `boundary_contact` SHALL be True
- **AND** this SHALL indicate touch sensation at grid edge

#### Scenario: Predator Physical Contact Detection
- **GIVEN** an agent within predator kill radius
- **WHEN** environment state is queried
- **THEN** `predator_contact` SHALL be True
- **AND** this SHALL be distinct from proximity (detection radius)
- **AND** this enables harsh touch response processing

#### Scenario: No Contact State
- **GIVEN** an agent at position (10, 10) in a 20x20 grid, not near any predator
- **WHEN** environment state is queried
- **THEN** `boundary_contact` SHALL be False
- **AND** `predator_contact` SHALL be False

### Requirement: Food Spawning Bias for Safe Zones
The system SHALL bias food spawning toward safe environmental zones when sensory modalities that define danger zones are enabled.

#### Scenario: Safe Zone Food Bias with Thermotaxis
- **GIVEN** thermotaxis enabled with comfort zone 15-25°C and `food_safe_zone_ratio: 0.8`
- **WHEN** food sources are spawned
- **THEN** 80% of food SHALL spawn in positions where temperature is within comfort or discomfort zones
- **AND** 20% of food MAY spawn anywhere regardless of temperature
- **AND** this SHALL create risk/reward trade-offs for dangerous areas

#### Scenario: Food Spawning Without Sensory Zones
- **GIVEN** no sensory modalities with danger zones enabled
- **WHEN** food sources are spawned
- **THEN** food SHALL spawn using existing Poisson disk sampling
- **AND** no safe zone bias SHALL be applied
- **AND** backward compatibility SHALL be maintained

### Requirement: Temperature Zone Visualization
The system SHALL render temperature zones using background colors in themes that support color, following a defined priority system for overlapping visual elements.

#### Scenario: Rich Theme Temperature Coloring
- **GIVEN** thermotaxis enabled and Rich theme selected
- **WHEN** the environment is rendered
- **THEN** cells SHALL have background colors based on temperature:
  - Lethal cold (<5°C): blue background
  - Danger cold (5-10°C): cyan background
  - Discomfort cold (10-15°C): light cyan background
  - Comfort (15-25°C): white background (default)
  - Discomfort hot (25-30°C): light yellow background
  - Danger hot (30-35°C): yellow background
  - Lethal hot (>35°C): red background

#### Scenario: Visual Element Priority
- **GIVEN** a cell containing agent, food, predator, and temperature background
- **WHEN** the cell is rendered
- **THEN** visual elements SHALL be rendered in priority order:
  1. Agent (highest - always visible)
  2. Predators
  3. Food
  4. Environmental hazards (toxic zones)
  5. Temperature background (lowest)
- **AND** higher priority elements SHALL override lower priority visuals

### Requirement: Extended Termination Reasons
The system SHALL support additional termination reasons for Phase 1 mechanics including health depletion and environmental hazards.

#### Scenario: Health Depleted Termination
- **GIVEN** health system enabled and agent HP reaches 0
- **WHEN** the episode terminates
- **THEN** termination reason SHALL be `TerminationReason.HEALTH_DEPLETED`
- **AND** this SHALL be distinct from PREDATOR (instant death) termination

#### Scenario: Temperature Extremes Termination
- **GIVEN** thermotaxis enabled with lethal temperature thresholds
- **WHEN** agent HP reaches 0 due to temperature damage
- **THEN** termination reason SHALL be `TerminationReason.HEALTH_DEPLETED`
- **AND** temperature as cause SHALL be logged in metrics

### Requirement: Hierarchical Benchmark Categories
The system SHALL support hierarchical benchmark category naming to organize benchmarks by task complexity and sensory modalities.

#### Scenario: Basic Foraging Category
- **GIVEN** a simulation with food collection only (no predators, no thermotaxis)
- **WHEN** benchmark category is determined
- **THEN** category SHALL be `basic/foraging_small/quantum` or `basic/foraging_small/classical`
- **AND** this replaces the flat `foraging_small` category

#### Scenario: Survival Category with Predators
- **GIVEN** a simulation with food collection and predators enabled
- **WHEN** benchmark category is determined
- **THEN** category SHALL be `survival/predator_small/quantum` or `survival/predator_small/classical`
- **AND** this replaces the flat `predator_small` category

#### Scenario: Backward Compatibility for Flat Categories
- **GIVEN** existing benchmark results using flat category names (e.g., `foraging_small`)
- **WHEN** benchmark system processes these results
- **THEN** flat names SHALL be accepted and mapped to hierarchical equivalents
- **AND** no existing benchmark data SHALL be invalidated

#### Scenario: Category Path Structure
- **GIVEN** any benchmark category
- **WHEN** the category path is constructed
- **THEN** path SHALL follow pattern: `{category}/{task}_{size}/{brain_type}`
- **AND** category SHALL be one of: `basic`, `survival`, `thermotaxis`, `multisensory`, `ablation`
- **AND** brain_type SHALL be one of: `quantum`, `classical`
