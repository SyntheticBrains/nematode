# environment-simulation Specification

## Purpose
TBD - created by archiving change add-dynamic-foraging-environment. Update Purpose after archive.
## Requirements
### Requirement: Dynamic Multi-Food Environment
The system SHALL provide a dynamic foraging environment that supports multiple simultaneous food sources in large-scale grids with persistent state across food collection events.

#### Scenario: Multiple Food Sources Active
- **GIVEN** a dynamic environment is configured with 10 food sources in a 50×50 grid
- **WHEN** the environment is initialized
- **THEN** all 10 food sources SHALL be placed on the grid
- **AND** each food source SHALL be separated by at least the configured minimum distance
- **AND** no food source SHALL spawn within the configured minimum distance of the agent's starting position

#### Scenario: Environment Persists After Food Collection
- **GIVEN** an agent collects one food source in a dynamic environment with 10 total foods
- **WHEN** the food is consumed
- **THEN** the environment SHALL remain active with 9 remaining foods
- **AND** a new food source SHALL spawn immediately at a valid location
- **AND** the new food SHALL be at least minimum distance from the agent and all other foods
- **AND** the agent's position SHALL not reset

#### Scenario: Large Environment Support
- **GIVEN** a dynamic environment configured with 100×100 grid size
- **WHEN** the simulation runs
- **THEN** the environment SHALL support grids up to at least 100×100 cells
- **AND** performance SHALL remain acceptable (< 100ms per step)

### Requirement: Gradient Field Superposition
The environment SHALL compute chemotaxis gradients as the superposition (sum) of gradients emitted by all active food sources.

#### Scenario: Single Food Gradient
- **GIVEN** a dynamic environment with one food source at position (10, 10)
- **WHEN** gradient strength is queried at position (5, 5)
- **THEN** the gradient SHALL be computed using exponential decay: `strength × exp(-distance / decay_constant)`
- **AND** the gradient direction SHALL point toward the food source using `arctan2(dy, dx)`

#### Scenario: Multiple Food Gradient Superposition
- **GIVEN** two food sources at (10, 10) and (20, 20)
- **WHEN** gradient is queried at position (15, 15)
- **THEN** the total gradient SHALL be the vector sum of individual gradients from both sources
- **AND** the resulting gradient SHALL naturally guide the agent toward the nearest/strongest source

#### Scenario: Gradient Updates When Food Consumed
- **GIVEN** an environment with 3 active food sources
- **WHEN** the agent consumes one food source
- **THEN** that food's gradient contribution SHALL be removed from all grid positions
- **AND** the superposed gradient field SHALL update to reflect only remaining food sources
- **AND** a new food's gradient SHALL be added when it spawns

### Requirement: Satiety-Based Agent Lifecycle
The system SHALL track agent satiety (hunger level) that decays over time and can be replenished by consuming food, with simulation terminating when satiety reaches zero.

#### Scenario: Satiety Decay Over Time
- **GIVEN** an agent with initial satiety of 200
- **WHEN** the agent takes 10 steps without consuming food
- **THEN** satiety SHALL decrease by `10 × decay_rate` (default decay_rate = 1.0)
- **AND** satiety SHALL be capped at minimum value of 0

#### Scenario: Satiety Restoration on Food Consumption
- **GIVEN** an agent with satiety of 150 and max satiety of 200
- **WHEN** the agent consumes food
- **THEN** satiety SHALL increase by configured percentage (default 20% of max = 40)
- **AND** satiety SHALL not exceed the maximum configured value (200)

#### Scenario: Starvation Termination
- **GIVEN** an agent with satiety approaching 0
- **WHEN** satiety reaches exactly 0
- **THEN** the episode SHALL terminate immediately
- **AND** the termination reason SHALL be recorded as "starvation"
- **AND** a penalty reward SHALL be applied (default -10)

#### Scenario: Satiety Configuration
- **GIVEN** a simulation configuration file
- **WHEN** satiety parameters are specified
- **THEN** the system SHALL accept `initial_satiety` (default 200)
- **AND** SHALL accept `satiety_decay_rate` per step (default 1.0)
- **AND** SHALL accept `satiety_gain_per_food` as percentage of max (default 0.2 = 20%)

### Requirement: Spatial Food Distribution
The environment SHALL use Poisson disk sampling to ensure food sources are well-distributed with minimum separation distances.

#### Scenario: Minimum Distance Enforcement
- **GIVEN** a configuration with `min_food_distance = 5` cells
- **WHEN** food sources are spawned
- **THEN** every pair of food sources SHALL be at least 5 cells apart (Manhattan distance)
- **AND** no food SHALL spawn within 5 cells of the agent's starting position

#### Scenario: Agent Exclusion Zone
- **GIVEN** an agent starting at position (25, 25) with exclusion radius of 10
- **WHEN** initial food sources are placed
- **THEN** no food SHALL be placed within 10 cells (Manhattan distance) of (25, 25)
- **AND** this ensures the agent must actively search rather than spawn on food

#### Scenario: Dynamic Spawn Location Validity
- **GIVEN** an environment with existing food sources
- **WHEN** a new food needs to spawn after consumption
- **THEN** the system SHALL find a valid location respecting minimum distances
- **AND** SHALL retry up to 100 attempts to find valid placement
- **AND** if no valid location found, SHALL log a warning and skip spawning

### Requirement: Viewport-Based Rendering
The system SHALL render only a configurable viewport window centered on the agent rather than the entire large environment.

#### Scenario: Agent-Centered Viewport
- **GIVEN** a 100×100 environment with viewport size 11×11
- **WHEN** the agent is at position (50, 50)
- **THEN** the rendering SHALL display cells from (45, 45) to (55, 55)
- **AND** the agent SHALL always be at the center of the viewport
- **AND** only food sources within the viewport SHALL be visually rendered

#### Scenario: Viewport Boundary Handling
- **GIVEN** an agent near the edge at position (2, 2) with 11×11 viewport
- **WHEN** rendering occurs
- **THEN** the viewport SHALL be clamped to valid grid boundaries
- **AND** SHALL display from (0, 0) to (10, 10) instead of going negative
- **AND** the agent SHALL appear off-center in the viewport

#### Scenario: Full Environment Logging
- **GIVEN** a new simulation session starting
- **WHEN** the first episode initializes
- **THEN** the system SHALL log a representation of the entire environment to the session log
- **AND** this full environment snapshot SHALL include all food positions
- **AND** subsequent rendering SHALL use viewport mode only

### Requirement: Multi-Food Foraging Metrics
The system SHALL track foraging efficiency metrics appropriate for multi-food scenarios including foraging rate and distance efficiency.

#### Scenario: Foraging Efficiency Rate
- **GIVEN** an agent collects 5 foods in 250 steps
- **WHEN** episode metrics are computed
- **THEN** foraging efficiency SHALL be calculated as `foods_collected / total_steps = 5/250 = 0.02`
- **AND** this metric SHALL be included in the performance report

#### Scenario: Distance Efficiency Per Food
- **GIVEN** an agent that was 20 cells from a food source when it became the target
- **WHEN** the agent reaches that food in 35 steps
- **THEN** distance efficiency SHALL be `(20 - 35) / 20 = -0.75` (inefficient path)
- **AND** this SHALL be averaged across all collected foods for the episode

#### Scenario: Greedy Baseline Tracking
- **GIVEN** an environment with multiple food sources
- **WHEN** an agent is navigating
- **THEN** the system SHALL track Manhattan distance to nearest food at each step
- **AND** SHALL record the initial distance when each food becomes the implicit target
- **AND** SHALL use this for computing distance efficiency relative to greedy optimal

#### Scenario: Survival Metrics
- **GIVEN** an episode that runs until satiety reaches 0
- **WHEN** the episode completes
- **THEN** metrics SHALL include `survival_time` (total steps before starvation)
- **AND** SHALL include `foods_collected` (count of successful consumptions)
- **AND** SHALL include termination reason ("starvation" or "max_steps" or "all_food_consumed")

### Requirement: Exploration Reward Bonus
The system SHALL provide a small reward bonus when the agent visits previously unvisited grid cells to encourage exploration behavior.

#### Scenario: Unvisited Cell Reward
- **GIVEN** an agent at position (10, 10) that has never visited (11, 10)
- **WHEN** the agent moves to (11, 10)
- **THEN** the agent SHALL receive an exploration bonus reward (default +0.05)
- **AND** cell (11, 10) SHALL be marked as visited

#### Scenario: Revisited Cell No Bonus
- **GIVEN** an agent that previously visited cell (15, 15)
- **WHEN** the agent moves to (15, 15) again
- **THEN** no exploration bonus SHALL be awarded
- **AND** the cell SHALL remain marked as visited

#### Scenario: Exploration Bonus Configuration
- **GIVEN** a simulation configuration
- **WHEN** exploration parameters are specified
- **THEN** the system SHALL accept `exploration_bonus` reward value (default 0.05)
- **AND** SHALL support disabling exploration bonus by setting to 0.0

### Requirement: Preset Environment Configurations
The system SHALL provide three preset configurations for curriculum learning: small, medium, and large dynamic environments.

#### Scenario: Small Dynamic Environment
- **GIVEN** the small preset configuration is loaded
- **WHEN** the environment initializes
- **THEN** grid size SHALL be 20×20
- **AND** initial food count SHALL be 5
- **AND** max active foods SHALL be 10
- **AND** initial satiety SHALL be 200 steps
- **AND** min food distance SHALL be 3 cells
- **AND** viewport size SHALL be 11×11

#### Scenario: Medium Dynamic Environment
- **GIVEN** the medium preset configuration is loaded
- **WHEN** the environment initializes
- **THEN** grid size SHALL be 50×50
- **AND** initial food count SHALL be 20
- **AND** max active foods SHALL be 30
- **AND** initial satiety SHALL be 500 steps
- **AND** min food distance SHALL be 5 cells
- **AND** viewport size SHALL be 11×11

#### Scenario: Large Dynamic Environment
- **GIVEN** the large preset configuration is loaded
- **WHEN** the environment initializes
- **THEN** grid size SHALL be 100×100
- **AND** initial food count SHALL be 50
- **AND** max active foods SHALL be 50
- **AND** initial satiety SHALL be 800 steps
- **AND** min food distance SHALL be 10 cells
- **AND** viewport size SHALL be 11×11

### Requirement: Backward Compatibility with Static Environments
The system SHALL maintain full backward compatibility with existing single-food StaticEnvironment configurations and all brain architectures.

#### Scenario: Existing Configuration Unchanged
- **GIVEN** an existing simulation configuration without `environment_type` field
- **WHEN** the simulation is run
- **THEN** the system SHALL use the original `StaticEnvironment` class
- **AND** behavior SHALL be identical to pre-dynamic-environment versions
- **AND** all metrics and outputs SHALL match historical format

#### Scenario: Brain Architecture Compatibility
- **GIVEN** any existing brain type (modular, qmodular, mlp, qmlp, spiking)
- **WHEN** used with a dynamic environment
- **THEN** the brain SHALL receive the same observation format (gradient strength, gradient direction)
- **AND** the brain SHALL produce the same action outputs
- **AND** no brain code changes SHALL be required

#### Scenario: Explicit Static Mode
- **GIVEN** a configuration with `environment_type: "static"`
- **WHEN** the simulation runs
- **THEN** the system SHALL use `StaticEnvironment` with single goal
- **AND** SHALL support explicit selection of legacy behavior
