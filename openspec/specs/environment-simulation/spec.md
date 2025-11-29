# environment-simulation Specification

## Purpose
This specification defines the environment simulation system for the Quantum Nematode project, which supports both dynamic multi-food foraging environments and legacy static single-goal navigation. The dynamic foraging environment provides a realistic simulation of C. elegans foraging behavior with multiple simultaneous food sources, satiety-based agent lifecycle management, chemotaxis gradient superposition, and spatial food distribution. This enables research into complex decision-making strategies, multi-objective reinforcement learning, and quantum advantages in sequential foraging tasks.
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
The system SHALL provide three preset configurations for curriculum learning: small, medium, and large dynamic environments, with optional predator support.

#### Scenario: Small Dynamic Environment
- **GIVEN** the small preset configuration is loaded
- **WHEN** the environment initializes
- **THEN** grid size SHALL be 20×20
- **AND** foraging configuration SHALL include:
  - foods on grid (constant): 5
  - target foods to collect (victory condition): 10
  - min food distance: 3 cells
  - gradient decay constant: 12.0
  - gradient strength: 1.0
- **AND** food SHALL respawn immediately after collection to maintain constant 5 on grid
- **AND** episode SHALL complete successfully when 10 foods have been collected
- **AND** initial satiety SHALL be 200 steps
- **AND** viewport size SHALL be 11×11
- **AND** predator configuration SHALL be optional (default disabled)

#### Scenario: Medium Dynamic Environment
- **GIVEN** the medium preset configuration is loaded
- **WHEN** the environment initializes
- **THEN** grid size SHALL be 50×50
- **AND** foraging configuration SHALL include:
  - foods on grid (constant): 20
  - target foods to collect (victory condition): 30
  - min food distance: 5 cells
  - gradient decay constant: 12.0
  - gradient strength: 1.0
- **AND** food SHALL respawn immediately after collection to maintain constant 20 on grid
- **AND** episode SHALL complete successfully when 30 foods have been collected
- **AND** initial satiety SHALL be 500 steps
- **AND** viewport size SHALL be 11×11
- **AND** predator configuration SHALL be optional (default disabled)

#### Scenario: Large Dynamic Environment
- **GIVEN** the large preset configuration is loaded
- **WHEN** the environment initializes
- **THEN** grid size SHALL be 100×100
- **AND** foraging configuration SHALL include:
  - foods on grid (constant): 50
  - target foods to collect (victory condition): 50
  - min food distance: 10 cells
  - gradient decay constant: 12.0
  - gradient strength: 1.0
- **AND** food SHALL respawn immediately after collection to maintain constant 50 on grid
- **AND** episode SHALL complete successfully when 50 foods have been collected
- **AND** initial satiety SHALL be 800 steps
- **AND** viewport size SHALL be 11×11
- **AND** predator configuration SHALL be optional (default disabled)

#### Scenario: Predator-Enabled Small Environment
- **GIVEN** a small preset configuration with predators enabled
- **WHEN** the environment initializes with `predators.enabled: true`
- **THEN** all small environment foraging parameters SHALL apply
- **AND** predator defaults SHALL be:
  - count: 2
  - speed: 1.0
  - detection radius: 8
  - kill radius: 0
  - movement pattern: "random"
- **AND** both foraging and predator mechanics SHALL be active

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

### Requirement: Predator Entities in Dynamic Environments
The system SHALL support configurable predator entities in dynamic foraging environments that move independently and pose a threat to the agent.

#### Scenario: Predator Initialization
- **GIVEN** a dynamic environment configured with `predators.enabled: true` and `predators.count: 3`
- **WHEN** the environment is initialized
- **THEN** exactly 3 predator entities SHALL be spawned at random valid positions
- **AND** each predator SHALL have independent position and movement state
- **AND** predators SHALL be tracked separately from food sources

#### Scenario: Predator Random Movement
- **GIVEN** a predator at position (10, 10) with `movement_pattern: "random"`
- **WHEN** the predator updates its position
- **THEN** the predator SHALL move one cell in a random direction (up, down, left, right)
- **AND** the predator SHALL stay within bounds of the environment
- **AND** movement SHALL occur at rate specified by `speed` parameter (default 1.0)

#### Scenario: Predator Speed Configuration
- **GIVEN** a predator configured with `speed: 0.5`
- **WHEN** the simulation advances
- **THEN** the predator SHALL move at half the rate of the agent
- **AND** this SHALL be implemented via accumulator-based movement (moves every 2 steps)

#### Scenario: Multiple Predators Independence
- **GIVEN** 3 predators in the environment
- **WHEN** each predator updates
- **THEN** each predator SHALL move independently
- **AND** predators MAY occupy the same cell
- **AND** predators SHALL not interact with each other

### Requirement: Unified Gradient System with Predator Repulsion
The environment SHALL compute a unified chemotaxis gradient that superimposes food attraction (positive values) and predator repulsion (negative values).

#### Scenario: Predator Negative Gradient Contribution
- **GIVEN** a predator at position (10, 10) with `gradient_strength: 1.0` and `gradient_decay_constant: 12.0`
- **WHEN** gradient is queried at position (5, 5)
- **THEN** the predator gradient SHALL be computed as `-gradient_strength × exp(-distance / decay_constant)`
- **AND** the negative value SHALL create a repulsive effect
- **AND** the gradient direction SHALL point away from the predator

#### Scenario: Food and Predator Gradient Superposition
- **GIVEN** one food source at (10, 10) and one predator at (20, 20)
- **WHEN** gradient is queried at position (15, 15)
- **THEN** the total gradient SHALL be the vector sum of food attraction and predator repulsion
- **AND** the resulting gradient direction SHALL balance both influences
- **AND** agent closer to predator SHALL experience stronger repulsion

#### Scenario: Multiple Predators Gradient Superposition
- **GIVEN** two predators at (5, 5) and (25, 25)
- **WHEN** gradient is queried at position (15, 15)
- **THEN** repulsive gradients from both predators SHALL be summed
- **AND** combined with food attraction gradients
- **AND** the agent SHALL receive a single unified gradient observation

#### Scenario: Predator Gradient Configuration
- **GIVEN** a configuration with predator gradient parameters
- **WHEN** the environment initializes
- **THEN** the system SHALL accept `gradient_decay_constant` for predator gradients (default 12.0)
- **AND** SHALL accept `gradient_strength` for predator gradients (default 1.0)
- **AND** these MAY differ from food gradient parameters

### Requirement: Predator Collision Detection and Termination
The system SHALL detect when the agent collides with a predator and terminate the episode immediately with appropriate termination reason.

#### Scenario: Predator Kill on Direct Collision
- **GIVEN** an agent at position (10, 10) and a predator at position (10, 10)
- **WHEN** collision is detected (same cell occupation)
- **THEN** the episode SHALL terminate immediately
- **AND** the termination reason SHALL be `TerminationReason.PREDATOR`
- **AND** a death penalty reward SHALL be applied (configurable, default -10)

#### Scenario: Kill Radius Configuration
- **GIVEN** a configuration with `kill_radius: 0`
- **WHEN** collision detection runs
- **THEN** collision SHALL be detected when agent is within 0 cells (Manhattan distance) of any predator
- **AND** kill_radius of 0 SHALL include the center cell only
- **AND** diagonal adjacency SHALL be excluded (Manhattan distance only)

#### Scenario: No Collision Outside Kill Radius
- **GIVEN** an agent at (10, 10) and predator at (11, 10) with `kill_radius: 0`
- **WHEN** positions are checked
- **THEN** no collision SHALL be detected (distance = 1 > kill_radius)
- **AND** the episode SHALL continue normally
- **AND** agent SHALL still sense predator gradient if within detection radius

### Requirement: Predator Detection and Proximity Penalty
The system SHALL track when the agent is within predator detection radius and apply a configurable proximity penalty to encourage proactive avoidance.

#### Scenario: Detection Radius Sensing
- **GIVEN** a predator at (10, 10) with `detection_radius: 8`
- **WHEN** the agent is at position (15, 15) (Manhattan distance = 10)
- **THEN** the agent SHALL NOT be considered within detection radius
- **AND** no proximity penalty SHALL apply

#### Scenario: Proximity Penalty Application
- **GIVEN** an agent at (10, 10), predator at (15, 12) (distance = 7), and `proximity_penalty: -0.1`
- **WHEN** reward is calculated for the step
- **THEN** a penalty of -0.1 SHALL be added to the step reward
- **AND** this penalty SHALL apply in addition to other reward components
- **AND** penalty SHALL be applied every step while within detection radius

#### Scenario: Multiple Predators Proximity Penalty
- **GIVEN** an agent within detection radius of 2 predators simultaneously
- **WHEN** proximity penalty is calculated
- **THEN** the penalty SHALL be applied once (not stacked per predator)
- **AND** the single penalty SHALL be -0.1 (configurable default)

#### Scenario: Proximity Penalty Disabled
- **GIVEN** a configuration with `proximity_penalty: 0.0`
- **WHEN** the agent is within detection radius
- **THEN** no proximity penalty SHALL be applied
- **AND** the agent SHALL still sense predator gradients normally

### Requirement: Predator Visualization and Rendering
The system SHALL render predators in the viewport with theme-appropriate symbols and optional detection radius visualization.

#### Scenario: Predator Emoji Rendering
- **GIVEN** a predator at position (10, 10) within the viewport and theme mode set to "emoji"
- **WHEN** the environment is rendered
- **THEN** the predator SHALL be displayed with spider emoji 🕷️
- **AND** the predator SHALL be visually distinct from food (🍎) and agent (🪱)

#### Scenario: Predator ASCII Rendering
- **GIVEN** a predator at position (10, 10) within the viewport and theme mode set to "ascii"
- **WHEN** the environment is rendered
- **THEN** the predator SHALL be displayed with hash symbol `#`
- **AND** the predator SHALL be visually distinct from food and agent symbols

#### Scenario: Agent Danger Status Display
- **GIVEN** an agent within detection radius of any predator
- **WHEN** the simulation run output is displayed
- **THEN** the agent status SHALL be shown as "IN DANGER"
- **AND** when no predator within detection radius, status SHALL show "SAFE"
- **AND** this SHALL be displayed alongside episode and step information

### Requirement: Predator-Specific Metrics Tracking
The system SHALL track predator-related performance metrics including encounters, evasions, and deaths for analysis and benchmarking.

#### Scenario: Predator Encounter Tracking
- **GIVEN** an agent that enters predator detection radius 5 times during an episode
- **WHEN** episode metrics are computed
- **THEN** `predator_encounters` metric SHALL equal 5
- **AND** each continuous period within detection radius SHALL count as one encounter
- **AND** leaving and re-entering detection radius SHALL increment the counter

#### Scenario: Successful Evasion Tracking
- **GIVEN** an agent that enters detection radius and successfully exits without collision
- **WHEN** episode metrics are computed
- **THEN** `successful_evasions` metric SHALL increment by 1
- **AND** evasion is defined as entering detection radius and leaving without death

#### Scenario: Death by Predator Tracking
- **GIVEN** an episode terminated by predator collision
- **WHEN** episode metrics are computed
- **THEN** `predator_deaths` metric SHALL equal 1
- **AND** termination reason SHALL be `TerminationReason.PREDATOR`
- **AND** this SHALL be distinct from starvation or timeout terminations

#### Scenario: Food Collected Before Death
- **GIVEN** an agent that collected 8 foods before predator collision
- **WHEN** episode metrics are computed
- **THEN** `foods_collected` metric SHALL equal 8
- **AND** this SHALL be tracked alongside predator death
- **AND** SHALL enable analysis of foraging-survival trade-offs

#### Scenario: Metrics When Predators Disabled
- **GIVEN** a configuration with `predators.enabled: false`
- **WHEN** episode metrics are computed
- **THEN** `predator_encounters` SHALL be None or 0
- **AND** `successful_evasions` SHALL be None or 0
- **AND** `predator_deaths` SHALL be None or 0
- **AND** this SHALL maintain backward compatibility

### Requirement: Predator-Enabled Benchmark Categories
The system SHALL provide separate benchmark categories for predator-enabled simulations to track learning performance on survival-foraging tasks.

#### Scenario: Predator Quantum Benchmarks
- **GIVEN** a simulation with quantum brain and `predators.enabled: true`
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `dynamic_predator_small_quantum`, `dynamic_predator_medium_quantum`, or `dynamic_predator_large_quantum`
- **AND** this SHALL be based on grid size using same thresholds as non-predator benchmarks
- **AND** small ≤ 20×20, medium ≤ 50×50, large > 50×50

#### Scenario: Predator Classical Benchmarks
- **GIVEN** a simulation with classical brain (MLP or Spiking) and `predators.enabled: true`
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `dynamic_predator_small_classical`, `dynamic_predator_medium_classical`, or `dynamic_predator_large_classical`
- **AND** this SHALL enable separate tracking of classical vs quantum performance on predator tasks

#### Scenario: Non-Predator Benchmark Unchanged
- **GIVEN** a simulation with `predators.enabled: false` or predators not configured
- **WHEN** benchmark category is determined
- **THEN** existing categories SHALL be used (`dynamic_small_quantum`, etc.)
- **AND** backward compatibility with existing benchmarks SHALL be maintained

### Requirement: Step Execution Order with Predators
The system SHALL execute each simulation step in a deterministic order to ensure consistent collision detection and predator movement behavior.

#### Scenario: Single Step Execution Sequence
- **GIVEN** a simulation step with predators enabled
- **WHEN** the step is executed
- **THEN** the following operations SHALL occur in order:
  1. Agent observes current state (including predator gradients)
  2. Agent selects and executes action (moves to new position)
  3. Food collection is checked and processed (if applicable)
  4. Predator encounter status is updated (detection radius entry/exit)
  5. Predator collision is checked at agent's new position (before predators move)
  6. If collision detected, episode terminates immediately
  7. If no collision, predators update their positions
  8. Satiety is decremented (if applicable)
  9. Other termination conditions checked (starvation, max steps)
- **AND** this order SHALL be deterministic and consistent across all episodes

#### Scenario: Collision Detection Before Predator Movement
- **GIVEN** an agent at (10, 10) after moving, and a predator at (10, 10) before predator update
- **WHEN** the step sequence executes
- **THEN** collision SHALL be detected at step 5 (before predator movement)
- **AND** the episode SHALL terminate with `TerminationReason.PREDATOR`
- **AND** predator movement (step 7) SHALL NOT occur after collision
- **AND** this prevents predators from moving away before collision is registered

#### Scenario: Predator Movement After Safe Step
- **GIVEN** an agent step where no collision occurs
- **WHEN** the step sequence completes
- **THEN** all predators SHALL update positions after collision check
- **AND** predator positions at step N+1 SHALL reflect movement from step N
- **AND** the new predator positions SHALL be used for gradient calculation in step N+1

### Requirement: Rendering Symbol Verification
The system SHALL use consistent, documented symbols for predators across all rendering themes to maintain visual clarity and documentation accuracy.

#### Scenario: Emoji Theme Predator Symbol
- **GIVEN** rendering with theme mode "emoji"
- **WHEN** a predator is rendered
- **THEN** the predator SHALL be displayed as spider emoji: 🕷️
- **AND** this SHALL match the documented spec exactly
- **AND** the symbol SHALL be visually distinct from food (🍎) and agent (🪱)

#### Scenario: ASCII Theme Predator Symbol
- **GIVEN** rendering with theme mode "ascii"
- **WHEN** a predator is rendered
- **THEN** the predator SHALL be displayed as hash symbol: #
- **AND** this SHALL match the documented spec exactly
- **AND** the symbol SHALL be visually distinct from food and agent ASCII symbols

#### Scenario: Benchmark Category Name Verification
- **GIVEN** a predator-enabled simulation for benchmarking
- **WHEN** the benchmark category is determined
- **THEN** category names SHALL exactly match the documented format:
  - Quantum brains: `dynamic_predator_small_quantum`, `dynamic_predator_medium_quantum`, `dynamic_predator_large_quantum`
  - Classical brains: `dynamic_predator_small_classical`, `dynamic_predator_medium_classical`, `dynamic_predator_large_classical`
- **AND** the underscore separator SHALL be used (not hyphen or space)
- **AND** category names SHALL match the implementation exactly to prevent doc-code drift
