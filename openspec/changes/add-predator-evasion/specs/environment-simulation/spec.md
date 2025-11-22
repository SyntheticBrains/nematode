# environment-simulation Delta

## ADDED Requirements

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
- **AND** the predator SHALL wrap around grid boundaries or stay within bounds based on environment settings
- **AND** movement SHALL occur at rate specified by `speed` parameter (default 1.0)

#### Scenario: Predator Speed Configuration
- **GIVEN** a predator configured with `speed: 0.5`
- **WHEN** the simulation advances
- **THEN** the predator SHALL move at half the rate of the agent
- **AND** this SHALL be implemented via probabilistic movement (50% chance to move each step)
- **OR** via accumulator-based movement (moves every 2 steps)

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
- **GIVEN** a configuration with `kill_radius: 1`
- **WHEN** collision detection runs
- **THEN** collision SHALL be detected when agent is within 1 cell (Manhattan distance) of any predator
- **AND** kill_radius of 1 SHALL include the 4 adjacent cells plus center
- **AND** diagonal adjacency SHALL be excluded (Manhattan distance only)

#### Scenario: No Collision Outside Kill Radius
- **GIVEN** an agent at (10, 10) and predator at (12, 10) with `kill_radius: 1`
- **WHEN** positions are checked
- **THEN** no collision SHALL be detected (distance = 2 > kill_radius)
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
- **THEN** `deaths_by_predator` metric SHALL equal 1
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
- **AND** `deaths_by_predator` SHALL be None or 0
- **AND** this SHALL maintain backward compatibility

### Requirement: Predator-Enabled Benchmark Categories
The system SHALL provide separate benchmark categories for predator-enabled simulations to track learning performance on survival-foraging tasks.

#### Scenario: Predator Quantum Benchmarks
- **GIVEN** a simulation with quantum brain and `predators.enabled: true`
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `dynamic_predator_quantum_small`, `dynamic_predator_quantum_medium`, or `dynamic_predator_quantum_large`
- **AND** this SHALL be based on grid size using same thresholds as non-predator benchmarks
- **AND** small ≤ 20×20, medium ≤ 50×50, large > 50×50

#### Scenario: Predator Classical Benchmarks
- **GIVEN** a simulation with classical brain (MLP or Spiking) and `predators.enabled: true`
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `dynamic_predator_classical_small`, `dynamic_predator_classical_medium`, or `dynamic_predator_classical_large`
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
  - Quantum brains: `dynamic_predator_quantum_small`, `dynamic_predator_quantum_medium`, `dynamic_predator_quantum_large`
  - Classical brains: `dynamic_predator_classical_small`, `dynamic_predator_classical_medium`, `dynamic_predator_classical_large`
- **AND** the underscore separator SHALL be used (not hyphen or space)
- **AND** category names SHALL match the implementation exactly to prevent doc-code drift

## MODIFIED Requirements

### Requirement: Preset Environment Configurations
The system SHALL provide three preset configurations for curriculum learning: small, medium, and large dynamic environments, with optional predator support.

#### Scenario: Small Dynamic Environment
- **GIVEN** the small preset configuration is loaded
- **WHEN** the environment initializes
- **THEN** grid size SHALL be 20×20
- **AND** foraging configuration SHALL include:
  - initial food count: 5
  - max active foods: 10
  - min food distance: 3 cells
  - gradient decay constant: 12.0
  - gradient strength: 1.0
- **AND** initial satiety SHALL be 200 steps
- **AND** viewport size SHALL be 11×11
- **AND** predator configuration SHALL be optional (default disabled)

#### Scenario: Medium Dynamic Environment
- **GIVEN** the medium preset configuration is loaded
- **WHEN** the environment initializes
- **THEN** grid size SHALL be 50×50
- **AND** foraging configuration SHALL include:
  - initial food count: 20
  - max active foods: 30
  - min food distance: 5 cells
  - gradient decay constant: 12.0
  - gradient strength: 1.0
- **AND** initial satiety SHALL be 500 steps
- **AND** viewport size SHALL be 11×11
- **AND** predator configuration SHALL be optional (default disabled)

#### Scenario: Large Dynamic Environment
- **GIVEN** the large preset configuration is loaded
- **WHEN** the environment initializes
- **THEN** grid size SHALL be 100×100
- **AND** foraging configuration SHALL include:
  - initial food count: 50
  - max active foods: 50
  - min food distance: 10 cells
  - gradient decay constant: 12.0
  - gradient strength: 1.0
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
  - kill radius: 1
  - movement pattern: "random"
- **AND** both foraging and predator mechanics SHALL be active
