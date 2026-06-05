## ADDED Requirements

### Requirement: Continuous-2D coordinate substrate

The system SHALL provide a continuous-2D environment in which agent position, food sources, and predators are represented as real-valued `(x, y)` coordinates within world bounds expressed in physical units (millimetres), distinct from and selectable alongside the discrete grid environment. The grid environment SHALL remain unchanged and byte-stable.

#### Scenario: Float agent position

- **WHEN** an agent acts in the continuous-2D environment
- **THEN** its position is stored and updated as real-valued `(x, y)` coordinates, not integer grid cells

#### Scenario: Physical scale

- **WHEN** a continuous-2D environment is constructed
- **THEN** world bounds, worm body length, per-step displacement, capture radius, and head-sweep amplitude are expressed relative to a physical scale (~1 mm body on a cm-scale arena), configurable with documented defaults

#### Scenario: Grid environment unaffected

- **WHEN** the continuous-2D environment is added
- **THEN** the discrete grid environment produces identical behaviour to before this change on its existing scenarios (verified by the T4 grid regression)

### Requirement: Kinematic continuous movement

The continuous-2D environment SHALL translate a continuous action `(speed, turn)` into a kinematic position update — a heading rotation by the turn angle followed by a forward displacement proportional to speed — bounded by the world extent, replacing the discrete one-cell cardinal-step movement.

#### Scenario: Heading-and-displacement update

- **WHEN** an agent emits a continuous action `(speed, turn)`
- **THEN** the agent's heading rotates by `turn` (clamped to `[−π, π]`) and the agent advances by a displacement proportional to `speed` (clamped to `[0, max]`) along the new heading

#### Scenario: World-bound clamping

- **WHEN** a movement would take the agent outside the world bounds
- **THEN** the agent is clamped to the bounds (or the step is rejected) without raising, and the episode continues

### Requirement: Capture-radius food consumption

The continuous-2D environment SHALL consume a food source when the agent is within a configurable capture radius of it (Euclidean distance), replacing exact grid-cell-equality consumption.

#### Scenario: Food consumed within radius

- **WHEN** the agent's position is within the capture radius of a food source
- **THEN** that food is consumed (removed) and the foraging reward/satiety update fires exactly once for it

#### Scenario: Food not consumed outside radius

- **WHEN** the agent is farther than the capture radius from every food source
- **THEN** no food is consumed that step

### Requirement: Continuous source placement and Euclidean fields

The continuous-2D environment SHALL place food and predator sources by continuous sampling over the world bounds (not integer grid sampling) and SHALL compute all spatial distances (pheromone, predator-mechano contact, nearest-food) using Euclidean distance.

#### Scenario: Continuous placement

- **WHEN** sources are initialised in the continuous-2D environment
- **THEN** their coordinates are drawn from a continuous distribution over the world bounds and respect the existing minimum-separation validity checks

#### Scenario: Euclidean distances

- **WHEN** any spatial distance is computed in the continuous-2D environment
- **THEN** it uses Euclidean distance, including for pheromone concentration, predator-mechanosensory contact intensity, and nearest-food queries

### Requirement: Physically-scaled klinotaxis sensing on continuous coordinates

Klinotaxis (head-sweep) sensing in the continuous-2D environment SHALL sample lateral concentrations at offsets scaled to a physical sweep amplitude (a fraction of body length) around the agent's continuous position and heading, rather than at fixed ±1 grid-cell offsets, and SHALL feed STAM the resulting continuous readings.

#### Scenario: Continuous lateral sampling

- **WHEN** klinotaxis sensing runs in the continuous-2D environment
- **THEN** left/right concentrations are sampled at continuous lateral offsets of the configured sweep amplitude relative to heading, and `dC` is the right-minus-left difference at those offsets

#### Scenario: STAM receives continuous readings

- **WHEN** continuous lateral readings are produced
- **THEN** STAM records them without integer-cell coercion and the dC/dt temporal derivative is computed from the continuous history
