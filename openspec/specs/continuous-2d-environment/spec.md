# continuous-2d-environment Specification

## Purpose

Defines the continuous-2D foraging environment: a square, millimetre-scale arena in which the worm's position, movement, food capture, and distances are real-valued (continuous), offered as an alternative substrate to the discrete grid environment and selected via the environment-type discriminator. It models *C. elegans* navigation on a cm-scale plate with a ~1 mm body, giving continuous-action PPO brains a kinematic `(speed, turn)` movement model with Euclidean capture and heading-aware klinotaxis sensing. It is consumed by the simulation/agent runners and continuous-action brains, and guarantees the discrete grid environment remains unchanged and byte-stable.

## Requirements

### Requirement: Continuous-2D coordinate substrate

The system SHALL provide a continuous-2D environment in which the **agent position** is represented as real-valued `(x, y)` coordinates within world bounds expressed in physical units (millimetres), distinct from and selectable alongside the discrete grid environment. The worm, its movement, capture, and distances are fully continuous; **source placement and Euclidean distances are specified in the *Source placement and Euclidean fields* requirement below**. The grid environment SHALL remain unchanged and byte-stable.

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
- **THEN** the agent's new position is clamped per-axis to `[0, world_size_mm]` — it advances as far as the bound allows (partial movement), the step is **not** rejected, no error is raised, and the episode continues

### Requirement: Capture-radius food consumption

The continuous-2D environment SHALL consume a food source when the agent is within a configurable capture radius of it (Euclidean distance), replacing exact grid-cell-equality consumption.

#### Scenario: Food consumed within radius

- **WHEN** the agent's position is within the capture radius of a food source
- **THEN** that food is consumed (removed) and the foraging reward/satiety update fires exactly once for it

#### Scenario: Food not consumed outside radius

- **WHEN** the agent is farther than the capture radius from every food source
- **THEN** no food is consumed that step

### Requirement: Source placement and Euclidean fields

The continuous-2D environment SHALL place **food sources** at **real-valued (float) coordinates** within the world bounds and SHALL compute food capture and nearest-food distance using **true Euclidean distance** (not Manhattan, and **not** rounded to integer). **Predator** placement and movement remain on the integer lattice in this iteration (predators-on-continuous are not yet exercised; continuous predator kinematics + Euclidean detection/damage are scheduled for T7 prep). The float source coordinates are confined to the continuous-2D environment; the discrete grid environment retains integer source coordinates and remains byte-stable.

#### Scenario: Food sources within bounds

- **WHEN** food sources are initialised in the continuous-2D environment
- **THEN** their coordinates are real-valued, lie within the world bounds, and respect the existing minimum-separation validity checks

#### Scenario: Euclidean distances

- **WHEN** food capture or nearest-food distance is computed in the continuous-2D environment
- **THEN** it uses real-valued Euclidean distance (replacing the grid's Manhattan distance and without integer rounding of the result)

#### Scenario: Grid environment unchanged

- **WHEN** the discrete grid environment places or measures sources
- **THEN** it retains integer source coordinates and Manhattan distance, byte-stable against the pre-change behaviour

### Requirement: Heading-aware klinotaxis sensing on the continuous substrate

Klinotaxis (head-sweep) sensing in the continuous-2D environment SHALL sample lateral concentrations perpendicular to the agent's continuous heading (`heading_rad`), scaled by the configured sweep amplitude (honoured as configured, including sub-cell amplitudes), rather than at the grid's fixed cardinal ±1-cell offsets, so the directional gradient rotates with the continuous heading, and SHALL feed STAM the resulting readings. Lateral samples SHALL be evaluated against the **continuous concentration field at real-valued sample positions** (no integer-cell snapping of the sample points).

#### Scenario: Heading-aware lateral sampling

- **WHEN** klinotaxis sensing runs in the continuous-2D environment
- **THEN** left/right samples are taken perpendicular to `heading_rad` (matching the grid offsets at cardinal headings, rotating smoothly otherwise) at real-valued sample positions, and `dC` is the right-minus-left difference

#### Scenario: Continuous-field sampling

- **WHEN** a lateral sample position is evaluated
- **THEN** the concentration is read from the continuous field at the real-valued position, without snapping the sample point to an integer cell

#### Scenario: STAM receives the readings

- **WHEN** lateral readings are produced
- **THEN** STAM records them and computes the dC/dt temporal derivative from its history
