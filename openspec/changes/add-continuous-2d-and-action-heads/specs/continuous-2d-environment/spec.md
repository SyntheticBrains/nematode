## ADDED Requirements

### Requirement: Continuous-2D coordinate substrate

The system SHALL provide a continuous-2D environment in which the **agent position** is represented as real-valued `(x, y)` coordinates within world bounds expressed in physical units (millimetres), distinct from and selectable alongside the discrete grid environment. In this iteration **food sources and predators are positioned on the integer lattice** (discrete integer `(x, y)` coordinates within the world bounds in millimetres); the worm, its movement, capture, and distances are fully continuous. The grid environment SHALL remain unchanged and byte-stable.

> Implementation note (non-normative): continuous real-valued *source* coordinates (float food/predator placement) are deferred to a future iteration — they would ripple the `self.foods: list[tuple[int, int]]` type — so sources remain lattice-positioned within the continuous arena for now.

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

### Requirement: Source placement and Euclidean fields

The continuous-2D environment SHALL place food and predator sources within the world bounds and SHALL compute food capture and nearest-food distance using Euclidean distance (not Manhattan). *(Implementation note, 2026-06-06: in this iteration sources sit on the **integer lattice within the continuous arena** — the worm and capture are fully continuous; float source placement is deferred to avoid the `self.foods: list[tuple[int, int]]` type ripple, and true continuous fields are the T6 Rung-2 env-fidelity work.)*

#### Scenario: Sources within bounds

- **WHEN** sources are initialised in the continuous-2D environment
- **THEN** their coordinates lie within the world bounds and respect the existing minimum-separation validity checks (on the integer lattice within the arena in this iteration)

#### Scenario: Euclidean distances

- **WHEN** food capture or nearest-food distance is computed in the continuous-2D environment
- **THEN** it uses Euclidean distance (replacing the grid's Manhattan distance)

### Requirement: Heading-aware klinotaxis sensing on the continuous substrate

Klinotaxis (head-sweep) sensing in the continuous-2D environment SHALL sample lateral concentrations perpendicular to the agent's continuous heading (`heading_rad`), scaled by the configured sweep amplitude (≥ 1 cell), rather than at the grid's fixed cardinal ±1-cell offsets, so the directional gradient rotates with the continuous heading, and SHALL feed STAM the resulting readings. *(Implementation note, 2026-06-06: samples the **integer cells** perpendicular to the heading — sources + sensing live on the integer lattice within the continuous arena (the worm itself moves continuously); true continuous-field sampling is the T6 Rung-2 work.)*

#### Scenario: Heading-aware lateral sampling

- **WHEN** klinotaxis sensing runs in the continuous-2D environment
- **THEN** left/right samples are taken perpendicular to `heading_rad` (matching the grid offsets at cardinal headings, rotating smoothly otherwise), and `dC` is the right-minus-left difference

#### Scenario: STAM receives the readings

- **WHEN** lateral readings are produced
- **THEN** STAM records them and computes the dC/dt temporal derivative from its history
