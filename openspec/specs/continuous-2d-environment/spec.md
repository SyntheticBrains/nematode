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

The continuous-2D environment SHALL place **food sources** at **real-valued (float) coordinates** within the world bounds and SHALL compute food capture and nearest-food distance using **true Euclidean distance** (not Manhattan, and **not** rounded to integer). **Predator** placement, movement, and detection/damage geometry on the continuous-2D substrate are **also** continuous — governed by the *Continuous predator kinematics*, *Euclidean predator detection, damage, and contact-zone geometry*, and *Float predator placement* requirements below (no longer deferred to a later iteration). The float source coordinates are confined to the continuous-2D environment; the discrete grid environment retains integer source coordinates and remains byte-stable.

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

### Requirement: Continuous predator kinematics

On the continuous-2D substrate, predators SHALL move with a continuous `(speed, heading)` kinematic update and SHALL carry a real-valued `pos_continuous` truth position, replacing the inherited cardinal one-cell integer-Manhattan step. A **pursuit** predator with an agent inside its (Euclidean) detection radius SHALL set its heading toward the bearing to the nearest agent's real-valued position and advance by a `speed`-scaled displacement along the new heading; a predator with no agent in range SHALL wander (heading perturbed by a bounded random angle, then advance); a **stationary** predator SHALL not move. Each updated position SHALL be clamped per-axis to the world bounds (partial movement, no error, episode continues). The predator's integer `position` SHALL be kept as a rounded, bound-clamped view of `pos_continuous` so inherited grid-coupled readers (rendering, metrics, logging) are unaffected. The discrete grid environment's integer-Manhattan predator movement SHALL remain unchanged and byte-stable.

#### Scenario: Pursuit predator steers toward the agent and advances smoothly

- **WHEN** a pursuit predator on the continuous-2D substrate has an agent within its detection radius
- **THEN** its heading rotates toward the bearing to that agent's real-valued position and it advances by a `speed`-scaled real-valued displacement along the new heading (no whole-cell Manhattan jump), with `pos_continuous` updated and the integer `position` set as its rounded, clamped view

#### Scenario: Wandering predator moves continuously when no agent is in range

- **WHEN** a non-stationary predator has no agent within its detection radius
- **THEN** it perturbs its heading by a bounded random angle and advances by a `speed`-scaled real-valued displacement, remaining within the world bounds

#### Scenario: World-bound clamping for predators

- **WHEN** a predator movement would cross the world bounds
- **THEN** the predator's new position is clamped per-axis to `[0, world_size_mm]`, the step is not rejected, no error is raised, and the episode continues

#### Scenario: Grid predator movement unchanged

- **WHEN** predators move in the discrete grid environment
- **THEN** they retain the integer-lattice cardinal-step Manhattan movement, byte-stable against the pre-change behaviour (the continuous fields are unused on the grid)

### Requirement: Euclidean predator detection, damage, and contact-zone geometry

On the continuous-2D substrate, predator **detection**, **damage**, and **contact-zone** classification SHALL be computed using true Euclidean distance between the predator's real-valued position and the agent's real-valued `pos_continuous`, not Manhattan distance against the agent's discretised integer position. The configured `detection_radius` and `damage_radius` SHALL be interpreted as Euclidean-millimetre thresholds (a Euclidean disc, not a Manhattan diamond). **Because `damage_radius` defaults to the integer `0` that means "same cell" on the grid — an unreachable Euclidean distance for real-valued positions — the continuous-2D substrate SHALL apply a body/contact-scale fallback damage radius (`Continuous2DParams.predator_damage_radius_mm`, default `1.0` mm) whenever the configured `damage_radius` is less than or equal to `0`; an explicitly-configured positive `damage_radius` SHALL take precedence.** The contact-zone approach angle SHALL be measured between the predator→agent direction and the worm's continuous forward heading (`heading_rad`), retaining the existing anterior/lateral/posterior cone classification. The predator reward **formula** is unchanged by this requirement — only the underlying distance **metric** changes, and `predator_damage_radius_mm` is a kinematics/substrate parameter, not a reward term. The discrete grid environment SHALL continue to use Manhattan distance against the integer position with `damage_radius = 0` as the same-cell contact rule, byte-stable.

#### Scenario: Euclidean detection on the continuous substrate

- **WHEN** danger or damage is evaluated for an agent in the continuous-2D environment
- **THEN** the predator-to-agent distance is the true Euclidean distance between the predator's real-valued position and the agent's `pos_continuous`, compared against the detection/damage radius as a Euclidean-millimetre threshold

#### Scenario: Damage reachable at the body/contact scale with the default damage radius

- **WHEN** damage is evaluated on the continuous-2D substrate and the configured `damage_radius` is `0` (the integer grid default)
- **THEN** the effective damage radius is the body/contact-scale fallback `predator_damage_radius_mm` (default `1.0` mm), so a predator that closes to within that Euclidean distance of the agent deals damage (rather than the unreachable zero-distance threshold)

#### Scenario: Explicit positive damage radius is honoured

- **WHEN** a scenario configures a positive `damage_radius` on the continuous-2D substrate
- **THEN** that configured value is used as the Euclidean-millimetre damage threshold and the fallback is not applied

#### Scenario: Contact zone uses the continuous heading

- **WHEN** a predator is within its (effective) damage radius of the agent on the continuous-2D substrate
- **THEN** the contact zone is classified by the approach angle between the predator→agent direction and the worm's continuous forward heading (`heading_rad`), using the existing anterior/lateral/posterior cones

#### Scenario: Grid detection and damage unchanged

- **WHEN** danger, damage, or contact zone is evaluated in the discrete grid environment
- **THEN** it retains Manhattan distance against the agent's integer position with `damage_radius = 0` as the same-cell contact rule, byte-stable against the pre-change behaviour

### Requirement: Float predator placement

On the continuous-2D substrate, predators SHALL be initialised at real-valued (float) coordinates within the world bounds, retaining the existing Euclidean minimum-separation-from-agent spawn validity check. The discrete grid environment SHALL retain integer-lattice predator placement, byte-stable.

#### Scenario: Predators spawn at float coordinates within bounds

- **WHEN** predators are initialised in the continuous-2D environment
- **THEN** their `pos_continuous` coordinates are real-valued, lie within the world bounds, and satisfy the existing minimum-separation-from-agent (Euclidean) spawn check, with the integer `position` set as the rounded view

#### Scenario: Grid predator placement unchanged

- **WHEN** predators are initialised in the discrete grid environment
- **THEN** they spawn on the integer lattice exactly as before, byte-stable against the pre-change behaviour
