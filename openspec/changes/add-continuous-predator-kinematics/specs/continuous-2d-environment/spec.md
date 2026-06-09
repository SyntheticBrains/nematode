## MODIFIED Requirements

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

## ADDED Requirements

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

On the continuous-2D substrate, predator **detection**, **damage**, and **contact-zone** classification SHALL be computed using true Euclidean distance between the predator's real-valued position and the agent's real-valued `pos_continuous`, not Manhattan distance against the agent's discretised integer position. The configured `detection_radius` and `damage_radius` SHALL be interpreted as Euclidean-millimetre thresholds (a Euclidean disc, not a Manhattan diamond). The contact-zone approach angle SHALL be measured between the predator→agent direction and the worm's continuous forward heading (`heading_rad`), retaining the existing anterior/lateral/posterior cone classification. The predator reward **formula** is unchanged by this requirement — only the underlying distance **metric** changes. The discrete grid environment SHALL continue to use Manhattan distance against the integer position, byte-stable.

#### Scenario: Euclidean detection on the continuous substrate

- **WHEN** danger or damage is evaluated for an agent in the continuous-2D environment
- **THEN** the predator-to-agent distance is the true Euclidean distance between the predator's real-valued position and the agent's `pos_continuous`, compared against the detection/damage radius as a Euclidean-millimetre threshold

#### Scenario: Contact zone uses the continuous heading

- **WHEN** a predator is within its damage radius of the agent on the continuous-2D substrate
- **THEN** the contact zone is classified by the approach angle between the predator→agent direction and the worm's continuous forward heading (`heading_rad`), using the existing anterior/lateral/posterior cones

#### Scenario: Grid detection and damage unchanged

- **WHEN** danger, damage, or contact zone is evaluated in the discrete grid environment
- **THEN** it retains Manhattan distance against the agent's integer position, byte-stable against the pre-change behaviour

### Requirement: Float predator placement

On the continuous-2D substrate, predators SHALL be initialised at real-valued (float) coordinates within the world bounds, retaining the existing Euclidean minimum-separation-from-agent spawn validity check. The discrete grid environment SHALL retain integer-lattice predator placement, byte-stable.

#### Scenario: Predators spawn at float coordinates within bounds

- **WHEN** predators are initialised in the continuous-2D environment
- **THEN** their `pos_continuous` coordinates are real-valued, lie within the world bounds, and satisfy the existing minimum-separation-from-agent (Euclidean) spawn check, with the integer `position` set as the rounded view

#### Scenario: Grid predator placement unchanged

- **WHEN** predators are initialised in the discrete grid environment
- **THEN** they spawn on the integer lattice exactly as before, byte-stable against the pre-change behaviour
