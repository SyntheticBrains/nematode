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

The continuous-2D environment SHALL translate a continuous action `(speed, turn)` into a kinematic position update — a heading rotation by the turn angle followed by a forward displacement proportional to speed — bounded by the world extent, replacing the discrete one-cell cardinal-step movement. Continuous-action brains emit a **normalized** action (`speed ∈ [0, 1]`, `turn ∈ [-1, 1]`); the environment SHALL rescale it to physical units — `speed_mm = speed_norm · max_step_mm` and **`turn_rad = turn_norm · max_turn_rad`**, where `max_turn_rad` is a configurable **maximum per-step angular velocity** (the rotational analogue of `max_step_mm`, default `0.5` rad ≈ 29°/step (real C. elegans reorients ~15–30°/step)). The resulting heading SHALL be wrapped to `[−π, π]`. So no single step rotates the heading by more than `max_turn_rad`, and the worm reorients in bounded sharp turns rather than rotating continuously ("helicopter" spinning). The discrete grid environment's cardinal-action movement SHALL remain unchanged and byte-stable.

#### Scenario: Heading-and-displacement update

- **WHEN** an agent emits a normalized continuous action `(speed_norm, turn_norm)`
- **THEN** the agent's heading rotates by `turn_norm · max_turn_rad` (the resulting heading wrapped to `[−π, π]`) and the agent advances by `speed_norm · max_step_mm` (clamped to `[0, max_step_mm]`) along the new heading

#### Scenario: Turn rate is bounded to a realistic maximum

- **WHEN** a continuous-action brain emits a normalized turn of magnitude up to `1.0`
- **THEN** the physical heading change for that step is at most `max_turn_rad` radians (default `0.5` rad ≈ 29°/step), so no single step performs a near-full heading reversal

#### Scenario: World-bound clamping

- **WHEN** a movement would take the agent outside the world bounds
- **THEN** the agent's new position is clamped per-axis to `[0, world_size_mm]` — it advances as far as the bound allows (partial movement), the step is **not** rejected, no error is raised, and the episode continues

### Requirement: Capture-radius food consumption

The continuous-2D environment SHALL consume a food source when the agent is within a configurable capture radius of it (Euclidean distance), replacing exact grid-cell-equality consumption. When the config-gated source-depletion dynamic is enabled, a consume event SHALL instead **decrement** the matched source's remaining amount once per event (the source persists at reduced amplitude), removing and respawning it only when it crosses the exhaustion threshold; with depletion disabled (the default) consumption removes the source outright as before.

#### Scenario: Food consumed within radius

- **WHEN** the agent's position is within the capture radius of a food source
- **THEN** that food is consumed (removed) and the foraging reward/satiety update fires exactly once for it

#### Scenario: Food not consumed outside radius

- **WHEN** the agent is farther than the capture radius from every food source
- **THEN** no food is consumed that step

#### Scenario: Depletion decrements instead of removing

- **WHEN** source-depletion is enabled and the agent consumes a source within the capture radius
- **THEN** the matched source's remaining amount SHALL be decremented (the source persists at its position with reduced amplitude), and the source SHALL be removed and respawned only when its remaining amount crosses the exhaustion threshold

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

On the continuous-2D substrate, predator **detection**, **damage**, **contact-zone** classification, and **nearest-predator-distance** queries SHALL be computed using true Euclidean distance between the predator's real-valued position and the agent's real-valued `pos_continuous`, not Manhattan distance against the agent's discretised integer position. The configured `detection_radius` and `damage_radius` SHALL be interpreted as Euclidean-millimetre thresholds (a Euclidean disc, not a Manhattan diamond). **Because `damage_radius` defaults to the integer `0` that means "same cell" on the grid — an unreachable Euclidean distance for real-valued positions — the continuous-2D substrate SHALL apply a body/contact-scale fallback damage radius (`Continuous2DParams.predator_damage_radius_mm`, default `1.0` mm) whenever the configured `damage_radius` is less than or equal to `0`; an explicitly-configured positive `damage_radius` SHALL take precedence.** The **`get_nearest_predator_distance_for` and `get_nearest_predator_distance` queries SHALL return the true Euclidean distance** between the agent's `pos_continuous` and the nearest predator's real-valued position, so the predator distance consumed by the reward calculator is coherent with the continuous geometry (the reward **formula** is unchanged — it queries the same method, which now returns a Euclidean distance on the continuous substrate). The contact-zone approach angle SHALL be measured between the predator→agent direction and the worm's continuous forward heading (`heading_rad`), retaining the existing anterior/lateral/posterior cone classification. The predator reward **formula** is unchanged by this requirement — only the underlying distance **metric** changes, and `predator_damage_radius_mm` is a kinematics/substrate parameter, not a reward term. The discrete grid environment SHALL continue to use Manhattan distance against the integer position with `damage_radius = 0` as the same-cell contact rule, byte-stable.

#### Scenario: Euclidean detection on the continuous substrate

- **WHEN** danger or damage is evaluated for an agent in the continuous-2D environment
- **THEN** the predator-to-agent distance is the true Euclidean distance between the predator's real-valued position and the agent's `pos_continuous`, compared against the detection/damage radius as a Euclidean-millimetre threshold

#### Scenario: Nearest-predator-distance is Euclidean on the continuous substrate

- **WHEN** `get_nearest_predator_distance_for` (or the single-agent `get_nearest_predator_distance`) is queried in the continuous-2D environment
- **THEN** it returns the true Euclidean distance between the agent's `pos_continuous` and the nearest predator's real-valued position (not Manhattan against the discretised integer position), so the predator distance the reward calculator consumes is coherent with the continuous geometry

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

- **WHEN** danger, damage, contact zone, or nearest-predator-distance is evaluated in the discrete grid environment
- **THEN** it retains Manhattan distance against the agent's integer position with `damage_radius = 0` as the same-cell contact rule, byte-stable against the pre-change behaviour

### Requirement: Float predator placement

On the continuous-2D substrate, predators SHALL be initialised at real-valued (float) coordinates within the world bounds, retaining the existing Euclidean minimum-separation-from-agent spawn validity check. The discrete grid environment SHALL retain integer-lattice predator placement, byte-stable.

#### Scenario: Predators spawn at float coordinates within bounds

- **WHEN** predators are initialised in the continuous-2D environment
- **THEN** their `pos_continuous` coordinates are real-valued, lie within the world bounds, and satisfy the existing minimum-separation-from-agent (Euclidean) spawn check, with the integer `position` set as the rounded view

#### Scenario: Grid predator placement unchanged

- **WHEN** predators are initialised in the discrete grid environment
- **THEN** they spawn on the integer lattice exactly as before, byte-stable against the pre-change behaviour

### Requirement: Native-metric distance-from-position queries

The environment SHALL expose `get_nearest_food_distance_from(pos)` and
`get_nearest_predator_distance_from(pos)` returning the nearest food / predator distance from an
**arbitrary** position in the environment's **native metric** — Manhattan on the discrete grid
(identical to `get_nearest_food_distance_for` / `get_nearest_predator_distance_for`) and **Euclidean**
on the continuous-2D substrate. These let a consumer measure a previous-step distance in the same metric
the environment uses for the current-step distance. Each SHALL return `None` when there are no foods /
no enabled predators.

#### Scenario: Euclidean distance-from-position on the continuous substrate

- **WHEN** `get_nearest_food_distance_from(pos)` (or the predator variant) is called on the continuous-2D
  environment with an arbitrary position
- **THEN** it returns the true Euclidean distance from that position to the nearest food (or predator),
  consistent with the substrate's `get_nearest_*_distance_for` metric

#### Scenario: Manhattan distance-from-position on the grid (byte-stable)

- **WHEN** the same query is called on the discrete grid environment
- **THEN** it returns the Manhattan distance from that position to the nearest food (or predator),
  identical to the grid's existing `get_nearest_*_distance_for` computation

### Requirement: Coherent-metric potential-based distance reward

The potential-based distance-reward terms SHALL compute the **previous-step** distance in the **same
metric** as the current-step distance, using the environment's native-metric distance-from-position
query for the previous position. This applies to the foraging term
(`reward_distance_scale · (prev_dist − curr_dist)`) and the `default`-mode predator-evasion delta. On
the continuous-2D substrate both distances are therefore Euclidean, so the term telescopes over an
episode (cumulative ≈ scale · net approach) and SHALL NOT accrue a spurious per-step reward from a
Manhattan-vs-Euclidean mismatch. The reward **formula** and coefficients are unchanged; only the
previous-step distance **metric** is made coherent with the current-step distance.

#### Scenario: Distance reward telescopes on the continuous substrate

- **WHEN** an agent wanders near a food on the continuous-2D substrate without net approach (e.g.
  tangential motion at roughly constant distance)
- **THEN** the cumulative distance-reward over those steps is approximately zero (the term telescopes),
  rather than a growing positive sum, so loitering near food is not rewarded

#### Scenario: Grid distance reward unchanged

- **WHEN** the distance-reward term is computed on the discrete grid environment
- **THEN** the previous-step distance uses Manhattan (as before), and the reward value is byte-stable
  with the pre-change computation

### Requirement: Euclidean predator contact-intensity query

The environment SHALL expose `predator_contact_intensity_at(pos)` returning the graded predator
contact intensity at a position — `max(0, 1 − dist / radius)` against the highest-intensity
predator within its (effective) damage radius, or `0.0` when predators are disabled, none exist,
or none are within range — computed in the environment's **native metric**. The discrete grid
SHALL use Manhattan distance against the predator's integer position and the predator's raw
`damage_radius` (skipping predators with `damage_radius <= 0`). The continuous-2D environment
SHALL use **Euclidean** distance against the predator's real-valued position and the **effective**
damage radius (the body/contact-scale `predator_damage_radius_mm` fallback applied when the
configured `damage_radius <= 0`, an explicit positive radius taking precedence), so the
`predator_mechano` sensory channel is **non-zero and metric-coherent** on the continuous
substrate rather than constantly zero. The contact-intensity formula and the channel's meaning
are unchanged; only the distance metric and the effective radius differ by substrate.

#### Scenario: Continuous contact intensity is non-zero within the effective radius

- **WHEN** a predator with the continuous default `damage_radius = 0` is within
  `predator_damage_radius_mm` (Euclidean) of the query position on the continuous-2D substrate
- **THEN** `predator_contact_intensity_at(pos)` returns a graded value in `(0, 1]`
  (`max(0, 1 − euclidean_dist / predator_damage_radius_mm)`), not `0.0` — the `predator_mechano`
  channel is no longer dead

#### Scenario: Continuous contact intensity is Euclidean

- **WHEN** the query is evaluated on the continuous-2D substrate
- **THEN** the distance to the predator is the true Euclidean distance against the predator's
  real-valued position (so an off-axis predator yields a different intensity than Manhattan would)

#### Scenario: Grid contact intensity is byte-stable

- **WHEN** `predator_contact_intensity_at(pos)` is evaluated on the discrete grid environment
- **THEN** it uses Manhattan distance against the predator's integer position and the raw
  `damage_radius` (skipping `damage_radius <= 0`), byte-identical to the prior inline computation

#### Scenario: Zero when out of range or no predators

- **WHEN** no predator is within its (effective) damage radius, or predators are disabled / absent
- **THEN** the query returns `0.0`

### Requirement: Sensory field sampling at the agent's true position

The environment SHALL expose `agent_sensing_position(agent_id)` returning the position at which
sensory and reward fields are sampled for that agent — the integer `.position` on the discrete
grid, and the real-valued `pos_continuous` (float truth) on the continuous-2D substrate. Scalar
chemo/predator/pheromone concentration queries, the separated-gradient query, the short-term
associative memory (STAM) channel scalars, and the reward's field-query terms (predator
concentration, temperature) SHALL sample at this position, so that on the continuous substrate two
agent positions within the same grid cell but at different real-valued coordinates yield different
sensed field values (sub-cell sensing), rather than the same rounded-cell value. The field
sampling kernels and sensor/reward definitions are unchanged; only the query position differs by
substrate.

#### Scenario: Sub-cell sensing on the continuous substrate

- **WHEN** an agent occupies two different real-valued positions within the same integer grid cell
  on the continuous-2D substrate (e.g. `(10.1, 10.0)` and `(10.9, 10.0)`)
- **THEN** the scalar concentration, separated-gradient, and STAM-recorded scalar sampled at
  `agent_sensing_position` differ between the two positions (they are sampled at the float truth,
  not rounded to the same cell)

#### Scenario: Grid sensing unchanged

- **WHEN** sensory fields are sampled on the discrete grid environment
- **THEN** `agent_sensing_position` returns the integer `.position` and the sensed values are
  byte-stable with the pre-change behaviour

### Requirement: Discrete cell-identity logic uses the integer position

Discrete cell-identity logic SHALL continue to use the agent's integer `.position`, distinct from
the float sensing position — specifically the anti-dithering exact-equality check (against the
integer position history) and the exploration-bonus visited-cells set. Moving these to the float
position would break their cell-granular semantics; only field-sampling query positions use the
float truth.

#### Scenario: Anti-dithering still detects an integer-cell repeat

- **WHEN** an agent returns to a previously occupied integer cell on the continuous substrate
- **THEN** the anti-dithering check (integer-position equality against the path history) still
  fires, independent of the float sensing position used for field queries
