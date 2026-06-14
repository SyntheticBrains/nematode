## MODIFIED Requirements

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
