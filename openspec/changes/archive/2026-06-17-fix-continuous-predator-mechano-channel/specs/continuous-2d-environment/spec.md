## ADDED Requirements

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
