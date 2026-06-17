## ADDED Requirements

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
