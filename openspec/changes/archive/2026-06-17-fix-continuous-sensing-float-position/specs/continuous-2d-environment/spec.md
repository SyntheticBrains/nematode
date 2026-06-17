## ADDED Requirements

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
