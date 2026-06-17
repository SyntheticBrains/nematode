## Why

The continuous-2D substrate's whole point is **sub-cell-resolution sensing** — a worm at
`(10.3, 8.7)` mm should sense a different field value than one at `(10.0, 8.0)`. But the
sensory/reward field queries sample at the agent's **rounded integer `.position`** (the grid
"discretised view") instead of the float truth `pos_continuous`. The continuous audit
(`tmp/continuous_audit.md`, verified against code) found this position-representation residual
across the live sensing path:

- scalar concentrations (`get_food/predator/pheromone_concentration`) sampled at int
  `.position` (agent.py ~771) — a staircase signal under sub-cell motion;
- the oracle separated-gradient query `get_separated_gradients(.position)` (agent.py ~1014) —
  gradient **direction** quantized, can flip near a source;
- the STAM channel scalars recorded at `pos_2d = (int(current_pos[0]), int(current_pos[1]))`
  (agent.py ~952) — every STAM channel + its dC/dt derived from quantized samples;
- the reward's predator-concentration term `get_predator_concentration(.position)`
  (reward_calculator ~137) and the temperature-avoidance term `get_temperature(.position)`
  (~340).

The field **kernels** are already Euclidean/analytic (the continuous distance/gradient math is
correct); **only the query position is rounded**. On the genuine continuous-action path
(`_kinematic_move`, fractional positions) this re-quantizes the substrate to ~1 mm cells,
degrading the sensor signal the brains act on — and it matters most for **C2 (predator)** and
**Stage 2 (the connectome's biologically-grounded sensor→neuron projections)**, which depend on
faithful continuous sensing. The klinotaxis path and the `get_nearest_*_distance_*` /
`predator_contact_intensity_at` methods already use the float truth; this change extends that to
the remaining scalar/oracle/STAM/reward field queries.

## What Changes

- **One accessor for the sensing query position.** Add `agent_sensing_position(agent_id)` to the
  environment — the position at which sensory fields SHALL be sampled: the integer `.position`
  on the discrete grid (unchanged), the float `pos_continuous` (via `_agent_xy`) on the
  continuous-2D substrate. This centralizes the float-vs-int choice where the geometry lives
  (mirroring the `get_nearest_*` overrides), so no consumer branches on substrate type.
- **Thread it through the field queries.** The chemo/predator `_for` field-query variants
  (`get_food_concentration_for`, `get_predator_concentration_for`, `get_separated_gradients_for`)
  and the agent's scalar/oracle/STAM sensing sites sample at `agent_sensing_position(...)` instead
  of `.position` (so scalar concentrations, the oracle separated-gradient + temperature/oxygen
  sensing, and the STAM channels sample at the float truth; STAM drops its `int(...)` cast on the
  continuous path). The reward's **current-position** predator-concentration term samples at the
  float position; the integer `.position` is kept for the anti-dithering exact-equality check.
- **Cell-identity uses keep the integer position.** The anti-dithering exact-equality check
  (`agent_pos == path[-3]`), the exploration-bonus `visited_cells` set, and any other discrete
  cell-identity logic SHALL continue to use the integer `.position` — only **field-sampling**
  query positions move to the float truth. (Grid behaviour is unchanged throughout; the accessor
  returns the integer position there.)
- **Sensor / reward definitions unchanged (RQ5-safe).** No new sensors, no formula changes —
  the same fields sampled at the substrate's true position rather than a rounded cell.

### Deferred (out of scope, flagged)

- **Reward prev-vs-curr telescoping terms (audit #1 + the temperature-avoidance term)** — the
  foraging-distance and temperature-avoidance reward terms compute a *previous*-step value from
  `path[-2]` (the integer position history); making the previous value float requires a float
  position history (the int `path` is also consumed by anti-dithering/`visited_cells` exact-cell
  semantics). These are **zero-mean and ≤~0.5 mm**, and C1 already converges at ~100% with them
  present, so they are deferred as a separate low-impact refinement rather than coupled to this
  live-sensing fix. (The temperature *sensing* the brain reads is fixed here — only the
  prev-vs-curr reward term is deferred.)
- **Pheromone Manhattan spatial kernel (audit #7)** — deferred until pheromones / social-feeding
  are enabled on the continuous substrate (out of the current C1/C2/C3 scope).

## Capabilities

### Modified Capabilities

- `continuous-2d-environment`: extends the float-truth guarantee to **sensory and reward field
  sampling** — the environment SHALL expose `agent_sensing_position(agent_id)` (float
  `pos_continuous` on continuous, integer `.position` on grid), and the scalar/oracle/STAM
  sensing and the reward's field-query terms SHALL sample fields at it, so continuous sensing is
  sub-cell-faithful rather than re-quantized to grid cells. Discrete cell-identity logic and grid
  behaviour are unchanged.

## Impact

- **Code:**
  - `env.py` — add base `agent_sensing_position(agent_id)` (integer `.position`); route the base
    `_for` field-query variants through it.
  - `continuous_2d.py` — override `agent_sensing_position` to return `_agent_xy` (float).
  - `agent.py` — scalar concentrations, oracle `get_separated_gradients`, and STAM channel
    sampling use `agent_sensing_position(...)`; drop the STAM `int(...)` cast on the float path.
  - `reward_calculator.py` — predator-concentration and temperature field terms sample at the
    float sensing position; keep the integer `.position` for the anti-dithering equality check.
- **Tests:** continuous scalar concentration / separated-gradient / STAM samples differ for two
  sub-cell positions within the same grid cell (sub-cell sensitivity restored); grid sensing +
  the anti-dithering/exploration cell-identity logic byte-stable; reward field terms sample at the
  float position on continuous.
- **Downstream:** faithful continuous sensing for C2 (predator) and Stage 2 (connectome sensor
  projections). No new dependencies.
