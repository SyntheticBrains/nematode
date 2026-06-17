# Design — continuous sensing samples at the float position

## Context

Continuous audit (`tmp/continuous_audit.md`) found a position-representation residual: sensory
and reward **field queries** sample at the agent's rounded integer `.position` rather than the
float `pos_continuous` truth (`_agent_xy`). The field kernels are already Euclidean/analytic, and
the klinotaxis path + `get_nearest_*_distance_*` + `predator_contact_intensity_at` already use the
float truth — this change extends the float truth to the remaining scalar/oracle/STAM/reward field
queries. On the continuous-action path this restores sub-cell sensing fidelity; it matters for C2
(predator) and the connectome's sensor projections (Stage 2).

## Decision 1 — One `agent_sensing_position(agent_id)` accessor

Add `agent_sensing_position(agent_id)` to the base env returning the integer `.position`;
override on `Continuous2DEnvironment` to return `_agent_xy(agent_id)` (the float `pos_continuous`,
falling back to the int view). Every **field-sampling** query position routes through it, so the
float-vs-int choice lives with the geometry (mirroring the `get_nearest_*` / `predator_contact_*`
overrides) and no consumer branches on substrate type. Grid returns the integer position → grid
byte-stable.

## Decision 2 — Route the base `_for` field variants + the agent/reward sites through it

- The chemo/predator `_for` field-query variants (`get_food_concentration_for`,
  `get_predator_concentration_for`, `get_separated_gradients_for`) sample at
  `self.agent_sensing_position(agent_id)`. (The temperature/oxygen/pheromone `_for` variants are
  currently uncalled, so not routed; the field-query signatures down the call chain —
  `get_separated_gradients`, `_compute_*_gradient_vector`, `_predator_contact_intensity_at`,
  `_previous_position` — are widened to accept the float position.)
- Agent sensing (agent.py): the scalar-concentration block, the oracle `get_separated_gradients`
  query, and the STAM channel-scalar sampling read `self.env.agent_sensing_position(self.agent_id)`.
- Reward (reward_calculator.py): the predator-concentration and temperature field terms sample at
  the float sensing position.

The position-accepting field methods (`get_food_concentration(position=...)`, etc.) are unchanged
— they already accept and correctly handle a float position; only the position passed to them
moves to the float truth.

## Decision 3 — Cell-identity logic keeps the integer position (do NOT blanket-replace)

`agent_pos` is overloaded in places: it feeds both field queries (→ float) and **discrete
cell-identity** checks (→ must stay int). Specifically preserved on the integer `.position`:

- anti-dithering exact equality `agent_pos == path[-3]` (reward_calculator) — float would never
  match the integer `path`, silently disabling anti-dithering;
- the exploration-bonus `visited_cells` set (cell-granular novelty by design);
- the integer `path` history itself (consumed by the above).

So the fix introduces a **separate** float sensing position for field queries rather than
replacing `agent_pos` wholesale. STAM's `pos_2d = (int(current_pos[0]), int(current_pos[1]))`
drops the `int(...)` on the continuous path (the channel fetchers — concentrations,
`predator_contact_intensity_at` — accept float).

## Decision 4 — Scope: live current-position field queries only

This change fixes the **current-position** field queries (audit #2 oracle gradient, #3 scalar
concentrations + the oracle temperature/oxygen *sensing*, #4 STAM, #5 predator-concentration
reward). Deferred:

- **Reward prev-vs-curr telescoping terms (audit #1 foraging-distance + the temperature-avoidance
  term)** — both compute a *previous*-step value from `path[-2]` (integer position history, shared
  with the cell-identity consumers above); float-ifying the previous value needs a float position
  history. Zero-mean ≤~0.5 mm, and C1 converges ~100% with them present. Tracked as a separate
  low-impact refinement. (Temperature *sensing* is fixed here; only the prev-vs-curr reward term is
  deferred.)
- **Audit #7 (pheromone Manhattan spatial kernel)** — only reachable if pheromones/social-feeding
  are enabled on continuous (not in current scope).

## Decision 5 — Validation

- **Sub-cell sensitivity (continuous):** for two agent positions in the same integer cell but
  different float coordinates (e.g. `(10.1, 10.0)` vs `(10.9, 10.0)`), the scalar food/predator
  concentration, the separated-gradient, and the STAM-recorded scalar differ — they were
  identical pre-fix (rounded to the same cell).
- **Grid byte-stability:** `agent_sensing_position` returns the integer `.position` on the grid;
  grid sensing values and the anti-dithering / exploration cell-identity logic are unchanged.
- **Reward:** the predator-concentration and temperature reward terms evaluate at the float
  position on continuous; the anti-dithering equality still fires on integer-cell repeats.
- **Regression:** agent-sensing, STAM, reward-calculator, and continuous-env suites pass; targeted
  pre-commit.

## Risks / alternatives considered

- **Blanket-replace `agent_pos` with the float position** — breaks the anti-dithering equality and
  the `visited_cells` cell-identity semantics. Rejected; the float position is introduced only for
  field sampling (Decision 3).
- **Override each `_for` variant on the continuous env individually** — more surface and easy to
  miss one; the single `agent_sensing_position` accessor centralizes the choice.
- **Store float positions in `path`** — would fix audit #1 too, but changes a representation
  consumed by anti-dithering/`visited_cells` exact-cell semantics; deferred with #1.
