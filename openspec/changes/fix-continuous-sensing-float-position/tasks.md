# Tasks — continuous sensing samples at the float position

## 1. Sensing-position accessor

- [x] 1.1 Add `agent_sensing_position(agent_id) -> tuple[float, float]` to the base
      `DynamicForagingEnvironment` (`env.py`) returning the integer `.position`.
- [x] 1.2 Override on `Continuous2DEnvironment` (`continuous_2d.py`) to return `_agent_xy(agent_id)`
      (float `pos_continuous`).

## 2. Route field queries through it

- [x] 2.1 The chemo/predator `_for` field-query variants (`get_food_concentration_for`,
      `get_predator_concentration_for`, `get_separated_gradients_for`) sample at
      `self.agent_sensing_position(agent_id)`. (The temperature/oxygen/pheromone `_for` variants
      are currently uncalled; not routed.)
- [x] 2.2 Agent sensing (`agent.py`): the scalar-concentration block + STAM use a `sensing_pos`
      (= `agent_sensing_position`) for the field queries while keeping the integer `agent_pos` for
      the grid klinotaxis cell-offset sampling; the oracle `get_separated_gradients` /
      temperature / contact-intensity queries sample at `agent_sensing_position`; STAM drops its
      `int(...)` cast (records the float sensing position). Widen the field-query signatures
      (`get_separated_gradients`, `_compute_*_gradient_vector`, `_predator_contact_intensity_at`,
      `_previous_position`) to accept the float position.
- [x] 2.3 Reward (`reward_calculator.py`): the current-position predator-concentration term samples
      at the float `sensing_pos`; the integer `agent_pos` is kept for the anti-dithering
      exact-equality check (`agent_pos == path[-3]`). The prev-vs-curr temperature-avoidance term is
      deferred with #1 (needs a float position history).

## 3. Tests

- [x] 3.1 Sub-cell sensitivity (continuous): two positions in the same integer cell but different
      float coordinates yield different scalar concentration / separated-gradient / STAM scalar.
- [x] 3.2 Grid byte-stability: `agent_sensing_position` returns the integer `.position`; grid
      sensing values unchanged; anti-dithering still fires on an integer-cell repeat (continuous).

## 4. Gates + tracking

- [x] 4.1 `openspec validate fix-continuous-sensing-float-position --strict`.
- [x] 4.2 Targeted `pre-commit` (ruff / pyright / markdownlint); agent-sensing / STAM /
      reward-calculator / continuous-env suites pass; full `pre-commit run -a` before push.
- [x] 4.3 Note in the continuous audit follow-up that #1 (reward prev-distance) and #7 (pheromone)
      remain deferred after this change.
