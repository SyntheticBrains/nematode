# Tasks

## 1. Implementation

- [x] 1.1 Override `get_nearest_predator_distance_for(self, agent_id)` on `Continuous2DEnvironment`
  (`env/continuous_2d.py`) to return the min Euclidean distance (`math.hypot`) between
  `_agent_xy(agent_id)` and each predator's `_predator_xy(pred)`; `None` when predators disabled/empty.
- [x] 1.2 Override the single-agent convenience `get_nearest_predator_distance(self)` to delegate to
  `get_nearest_predator_distance_for(DEFAULT_AGENT_ID)`.
- [x] 1.3 Confirm `reward_calculator.py` is unchanged (the reward formula queries the same method).

## 2. Tests

- [x] 2.1 Continuous nearest-predator-distance is Euclidean: place a predator at a known offset from the
  agent's `pos_continuous` and assert the returned distance equals `hypot(Δx, Δy)` (and differs from the
  Manhattan `|Δx| + |Δy|` for a diagonal offset).
- [x] 2.2 The convenience `get_nearest_predator_distance` matches the `_for(DEFAULT_AGENT_ID)` result.
- [x] 2.3 Grid byte-stability: the discrete-grid nearest-predator-distance remains integer-Manhattan
  (existing predator suite stays green).

## 3. Validate + gate

- [x] 3.1 `openspec validate make-continuous-predator-distance-euclidean --strict`.
- [x] 3.2 Targeted `pre-commit` on changed files; full `pre-commit run -a` before push;
  `uv run pytest -m "not nightly"` for the env suite.
