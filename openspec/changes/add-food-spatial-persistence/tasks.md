## 1. Type Definition

- [x] 1.1 Add `FoodHotspot = tuple[int, int, float]` type alias to `dtypes.py` (after `OxygenSpot`)
- [x] 1.2 Export `FoodHotspot` from `env/__init__.py`

## 2. ForagingParams Extension

- [x] 2.1 Add `food_hotspots: list[FoodHotspot] | None = None` field to `ForagingParams`
- [x] 2.2 Add `food_hotspot_bias: float = 0.0` field to `ForagingParams`
- [x] 2.3 Add `food_hotspot_decay: float = 8.0` field to `ForagingParams`
- [x] 2.4 Add `no_respawn: bool = False` field to `ForagingParams`
- [x] 2.5 Add `satiety_food_threshold: float | None = None` field to `ForagingParams`

## 3. Hotspot-Biased Sampling

- [x] 3.1 Add `_sample_hotspot_candidate()` method to `DynamicForagingEnvironment` — selects hotspot weighted by weight, samples position with exponential decay, clamps to grid bounds
- [x] 3.2 Modify `_initialize_foods()` to use hotspot bias for candidate generation (matching `safe_zone_food_bias` pattern)
- [x] 3.3 Modify `spawn_food()` to use hotspot bias for candidate generation
- [x] 3.4 Add `no_respawn` guard at top of `spawn_food()` — return False immediately when enabled

## 4. Satiety-Gated Food Collection

- [x] 4.1 Add satiety gate in `check_and_consume_food()` (food_handler.py) — check agent satiety against `env.foraging.satiety_food_threshold * max_satiety` before calling `env.consume_food()` (note: single-agent uses `consume_food()`, not `consume_food_for()`), return food_consumed=False if sated
- [x] 4.2 Add satiety gate in `_resolve_food_step()` (multi_agent.py) — pre-filter sated agents from the `contested` map so they don't compete for food, leaving food available for hungry agents
- [x] 4.3 Add `can_eat: bool = True` parameter to `calculate_reward()` (reward_calculator.py) — suppress goal bonus when can_eat=False. Callers pass False when agent is on food but sated
- [x] 4.4 Pass `can_eat=False` through the reward chain: `agent.calculate_reward()` (agent.py:922) delegates to `RewardCalculator.calculate_reward()` — add `can_eat` parameter to both. Single-agent runner (runners.py) and multi-agent step loop pass False when satiety gate blocks consumption

## 5. Config Loader

- [x] 5.1 Add `food_hotspots`, `food_hotspot_bias`, `food_hotspot_decay`, `no_respawn`, `satiety_food_threshold` fields to `ForagingConfig` Pydantic model
- [x] 5.2 Add validation: hotspot coordinates within grid bounds, bias in [0.0, 1.0], decay > 0, threshold in (0.0, 1.0\] if set
- [x] 5.3 Convert `list[list[float]]` to `list[FoodHotspot]` in `to_params()`

## 6. Tests

- [x] 6.1 Test backward compatibility: default ForagingParams produces identical behavior
- [x] 6.2 Test hotspot spawning with bias=1.0: food spawns significantly closer to hotspot centers (statistical, seeded RNG)
- [x] 6.3 Test multiple hotspots: food distributes across hotspots proportional to weights
- [x] 6.4 Test hotspot respawn: after consumption, replacement food spawns near hotspots
- [x] 6.5 Test no_respawn mode: food count decreases after consumption, never increases
- [x] 6.6 Test partial bias (0.5): mix of hotspot and uniform spawning
- [x] 6.7 Test YAML config loading with food_hotspots and satiety_food_threshold
- [x] 6.8 Test hotspot bias composes with safe_zone_food_bias
- [x] 6.9 Test satiety gate: agent at high satiety cannot consume food (food remains on grid)
- [x] 6.10 Test satiety gate: agent below threshold can consume normally
- [x] 6.11 Test satiety gate disabled (None): no restriction on consumption
- [x] 6.12 Test multi-agent satiety gate: sated agents excluded from food competition, hungry agent at same position gets food
- [x] 6.13 Test reward suppression: no goal bonus when agent on food but can't eat

## 7. Example Configs

- [ ] 7.1 Create small (20×20) multi-agent hotspot + satiety gate config for mechanics verification and logbook 011 baseline comparison
- [ ] 7.2 Create medium (50×50) multi-agent hotspot + satiety gate config for food-marking pheromone evaluation (primary evaluation environment)

## 8. Verification

- [ ] 8.1 All existing tests pass (`uv run pytest -m "not nightly"`)
- [ ] 8.2 Pre-commit hooks pass (`uv run pre-commit run -a`)
- [ ] 8.3 Smoke test: small multi-agent hotspot config runs without error
- [ ] 8.4 Smoke test: medium multi-agent hotspot config runs without error
