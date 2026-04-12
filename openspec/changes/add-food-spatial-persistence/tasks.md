## 1. Type Definition

- [ ] 1.1 Add `FoodHotspot = tuple[int, int, float]` type alias to `dtypes.py` (after `OxygenSpot`)
- [ ] 1.2 Export `FoodHotspot` from `env/__init__.py`

## 2. ForagingParams Extension

- [ ] 2.1 Add `food_hotspots: list[FoodHotspot] | None = None` field to `ForagingParams`
- [ ] 2.2 Add `food_hotspot_bias: float = 0.0` field to `ForagingParams`
- [ ] 2.3 Add `food_hotspot_decay: float = 8.0` field to `ForagingParams`
- [ ] 2.4 Add `no_respawn: bool = False` field to `ForagingParams`
- [ ] 2.5 Add `satiety_food_threshold: float | None = None` field to `ForagingParams`

## 3. Hotspot-Biased Sampling

- [ ] 3.1 Add `_sample_hotspot_candidate()` method to `DynamicForagingEnvironment` — selects hotspot weighted by weight, samples position with exponential decay, clamps to grid bounds
- [ ] 3.2 Modify `_initialize_foods()` to use hotspot bias for candidate generation (matching `safe_zone_food_bias` pattern)
- [ ] 3.3 Modify `spawn_food()` to use hotspot bias for candidate generation
- [ ] 3.4 Add `no_respawn` guard at top of `spawn_food()` — return False immediately when enabled

## 4. Satiety-Gated Food Collection

- [ ] 4.1 Add `satiety_food_threshold` check in `consume_food_for()` (env.py) — refuse collection when agent satiety > threshold × max_satiety, return None
- [ ] 4.2 Update `_resolve_food_step()` in `multi_agent.py` — if winner can't eat due to satiety gate, re-offer food to other contestants at same position
- [ ] 4.3 Suppress goal bonus in `reward_calculator.py` when agent is on food but can't eat due to satiety gate (avoid perverse reward for standing on food while sated)
- [ ] 4.4 Pass `satiety_food_threshold` from ForagingParams to the satiety check (env has access to both agent satiety via agent state and foraging config)

## 5. Config Loader

- [ ] 5.1 Add `food_hotspots`, `food_hotspot_bias`, `food_hotspot_decay`, `no_respawn`, `satiety_food_threshold` fields to `ForagingConfig` Pydantic model
- [ ] 5.2 Add validation: hotspot coordinates within grid bounds, bias in [0.0, 1.0], decay > 0, threshold in (0.0, 1.0\] if set
- [ ] 5.3 Convert `list[list[float]]` to `list[FoodHotspot]` in `to_params()`

## 6. Tests

- [ ] 6.1 Test backward compatibility: default ForagingParams produces identical behavior
- [ ] 6.2 Test hotspot spawning with bias=1.0: food spawns significantly closer to hotspot centers (statistical, seeded RNG)
- [ ] 6.3 Test multiple hotspots: food distributes across hotspots proportional to weights
- [ ] 6.4 Test hotspot respawn: after consumption, replacement food spawns near hotspots
- [ ] 6.5 Test no_respawn mode: food count decreases after consumption, never increases
- [ ] 6.6 Test partial bias (0.5): mix of hotspot and uniform spawning
- [ ] 6.7 Test YAML config loading with food_hotspots and satiety_food_threshold
- [ ] 6.8 Test hotspot bias composes with safe_zone_food_bias
- [ ] 6.9 Test satiety gate: agent at high satiety cannot consume food (food remains on grid)
- [ ] 6.10 Test satiety gate: agent below threshold can consume normally
- [ ] 6.11 Test satiety gate disabled (None): no restriction on consumption
- [ ] 6.12 Test multi-agent satiety gate: sated winner's food re-offered to hungry contestant
- [ ] 6.13 Test reward suppression: no goal bonus when agent on food but can't eat

## 7. Example Configs

- [ ] 7.1 Create small (20×20) multi-agent hotspot + satiety gate config for mechanics verification and logbook 011 baseline comparison
- [ ] 7.2 Create medium (50×50) multi-agent hotspot + satiety gate config for food-marking pheromone evaluation (primary evaluation environment)

## 8. Verification

- [ ] 8.1 All existing tests pass (`uv run pytest -m "not nightly"`)
- [ ] 8.2 Pre-commit hooks pass (`uv run pre-commit run -a`)
- [ ] 8.3 Smoke test: small multi-agent hotspot config runs without error
- [ ] 8.4 Smoke test: medium multi-agent hotspot config runs without error
