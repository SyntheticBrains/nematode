# Tasks: Phase 4 Evaluation Campaign

## Phase 1: New Metrics

**Dependencies**: None
**Parallelizable**: No (foundational)

- [x] 1.1 Add `_per_agent_food_positions: dict[str, list[tuple[int, int]]]` tracking to `MultiAgentSimulation`

  - Initialize in `__post_init__` or episode reset
  - Populate in `_resolve_food_step()` when food is consumed

- [x] 1.2 Implement `_compute_territorial_index()` helper

  - Per-agent: compute foraging spread (mean distance from centroid of food positions)
  - Gini coefficient across agent spreads
  - Return 0.0 when < 2 agents collected food

- [x] 1.3 Add `territorial_index: float = 0.0` to `MultiAgentEpisodeResult`

  - Wire into `_build_result()`

- [x] 1.4 Add `ALARM_RESPONSE_WINDOW = 5` constant

- [x] 1.5 Implement alarm response rate tracking

  - Buffer alarm emissions with nearby agent directions at emission time
  - Track direction changes within window
  - Compute rate = changes / opportunities

- [x] 1.6 Add `alarm_response_rate: float = 0.0` to `MultiAgentEpisodeResult`

  - Wire into `_build_result()`

- [x] 1.7 Update `_MULTI_AGENT_SUMMARY_FIELDNAMES` in `csv_export.py`

  - Add `territorial_index` and `alarm_response_rate`

- [x] 1.8 Update `write_multi_agent_summary_row()` signature and body

  - Add `territorial_index: float = 0.0` and `alarm_response_rate: float = 0.0` parameters

- [x] 1.9 Update `run_simulation.py` to pass new fields

- [x] 1.10 Add docstrings for new fields on `MultiAgentEpisodeResult`

- [x] 1.11 Unit tests for territorial_index computation

  - 0.0 when < 2 agents have food
  - 0.0 when all agents forage identically
  - > 0 when agents forage in different regions
  - Gini = 1.0 when one agent tight, another spread

- [x] 1.12 Unit tests for alarm_response_rate

  - 0.0 when no alarm emissions
  - 0.0 when no nearby agents at emission time
  - Correct rate computation with known direction changes

## Phase 2: Scenario Configs

**Dependencies**: None (can be done in parallel with Phase 1)
**Parallelizable**: Yes

- [x] 2.1 Create `mlpppo_small_5agents_competition_full_oracle.yml`

  - 20x20, 5 agents, 3 food, all features (pheromones + social feeding + aggregation)

- [x] 2.2 Create `mlpppo_small_2agents_competition_oracle.yml`

  - 20x20, 2 agents, 3 food, baseline

- [x] 2.3 Create `mlpppo_small_10agents_competition_oracle.yml`

  - 20x20, 10 agents, 3 food, baseline

- [x] 2.4 Create `mlpppo_small_5agents_pursuit_oracle.yml`

  - 20x20, 5 agents, 3 food, 2 pursuit predators, alarm pheromones enabled

- [x] 2.5 Create `mlpppo_small_5agents_pursuit_no_alarm_oracle.yml`

  - 20x20, 5 agents, 3 food, 2 pursuit predators, pheromones disabled

- [x] 2.6 Create `configs/scenarios/foraging/mlpppo_small_1agent_foraging_oracle.yml`

  - 20x20, 1 agent, 3 food, single-agent baseline (no multi_agent block)
  - Placed in `foraging/` (single-agent), not `multi_agent_foraging/`
  - Environment params (grid_size, food, reward, satiety) must match multi-agent competition configs

- [x] 2.7 Create `lstmppo_small_5agents_competition_temporal.yml`

  - 20x20, 5 agents, 3 food, LSTM PPO GRU, temporal chemotaxis + STAM

- [x] 2.8 Create `configs/scenarios/foraging/lstmppo_small_1agent_foraging_temporal.yml`

  - 20x20, 1 agent, 3 food, LSTM PPO GRU, temporal chemotaxis + STAM
  - Placed in `foraging/` (single-agent)
  - Environment params must match multi-agent temporal competition configs

- [x] 2.9 Create `lstmppo_small_5agents_competition_pheromone_temporal.yml`

  - 20x20, 5 agents, 3 food, LSTM PPO GRU, temporal + all pheromones + STAM

## Phase 3: Verification & Sanity Checks

**Dependencies**: Phases 1 and 2

- [x] 3.1 Run `uv run pytest -m "not nightly"` — all tests pass

- [x] 3.2 Run `uv run pre-commit run -a` — all hooks pass

- [x] 3.3 Sanity check each new config (10 eps, seed 42) — no crashes, CSVs written

- [x] 3.4 Verify territorial_index and alarm_response_rate appear in summary CSVs

## Phase 4: Evaluation Campaigns

**Dependencies**: Phase 3

- [x] 4.1 Campaign A: Post-bug-fix validation (2000 eps, 4 seeds, 4 configs)

- [x] 4.2 Campaign B1: Oracle classical strain (4000 eps, 4 seeds, 4 configs)

- [x] 4.3 Campaign B2+D combined: Temporal strain + pheromone value (2000 eps, 2 seeds, 3 configs)

  - Proper Phase 3 temporal configs (400 satiety, 1000 max_steps, LR warmup/decay)
  - Pheromone treatment: alarm + aggregation only (food-marking excluded, issue #116)

- [x] 4.4 Campaign C: Collective predator response (2000 eps, 2 seeds, 2 configs)

- [x] 4.5 Campaign E: Social feeding under extreme scarcity (2000 eps, 2 seeds, 2 configs)

- [x] 4.6 Campaign F: Mixed phenotype competition (2000 eps, 2 seeds, 1 config)

- [x] 4.7 Campaign G: Proportional food scaling (2000 eps, 2 seeds, 3 configs)

- [x] 4.8 Game-theoretic indicator analysis from CSV data

  - Food Gini trajectory (from Campaign E data — scarcity creates inequality)
  - Territorial index (tracked but near-zero across all campaigns)

- [x] 4.9 Best-response analysis (deferred — requires additional infrastructure)

  - Deferred: multi-agent weight loading + freeze-at-step-0 not yet implemented

## Phase 5: Additional Bug Fixes (Discovered During Evaluation)

- [x] 5.1 Fix reward calculator bug #115 (agent_id parameter throughout)

- [x] 5.2 Add `get_nearest_food_distance_for()` and `get_nearest_predator_distance_for()` to env

- [x] 5.3 Fix STAM dimension mismatch (`_infer_stam_dim()` in modules.py)

- [x] 5.4 Gate food_sharing_buffer on pheromones_enabled

- [x] 5.5 Add config_name to both multi-agent CSV exports

- [x] 5.6 Expand per-agent CSV (total_reward, success, satiety_remaining, foods_available)

## Phase 6: Documentation & Conclusion

**Dependencies**: Phases 4, 5

- [x] 6.1 Write Logbook 011 (in-progress status, future work section open)

- [x] 6.2 Copy artifacts to `artifacts/logbooks/011/`

- [x] 6.3 Update docs/roadmap.md with exit criteria evidence

- [x] 6.4 Quantum checkpoint assessment (do not trigger)

- [x] 6.5 Go/No-Go decision (GO to Phase 5, deferred environment work)

- [x] 6.6 Update experiment index (docs/experiments/README.md)

- [x] 6.7 Add food spatial persistence to roadmap (issue #116)

______________________________________________________________________

## Summary

| Phase | Tasks | Dependencies |
|-------|-------|-------------|
| 1. New Metrics | 12 | None |
| 2. Scenario Configs | 9 | None |
| 3. Verification | 4 | Phases 1, 2 |
| 4. Evaluation Campaigns | 9 (1 deferred) | Phase 3 |
| 5. Bug Fixes | 6 | Discovered during Phase 4 |
| 6. Documentation | 7 | Phases 4, 5 |

**Total: 47 tasks across 6 phases (46 complete, 1 deferred)**
