# Tasks: Phase 4 Evaluation Campaign

## Phase 1: New Metrics

**Dependencies**: None
**Parallelizable**: No (foundational)

- [ ] 1.1 Add `_per_agent_food_positions: dict[str, list[tuple[int, int]]]` tracking to `MultiAgentSimulation`

  - Initialize in `__post_init__` or episode reset
  - Populate in `_resolve_food_step()` when food is consumed

- [ ] 1.2 Implement `_compute_territorial_index()` helper

  - Per-agent: compute foraging spread (mean distance from centroid of food positions)
  - Gini coefficient across agent spreads
  - Return 0.0 when < 2 agents collected food

- [ ] 1.3 Add `territorial_index: float = 0.0` to `MultiAgentEpisodeResult`

  - Wire into `_build_result()`

- [ ] 1.4 Add `ALARM_RESPONSE_WINDOW = 5` constant

- [ ] 1.5 Implement alarm response rate tracking

  - Buffer alarm emissions with nearby agent directions at emission time
  - Track direction changes within window
  - Compute rate = changes / opportunities

- [ ] 1.6 Add `alarm_response_rate: float = 0.0` to `MultiAgentEpisodeResult`

  - Wire into `_build_result()`

- [ ] 1.7 Update `_MULTI_AGENT_SUMMARY_FIELDNAMES` in `csv_export.py`

  - Add `territorial_index` and `alarm_response_rate`

- [ ] 1.8 Update `write_multi_agent_summary_row()` signature and body

  - Add `territorial_index: float = 0.0` and `alarm_response_rate: float = 0.0` parameters

- [ ] 1.9 Update `run_simulation.py` to pass new fields

- [ ] 1.10 Add docstrings for new fields on `MultiAgentEpisodeResult`

- [ ] 1.11 Unit tests for territorial_index computation

  - 0.0 when < 2 agents have food
  - 0.0 when all agents forage identically
  - > 0 when agents forage in different regions
  - Gini = 1.0 when one agent tight, another spread

- [ ] 1.12 Unit tests for alarm_response_rate

  - 0.0 when no alarm emissions
  - 0.0 when no nearby agents at emission time
  - Correct rate computation with known direction changes

## Phase 2: Scenario Configs

**Dependencies**: None (can be done in parallel with Phase 1)
**Parallelizable**: Yes

- [ ] 2.1 Create `mlpppo_small_5agents_competition_full_oracle.yml`

  - 20x20, 5 agents, 3 food, all features (pheromones + social feeding + aggregation)

- [ ] 2.2 Create `mlpppo_small_2agents_competition_oracle.yml`

  - 20x20, 2 agents, 3 food, baseline

- [ ] 2.3 Create `mlpppo_small_10agents_competition_oracle.yml`

  - 20x20, 10 agents, 3 food, baseline

- [ ] 2.4 Create `mlpppo_small_5agents_pursuit_oracle.yml`

  - 20x20, 5 agents, 3 food, 2 pursuit predators, alarm pheromones enabled

- [ ] 2.5 Create `mlpppo_small_5agents_pursuit_no_alarm_oracle.yml`

  - 20x20, 5 agents, 3 food, 2 pursuit predators, pheromones disabled

- [ ] 2.6 Create `mlpppo_small_1agent_foraging_oracle.yml`

  - 20x20, 1 agent, 3 food, single-agent baseline (no multi_agent block)

- [ ] 2.7 Create `lstmppo_small_5agents_competition_temporal.yml`

  - 20x20, 5 agents, 3 food, LSTM PPO GRU, temporal chemotaxis + STAM

- [ ] 2.8 Create `lstmppo_small_1agent_foraging_temporal.yml`

  - 20x20, 1 agent, 3 food, LSTM PPO GRU, temporal chemotaxis + STAM

- [ ] 2.9 Create `lstmppo_small_5agents_competition_pheromone_temporal.yml`

  - 20x20, 5 agents, 3 food, LSTM PPO GRU, temporal + all pheromones + STAM

## Phase 3: Verification & Sanity Checks

**Dependencies**: Phases 1 and 2

- [ ] 3.1 Run `uv run pytest -m "not nightly"` — all tests pass

- [ ] 3.2 Run `uv run pre-commit run -a` — all hooks pass

- [ ] 3.3 Sanity check each new config (10 eps, seed 42) — no crashes, CSVs written

- [ ] 3.4 Verify territorial_index and alarm_response_rate appear in summary CSVs

## Phase 4: Evaluation Campaigns

**Dependencies**: Phase 3

- [ ] 4.1 Campaign A: Post-bug-fix validation (2000 eps, 4 seeds, 4 configs)

- [ ] 4.2 Campaign B1: Oracle classical strain (4000 eps, 4 seeds, 4 configs)

- [ ] 4.3 Campaign B2: Temporal classical strain (4000 eps, 4 seeds, 2 configs)

- [ ] 4.4 Campaign C: Collective predator response (2000 eps, 4 seeds, 2 configs)

- [ ] 4.5 Campaign D: Temporal pheromone value (4000 eps, 4 seeds, 2 configs)

- [ ] 4.6 Game-theoretic indicator analysis (post-hoc from CSV data)

## Phase 5: Documentation & Conclusion

**Dependencies**: Phase 4

- [ ] 5.1 Write Logbook 011: Multi-Agent Phase 4 Evaluation

- [ ] 5.2 Update docs/roadmap.md with Phase 4 exit criteria evidence

- [ ] 5.3 Quantum checkpoint assessment

- [ ] 5.4 Go/No-Go decision text

- [ ] 5.5 Update AGENTS.md with new scenario configs

- [ ] 5.6 Update configs/README.md

- [ ] 5.7 Update openspec/config.yaml

______________________________________________________________________

## Summary

| Phase | Tasks | Dependencies |
|-------|-------|-------------|
| 1. New Metrics | 12 | None |
| 2. Scenario Configs | 9 | None |
| 3. Verification | 4 | Phases 1, 2 |
| 4. Evaluation Campaigns | 6 | Phase 3 |
| 5. Documentation | 7 | Phase 4 |

**Total: 38 tasks across 5 phases**
