# Tasks: Social Dynamics (Social Feeding, Aggregation Pheromones, Collective Metrics)

## Phase 1: Social Feeding Mechanics

**Dependencies**: None
**Parallelizable**: No (foundational)

- [ ] 1.1 Add `SocialFeedingParams` dataclass to `env/env.py`

  - Fields: `enabled: bool = False`, `decay_reduction: float = 0.7`, `solitary_decay: float = 1.0`

- [ ] 1.2 Wire `SocialFeedingParams` into `DynamicForagingEnvironment.__init__()` and `copy()`

  - `self.social_feeding: SocialFeedingParams` attribute
  - Pass through in `copy()` (same pattern as PheromoneParams)

- [ ] 1.3 Add `decay_multiplier` parameter to `SatietyManager.decay_satiety()`

  - Signature: `decay_satiety(self, multiplier: float = 1.0)`
  - Multiply the decay amount by the multiplier before applying
  - Default 1.0 preserves backward compatibility

- [ ] 1.4 Add `SocialFeedingConfig` to `config_loader.py`

  - Pydantic model with validation (decay_reduction > 0)
  - `to_params() -> SocialFeedingParams` conversion method
  - Wire into `EnvironmentConfig` as optional field

- [ ] 1.5 Add `social_phenotype` to `AgentConfig` in `config_loader.py`

  - `social_phenotype: Literal["social", "solitary"] = "social"`

- [ ] 1.6 Apply social feeding decay reduction in `multi_agent.py`

  - In `run_episode()` section 5 (EFFECTS), compute decay_multiplier from social_feeding config and nearby_per_agent
  - Store agent phenotypes as `_agent_phenotypes: dict[str, str]` on MultiAgentSimulation
  - Social phenotype: use `decay_reduction` when nearby > 0
  - Solitary phenotype: use `solitary_decay` when nearby > 0
  - Pass multiplier to `agent._satiety_manager.decay_satiety(multiplier=decay_mult)`

- [ ] 1.7 Track `social_feeding_events` counter in MultiAgentSimulation

  - Increment when decay reduction is applied (multiplier != 1.0 and nearby > 0)

- [ ] 1.8 Export `SocialFeedingParams` from `env/__init__.py`

- [ ] 1.9 Unit tests for social feeding

  - Decay reduction applied when nearby agents > 0
  - No reduction when nearby agents = 0
  - Solitary phenotype uses solitary_decay multiplier
  - Backward compatible: disabled by default, decay_satiety(1.0) = original behavior
  - Config validation tests

## Phase 2: Aggregation Pheromone Infrastructure

**Dependencies**: None (PheromoneField already exists)
**Parallelizable**: Yes (with Phase 1)

- [ ] 2.1 Add `AGGREGATION = "aggregation"` to `PheromoneType` enum in `env/pheromone.py`

- [ ] 2.2 Add `aggregation: PheromoneTypeConfig` to `PheromoneParams` in `env/env.py`

  - Defaults: `emission_strength=0.5, spatial_decay_constant=10.0, temporal_half_life=10.0, max_sources=200`

- [ ] 2.3 Create `pheromone_field_aggregation: PheromoneField | None` in `DynamicForagingEnvironment.__init__()`

  - Created when pheromones enabled and aggregation config present
  - Update `copy()` to include aggregation field

- [ ] 2.4 Add aggregation pheromone methods to environment

  - `emit_aggregation_pheromone(position, current_step, emitter_id, strength=None)`
  - `get_pheromone_aggregation_concentration(position, current_step) -> float`
  - `get_pheromone_aggregation_gradient(position, current_step) -> tuple[float, float] | None`
  - `get_pheromone_aggregation_gradient_polar(position, current_step) -> tuple[float, float] | None`
  - `*_for(agent_id)` variants

- [ ] 2.5 Update `update_pheromone_fields()` to prune aggregation field

- [ ] 2.6 Add continuous aggregation emission in `multi_agent.py` step loop

  - After movement (section 2), for each alive agent, emit aggregation pheromone at agent's current position
  - Only when pheromones enabled and aggregation field exists

- [ ] 2.7 Add 4 BrainParams fields for aggregation sensing in `brain/arch/_brain.py`

  - Oracle: `pheromone_aggregation_gradient_strength: float | None`, `pheromone_aggregation_gradient_direction: float | None`
  - Temporal: `pheromone_aggregation_concentration: float | None`, `pheromone_aggregation_dconcentration_dt: float | None`

- [ ] 2.8 Add `PHEROMONE_AGGREGATION` and `PHEROMONE_AGGREGATION_TEMPORAL` sensing modules in `brain/modules.py`

  - Oracle: gradient strength + relative angle (classical_dim=2)
  - Temporal: concentration + tanh(dC/dt * derivative_scale) (classical_dim=2)
  - Pattern: identical to PHEROMONE_FOOD / PHEROMONE_FOOD_TEMPORAL

- [ ] 2.9 Populate aggregation pheromone fields in `agent/agent.py`

  - `_create_brain_params()`: read aggregation gradient/concentration like food/alarm pheromones
  - `_compute_temporal_data()`: include aggregation pheromone concentration in temporal buffer

- [ ] 2.10 Extend STAM for 7-channel mode in `agent/stam.py`

  - `CHANNELS_PHEROMONE_FULL = 7`
  - `IDX_PHEROMONE_AGGREGATION = 6`
  - Update channel detection: 7 channels when aggregation pheromone configured

- [ ] 2.11 Update `config_loader.py` for aggregation pheromone

  - Add `aggregation: PheromoneTypeConfigYAML | None` to `PheromoneConfig`
  - Add `pheromone_aggregation_mode: SensingMode` to `SensingConfig`
  - Update `apply_sensing_mode()` for aggregation module translation
  - Update `validate_sensing_config()` to include aggregation mode

- [ ] 2.12 Unit tests for aggregation pheromone

  - AGGREGATION type in PheromoneType enum
  - Aggregation field created when configured
  - Continuous emission produces detectable concentration
  - Gradient points toward cluster of emitters
  - Sensing modules produce correct features
  - STAM 7-channel mode produces 17-dim memory state
  - Config validation

## Phase 3: Collective Behavior Metrics

**Dependencies**: Phases 1 and 2
**Parallelizable**: No (integrates prior work)

- [ ] 3.1 Extend `MultiAgentEpisodeResult` with 4 new metric fields

  - `social_feeding_events: int`
  - `aggregation_index: float`
  - `alarm_evasion_events: int`
  - `food_sharing_events: int`

- [ ] 3.2 Implement `aggregation_index` computation

  - Per-step: compute mean normalized inverse pairwise distance for all alive agents
  - Accumulate per step, average at episode end
  - 0.0 when < 2 agents alive

- [ ] 3.3 Implement `alarm_evasion_events` tracking

  - Track per-agent alarm pheromone concentration from previous step
  - After movement, if previous alarm concentration > threshold AND agent moved away from alarm gradient, increment counter
  - Threshold: configurable, default 0.1

- [ ] 3.4 Implement `food_sharing_events` tracking

  - Buffer recent food-marking emissions: (position, step, emitter_id)
  - After movement, check if any non-emitter agent moved within detection_radius of a recent emission (within lookback window, default 20 steps)
  - Increment counter for each such event, then remove the emission from buffer

- [ ] 3.5 Wire metrics into `_build_result()` in multi_agent.py

- [ ] 3.6 Update `csv_export.py` with new summary columns

  - Add `social_feeding_events`, `aggregation_index`, `alarm_evasion_events`, `food_sharing_events` to multi_agent_summary fieldnames

- [ ] 3.7 Unit tests for collective metrics

  - Aggregation index = 0 when agents far apart, > 0 when clustered, = 1 when co-located
  - Alarm evasion correctly detected when agent moves away from alarm source
  - Food sharing events counted when non-emitter approaches emission site
  - Social feeding events match counter from Phase 1

## Phase 4: Configuration and Integration

**Dependencies**: All prior phases
**Parallelizable**: Partially

- [ ] 4.1 Create `configs/scenarios/multi_agent_foraging/mlpppo_medium_5agents_social_oracle.yml`

  - 5 agents, social feeding enabled, no aggregation pheromone
  - Tests social feeding in isolation

- [ ] 4.2 Create `configs/scenarios/multi_agent_foraging/mlpppo_medium_5agents_aggregation_oracle.yml`

  - 5 agents, aggregation pheromone enabled, no social feeding
  - Tests aggregation pheromone in isolation

- [ ] 4.3 Create `configs/scenarios/multi_agent_foraging/mlpppo_medium_5agents_full_social_oracle.yml`

  - 5 agents, social feeding + all pheromones (food-marking, alarm, aggregation)
  - Full social dynamics stack

- [ ] 4.4 Run `uv run pytest -m "not nightly"` — all tests pass

- [ ] 4.5 Run `uv run pre-commit run -a` — all hooks pass

- [ ] 4.6 Update `docs/roadmap.md` Phase 4 status

- [ ] 4.7 Update `AGENTS.md` with social dynamics info

- [ ] 4.8 Update `openspec/config.yaml` Architecture section

- [ ] 4.9 Update `configs/README.md` with new scenario configs

______________________________________________________________________

## Summary

| Phase | Tasks | Dependencies | Parallelizable |
|-------|-------|-------------|----------------|
| 1. Social Feeding | 9 | None | No |
| 2. Aggregation Pheromone | 12 | None | Yes (with 1) |
| 3. Collective Metrics | 7 | Phases 1, 2 | No |
| 4. Config & Integration | 9 | All | Partially |

**Total: 37 tasks across 4 phases**
