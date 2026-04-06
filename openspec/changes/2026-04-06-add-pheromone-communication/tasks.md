# Tasks: Pheromone Communication System

## Phase 1: PheromoneField Core

**Dependencies**: None
**Parallelizable**: No (foundational)

- [ ] 1.1 Create `quantumnematode/env/pheromone.py` with `PheromoneType` enum

  - Values: `FOOD_MARKING`, `ALARM`
  - StrEnum for YAML compatibility

- [ ] 1.2 Create `PheromoneSource` dataclass

  - Fields: `position: tuple[int, int]`, `pheromone_type: PheromoneType`, `strength: float`, `emission_step: int`, `emitter_id: str`

- [ ] 1.3 Create `PheromoneField` class

  - Constructor: `spatial_decay_constant: float`, `temporal_half_life: float`, `max_sources: int`
  - Internal: `_sources: list[PheromoneSource]`, `_max_age: int` (derived from temporal_half_life * 5)

- [ ] 1.4 Implement `add_source(source: PheromoneSource)`

  - Append to `_sources`
  - If len exceeds `max_sources`, remove oldest sources

- [ ] 1.5 Implement `get_concentration(position, current_step) -> float`

  - Superposition of all active sources: `Σ strength * exp(-dist / spatial_decay) * exp(-age * ln(2) / half_life)`
  - Tanh-normalize to [0, 1]
  - Use Manhattan distance (consistent with grid world)

- [ ] 1.6 Implement `get_gradient(position, current_step) -> tuple[float, float]`

  - Central differences: `(C(x+1,y) - C(x-1,y)) / 2, (C(x,y+1) - C(x,y-1)) / 2`
  - Return `(dx, dy)` vector
  - Pattern: same as OxygenField.get_gradient()

- [ ] 1.7 Implement `get_gradient_polar(position, current_step) -> tuple[float, float]`

  - Convert gradient vector to `(magnitude, direction_radians)`
  - Magnitude via tanh normalization
  - Pattern: same as OxygenField.get_gradient_polar()

- [ ] 1.8 Implement `prune(current_step)`

  - Remove sources where `current_step - emission_step > max_age`

- [ ] 1.9 Add comprehensive tests in `tests/.../env/test_pheromone.py`

  - Empty field returns 0 concentration
  - Single source decays with distance
  - Single source decays with age
  - Multiple sources superpose
  - Gradient points toward source
  - Pruning removes expired sources
  - Max sources enforced (oldest removed)

## Phase 2: Environment Integration

**Dependencies**: Phase 1
**Parallelizable**: No

- [ ] 2.1 Create `PheromoneTypeConfig` dataclass in `env/env.py`

  - Fields: `emission_strength: float = 1.0`, `spatial_decay_constant: float = 8.0`, `temporal_half_life: float = 50.0`, `max_sources: int = 100`

- [ ] 2.2 Create `PheromoneParams` dataclass in `env/env.py`

  - Fields: `enabled: bool = False`, `food_marking: PheromoneTypeConfig`, `alarm: PheromoneTypeConfig`
  - Default alarm config: emission_strength=2.0, spatial_decay_constant=5.0, temporal_half_life=20.0, max_sources=50

- [ ] 2.3 Add `pheromones: PheromoneParams` to `DynamicForagingEnvironment.__init__()` and create PheromoneField instances when enabled

  - `self.pheromone_field_food: PheromoneField | None`
  - `self.pheromone_field_alarm: PheromoneField | None`
  - Also update `copy()` to pass `pheromones=self.pheromones` to the new env constructor (same pattern as aerotaxis/thermotaxis)

- [ ] 2.4 Add concentration and gradient methods

  - `get_pheromone_food_concentration(position=None) -> float`
  - `get_pheromone_alarm_concentration(position=None) -> float`
  - `get_pheromone_food_gradient(position=None) -> GradientPolar | None`
  - `get_pheromone_alarm_gradient(position=None) -> GradientPolar | None`
  - Default to agent_pos when position is None (same pattern as get_temperature)

- [ ] 2.5 Add `*_for(agent_id)` variants of pheromone methods

  - `get_pheromone_food_concentration_for(agent_id)`, etc.
  - Resolve agent_id to position, call existing method

- [ ] 2.6 Add emission methods

  - `emit_food_pheromone(position, current_step, emitter_id, strength=None)`
  - `emit_alarm_pheromone(position, current_step, emitter_id, strength=None)`
  - Use config emission_strength as default when strength not provided

- [ ] 2.7 Add `update_pheromone_fields(current_step)` method

  - Prune expired sources on both fields
  - Called once per step from orchestrator

- [ ] 2.8 Export `PheromoneField`, `PheromoneType`, `PheromoneSource`, `PheromoneParams` from `env/__init__.py`

## Phase 3: BrainParams Pheromone Fields

**Dependencies**: None (independent)
**Parallelizable**: Yes (with Phases 1-2)

- [ ] 3.1 Add 8 pheromone fields to `BrainParams` in `brain/arch/_brain.py`

  - Oracle: `pheromone_food_gradient_strength: float | None`, `pheromone_food_gradient_direction: float | None`, `pheromone_alarm_gradient_strength: float | None`, `pheromone_alarm_gradient_direction: float | None`
  - Temporal: `pheromone_food_concentration: float | None`, `pheromone_alarm_concentration: float | None`
  - Derivative: `pheromone_food_dconcentration_dt: float | None`, `pheromone_alarm_dconcentration_dt: float | None`
  - All default to None (no pheromone data in single-agent mode or when pheromones disabled)

- [ ] 3.2 Update `_create_brain_params()` in `agent/agent.py` to populate pheromone fields

  - If pheromone fields exist on env, read concentration/gradient and set in BrainParams
  - Use separate methods (`get_pheromone_food_gradient`, `get_pheromone_alarm_gradient`) — do NOT extend `get_separated_gradients` (keep pheromone field decoupled from food/predator gradient system)
  - Use sensing mode (oracle/temporal/derivative) to determine which fields to populate
  - Oracle: set gradient fields, clear temporal/derivative fields
  - Temporal: set concentration fields, clear gradient fields
  - Derivative: set concentration + dC/dt fields

## Phase 4: Sensing Modules

**Dependencies**: Phase 3
**Parallelizable**: Yes (with Phases 1-2)

- [ ] 4.1 Add `PHEROMONE_FOOD` and `PHEROMONE_ALARM` to `ModuleName` enum

- [ ] 4.2 Add `PHEROMONE_FOOD_TEMPORAL` and `PHEROMONE_ALARM_TEMPORAL` to `ModuleName` enum

- [ ] 4.3 Implement `_pheromone_food_core(params) -> CoreFeatures`

  - Oracle: strength = gradient_strength, angle = relative angle from agent heading
  - Pattern: identical to `_food_chemotaxis_core()`

- [ ] 4.4 Implement `_pheromone_alarm_core(params) -> CoreFeatures`

  - Oracle: strength = gradient_strength, angle = relative angle from agent heading
  - Pattern: identical to `_nociception_core()`

- [ ] 4.5 Implement `_pheromone_food_temporal_core(params) -> CoreFeatures`

  - Temporal: strength = concentration [0,1], angle = tanh(dC/dt * derivative_scale)
  - Pattern: identical to `_food_chemotaxis_temporal_core()`

- [ ] 4.6 Implement `_pheromone_alarm_temporal_core(params) -> CoreFeatures`

  - Temporal: strength = concentration [0,1], angle = tanh(dC/dt * derivative_scale)
  - Pattern: identical to `_nociception_temporal_core()`

- [ ] 4.7 Register all 4 modules in `SENSORY_MODULES` dict

  - Each with appropriate `classical_dim=2`, descriptions referencing C. elegans ASK/ADL neurons

- [ ] 4.8 Update `apply_sensing_mode()` in `config_loader.py`

  - `pheromone_food` → `pheromone_food_temporal` when pheromone_food_mode != ORACLE
  - `pheromone_alarm` → `pheromone_alarm_temporal` when pheromone_alarm_mode != ORACLE

- [ ] 4.9 Add tests in `tests/.../brain/test_pheromone_modules.py`

  - Registration check for all 4 modules
  - Feature extraction with/without pheromone data
  - Classical dim = 2 for all modules
  - Quantum transform produces valid angles

## Phase 5: STAM Extension

**Dependencies**: Phase 3
**Parallelizable**: No

- [ ] 5.1 Make `STAMBuffer.num_channels` configurable (4 or 6)

  - Constructor: `num_channels: int = 4` (backward compat)
  - Remove the hard assertion that num_channels must be 4
  - Derive `MEMORY_DIM = num_channels * 2 + 2 + 1`

- [ ] 5.2 Update channel index constants

  - When num_channels=6: IDX_WEIGHTED_PHEROMONE_FOOD=4, IDX_WEIGHTED_PHEROMONE_ALARM=5, IDX_DERIV_PHEROMONE_FOOD=10, IDX_DERIV_PHEROMONE_ALARM=11, IDX_POS_DELTA_X=12, IDX_POS_DELTA_Y=13, IDX_ACTION_ENTROPY=14
  - Use dynamic indexing based on num_channels rather than hardcoded constants

- [ ] 5.3 Update `record()` to accept variable-length scalars

  - `scalars: np.ndarray` shape (num_channels,) instead of fixed (4,)
  - Validate length matches num_channels

- [ ] 5.4 Update `get_memory_state()` for dynamic dimensions

  - Weighted means: indices [0:num_channels]
  - Derivatives: indices [num_channels:2\*num_channels]
  - Position deltas: indices \[2*num_channels:2*num_channels+2\]
  - Action entropy: index [2\*num_channels+2]

- [ ] 5.5 Update `STAMSensoryModule` in `brain/modules.py`

  - STAMSensoryModule is registered statically at module load time with `classical_dim=11`
  - For 6-channel mode: the agent must override `classical_dim` when creating the module list, based on SensingConfig pheromone state
  - Approach: make STAMSensoryModule read dim from a class-level variable that can be set at agent init, OR register two variants (STAM with dim=11, STAM_PHEROMONE with dim=15), OR derive dim from the actual STAMBuffer instance at extraction time via BrainParams
  - Ensure `to_classical()` returns correct-length array for both 11-dim and 15-dim modes

- [ ] 5.6 Update `_compute_temporal_data()` in `agent/agent.py`

  - Pass pheromone concentrations as channels 4 and 5 when pheromones enabled
  - Record 6-element scalar array to STAM when pheromones enabled

- [ ] 5.7 Add STAM extension tests

  - 4-channel mode produces 11-dim output (backward compat)
  - 6-channel mode produces 15-dim output
  - Pheromone derivatives computed correctly
  - Channel indices correct for both modes

## Phase 6: MultiAgentSimulation Emission Integration

**Dependencies**: Phases 2, 5
**Parallelizable**: No (integrates prior work)

- [ ] 6.1 Add step counter to `run_episode()` for pheromone aging

  - Track `current_step` starting from 0 each episode
  - Note: pheromone fields are implicitly reset between episodes because the env is recreated per episode in `_run_multi_agent`. No explicit pheromone reset needed.

- [ ] 6.2 Emit food-marking pheromone after food consumption in `_resolve_food_step()`

  - For each food winner: `env.emit_food_pheromone(consumed_position, current_step, agent_id)`
  - Only when pheromones enabled

- [ ] 6.3 Emit alarm pheromone after predator damage

  - Capture return value of `apply_predator_damage_for(aid)` — currently discarded in orchestrator
  - When actual_damage > 0: `env.emit_alarm_pheromone(agent_position, current_step, agent_id)`
  - Only when pheromones enabled

- [ ] 6.4 Update pheromone fields each step

  - Call `env.update_pheromone_fields(current_step)` once per step (after emission, before next perception)

- [ ] 6.5 Pass pheromone data to `_create_brain_params()` during perception

  - Ensure pheromone concentration/gradient is read per agent using `*_for(agent_id)` methods

- [ ] 6.6 Add pheromone emission tests

  - Food consumption → food-marking source added to field
  - Predator damage → alarm source added to field
  - Pheromone concentration increases near emission site
  - Pheromone decays over steps

## Phase 7: Configuration Schema

**Dependencies**: Phase 2
**Parallelizable**: Yes (with Phases 5-6)

- [ ] 7.1 Create `PheromoneTypeConfig` and `PheromoneConfig` Pydantic models in `config_loader.py`

  - Mirror the PheromoneTypeConfig/PheromoneParams dataclasses
  - Validation: emission_strength > 0, spatial_decay_constant > 0, temporal_half_life > 0, max_sources > 0

- [ ] 7.2 Add `pheromones: PheromoneConfig | None = None` to `EnvironmentConfig`

- [ ] 7.3 Add `pheromone_food_mode: SensingMode` and `pheromone_alarm_mode: SensingMode` to `SensingConfig`

  - Default: `SensingMode.ORACLE`

- [ ] 7.4 ~~Update `apply_sensing_mode()` for pheromone modules~~ (covered by task 4.8)

- [ ] 7.5 Update `create_env_from_config()` to pass pheromone params to environment

- [ ] 7.6 Add config validation tests

  - Pheromone config with valid/invalid values
  - Sensing mode translation for pheromone modules

## Phase 8: CSV Export + Session Summary (Deferred Items)

**Dependencies**: None (independent)
**Parallelizable**: Yes

- [ ] 8.1 Add `agent_id` to `_SIMULATION_RESULTS_FIELDNAMES` in `report/csv_export.py`

  - Insert as second field (after "run")
  - Update `_simulation_result_to_row()` to include agent_id

- [ ] 8.2 Write per-agent CSV rows in `_run_multi_agent()`

  - After each episode, write one row per agent to simulation_results.csv
  - Include agent_id, steps, reward, termination_reason, foods_collected per agent

- [ ] 8.3 Create `multi_agent_summary.csv` writer

  - One row per episode: run, total_food, competition_events, proximity_events, alive_at_end, mean_success, gini

- [ ] 8.4 Print session summary table at end of `_run_multi_agent()`

  - Total episodes, mean food/episode, mean competition events, mean Gini
  - Per-agent success rates

- [ ] 8.5 Ensure single-agent CSV export is unchanged

  - agent_id column present but value is "default" for backward compat

## Phase 9: Weight Save/Load Round-Trip Test (Deferred Item)

**Dependencies**: None
**Parallelizable**: Yes

- [ ] 9.1 Add test: save 3 multi-agent weights → verify 3 .pt files created

- [ ] 9.2 Add test: load saved weights → verify model state matches original

## Phase 10: Scenario Configurations

**Dependencies**: Phase 7
**Parallelizable**: Yes
**Note**: Minimal configs for functional verification and convergence checks. Comprehensive cross-scenario evaluations deferred to post-Phase 4 completion.

- [ ] 10.1 Create `configs/scenarios/multi_agent_foraging/mlpppo_medium_2agents_pheromone_oracle.yml`

  - Directory: `multi_agent_foraging/` (foraging only, no predators)
  - Naming: `{brain}_{size}_{variant}_{sensing}` = `mlpppo_medium_2agents_pheromone_oracle`
  - 2 agents, MLP PPO, 50x50, oracle pheromone sensing
  - Sensory modules: food_chemotaxis, pheromone_food, pheromone_alarm

- [ ] 10.2 Create `configs/scenarios/multi_agent_pursuit/mlpppo_medium_5agents_pheromone_oracle.yml`

  - Directory: `multi_agent_pursuit/` (NEW directory — pursuit predators + multi-agent)
  - 5 agents, MLP PPO, 50x50, oracle pheromone sensing, pursuit predators
  - Sensory modules: food_chemotaxis, nociception, pheromone_food, pheromone_alarm

- [ ] 10.3 Create `configs/scenarios/multi_agent_foraging/lstmppo_medium_2agents_pheromone_temporal.yml`

  - Directory: `multi_agent_foraging/`
  - 2 agents, LSTM PPO (GRU variant), 50x50, temporal pheromone sensing
  - Sensory modules: food_chemotaxis_temporal, pheromone_food_temporal, pheromone_alarm_temporal, stam

- [ ] 10.4 Create `configs/scenarios/multi_agent_pursuit/lstmppo_medium_2agents_pheromone_temporal.yml`

  - Directory: `multi_agent_pursuit/`
  - 2 agents, LSTM PPO (GRU variant), 50x50, temporal pheromone + nociception, pursuit predators
  - Sensory modules: food_chemotaxis_temporal, nociception_temporal, pheromone_food_temporal, pheromone_alarm_temporal, stam

## Phase 11: Verification

**Dependencies**: All phases
**Parallelizable**: Partially

- [ ] 11.1 Run `uv run pytest -m "not nightly"` — ALL tests pass

- [ ] 11.2 Run `uv run pre-commit run -a` — all hooks pass

- [ ] 11.3 Backward compatibility: existing multi-agent configs without pheromones produce identical behavior

- [ ] 11.4 Sanity check: 2-agent with food-marking pheromones, verify pheromone sources created on food consumption

- [ ] 11.5 Sanity check: multi-agent with alarm pheromones + predators, verify alarm sources created on predator damage

- [ ] 11.6 Update `docs/roadmap.md`

  - Update Phase 4 status with pheromone communication progress

- [ ] 11.7 Update `AGENTS.md`

  - Add `multi_agent_pursuit` to scenario list
  - Add pheromone variant suffix to variant list
  - Update description of multi-agent capabilities

- [ ] 11.8 Update `openspec/config.yaml`

  - Mention pheromone communication in Architecture section

- [ ] 11.9 Update `configs/README.md`

  - Add `multi_agent_foraging/` and `multi_agent_pursuit/` to scenario table

______________________________________________________________________

## Summary

| Phase | Tasks | Dependencies | Parallelizable |
|-------|-------|-------------|----------------|
| 1. PheromoneField Core | 9 | None | No |
| 2. Environment Integration | 8 | Phase 1 | No |
| 3. BrainParams Fields | 2 | None | Yes (with 1-2) |
| 4. Sensing Modules | 9 | Phase 3 | Yes (with 1-2) |
| 5. STAM Extension | 7 | Phase 3 | No |
| 6. Orchestrator Emission | 6 | Phases 2, 5 | No |
| 7. Config Schema | 6 | Phase 2 | Yes (with 5-6) |
| 8. CSV Export + Summary | 5 | None | Yes |
| 9. Weight Round-Trip Test | 2 | None | Yes |
| 10. Scenario Configs | 4 | Phase 7 | Yes |
| 11. Verification + Docs | 9 | All | Partially |

**Total: 67 tasks across 11 phases**
