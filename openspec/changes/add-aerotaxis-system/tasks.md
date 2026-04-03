## 1. OxygenField Class

- [ ] 1.1 Add `OxygenSpot` type alias to `quantumnematode/dtypes.py` (parallel to `TemperatureSpot`)
- [ ] 1.2 Create `quantumnematode/env/oxygen.py` with `OxygenZone` enum (5 zones: LETHAL_HYPOXIA, DANGER_HYPOXIA, COMFORT, DANGER_HYPEROXIA, LETHAL_HYPEROXIA)
- [ ] 1.3 Implement `OxygenZoneThresholds` dataclass with absolute percentage thresholds (defaults: lethal_hypoxia_upper=2.0, danger_hypoxia_upper=5.0, comfort_lower=5.0, comfort_upper=12.0, danger_hyperoxia_upper=17.0)
- [ ] 1.4 Implement `OxygenField` dataclass with `get_oxygen(position)` (linear gradient + high/low spots with exponential decay, clamped to [0.0, 21.0])
- [ ] 1.5 Implement `OxygenField.get_gradient(position)` via central difference and `get_gradient_polar(position)` returning (magnitude, direction)
- [ ] 1.6 Implement `OxygenField.get_zone(oxygen, thresholds)` classifying O2 percentage into OxygenZone
- [ ] 1.7 Export `OxygenField`, `OxygenZone`, `OxygenZoneThresholds` from `quantumnematode/env/__init__.py` (parallel to TemperatureField exports)

## 2. Environment Integration

- [ ] 2.1 Add default constants for oxygen to `quantumnematode/env/env.py` (DEFAULT_BASE_OXYGEN=10.0, DEFAULT_OXYGEN_GRADIENT_STRENGTH=0.1, default penalties/damage matching thermotaxis)
- [ ] 2.2 Create `AerotaxisParams` dataclass in `env.py` (parallel to `ThermotaxisParams`: enabled, base_oxygen, gradient_direction, gradient_strength, high/low_oxygen_spots, spot_decay_constant, zone thresholds, comfort_reward, danger_penalty, danger_hp_damage, lethal_hp_damage, reward_discomfort_food)
- [ ] 2.3 Add `aerotaxis: AerotaxisParams | None = None` parameter to `DynamicForagingEnvironment.__init__()` signature, store as `self.aerotaxis`, create `self._oxygen_field: OxygenField | None` from AerotaxisParams when enabled
- [ ] 2.4 Implement `get_oxygen() -> float | None` and `get_oxygen_gradient() -> GradientPolar | None` methods on the environment
- [ ] 2.5 Implement `get_oxygen_zone() -> OxygenZone | None` method using OxygenField.get_zone()
- [ ] 2.6 Implement `apply_oxygen_effects() -> tuple[float, float]` with zone-based rewards/penalties and HP damage (parallel to `apply_temperature_effects()`)
- [ ] 2.7 Add oxygen comfort score tracking: `steps_in_oxygen_comfort_zone`, `total_aerotaxis_steps`, `get_oxygen_comfort_score()`, `reset_aerotaxis()`
- [ ] 2.8 Integrate oxygen gradient into `get_separated_gradients()` when aerotaxis enabled in oracle mode
- [ ] 2.9 Implement `get_oxygen_concentration() -> float | None` for scalar O2 at agent position (raw percentage, NOT tanh-normalized)

## 3. Configuration Layer

- [ ] 3.1 Create `AerotaxisConfig` Pydantic BaseModel in `quantumnematode/utils/config_loader.py` (parallel to `ThermotaxisConfig`, all AerotaxisParams fields, `enabled: bool = False`)
- [ ] 3.2 Add `aerotaxis: AerotaxisConfig | None = None` to `EnvironmentConfig` and `get_aerotaxis_config()` method
- [ ] 3.3 Add `aerotaxis_mode: SensingMode = SensingMode.ORACLE` to `SensingConfig`
- [ ] 3.4 Update `apply_sensing_mode()` to handle `aerotaxis → aerotaxis_temporal` substitution when `aerotaxis_mode != ORACLE`
- [ ] 3.5 Update `validate_sensing_config()` to include `aerotaxis_mode` in derivative/temporal mode checks

## 4. BrainParams and Sensory Modules

- [ ] 4.1 Add oxygen fields to `BrainParams` in `quantumnematode/brain/arch/_brain.py`: `oxygen_concentration`, `oxygen_gradient_strength`, `oxygen_gradient_direction`, `oxygen_dconcentration_dt` (all `float | None = None`)
- [ ] 4.2 Add `AEROTAXIS = "aerotaxis"` and `AEROTAXIS_TEMPORAL = "aerotaxis_temporal"` to `ModuleName` enum in `modules.py`
- [ ] 4.3 Implement `_aerotaxis_core(params)` in `modules.py` returning CoreFeatures (strength=tanh(gradient_strength), angle=egocentric relative angle, binary=clip((O2-8.5)/12.5, -1, 1))
- [ ] 4.4 Implement `_aerotaxis_temporal_core(params)` in `modules.py` returning CoreFeatures (strength=abs deviation, angle=tanh(dO2/dt * derivative_scale), binary=signed deviation)
- [ ] 4.5 Register both modules in `SENSORY_MODULES` dict with `classical_dim=3`, `transform_type="standard"`

## 5. STAM Expansion (3 → 4 Channels)

- [ ] 5.1 Update `STAMBuffer` in `quantumnematode/agent/stam.py`: `MEMORY_DIM = 11`, `num_channels` default to 4, update validation assertion
- [ ] 5.2 Add named indices: `IDX_WEIGHTED_OXYGEN = 3`, `IDX_DERIV_OXYGEN = 7`, shift `IDX_POS_DELTA_X` to 8, `IDX_POS_DELTA_Y` to 9, `IDX_ACTION_ENTROPY` to 10
- [ ] 5.3 Update `get_memory_state()` for 4-channel layout (weighted means [0:4], derivatives [4:8], pos deltas [8:10], entropy [10])
- [ ] 5.4 Update `STAMSensoryModule.classical_dim` from 9 to 11 in `modules.py`
- [ ] 5.5 Update `STAMSensoryModule.to_classical()` to return 11-float vector
- [ ] 5.6 Update `STAMSensoryModule.to_quantum()` compression: mean of [0:4], mean of [4:8], entropy at [10]

## 6. Agent Integration

- [ ] 6.1 Update `_create_brain_params()` in `quantumnematode/agent/agent.py` to populate `oxygen_concentration`, `oxygen_gradient_strength`, `oxygen_gradient_direction` when aerotaxis enabled (oracle mode: include gradients; temporal/derivative: scalar only)
- [ ] 6.2 Update `_compute_temporal_data()` to add oxygen as STAM channel 3 (scalars array: `[food, temp, pred, oxygen]`), compute `oxygen_dconcentration_dt` for derivative mode
- [ ] 6.3 Update STAM initialization in agent to use `num_channels=4`
- [ ] 6.4 Call `apply_oxygen_effects()` in the step loop in `quantumnematode/agent/runners.py` (parallel to `apply_temperature_effects()`)
- [ ] 6.5 Add oxygen brave foraging bonus in `runners.py` food collection handler (parallel to thermotaxis `reward_discomfort_food` at lines 266-282): apply bonus when food collected in oxygen danger zone
- [ ] 6.6 Add `reset_aerotaxis()` call in environment reset path
- [ ] 6.7 Update `reset_environment()` in `agent.py` to pass `aerotaxis=self.env.aerotaxis` to preserve oxygen field config across episode resets (parallel to `thermotaxis=self.env.thermotaxis` at line 722)

## 7. Experiment Tracking and Data Pipeline

- [ ] 7.1 Add `oxygen_history: list[float]` field to `EpisodeData` dataclass in `agent/runners.py` (parallel to `temperature_history` at line 61) AND to `EpisodeTrackingData` in `report/dtypes.py` (parallel to line 162)
- [ ] 7.2 Add `track_oxygen(oxygen: float)` method to `EpisodeTracker` in `agent/tracker.py` (parallel to `track_temperature` at line 138), add `oxygen_history` property, and include `oxygen_history=[]` in `__init__()` reset
- [ ] 7.3 Add `oxygen_history: list[float] | None = None` and `oxygen_comfort_score: float | None = None` to `SimulationResult` in `report/dtypes.py`
- [ ] 7.4 Add `oxygen_history` field to runner step data (`agent/runners.py`): call `track_oxygen()` each step when aerotaxis enabled (parallel to `track_temperature` at line 443)
- [ ] 7.5 Populate `oxygen_history` and `oxygen_comfort_score` in `run_simulation.py`: (a) copy oxygen_history from tracker (parallel to temperature at lines 621-626), (b) compute oxygen_comfort_score from env (parallel to lines 639-640), (c) include both in `SimulationResult` constructor, (d) include oxygen_history in `EpisodeTrackingData` construction in the `track_per_run` block (parallel to temperature_history at line 734)
- [ ] 7.6 Add `oxygen_comfort_score` to `_SIMULATION_RESULTS_FIELDNAMES` and `_simulation_result_to_row()` in `report/csv_export.py`
- [ ] 7.7 Add `oxygen_history.csv` export in per-run data export in `report/csv_export.py` (parallel to `temperature_history.csv` at lines 902-909)
- [ ] 7.8 Add oxygen metrics (final_oxygen, mean_oxygen, min_oxygen, max_oxygen) to `foraging_summary.csv` in `report/csv_export.py` (parallel to temperature metrics at lines 948-957)
- [ ] 7.9 Add `oxygen_progression.png` plot in `report/plots.py` (parallel to `temperature_progression.png` at lines 483-512)
- [ ] 7.10 Add `oxygen_comfort_score: float | None` to `PerRunResult` in `experiment/metadata.py`
- [ ] 7.11 Add `avg_oxygen_comfort_score: float | None` and `post_convergence_oxygen_comfort_score: float | None` to `ResultsMetadata` in `experiment/metadata.py`
- [ ] 7.12 Add oxygen comfort score aggregation in `experiment/tracker.py` (parallel to temperature_comfort_score at lines 339-343)
- [ ] 7.13 Add `result.oxygen_history = None` in the per-step data flushing section of `run_simulation.py` (parallel to temperature_history flushing at line 780)

## 8. Visualization

- [ ] 8.1 Add oxygen zone overlay colors to `quantumnematode/env/sprites.py` — ZONE_LETHAL_HYPOXIA (180,40,40,90), ZONE_DANGER_HYPOXIA (200,80,60,70), ZONE_COMFORT_OXYGEN (transparent), ZONE_DANGER_HYPEROXIA (80,200,220,70), ZONE_LETHAL_HYPEROXIA (40,180,220,90) — and update `create_zone_overlay()` color_map
- [ ] 8.2 Add `_render_oxygen_zones(env, viewport)` method to `PygameRenderer` in `pygame_renderer.py` (parallel to `_render_temperature_zones()`)
- [ ] 8.3 Insert `_render_oxygen_zones()` call in render pipeline between temperature zones and toxic zones
- [ ] 8.4 Update `render_frame()` signature to accept `oxygen: float | None` and `oxygen_zone_name: str | None`, update `_render_status_bar()` to display O2 reading
- [ ] 8.5 Add oxygen zone symbols to `ThemeSymbolSet` and oxygen zone colors to `DarkColorRichStyleConfig` in `theme.py`
- [ ] 8.6 Update `_render_step_pygame()` in `agent.py` to fetch oxygen value and zone name from environment and pass to renderer

## 9. Scenario Configurations

- [ ] 9.1 Create `configs/scenarios/oxygen_foraging/mlpppo_medium_oracle.yml` (50×50, aerotaxis+food, oracle sensing)
- [ ] 9.2 Create `configs/scenarios/oxygen_foraging/mlpppo_large_oracle.yml` (100×100, aerotaxis+food with high/low O2 spots)
- [ ] 9.3 Create `configs/scenarios/oxygen_foraging/mlpppo_medium_temporal.yml` (50×50, temporal sensing for both chemotaxis and aerotaxis)
- [ ] 9.4 Create `configs/scenarios/oxygen_pursuit/mlpppo_large_oracle.yml` (100×100, aerotaxis+food+pursuit predators)
- [ ] 9.5 Create `configs/scenarios/oxygen_stationary/mlpppo_large_oracle.yml` (100×100, aerotaxis+food+stationary predators)
- [ ] 9.6 Create `configs/scenarios/oxygen_thermal_foraging/mlpppo_large_oracle.yml` (100×100, combined O2+thermal+food, orthogonal gradients)
- [ ] 9.7 Create `configs/scenarios/oxygen_thermal_foraging/mlpppo_large_temporal.yml` (100×100, combined temporal sensing)
- [ ] 9.8 Create `configs/scenarios/oxygen_thermal_foraging/lstmppo_large_temporal.yml` (100×100, LSTM PPO with temporal sensing)
- [ ] 9.9 Create `configs/scenarios/oxygen_thermal_pursuit/mlpppo_large_oracle.yml` (100×100, combined+pursuit predators)
- [ ] 9.10 Create `configs/scenarios/oxygen_thermal_pursuit/lstmppo_large_temporal.yml`
- [ ] 9.11 Create `configs/scenarios/oxygen_thermal_stationary/mlpppo_large_oracle.yml` (100×100, combined+stationary predators)
- [ ] 9.12 Create `configs/scenarios/oxygen_thermal_stationary/lstmppo_large_temporal.yml`

## 10. Tests

- [ ] 10.1 Create `tests/quantumnematode_tests/env/test_oxygen.py` with TestOxygenField (base oxygen, linear gradient, high/low spots, clamping), TestOxygenZones (all 5 zones, boundary cases), TestOxygenGradient (central difference, polar conversion)
- [ ] 10.2 Add oxygen environment integration tests to `test_env.py` (AerotaxisParams initialization, get_oxygen, get_oxygen_gradient, apply_oxygen_effects rewards/HP)
- [ ] 10.3 Add combined thermal+oxygen environment tests to `test_env.py` (both fields coexist, effects are additive)
- [ ] 10.4 Update STAM tests in `test_stam.py` for 4-channel buffer (11-dim state, oxygen channel recording, oxygen derivative)
- [ ] 10.5 Add aerotaxis module tests to `test_modules.py` (oracle and temporal feature extraction, classical_dim=3, quantum output)
- [ ] 10.6 Add or update config loader tests for AerotaxisConfig parsing, aerotaxis_mode in SensingConfig, apply_sensing_mode with aerotaxis
- [ ] 10.7 Run full test suite: `uv run pytest -m "not nightly"` — all pass
- [ ] 10.8 Run linting: `uv run pre-commit run -a` — clean

## 11. Skills and Documentation

- [ ] 11.1 Update `.claude/skills/nematode-evaluate/skill.md`: add aerotaxis diagnostic at line 66 ("**Aerotaxis**: Check for oxygen-related deaths, comfort zone (5-12% O2) navigation"), update guardrail at line 88 to mention aerotaxis columns
- [ ] 11.2 Update `README.md`: add aerotaxis feature entry (line ~19), add oxygen zone visual description to pixel theme table (line ~287), update status bar description (line ~292), mention aerotaxis in sensory systems list (line ~349)
- [ ] 11.3 Update `AGENTS.md` line 38: add new scenario directories to Scenarios list (oxygen_foraging, oxygen_pursuit, oxygen_stationary, oxygen_thermal_foraging, oxygen_thermal_pursuit, oxygen_thermal_stationary)
- [ ] 11.4 Update `docs/roadmap.md`: mark oxygen sensing stretch goal as completed in Phase 3 exit criteria, update deferred items

## 12. Verification

- [ ] 12.1 Smoke test: run a short oxygen_foraging simulation (`uv run ./scripts/run_simulation.py --config ./configs/scenarios/oxygen_foraging/mlpppo_medium_oracle.yml --runs 3 --theme headless`)
- [ ] 12.2 Combined scenario test: run oxygen_thermal_foraging to verify multi-field coexistence
- [ ] 12.3 Backward compatibility: run existing thermal_foraging and plain foraging configs — unchanged behavior
- [ ] 12.4 STAM regression: run an existing temporal config (e.g., `thermal_pursuit/mlpppo_small_temporal.yml`) — confirm it still works with 4-channel STAM
- [ ] 12.5 Verify CSV exports: check that oxygen_history.csv and oxygen_comfort_score appear in export output for oxygen scenarios
