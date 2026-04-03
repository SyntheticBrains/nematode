## 1. OxygenField Class

- [ ] 1.1 Add `OxygenSpot` type alias to `quantumnematode/dtypes.py` (parallel to `TemperatureSpot`)
- [ ] 1.2 Create `quantumnematode/env/oxygen.py` with `OxygenZone` enum (5 zones: LETHAL_HYPOXIA, DANGER_HYPOXIA, COMFORT, DANGER_HYPEROXIA, LETHAL_HYPEROXIA)
- [ ] 1.3 Implement `OxygenZoneThresholds` dataclass with absolute percentage thresholds (defaults: lethal_hypoxia_upper=2.0, danger_hypoxia_upper=5.0, comfort_lower=5.0, comfort_upper=12.0, danger_hyperoxia_upper=17.0)
- [ ] 1.4 Implement `OxygenField` dataclass with `get_oxygen(position)` (linear gradient + high/low spots with exponential decay, clamped to [0.0, 21.0])
- [ ] 1.5 Implement `OxygenField.get_gradient(position)` via central difference and `get_gradient_polar(position)` returning (magnitude, direction)
- [ ] 1.6 Implement `OxygenField.get_zone(oxygen, thresholds)` classifying O2 percentage into OxygenZone

## 2. Environment Integration

- [ ] 2.1 Add default constants for oxygen to `quantumnematode/env/env.py` (DEFAULT_BASE_OXYGEN=10.0, DEFAULT_OXYGEN_GRADIENT_STRENGTH=0.1, default penalties/damage matching thermotaxis)
- [ ] 2.2 Create `AerotaxisParams` dataclass in `env.py` (parallel to `ThermotaxisParams`: enabled, base_oxygen, gradient_direction, gradient_strength, high/low_oxygen_spots, spot_decay_constant, zone thresholds, reward/penalty/HP damage params)
- [ ] 2.3 Add `_oxygen_field: OxygenField | None` attribute and initialization to `DynamicForagingEnvironment.__init__()`, creating OxygenField from AerotaxisParams when enabled
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
- [ ] 6.5 Add `reset_aerotaxis()` call in environment reset path

## 7. Visualization

- [ ] 7.1 Add oxygen zone overlay colors to `quantumnematode/env/sprites.py` (ZONE_LETHAL_HYPOXIA, ZONE_DANGER_HYPOXIA, ZONE_COMFORT_OXYGEN, ZONE_DANGER_HYPEROXIA, ZONE_LETHAL_HYPEROXIA) and update `create_zone_overlay()` color_map
- [ ] 7.2 Add `_render_oxygen_zones(env, viewport)` method to `PygameRenderer` in `pygame_renderer.py` (parallel to `_render_temperature_zones()`)
- [ ] 7.3 Insert `_render_oxygen_zones()` call in render pipeline between temperature zones and toxic zones
- [ ] 7.4 Update `render_frame()` signature to accept `oxygen: float | None` and `oxygen_zone_name: str | None`, update `_render_status_bar()` to display O2 reading
- [ ] 7.5 Add oxygen zone symbols to `ThemeSymbolSet` and oxygen zone colors to `DarkColorRichStyleConfig` in `theme.py`
- [ ] 7.6 Update `_render_step_pygame()` in `agent.py` to fetch oxygen value and zone name from environment and pass to renderer

## 8. Scenario Configurations

- [ ] 8.1 Create `configs/scenarios/oxygen_foraging/mlpppo_medium_oracle.yml` (50×50, aerotaxis+food, oracle sensing)
- [ ] 8.2 Create `configs/scenarios/oxygen_foraging/mlpppo_large_oracle.yml` (100×100, aerotaxis+food with high/low O2 spots)
- [ ] 8.3 Create `configs/scenarios/oxygen_foraging/mlpppo_medium_temporal.yml` (50×50, temporal sensing for both chemotaxis and aerotaxis)
- [ ] 8.4 Create `configs/scenarios/oxygen_pursuit/mlpppo_large_oracle.yml` (100×100, aerotaxis+food+pursuit predators)
- [ ] 8.5 Create `configs/scenarios/oxygen_stationary/mlpppo_large_oracle.yml` (100×100, aerotaxis+food+stationary predators)
- [ ] 8.6 Create `configs/scenarios/oxygen_thermal_foraging/mlpppo_large_oracle.yml` (100×100, combined O2+thermal+food, orthogonal gradients)
- [ ] 8.7 Create `configs/scenarios/oxygen_thermal_foraging/mlpppo_large_temporal.yml` (100×100, combined temporal sensing)
- [ ] 8.8 Create `configs/scenarios/oxygen_thermal_foraging/lstmppo_large_temporal.yml` (100×100, LSTM PPO with temporal sensing)
- [ ] 8.9 Create `configs/scenarios/oxygen_thermal_pursuit/mlpppo_large_oracle.yml` (100×100, combined+pursuit predators)
- [ ] 8.10 Create `configs/scenarios/oxygen_thermal_pursuit/lstmppo_large_temporal.yml`
- [ ] 8.11 Create `configs/scenarios/oxygen_thermal_stationary/mlpppo_large_oracle.yml` (100×100, combined+stationary predators)
- [ ] 8.12 Create `configs/scenarios/oxygen_thermal_stationary/lstmppo_large_temporal.yml`

## 9. Tests

- [ ] 9.1 Create `tests/quantumnematode_tests/env/test_oxygen.py` with TestOxygenField (base oxygen, linear gradient, high/low spots, clamping), TestOxygenZones (all 5 zones, boundary cases), TestOxygenGradient (central difference, polar conversion)
- [ ] 9.2 Add oxygen environment integration tests to `test_env.py` (AerotaxisParams initialization, get_oxygen, get_oxygen_gradient, apply_oxygen_effects rewards/HP)
- [ ] 9.3 Add combined thermal+oxygen environment tests to `test_env.py` (both fields coexist, effects are additive)
- [ ] 9.4 Update STAM tests in `test_stam.py` for 4-channel buffer (11-dim state, oxygen channel recording, oxygen derivative)
- [ ] 9.5 Add aerotaxis module tests to `test_modules.py` (oracle and temporal feature extraction, classical_dim=3, quantum output)
- [ ] 9.6 Add or update config loader tests for AerotaxisConfig parsing, aerotaxis_mode in SensingConfig, apply_sensing_mode with aerotaxis
- [ ] 9.7 Run full test suite: `uv run pytest -m "not nightly"` — all pass
- [ ] 9.8 Run linting: `uv run pre-commit run -a` — clean

## 10. Skills and Documentation

- [ ] 10.1 Update `.claude/skills/nematode-evaluate/skill.md`: add aerotaxis diagnostic at line 66 ("**Aerotaxis**: Check for oxygen-related deaths, comfort zone (5-12% O2) navigation"), update guardrail at line 88 to mention aerotaxis columns
- [ ] 10.2 Update `README.md`: add aerotaxis feature entry (line ~19), add oxygen zone visual description to pixel theme table (line ~287), update status bar description (line ~292), mention aerotaxis in sensory systems list (line ~349)
- [ ] 10.3 Update `AGENTS.md` line 38: add new scenario directories to Scenarios list (oxygen_foraging, oxygen_pursuit, oxygen_stationary, oxygen_thermal_foraging, oxygen_thermal_pursuit, oxygen_thermal_stationary)
- [ ] 10.4 Update `docs/roadmap.md`: mark oxygen sensing stretch goal as completed in Phase 3 exit criteria, update deferred items

## 11. Verification

- [ ] 11.1 Smoke test: run a short oxygen_foraging simulation (`uv run ./scripts/run_simulation.py --config ./configs/scenarios/oxygen_foraging/mlpppo_medium_oracle.yml --runs 3 --theme headless`)
- [ ] 11.2 Combined scenario test: run oxygen_thermal_foraging to verify multi-field coexistence
- [ ] 11.3 Backward compatibility: run existing thermal_foraging and plain foraging configs — unchanged behavior
- [ ] 11.4 STAM regression: run an existing temporal config (e.g., `thermal_pursuit/mlpppo_small_temporal.yml`) — confirm it still works with 4-channel STAM
