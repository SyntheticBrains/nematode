## 1. STAM Buffer Module

- [x] 1.1 Create `packages/quantum-nematode/quantumnematode/agent/stam.py` with `STAMBuffer` class: deque-based circular buffer, precomputed exponential decay weights, `record(scalars, position_delta, action)` (position_delta is step-to-step dx/dy, NOT absolute coordinates), `reset()`, `compute_temporal_derivative(channel)`, `get_memory_state()` returning fixed 9-float vector (3 weighted means, 3 derivatives, 2 position deltas from step-to-step movements, 1 action entropy), `memory_dimension` property. Document biological timescale mapping in comments (buffer_size=30 with decay_rate=0.1 ≈ 10-15s biological STAM). Document action entropy as computational convenience, not biologically motivated.
- [x] 1.2 Add `STAMBuffer` to `agent/__init__.py` exports
- [x] 1.3 Create `tests/quantumnematode_tests/agent/test_stam.py` — unit tests for buffer record/retrieve, decay weights, temporal derivative computation (positive/negative/zero), episode reset, empty/partial buffer edge cases, memory state shape, position deltas use step-to-step differences not absolute coordinates

## 2. Environment Scalar Concentration Methods

- [x] 2.1 Add `get_food_concentration(position)` to `env.py` — sum of exponential decay magnitudes from all food sources, normalized via `tanh(raw * GRADIENT_SCALING_TANH_FACTOR)` to [0, 1], reusing existing decay parameters
- [x] 2.2 Add `get_predator_concentration(position)` to `env.py` — same pattern for predators with predator-specific parameters, normalized via tanh, returns 0.0 when predators disabled
- [x] 2.3 Add unit tests for scalar concentration methods in existing env test file — single source, multiple sources, tanh normalization range, consistency with gradient magnitude normalization, food collection updates

## 3. BrainParams Extensions

- [x] 3.1 Add new optional fields to `BrainParams` in `_brain.py`: `food_concentration`, `predator_concentration`, `food_dconcentration_dt`, `predator_dconcentration_dt`, `temperature_ddt`, `stam_state` — all defaulting to `None`

## 4. Sensing Configuration

- [x] 4.1 Add `SensingMode` enum and `SensingConfig` Pydantic model to `config_loader.py` with fields: `chemotaxis_mode`, `thermotaxis_mode`, `nociception_mode`, `stam_enabled`, `stam_buffer_size`, `stam_decay_rate`
- [x] 4.2 Add `sensing: SensingConfig | None = None` to `EnvironmentConfig` with `get_sensing_config()` accessor
- [x] 4.3 Add `_apply_sensing_mode()` function to auto-replace oracle module names with temporal variants, handle `chemotaxis` → `food_chemotaxis_temporal` + nociception split, and append STAM module when enabled
- [x] 4.4 Add config validation: invalid sensing mode error, temporal-without-STAM warning, derivative-mode auto-enables STAM with info log, buffer_size > 0, decay_rate > 0
- [x] 4.5 Add config loader tests — SensingConfig parsing, defaults, validation errors, module translation including combined chemotaxis split, derivative-mode STAM auto-enable

## 5. Temporal Sensory Modules

- [x] 5.1 Add `FOOD_CHEMOTAXIS_TEMPORAL`, `NOCICEPTION_TEMPORAL`, `THERMOTAXIS_TEMPORAL`, `STAM` to `ModuleName` enum in `modules.py`
- [x] 5.2 Implement `_food_chemotaxis_temporal_core()` — strength directly from food_concentration (already tanh-normalized by env, do NOT re-normalize), angle from `tanh(food_dconcentration_dt)` to clamp derivative to [-1, 1]
- [x] 5.3 Implement `_nociception_temporal_core()` — strength directly from predator_concentration (already tanh-normalized by env), angle from `tanh(predator_dconcentration_dt)`
- [x] 5.4 Implement `_thermotaxis_temporal_core()` — strength from temp deviation, angle from `tanh(temperature_ddt)`, binary from temp deviation (classical_dim=3)
- [x] 5.5 Implement `STAMSensoryModule` subclass — `to_classical()` returns full 9-float state, `to_quantum()` compresses to 3-float summary, `classical_dim=9`
- [x] 5.6 Register all new modules in `SENSORY_MODULES` dict
- [x] 5.7 Create `tests/quantumnematode_tests/brain/test_temporal_modules.py` — each temporal module returns correct features, STAM module output shape, quantum/classical transforms, None-field handling

## 6. Agent Integration

- [x] 6.1 Accept `SensingConfig` in `QuantumNematodeAgent` constructor, create `STAMBuffer` when `stam_enabled`
- [x] 6.2 Modify `_create_brain_params()` to follow this step-0-safe ordering: (a) get scalar concentrations from env, (b) compute position delta from previous position, (c) record to STAM, (d) get temporal derivatives from STAM, (e) build BrainParams with all fields. Conditionally populate based on sensing mode: oracle (existing gradient fields), temporal (scalar concentration only), derivative (scalar + dC/dt from STAM)
- [x] 6.3 Reset STAM buffer in `prepare_episode()`
- [x] 6.4 Wire `SensingConfig` from loaded config through runner to agent construction — trace the config→runner→agent path and modify as needed

## 7. Example Configurations

- [x] 7.1 Create `configs/examples/mlpppo_foraging_small_temporal.yml` — Mode A chemotaxis + STAM, 20×20 grid. Include comments documenting sensing mode and biological calibration notes
- [x] 7.2 Create `configs/examples/mlpppo_thermotaxis_foraging_small_temporal.yml` — Mode B thermotaxis + temporal chemotaxis + STAM
- [x] 7.3 Create `configs/examples/mlpppo_pursuit_predators_small_temporal.yml` — Mode A nociception + chemotaxis + STAM, pursuit predators
- [x] 7.4 Create `configs/examples/mlpppo_thermotaxis_pursuit_predators_small_temporal.yml` — all three modalities temporal + STAM, pursuit predators

## 8. Verification

- [x] 8.1 Run `uv run pytest -m "not nightly"` — all existing + new tests pass
- [x] 8.2 Run `uv run pre-commit run -a` — lint and format pass
- [x] 8.3 Run simulation with existing oracle config (`mlpppo_foraging_small.yml`) — verify behavior unchanged
- [x] 8.4 Run simulation with temporal config (`mlpppo_foraging_small_temporal.yml`) — verify it runs successfully
- [x] 8.5 Add one `@pytest.mark.smoke` test that runs a temporal sensing config end-to-end via CLI (short episode, verify no crash)
