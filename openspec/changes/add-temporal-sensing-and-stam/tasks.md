## 1. STAM Buffer Module

- [ ] 1.1 Create `packages/quantum-nematode/quantumnematode/agent/stam.py` with `STAMBuffer` class: deque-based circular buffer, precomputed exponential decay weights, `record()`, `reset()`, `compute_temporal_derivative()`, `get_memory_state()` returning fixed 9-float vector, `memory_dimension` property
- [ ] 1.2 Create `tests/quantumnematode_tests/agent/test_stam.py` — unit tests for buffer record/retrieve, decay weights, temporal derivative computation (positive/negative/zero), episode reset, empty/partial buffer edge cases, memory state shape

## 2. Environment Scalar Concentration Methods

- [ ] 2.1 Add `get_food_concentration(position)` to `env.py` — sum of exponential decay magnitudes from all food sources, reusing existing decay parameters
- [ ] 2.2 Add `get_predator_concentration(position)` to `env.py` — same pattern for predators, returns 0.0 when predators disabled
- [ ] 2.3 Add unit tests for scalar concentration methods in existing env test file — single source, multiple sources, consistency with gradient computation, food collection updates

## 3. BrainParams Extensions

- [ ] 3.1 Add new optional fields to `BrainParams` in `_brain.py`: `food_concentration`, `predator_concentration`, `food_dconcentration_dt`, `predator_dconcentration_dt`, `temperature_ddt`, `stam_state` — all defaulting to `None`

## 4. Sensing Configuration

- [ ] 4.1 Add `SensingMode` enum and `SensingConfig` Pydantic model to `config_loader.py` with fields: `chemotaxis_mode`, `thermotaxis_mode`, `nociception_mode`, `stam_enabled`, `stam_buffer_size`, `stam_decay_rate`
- [ ] 4.2 Add `sensing: SensingConfig | None = None` to `EnvironmentConfig` with `get_sensing_config()` accessor
- [ ] 4.3 Add `_apply_sensing_mode()` function to auto-replace oracle module names with temporal variants and append STAM module when enabled
- [ ] 4.4 Add config validation: invalid sensing mode error, temporal-without-STAM warning, buffer_size > 0, decay_rate > 0
- [ ] 4.5 Add config loader tests — SensingConfig parsing, defaults, validation errors, module translation

## 5. Temporal Sensory Modules

- [ ] 5.1 Add `FOOD_CHEMOTAXIS_TEMPORAL`, `NOCICEPTION_TEMPORAL`, `THERMOTAXIS_TEMPORAL`, `STAM` to `ModuleName` enum in `modules.py`
- [ ] 5.2 Implement `_food_chemotaxis_temporal_core()` — strength from normalized food_concentration, angle from food_dconcentration_dt
- [ ] 5.3 Implement `_nociception_temporal_core()` — strength from normalized predator_concentration, angle from predator_dconcentration_dt
- [ ] 5.4 Implement `_thermotaxis_temporal_core()` — strength from temp deviation, angle from temperature_ddt, binary from temp deviation (classical_dim=3)
- [ ] 5.5 Implement `STAMSensoryModule` subclass — `to_classical()` returns full 9-float state, `to_quantum()` compresses to 3-float summary, `classical_dim=9`
- [ ] 5.6 Register all new modules in `SENSORY_MODULES` dict
- [ ] 5.7 Create `tests/quantumnematode_tests/brain/test_temporal_modules.py` — each temporal module returns correct features, STAM module output shape, quantum/classical transforms, None-field handling

## 6. Agent Integration

- [ ] 6.1 Accept `SensingConfig` in `QuantumNematodeAgent` constructor, create `STAMBuffer` when `stam_enabled`
- [ ] 6.2 Modify `_create_brain_params()` to conditionally populate fields based on sensing mode: oracle (existing), temporal (scalar only), derivative (scalar + dC/dt from STAM)
- [ ] 6.3 Call `stam.record()` before building BrainParams each step, set `stam_state` from `stam.get_memory_state()`
- [ ] 6.4 Reset STAM buffer in `prepare_episode()`
- [ ] 6.5 Wire `SensingConfig` from loaded config through runner to agent construction

## 7. Example Configurations

- [ ] 7.1 Create `configs/examples/mlpppo_foraging_small_temporal.yml` — Mode A chemotaxis + STAM, 20×20 grid
- [ ] 7.2 Create `configs/examples/mlpppo_thermotaxis_foraging_small_temporal.yml` — Mode B thermotaxis + temporal chemotaxis + STAM
- [ ] 7.3 Create `configs/examples/mlpppo_pursuit_predators_small_temporal.yml` — Mode A nociception + chemotaxis + STAM, pursuit predators
- [ ] 7.4 Create `configs/examples/mlpppo_thermotaxis_pursuit_predators_small_temporal.yml` — all three modalities temporal + STAM, pursuit predators

## 8. Verification

- [ ] 8.1 Run `uv run pytest -m "not nightly"` — all existing + new tests pass
- [ ] 8.2 Run `uv run pre-commit run -a` — lint and format pass
- [ ] 8.3 Run simulation with existing oracle config (`mlpppo_foraging_small.yml`) — verify behavior unchanged
- [ ] 8.4 Run simulation with temporal config (`mlpppo_foraging_small_temporal.yml`) — verify it runs, observe performance gap vs oracle
