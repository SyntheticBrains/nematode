## 1. Sensing Mode and Config

- [ ] 1.1 Add `KLINOTAXIS = "klinotaxis"` to `SensingMode` enum in `config_loader.py`
- [ ] 1.2 Add `lateral_scale: float = Field(default=50.0, gt=0.0)` to `SensingConfig`
- [ ] 1.3 Refine `apply_sensing_mode()` to use explicit mode matching: `== KLINOTAXIS` → `*_klinotaxis`, `!= ORACLE and != KLINOTAXIS` → `*_temporal`
- [ ] 1.4 Update `validate_sensing_config()` to auto-enable STAM when any modality uses klinotaxis
- [ ] 1.5 Add tests for SensingMode.KLINOTAXIS enum, apply_sensing_mode substitution, STAM auto-enable

## 2. BrainParams Extension

- [ ] 2.1 Add 7 lateral gradient fields to BrainParams: food, predator, temperature, oxygen, pheromone_food, pheromone_alarm, pheromone_aggregation (all `float | None = None`)
- [ ] 2.2 Add `lateral_scale: float = 50.0` field to BrainParams
- [ ] 2.3 Add tests verifying default None values and field presence

## 3. Lateral Sampling in Agent

- [ ] 3.1 Add `_compute_lateral_offsets(direction, position, grid_size)` helper function returning `(left_pos, right_pos)` tuples for all 4 directions + STAY
- [ ] 3.2 Add `_last_heading: Direction` tracking on agent, updated when direction != STAY, default UP
- [ ] 3.3 Add lateral concentration queries in `_compute_temporal_data()` for each modality when mode is KLINOTAXIS — query left/right positions, compute `right - left`
- [ ] 3.4 Store lateral gradients in result dict (`food_lateral_gradient`, etc.)
- [ ] 3.5 Pass `lateral_scale` from SensingConfig to BrainParams in `_create_brain_params()`
- [ ] 3.6 Ensure klinotaxis mode triggers STAM derivative computation (same as DERIVATIVE mode)
- [ ] 3.7 Add tests for `_compute_lateral_offsets()` — all directions, STAY fallback, edge clamping
- [ ] 3.8 Add integration test: klinotaxis mode populates lateral gradient fields in BrainParams
- [ ] 3.9 Add test: klinotaxis mode suppresses oracle gradient fields (food_gradient_strength/direction = None)

## 4. Klinotaxis Sensory Modules

- [ ] 4.1 Add 7 ModuleName entries: FOOD_CHEMOTAXIS_KLINOTAXIS, NOCICEPTION_KLINOTAXIS, THERMOTAXIS_KLINOTAXIS, AEROTAXIS_KLINOTAXIS, PHEROMONE_FOOD_KLINOTAXIS, PHEROMONE_ALARM_KLINOTAXIS, PHEROMONE_AGGREGATION_KLINOTAXIS
- [ ] 4.2 Add extraction functions for each module (strength=concentration, angle=lateral gradient, binary=dC/dt) with classical_dim=3
- [ ] 4.3 Register all 7 modules in SENSORY_MODULES dict
- [ ] 4.4 Handle thermotaxis klinotaxis: strength=|temp_deviation/15|, angle=tanh((temp_right - temp_left) / 15.0 * lateral_scale), binary=dT/dt — normalize lateral gradient by same divisor (15.0) as center value
- [ ] 4.5 Handle aerotaxis klinotaxis: strength=O2 concentration, angle=tanh((o2_right - o2_left) / 21.0 * lateral_scale), binary=dO2/dt — normalize lateral gradient by 21.0 to match O2 percentage scale
- [ ] 4.6 Update `_infer_stam_dim_from_modules()` to include klinotaxis variants in modality_pairs and pheromone_pairs (each pair becomes a triple: oracle, temporal, klinotaxis)
- [ ] 4.7 Add tests: module registration, feature extraction, classical_dim=3, None field handling, STAM dim inference with klinotaxis modules

## 5. Single-Agent Evaluation Configs

- [ ] 5.1 Create `foraging/lstmppo_small_klinotaxis.yml` — 20×20, food chemotaxis klinotaxis
- [ ] 5.2 Create `pursuit/lstmppo_small_klinotaxis.yml` — 20×20, food + predator klinotaxis
- [ ] 5.3 Create `thermal_pursuit/lstmppo_large_klinotaxis.yml` — 100×100, food + predator + temperature klinotaxis
- [ ] 5.4 Create `thermal_stationary/lstmppo_large_klinotaxis.yml` — 100×100, food + predator + temperature klinotaxis
- [ ] 5.5 Create `oxygen_foraging/lstmppo_large_klinotaxis.yml` — 100×100, food + oxygen klinotaxis
- [ ] 5.6 Create `oxygen_thermal_foraging/lstmppo_large_klinotaxis.yml` — 100×100, food + temperature + oxygen klinotaxis

## 6. Multi-Agent Pheromone Evaluation Configs

- [ ] 6.1 Create `multi_agent_foraging/lstmppo_medium_5agents_hotspot_pheromone_klinotaxis.yml` — hotspots, ALL pheromones klinotaxis
- [ ] 6.2 Create `multi_agent_foraging/lstmppo_medium_5agents_hotspot_no_pheromone_klinotaxis.yml` — hotspots, no pheromones (control)
- [ ] 6.3 Create `multi_agent_foraging/lstmppo_medium_5agents_pheromone_klinotaxis.yml` — no hotspots, ALL pheromones (control)

## 7. Documentation

- [ ] 7.1 Update `README.md` — add klinotaxis to Temporal Sensing feature, perception modes table
- [ ] 7.2 Update `AGENTS.md` — add `_klinotaxis` to sensing suffixes
- [ ] 7.3 Update `CONTRIBUTING.md` — add klinotaxis example in testing section
- [ ] 7.4 Update `docs/roadmap.md` — Phase 4 sensing model notes, klinotaxis as resolution for issue #125
- [ ] 7.5 Update `.claude/skills/nematode-run-experiments/skill.md` — add klinotaxis episode guidance

## 8. Verification

- [ ] 8.1 All existing tests pass (`uv run pytest -m "not nightly"`)
- [ ] 8.2 Pre-commit hooks pass (`uv run pre-commit run -a`)
- [ ] 8.3 Sanity: foraging small klinotaxis runs 100 episodes without error
- [ ] 8.4 Sanity: multi-agent hotspot klinotaxis runs 50 episodes without error
