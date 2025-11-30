# Implementation Tasks

## Phase 1: Trajectory Learning (Priority - 1-2 days)

### Task 1.1: Add Episode Buffer to ModularBrain
- [ ] Add `EpisodeBuffer` dataclass to store params, actions, rewards per step
- [ ] Add episode buffer initialization in `ModularBrain.__init__` when `use_trajectory_learning=True`
- [ ] Modify `run_brain()` to append to buffer instead of immediate learning
- [ ] Add buffer clearing logic at episode boundaries
- **Validation**: Unit test episode data accumulation
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py`

### Task 1.2: Implement Discounted Return Computation
- [ ] Add `compute_discounted_returns(rewards: list[float], gamma: float) -> list[float]` method
- [ ] Implement backward iteration: `G_t = r_t + gamma * G_{t+1}`
- [ ] Handle terminal states correctly (G_T = r_T)
- [ ] Add input validation for gamma range [0, 1]
- **Validation**: Unit tests for various reward sequences and gamma values
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py`

### Task 1.3: Trajectory Parameter-Shift Gradients
- [ ] Create `trajectory_parameter_shift_gradients()` method
- [ ] Re-evaluate circuits for each timestep's parameters (both shifted)
- [ ] Accumulate gradient contributions: `sum_t (P_+ - P_-) * G_t`
- [ ] Maintain mathematical equivalence to single-step for validation
- **Validation**: Test gradient linearity property, compare with analytical results
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py`

### Task 1.4: Integrate Learning Trigger
- [ ] Modify `post_process_episode()` to compute returns and update parameters
- [ ] Add trajectory update call in episode completion path
- [ ] Ensure buffer is cleared after update
- [ ] Add timing/logging for trajectory updates
- **Validation**: Integration test with full episode execution
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py`, `packages/quantum-nematode/quantumnematode/agent/runners.py`

### Task 1.5: Configuration Schema Extension
- [ ] Add `use_trajectory_learning: bool` to ModularBrainConfig
- [ ] Add `gamma: float` with default 0.99 and validation
- [ ] Add optional `baseline_alpha: float` for future variance reduction
- [ ] Add optional `normalize_returns: bool` for future normalization
- [ ] Update config validation and error messages
- **Validation**: Config parsing tests, validation error tests
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py`

### Task 1.6: Testing Suite for Trajectory Learning
- [ ] Test: `test_episode_buffer_accumulation()` - verify correct data storage
- [ ] Test: `test_discounted_returns_simple()` - verify G_t computation
- [ ] Test: `test_discounted_returns_terminal()` - verify terminal state handling
- [ ] Test: `test_trajectory_gradients_linearity()` - verify gradient math
- [ ] Test: `test_backward_compatibility_single_step()` - verify flag=False behavior
- [ ] Test: `test_trajectory_learning_convergence()` - benchmark integration test
- **Validation**: All tests pass, coverage >90%
- **Files**: `packages/quantum-nematode/tests/quantumnematode_tests/brain/test_modular.py`

### Task 1.7: Documentation for Trajectory Learning
- [ ] Update ModularBrain docstring with trajectory learning description
- [ ] Add example config snippet for trajectory learning
- [ ] Document mathematical background (parameter-shift + returns)
- [ ] Update CLAUDE.md with trajectory learning patterns
- **Validation**: Documentation review, example configs tested
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py`, `CLAUDE.md`

## Phase 2: Appetitive/Aversive Modules (After Phase 1 evaluation - 1-2 days)

### Task 2.1: Module Name Migration
- [ ] Rename `ModuleName.CHEMOTAXIS` to `ModuleName.APPETITIVE` in modules.py
- [ ] Add `ModuleName.AVERSIVE` to enum
- [ ] Add backward compatibility alias: `CHEMOTAXIS = APPETITIVE` with deprecation warning
- [ ] Update feature registry to map "appetitive" → appetitive_features
- **Validation**: Existing configs still work with deprecation warning
- **Files**: `packages/quantum-nematode/quantumnematode/brain/modules.py`

### Task 2.2: Implement Aversive Feature Extraction
- [ ] Create `aversive_features(params: BrainParams) -> dict[RotationAxis, float]`
- [ ] Extract abs(predator_gradient_strength) for RX (scaled to [-π/2, π/2])
- [ ] Compute relative angle to predator for RY (same logic as appetitive)
- [ ] Use RZ = 0.0 (mirrors appetitive, reserved for future)
- [ ] Register in feature extraction mapping
- [ ] Ensure biologically accurate: only use gradient data, no external state
- **Validation**: Unit tests for feature encoding, value ranges, biological accuracy
- **Files**: `packages/quantum-nematode/quantumnematode/brain/modules.py`

### Task 2.3: Gradient Mode Configuration
- [ ] Add `gradient_mode: str` to environment config with validation ("unified" | "split")
- [ ] Default gradient_mode to "unified" for backward compatibility
- [ ] Add validation error for invalid gradient_mode values
- **Validation**: Config validation tests
- **Files**: `packages/quantum-nematode/quantumnematode/env/env.py`, config dtypes

### Task 2.4: Split Gradient Computation in Environment
- [ ] Modify `get_state()` to compute food and predator gradients separately when mode="split"
- [ ] Track food_vector_x, food_vector_y independently
- [ ] Track predator_vector_x, predator_vector_y independently
- [ ] Compute magnitudes and directions for both
- [ ] Pass both gradient sets to BrainParams
- **Validation**: Test unified mode preserves current behavior, split mode separates correctly
- **Files**: `packages/quantum-nematode/quantumnematode/env/env.py`

### Task 2.5: BrainParams Extension for Split Gradients
- [ ] Add optional `food_gradient_strength: float | None`
- [ ] Add optional `food_gradient_direction: float | None`
- [ ] Add optional `predator_gradient_strength: float | None`
- [ ] Add optional `predator_gradient_direction: float | None`
- [ ] Maintain backward compatibility with existing gradient fields
- [ ] Ensure all new fields represent biologically realistic sensory data
- **Validation**: Type checking passes, backward compatible with existing code
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py` or similar
- **Note**: Removed predator_proximity and danger_flag (not biologically realistic)

### Task 2.6: Config Migration for Existing Files
- [ ] Update `configs/examples/modular_dynamic_small.yml` - rename chemotaxis → appetitive
- [ ] Update `configs/examples/modular_dynamic_medium.yml`
- [ ] Update `configs/examples/modular_dynamic_large.yml`
- [ ] Update all predator config variants
- [ ] Add example 4-qubit config with appetitive + aversive modules
- [ ] Set gradient_mode: unified in existing configs (explicit default)
- **Validation**: All updated configs load successfully
- **Files**: `configs/examples/*.yml` (10+ files)

### Task 2.7: Testing Suite for Modular Architecture
- [ ] Test: `test_appetitive_module_features()` - verify food gradient encoding
- [ ] Test: `test_aversive_module_features()` - verify predator gradient encoding
- [ ] Test: `test_gradient_mode_unified()` - verify superposition behavior
- [ ] Test: `test_gradient_mode_split()` - verify separate computation
- [ ] Test: `test_module_qubit_allocation()` - verify no overlap
- [ ] Test: `test_backward_compatibility_chemotaxis()` - verify alias works
- [ ] Test: `test_4qubit_appetitive_aversive()` - integration test
- **Validation**: All tests pass, coverage >90%
- **Files**: `packages/quantum-nematode/tests/quantumnematode_tests/brain/test_modules.py`, environment tests

### Task 2.8: Documentation for Modular Architecture
- [ ] Update modules.py docstring explaining appetitive/aversive distinction
- [ ] Add biological context (C. elegans appetitive vs aversive learning)
- [ ] Document gradient mode configuration and when to use split vs unified
- [ ] Create example 4-qubit predator config with comments
- **Validation**: Documentation review
- **Files**: `packages/quantum-nematode/quantumnematode/brain/modules.py`, `docs/`, `configs/examples/`

## Phase 3: Integration and Benchmarking (After Phase 2 - 1 day)

### Task 3.1: Combined Feature Benchmark
- [ ] Create benchmark config with trajectory_learning=True, appetitive+aversive, gradient_mode=split
- [ ] Run 200-episode benchmark on dynamic_predator_small environment
- [ ] Compare with baseline (current best quantum: 26.5%, classical MLP: 85%)
- [ ] Analyze convergence, success rate, predator encounters
- **Validation**: Success rate > 60% target, documented in benchmark JSON
- **Files**: New benchmark result in `benchmarks/dynamic_predator_small/quantum/`

### Task 3.2: Ablation Studies
- [ ] Benchmark: trajectory learning only (no aversive module)
- [ ] Benchmark: aversive module only (no trajectory learning)
- [ ] Benchmark: both features combined
- [ ] Document performance contribution of each feature
- **Validation**: Clear performance attribution, regression if individual features < combined
- **Files**: Multiple benchmark results, analysis document

### Task 3.3: Performance Documentation
- [ ] Document memory overhead (episode buffer size)
- [ ] Document computational cost (4-qubit vs 2-qubit timing)
- [ ] Document recommended shot counts for 2-qubit vs 4-qubit
- [ ] Create performance tuning guide
- **Validation**: Documented and added to project docs
- **Files**: `docs/performance.md` or similar

## Dependencies and Parallelization

### Can be done in parallel:
- Task 1.1-1.3 (core trajectory learning implementation)
- Task 1.5 (config schema) - independent of implementation
- Task 2.1-2.2 (module renaming and aversive features) - independent of trajectory learning

### Must be sequential:
- Task 1.4 depends on 1.1-1.3 (integration requires core implementation)
- Task 1.6 depends on 1.1-1.4 (tests require complete implementation)
- Task 2.4 depends on 2.3 (gradient computation requires config)
- Task 2.6 depends on 2.1 (config migration requires enum changes)
- Task 3.1-3.3 depend on all of Phase 1 and Phase 2 (benchmarking requires complete features)

### Critical path:
1. Phase 1 core (Tasks 1.1-1.4) → Phase 1 validation (Tasks 1.6-1.7)
2. Evaluate Phase 1 benchmark results
3. Decide whether to proceed with Phase 2
4. Phase 2 core (Tasks 2.1-2.5) → Phase 2 migration (Tasks 2.6-2.8)
5. Phase 3 integration and benchmarking

## Validation Criteria

### Phase 1 Success Criteria:
- [ ] All trajectory learning tests pass (>90% coverage)
- [ ] Benchmark with trajectory_learning=True shows +20-30% success rate improvement
- [ ] No regression in non-predator environments
- [ ] Backward compatibility verified (trajectory_learning=False matches current behavior)

### Phase 2 Success Criteria:
- [ ] All modular architecture tests pass (>90% coverage)
- [ ] Existing configs work with deprecation warning
- [ ] New 4-qubit configs load and execute correctly
- [ ] Gradient separation verified (split mode != unified mode in behavior)

### Phase 3 Success Criteria:
- [ ] Combined features achieve >60% success rate target
- [ ] Ablation studies show additive/complementary effects
- [ ] Performance overhead documented and acceptable
- [ ] All documentation updated and reviewed
