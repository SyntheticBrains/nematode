# Implementation Tasks

## Status: Implemented but Deferred

**Summary**: Core trajectory learning and appetitive/aversive modules have been fully implemented and tested. However, evaluations showed no performance improvement over baseline. The feature remains available behind config flags for potential future experimentation.

**Evaluation Results**:
- Trajectory learning: No measurable improvement in success rate or convergence
- Appetitive/Aversive 4-qubit: No improvement over 2-qubit chemotaxis baseline
- Both features add complexity without demonstrated benefit in current environments

**Decision**: Keep implementation dormant (flags default to false) for future research.

---

## Phase 1: Trajectory Learning (Priority - 1-2 days)

### Task 1.1: Add Episode Buffer to ModularBrain
- [x] Add `EpisodeBuffer` dataclass to store params, actions, rewards per step
- [x] Add episode buffer initialization in `ModularBrain.__init__` when `use_trajectory_learning=True`
- [x] Modify `run_brain()` to append to buffer instead of immediate learning
- [x] Add buffer clearing logic at episode boundaries
- **Validation**: Unit test episode data accumulation ✓
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py` (lines 147-184)

### Task 1.2: Implement Discounted Return Computation
- [x] Add `compute_discounted_returns(rewards: list[float], gamma: float) -> list[float]` method
- [x] Implement backward iteration: `G_t = r_t + gamma * G_{t+1}`
- [x] Handle terminal states correctly (G_T = r_T)
- [x] Add input validation for gamma range [0, 1]
- **Validation**: Unit tests for various reward sequences and gamma values ✓
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py` (lines 787-822)

### Task 1.3: Trajectory Parameter-Shift Gradients
- [x] Create `trajectory_parameter_shift_gradients()` method
- [x] Re-evaluate circuits for each timestep's parameters (both shifted)
- [x] Accumulate gradient contributions: `sum_t (P_+ - P_-) * G_t`
- [x] Maintain mathematical equivalence to single-step for validation
- **Validation**: Test gradient linearity property ✓
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py`

### Task 1.4: Integrate Learning Trigger
- [x] Modify `post_process_episode()` to compute returns and update parameters
- [x] Add trajectory update call in episode completion path
- [x] Ensure buffer is cleared after update
- [x] Add timing/logging for trajectory updates
- **Validation**: Integration test with full episode execution ✓
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py` (lines 881-969)

### Task 1.5: Configuration Schema Extension
- [x] Add `use_trajectory_learning: bool` to ModularBrainConfig
- [x] Add `gamma: float` with default 0.99 and validation
- [x] Add `learn_only_from_success: bool` for selective learning
- [x] Add optional `normalize_returns` via `_normalize_returns()` method
- [x] Update config validation and error messages
- **Validation**: Config parsing tests ✓
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py` (lines 128-131)

### Task 1.6: Testing Suite for Trajectory Learning
- [x] Test: `test_trajectory_learning_config()` - verify config parsing
- [x] Test: `test_trajectory_brain_initialization()` - verify brain init with trajectory
- [x] Test: `test_episode_buffer_accumulation()` - verify correct data storage
- [x] Test: `test_discounted_returns_simple()` - verify G_t computation
- [x] Test: `test_discounted_returns_terminal()` - verify terminal state handling
- [x] Test: `test_discounted_returns_gamma_validation()` - verify gamma validation
- [x] Test: `test_discounted_returns_empty_rewards()` - edge case
- [x] Test: `test_trajectory_learning_convergence()` - benchmark integration test
- [x] Test: `test_episode_buffer_clearing()` - verify buffer cleanup
- [x] Test: `test_trajectory_gradient_with_varying_returns()` - gradient math
- **Validation**: All tests pass ✓
- **Files**: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_modular.py` (lines 466-650+)

### Task 1.7: Documentation for Trajectory Learning
- [x] Update ModularBrain docstring with trajectory learning description
- [x] Add example config snippet for trajectory learning
- [x] Document mathematical background (parameter-shift + returns)
- [x] Update CLAUDE.md with trajectory learning patterns (not needed - dormant feature)
- **Validation**: Documentation included in modular.py docstring ✓
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/modular.py` (lines 1-68)

### Task 1.8: Example Configuration
- [x] Create trajectory learning config for predators environment
- **Files**: `configs/examples/modular_predators_small_trajectory.yml`

## Phase 2: Appetitive/Aversive Modules (After Phase 1 evaluation - 1-2 days)

### Task 2.1: Module Name Migration
- [x] Add `ModuleName.APPETITIVE` to enum
- [x] Add `ModuleName.AVERSIVE` to enum
- [x] Update feature registry to map modules → extractors
- [x] Add backward compatibility alias: `CHEMOTAXIS = APPETITIVE` (not done - kept separate)
- **Note**: Kept CHEMOTAXIS as separate module (uses combined gradient), APPETITIVE uses food-only gradient
- **Files**: `packages/quantum-nematode/quantumnematode/brain/modules.py` (lines 306-319)

### Task 2.2: Implement Appetitive Feature Extraction
- [x] Create `appetitive_features(params: BrainParams) -> dict[RotationAxis, float]`
- [x] Extract food_gradient_strength for RX (scaled to [-π/2, π/2])
- [x] Compute relative angle to food for RY
- [x] Use RZ = 0.0
- [x] Register in feature extraction mapping
- [x] Ensure biologically accurate: only use gradient data, no external state
- **Validation**: Unit tests for feature encoding, value ranges ✓
- **Files**: `packages/quantum-nematode/quantumnematode/brain/modules.py` (lines 156-210)

### Task 2.3: Implement Aversive Feature Extraction
- [x] Create `aversive_features(params: BrainParams) -> dict[RotationAxis, float]`
- [x] Extract predator_gradient_strength for RX (threat level)
- [x] Compute relative escape direction for RY
- [x] Use RZ = 0.0
- [x] Register in feature extraction mapping
- **Validation**: Unit tests for feature encoding, value ranges ✓
- **Files**: `packages/quantum-nematode/quantumnematode/brain/modules.py` (lines 213-267)

### Task 2.4: Gradient Mode Configuration
- [x] Add `use_separated_gradients: bool` to environment config
- [x] Default to false for backward compatibility
- **Files**: `packages/quantum-nematode/quantumnematode/utils/config_loader.py` (line 164)

### Task 2.5: Split Gradient Computation in Environment
- [x] Modify agent to compute food and predator gradients separately when enabled
- [x] Track food_vector_x, food_vector_y independently
- [x] Track predator_vector_x, predator_vector_y independently
- [x] Pass both gradient sets to BrainParams
- **Files**: `packages/quantum-nematode/quantumnematode/agent/agent.py` (lines 129-156, 356+)

### Task 2.6: BrainParams Extension for Split Gradients
- [x] Add `food_gradient_strength: float | None`
- [x] Add `food_gradient_direction: float | None`
- [x] Add `predator_gradient_strength: float | None`
- [x] Add `predator_gradient_direction: float | None`
- [x] Maintain backward compatibility with existing gradient fields
- **Files**: `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py` (lines 137-150+)

### Task 2.7: Testing Suite for Modular Architecture
- [x] Test: `test_appetitive_deterministic()` - verify deterministic features
- [x] Test: `test_aversive_deterministic()` - verify deterministic features
- [x] Test: `test_appetitive_rx_bounded()` - verify value ranges
- [x] Test: `test_appetitive_ry_bounded()` - verify value ranges
- [x] Test: `test_appetitive_rz_bounded()` - verify value ranges
- [x] Test: `test_aversive_rx_bounded()` - verify value ranges
- [x] Test: `test_aversive_ry_bounded()` - verify value ranges
- [x] Test: `test_aversive_rz_bounded()` - verify value ranges
- **Validation**: All tests pass ✓
- **Files**: `packages/quantum-nematode/tests/quantumnematode_tests/brain/test_modules.py` (lines 566-742)

### Task 2.8: Example Configuration
- [x] Create 4-qubit appetitive+aversive config for predators environment
- **Files**: `configs/examples/modular_predators_small_appetitive_aversive.yml`

### Task 2.9: Documentation for Modular Architecture
- [x] Add biological context (C. elegans appetitive vs aversive learning) in module docstrings
- [x] Document feature extraction in appetitive_features() and aversive_features()
- **Files**: `packages/quantum-nematode/quantumnematode/brain/modules.py`

## Phase 3: Integration and Benchmarking (After Phase 2 - 1 day)

**Status: Completed but showed no improvement**

### Task 3.1: Combined Feature Benchmark
- [x] Created benchmark configs with trajectory_learning and appetitive+aversive
- [x] Ran evaluations on dynamic_predator_small environment
- [x] Compared with baseline (current best quantum: 88% evolved, classical MLP: 92%)
- **Result**: No improvement observed - both features perform at or below baseline
- **Files**: `configs/examples/modular_predators_small_trajectory.yml`, `configs/examples/modular_predators_small_appetitive_aversive.yml`

### Task 3.2: Ablation Studies
- [x] Benchmark: trajectory learning only (no aversive module) - no improvement
- [x] Benchmark: aversive module only (no trajectory learning) - no improvement
- [x] Benchmark: both features combined - no improvement
- **Conclusion**: Neither feature provides measurable benefit in current environment setup

### Task 3.3: Performance Documentation
- [x] Document memory overhead (episode buffer size) - deferred, feature dormant
- [x] Document computational cost (4-qubit vs 2-qubit timing) - deferred
- [x] Document recommended shot counts - deferred
- **Reason**: Feature kept dormant; documentation deferred until activation needed

## Summary of Implementation Status

### Fully Implemented and Tested ✓
1. **EpisodeBuffer** - Complete dataclass for trajectory data storage
2. **compute_discounted_returns()** - Backward iteration with gamma validation
3. **trajectory_parameter_shift_gradients()** - Trajectory-level gradient computation
4. **post_process_episode()** - Integration with trajectory learning path
5. **appetitive_features()** - Food-seeking feature extraction
6. **aversive_features()** - Predator-avoidance feature extraction
7. **Separated gradients** - food_gradient_* and predator_gradient_* in BrainParams
8. **Test coverage** - All trajectory and appetitive/aversive tests passing

### Config Options (defaults off)
- `use_trajectory_learning: false` - Trajectory REINFORCE learning
- `gamma: 0.99` - Discount factor for returns
- `learn_only_from_success: false` - Selective learning from successful episodes
- `use_separated_gradients: false` - Split food/predator gradient computation

### Not Implemented (Deferred)
- CHEMOTAXIS → APPETITIVE backward compatibility alias (kept as separate modules)
- Detailed performance/memory documentation
- CLAUDE.md updates for dormant features
