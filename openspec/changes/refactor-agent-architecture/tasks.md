# Tasks: Refactor Agent Architecture

## Phase 1: Foundation - Data Transfer Objects and Protocols

### Task 1.1: Create DTOs for component interfaces
- [ ] Create `StepResult` dataclass in `quantumnematode/agent.py`
- [ ] Create `FoodConsumptionResult` dataclass in `quantumnematode/agent.py`
- [ ] Create `EpisodeResult` dataclass in `quantumnematode/agent.py`
- [ ] Add type hints and docstrings
- [ ] Validate with pyright and ruff

**Dependencies**: None
**Validation**: Type checking passes, DTOs are importable
**Parallelizable**: No (foundation for other tasks)

### Task 1.2: Create EpisodeRunner protocol
- [ ] Define `EpisodeRunner` Protocol class with `run()` method signature
- [ ] Add comprehensive docstring explaining the protocol
- [ ] Validate with pyright for protocol compliance

**Dependencies**: Task 1.1
**Validation**: Protocol compiles, pyright accepts protocol
**Parallelizable**: No

## Phase 2: Component Extraction - Independent Classes

### Task 2.1: Implement SatietyManager
- [ ] Create `SatietyManager` class in new file `quantumnematode/satiety.py`
- [ ] Implement `__init__` with SatietyConfig
- [ ] Implement `decay_satiety()` method with clamping at 0.0
- [ ] Implement `restore_satiety(amount)` method with clamping at 1.0
- [ ] Implement `is_starved()` method
- [ ] Add `current_satiety` read-only property
- [ ] Write comprehensive unit tests (~8 tests) in `tests/quantumnematode_tests/test_satiety.py`
- [ ] Achieve >95% coverage for SatietyManager

**Dependencies**: Task 1.1
**Validation**: All tests pass, coverage >95%, ruff/pyright clean
**Parallelizable**: Can be done in parallel with Tasks 2.2, 2.3, 2.4

### Task 2.2: Implement MetricsTracker
- [ ] Create `MetricsTracker` class in new file `quantumnematode/metrics.py`
- [ ] Implement `__init__` with counter initialization
- [ ] Implement `track_episode_completion(success, steps, total_reward)` method
- [ ] Implement `track_food_collection(distance_efficiency)` method
- [ ] Implement `track_step(reward)` method
- [ ] Implement `calculate_metrics(total_runs)` returning PerformanceMetrics
- [ ] Write comprehensive unit tests (~12 tests) in `tests/quantumnematode_tests/test_metrics.py`
- [ ] Achieve >95% coverage for MetricsTracker

**Dependencies**: Task 1.1
**Validation**: All tests pass, coverage >95%, ruff/pyright clean
**Parallelizable**: Can be done in parallel with Tasks 2.1, 2.3, 2.4

### Task 2.3: Implement EpisodeRenderer
- [ ] Create `EpisodeRenderer` class in new file `quantumnematode/rendering.py`
- [ ] Implement `__init__` with rendering configuration
- [ ] Implement `render_frame(env, step, max_steps, text)` method
- [ ] Implement `render_if_needed(env, step, max_steps, show_last_frame_only)` method
- [ ] Implement `clear_screen()` method
- [ ] Support headless mode (enabled=False)
- [ ] Write unit tests (~6 tests) in `tests/quantumnematode_tests/test_rendering.py`
- [ ] Achieve >90% coverage for EpisodeRenderer

**Dependencies**: Task 1.1
**Validation**: All tests pass, coverage >90%, ruff/pyright clean
**Parallelizable**: Can be done in parallel with Tasks 2.1, 2.2, 2.4

### Task 2.4: Implement FoodConsumptionHandler
- [ ] Create `FoodConsumptionHandler` class in new file `quantumnematode/food_handler.py`
- [ ] Implement `__init__(env, satiety_manager)` with dependency injection
- [ ] Implement `check_and_consume_food(agent_pos)` returning FoodConsumptionResult
- [ ] Add logic to detect environment type (static vs dynamic)
- [ ] Implement distance efficiency calculation for dynamic environments
- [ ] Implement satiety restoration integration
- [ ] Write unit tests (~10 tests) in `tests/quantumnematode_tests/test_food_handler.py`
- [ ] Test both static and dynamic environment behaviors
- [ ] Achieve >95% coverage for FoodConsumptionHandler

**Dependencies**: Task 2.1 (requires SatietyManager)
**Validation**: All tests pass, coverage >95%, ruff/pyright clean
**Parallelizable**: Can start after Task 2.1, parallel with 2.2, 2.3

## Phase 3: Step Processing and Episode Runners

### Task 3.1: Implement StepProcessor
- [ ] Create `StepProcessor` class in new file `quantumnematode/step_processor.py`
- [ ] Implement `__init__(brain, env, food_handler, satiety_manager)` with DI
- [ ] Implement `prepare_brain_params(gradient_strength, gradient_direction, previous_action, previous_reward)` method
- [ ] Implement `process_step(gradient_strength, gradient_direction, previous_action, previous_reward)` returning StepResult
- [ ] Ensure stateless design (no episode state retained)
- [ ] Write unit tests (~15 tests) in `tests/quantumnematode_tests/test_step_processor.py`
- [ ] Test with mocked brain, env, food_handler, satiety_manager
- [ ] Achieve >95% coverage for StepProcessor

**Dependencies**: Tasks 2.1, 2.4
**Validation**: All tests pass, coverage >95%, ruff/pyright clean
**Parallelizable**: No (needs Phase 2 complete)

### Task 3.2: Implement StandardEpisodeRunner
- [ ] Create `StandardEpisodeRunner` class in new file `quantumnematode/runners.py`
- [ ] Implement `__init__(step_processor, metrics_tracker, renderer)` with DI
- [ ] Implement `run(agent, reward_config, max_steps, **kwargs)` returning EpisodeResult
- [ ] Implement episode loop with step execution delegation
- [ ] Implement termination logic (goal reached, max steps, starvation)
- [ ] Integrate rendering calls via renderer
- [ ] Integrate metrics tracking via metrics_tracker
- [ ] Write unit tests (~12 tests) in `tests/quantumnematode_tests/test_runners.py`
- [ ] Test episode termination scenarios
- [ ] Achieve >90% coverage for StandardEpisodeRunner

**Dependencies**: Tasks 3.1, 2.2, 2.3
**Validation**: All tests pass, coverage >90%, ruff/pyright clean
**Parallelizable**: Can be done in parallel with Task 3.3

### Task 3.3: Implement ManyworldsEpisodeRunner
- [ ] Create `ManyworldsEpisodeRunner` class in `quantumnematode/runners.py`
- [ ] Implement `__init__(step_processor, metrics_tracker, renderer)` with DI
- [ ] Implement `run(agent, reward_config, manyworlds_config, max_steps, **kwargs)` returning EpisodeResult
- [ ] Implement branching logic with probability-weighted trajectories
- [ ] Implement trajectory selection (highest reward)
- [ ] Integrate rendering for the selected trajectory
- [ ] Write unit tests (~8 tests) in `tests/quantumnematode_tests/test_runners.py`
- [ ] Test branching and trajectory selection
- [ ] Achieve >85% coverage for ManyworldsEpisodeRunner

**Dependencies**: Tasks 3.1, 2.2, 2.3
**Validation**: All tests pass, coverage >85%, ruff/pyright clean
**Parallelizable**: Can be done in parallel with Task 3.2

## Phase 4: Integration - Refactor QuantumNematodeAgent

### Task 4.1: Refactor QuantumNematodeAgent.__init__
- [ ] Update `__init__` to instantiate all component classes
- [ ] Create SatietyManager with satiety_config
- [ ] Create MetricsTracker
- [ ] Create FoodConsumptionHandler with env and satiety_manager
- [ ] Create StepProcessor with brain, env, food_handler, satiety_manager
- [ ] Create EpisodeRenderer with rendering config
- [ ] Maintain all existing public attributes for backward compatibility
- [ ] Add read-only properties for component access

**Dependencies**: All of Phase 3
**Validation**: Agent instantiates without errors, existing tests pass
**Parallelizable**: No (integration point)

### Task 4.2: Refactor QuantumNematodeAgent.run_episode
- [ ] Replace 268-line implementation with delegation to StandardEpisodeRunner
- [ ] Create StandardEpisodeRunner instance with components
- [ ] Call runner.run() with appropriate parameters
- [ ] Transform EpisodeResult back to list[tuple] return type for compatibility
- [ ] Update agent state (path, success_count, etc.) from EpisodeResult
- [ ] Remove duplicated code now handled by components
- [ ] Reduce method to <50 lines (mostly orchestration)

**Dependencies**: Task 4.1, Task 3.2
**Validation**: All existing integration tests pass, method <50 lines
**Parallelizable**: Can be done in parallel with Task 4.3

### Task 4.3: Refactor QuantumNematodeAgent.run_manyworlds_mode
- [ ] Replace 192-line implementation with delegation to ManyworldsEpisodeRunner
- [ ] Create ManyworldsEpisodeRunner instance with components
- [ ] Call runner.run() with appropriate parameters
- [ ] Transform EpisodeResult back to list[tuple] return type for compatibility
- [ ] Update agent state from EpisodeResult
- [ ] Remove duplicated code now handled by components
- [ ] Reduce method to <50 lines (mostly orchestration)

**Dependencies**: Task 4.1, Task 3.3
**Validation**: All existing integration tests pass, method <50 lines
**Parallelizable**: Can be done in parallel with Task 4.2

### Task 4.4: Refactor helper methods to use components
- [ ] Update `calculate_reward` to delegate to components where appropriate
- [ ] Update `reset_environment` to reset component state
- [ ] Update `reset_brain` to reset component state
- [ ] Update `calculate_metrics` to delegate to MetricsTracker
- [ ] Remove any remaining duplicated logic

**Dependencies**: Tasks 4.1, 4.2, 4.3
**Validation**: All tests pass, no code duplication detected
**Parallelizable**: No

## Phase 5: Testing and Documentation

### Task 5.1: Integration testing
- [ ] Run full test suite and ensure all existing tests pass
- [ ] Add new integration tests for component interactions
- [ ] Test both static and dynamic environment scenarios end-to-end
- [ ] Verify backward compatibility with existing configs and scripts
- [ ] Ensure coverage target >70% for agent module is met

**Dependencies**: All of Phase 4
**Validation**: All tests pass, coverage >70%
**Parallelizable**: No

### Task 5.2: Performance validation
- [ ] Benchmark run_episode before and after refactoring
- [ ] Benchmark run_manyworlds_mode before and after refactoring
- [ ] Ensure no >5% performance regression
- [ ] Document any performance improvements

**Dependencies**: Task 5.1
**Validation**: Performance benchmarks show <5% regression
**Parallelizable**: Can be done in parallel with Task 5.3

### Task 5.3: Update documentation
- [ ] Update QuantumNematodeAgent class docstring to reflect new architecture
- [ ] Add architecture diagram showing component relationships
- [ ] Document component responsibilities and interactions
- [ ] Update README if necessary
- [ ] Add migration guide for anyone extending the agent

**Dependencies**: Task 5.1
**Validation**: Documentation reviewed and accurate
**Parallelizable**: Can be done in parallel with Task 5.2

## Phase 6: Cleanup

### Task 6.1: Remove old code and noqa directives
- [ ] Remove old code that has been replaced by components
- [ ] Remove `# noqa: C901, PLR0912, PLR0915` directives (complexity should be fixed)
- [ ] Verify pyright and ruff pass without warnings
- [ ] Remove any unused imports or dead code

**Dependencies**: All of Phase 5
**Validation**: Clean ruff/pyright output, no noqa directives needed
**Parallelizable**: No

### Task 6.2: Final review and validation
- [ ] Run `openspec validate refactor-agent-architecture --strict`
- [ ] Review all code changes for quality and consistency
- [ ] Ensure all spec requirements are implemented
- [ ] Prepare change summary for approval

**Dependencies**: Task 6.1
**Validation**: OpenSpec validation passes, all requirements met
**Parallelizable**: No

---

## Summary

**Total Tasks**: 24
**Estimated Effort**: 3-5 days
**Parallelizable Phases**: Phase 2 (4 tasks can run in parallel)
**Critical Path**: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
**Test Coverage Goal**: From ~0% to >70% for agent module
**Expected LOC Reduction**: ~300 lines (from component extraction and duplication removal)
