# Implementation Tasks

## 1. Core Environment Implementation
- [x] 1.1 Create `BaseEnvironment` abstract class extracting common functionality from `MazeEnvironment`
- [x] 1.2 Refactor `MazeEnvironment` to inherit from `BaseEnvironment`
- [x] 1.3 Implement `DynamicForagingEnvironment` class with multi-food support
- [x] 1.4 Implement gradient field superposition in `get_state()` method
- [x] 1.5 Add Poisson disk sampling algorithm for spatial food distribution
- [x] 1.6 Implement food spawning logic with distance constraints
- [x] 1.7 Add visited cell tracking for exploration rewards

## 2. Satiety System
- [x] 2.1 Add `satiety` attribute to agent state
- [x] 2.2 Implement satiety decay per step in agent loop
- [x] 2.3 Implement satiety restoration on food consumption
- [x] 2.4 Add starvation termination condition
- [x] 2.5 Add satiety-related rewards and penalties

## 3. Viewport Rendering System
- [x] 3.1 Implement viewport calculation with agent-centered logic
- [x] 3.2 Add viewport boundary clamping for edge cases
- [x] 3.3 Update render methods to use viewport instead of full grid
- [x] 3.4 Add full environment logging at episode start
- [x] 3.5 Update all rendering themes (ASCII, EMOJI, UNICODE, etc.) for viewport mode

## 4. Metrics and Reward System
- [x] 4.1 Implement foraging efficiency rate metric (`foods_collected / total_steps`)
- [x] 4.2 Implement distance efficiency per food metric
- [x] 4.3 Add greedy baseline distance tracking
- [x] 4.4 Extend `PerformanceMetrics` dataclass with new multi-food metrics
- [x] 4.5 Update reward calculation to include exploration bonus
- [x] 4.6 Add survival time and termination reason to metrics
- [x] 4.7 Update CSV export format to include new metrics

## 5. Configuration System
- [x] 5.1 Create `DynamicEnvironmentConfig` Pydantic model
- [x] 5.2 Create `SatietyConfig` Pydantic model
- [x] 5.3 Add `environment_type` field to `SimulationConfig`
- [x] 5.4 Implement configuration validation rules
- [x] 5.5 Add default formula for food count based on grid size
- [x] 5.6 Implement viewport size validation (odd numbers)

## 6. Preset Configurations
- [x] 6.1 Create `configs/examples/modular_dynamic_small.yml` (20×20, 5 foods, 200 satiety)
- [x] 6.2 Create `configs/examples/modular_dynamic_medium.yml` (50×50, 20 foods, 500 satiety)
- [x] 6.3 Create `configs/examples/modular_dynamic_large.yml` (100×100, 50 foods, 800 satiety)
- [x] 6.4 Add detailed comments explaining each parameter

## 7. Integration and Compatibility
- [x] 7.1 Update `run_simulation.py` to handle environment type selection
- [x] 7.2 Ensure brain architectures work with both environment types
- [x] 7.3 Verify observation space remains consistent (gradient strength, direction)
- [x] 7.4 Test all existing configurations still work unchanged
- [x] 7.5 Add environment type detection logic in agent initialization

## 8. Testing
- [x] 8.1 Unit tests for gradient superposition
- [x] 8.2 Unit tests for Poisson disk sampling
- [x] 8.3 Unit tests for satiety system
- [x] 8.4 Unit tests for viewport calculations
- [x] 8.5 Integration tests with each preset configuration
- [x] 8.6 Backward compatibility tests with existing configs
- [ ] 8.7 Performance benchmarks for large environments
- [x] 8.8 Test exploration reward mechanics

## 9. Documentation
- [x] 9.1 Add docstrings to all new classes and methods (NumPy style)
- [x] 9.2 Update type hints throughout
- [x] 9.3 Add inline comments for complex algorithms (gradient superposition, Poisson sampling)
- [x] 9.4 Ensure all code passes Ruff and Pyright validation

## 10. Validation and Testing
- [x] 10.1 Run full test suite and ensure all tests pass
- [ ] 10.2 Run simulations with all three preset configs to verify behavior
- [x] 10.3 Validate metrics are computed correctly
- [x] 10.4 Check rendering works correctly in all themes
- [x] 10.5 Verify backward compatibility with legacy configurations
- [ ] 10.6 Performance test with 100×100 grid and 50 foods
