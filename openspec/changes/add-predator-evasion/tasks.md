# Implementation Tasks

## 1. Core Predator Mechanics

- [ ] 1.1 Create `Predator` class in `env.py` with position, speed, and movement methods
- [ ] 1.2 Implement random movement pattern for predators with configurable speed
- [ ] 1.3 Add predator collection to `DynamicForagingEnvironment.__init__`
- [ ] 1.4 Implement predator spawning with random valid positions
- [ ] 1.5 Add predator position update method called each environment step
- [ ] 1.6 Implement collision detection between agent and predators (kill_radius check)
- [ ] 1.7 Add `TerminationReason.PREDATOR` enum value to termination reasons
- [ ] 1.8 Implement episode termination on predator collision with death penalty reward

## 2. Unified Gradient System

- [ ] 2.1 Rename existing gradient methods for clarity (`calculate_gradient` → `calculate_food_gradient`)
- [ ] 2.2 Implement `calculate_predator_gradient` method using negative exponential decay
- [ ] 2.3 Implement `calculate_combined_gradient` method for vector superposition
- [ ] 2.4 Update environment observation to use combined gradient (food + predator)
- [ ] 2.5 Add gradient_decay_constant and gradient_strength to predator configuration
- [ ] 2.6 Ensure gradient direction points away from predators (repulsive)
- [ ] 2.7 Test gradient superposition with multiple foods and multiple predators

## 3. Detection and Proximity System

- [ ] 3.1 Implement `is_agent_in_danger` method checking if agent within any predator's detection_radius
- [ ] 3.2 Add proximity penalty calculation in `RewardCalculator` when agent in danger
- [ ] 3.3 Make proximity_penalty configurable (default -0.1)
- [ ] 3.4 Ensure proximity penalty applied per step, not stacked per predator
- [ ] 3.5 Add agent danger status tracking for visualization output

## 4. Configuration System Updates

- [ ] 4.1 Create `PredatorConfig` Pydantic model with all predator parameters
- [ ] 4.2 Create `ForagingConfig` Pydantic model for nested foraging parameters
- [ ] 4.3 Update `DynamicEnvironmentConfig` to include `foraging` and `predators` subsections
- [ ] 4.4 Implement automatic migration from flat to nested configuration structure
- [ ] 4.5 Add deprecation warning logging for legacy flat configurations
- [ ] 4.6 Add validation for `movement_pattern` (only "random" allowed, clear error for future patterns)
- [ ] 4.7 Set default values: `enabled: false`, `count: 2`, `speed: 1.0`, `detection_radius: 8`, `kill_radius: 1`
- [ ] 4.8 Add `show_detection_radius: true` configuration parameter

## 5. Metrics and Tracking

- [ ] 5.1 Add predator metrics fields to `PerformanceMetrics` dataclass:
  - `predator_encounters: int | None`
  - `successful_evasions: int | None`
  - `deaths_by_predator: int | None`
  - `foods_collected_before_death: int | None`
- [ ] 5.2 Implement encounter tracking logic (count when entering detection radius)
- [ ] 5.3 Implement evasion tracking (increment when exiting detection radius without death)
- [ ] 5.4 Set predator metrics to None when `predators.enabled: false` (backward compatibility)
- [ ] 5.5 Add predator metrics to `EpisodeTracker` tracking data
- [ ] 5.6 Update experiment metadata JSON to include predator configuration
- [ ] 5.7 Ensure termination reason correctly recorded as "predator" in tracking data

## 6. Visualization and Rendering

- [ ] 6.1 Add predator rendering in emoji theme mode using 🕷️ spider emoji
- [ ] 6.2 Add predator rendering in ASCII theme mode using `#` symbol
- [ ] 6.3 Implement detection radius visualization (configurable, default enabled)
- [ ] 6.4 Add danger status display to simulation run output ("SAFE" / "IN DANGER")
- [ ] 6.5 Ensure predators within viewport are rendered correctly
- [ ] 6.6 Test rendering with multiple predators and overlapping detection radii

## 7. Benchmark Categories

- [ ] 7.1 Add six new benchmark categories to `categorization.py`:
  - `dynamic_predator_quantum_small`
  - `dynamic_predator_quantum_medium`
  - `dynamic_predator_quantum_large`
  - `dynamic_predator_classical_small`
  - `dynamic_predator_classical_medium`
  - `dynamic_predator_classical_large`
- [ ] 7.2 Update categorization logic to check `predators.enabled` flag
- [ ] 7.3 Use same grid size thresholds as non-predator benchmarks (≤20, ≤50, >50)
- [ ] 7.4 Ensure non-predator simulations still use existing categories (backward compatibility)
- [ ] 7.5 Add predator configuration metadata to benchmark submissions
- [ ] 7.6 Update benchmark comparison to include predator-specific metrics

## 8. Plotting and Exports

- [ ] 8.1 Add predator encounters over time plot to `plots.py`
- [ ] 8.2 Add evasion success rate over time plot (successful_evasions / encounters)
- [ ] 8.3 Add survival rate vs food collection scatter plot showing trade-offs
- [ ] 8.4 Update CSV export in `csv_export.py` to include predator metric columns
- [ ] 8.5 Ensure predator columns are empty/null when predators disabled
- [ ] 8.6 Add predator configuration to experiment JSON exports

## 9. Example Configurations

- [ ] 9.1 Create `configs/examples/mlp_dynamic_small_predators.yml` with:
  - 20×20 grid
  - 2 predators
  - All parameters explicitly shown with inline comments
  - Nested foraging/predators structure
- [ ] 9.2 Create `configs/examples/modular_dynamic_medium_predators.yml` with:
  - 50×50 grid
  - 3 predators
  - ModularBrain configuration
- [ ] 9.3 Create `configs/examples/mlp_dynamic_large_predators.yml` with:
  - 100×100 grid
  - 5 predators
  - Advanced difficulty demonstration
- [ ] 9.4 Add comprehensive inline documentation to all example configs explaining predator mechanics

## 10. Testing

- [ ] 10.1 Unit tests for `Predator` class movement and position updates
- [ ] 10.2 Unit tests for predator gradient calculation (negative, exponential decay)
- [ ] 10.3 Unit tests for gradient superposition (food + predator vectors)
- [ ] 10.4 Unit tests for collision detection (kill_radius boundaries)
- [ ] 10.5 Unit tests for proximity detection (detection_radius boundaries)
- [ ] 10.6 Unit tests for proximity penalty calculation
- [ ] 10.7 Integration tests for complete episode with predators
- [ ] 10.8 Integration tests for predator collision termination
- [ ] 10.9 Tests for configuration migration from flat to nested structure
- [ ] 10.10 Tests for backward compatibility (predators disabled by default)
- [ ] 10.11 Tests for predator metrics tracking (encounters, evasions, deaths)
- [ ] 10.12 Tests for benchmark categorization with predators
- [ ] 10.13 Performance tests ensuring <100ms step time with predators enabled

## 11. Documentation

- [ ] 11.1 Update README.md with predator evasion feature description
- [ ] 11.2 Add predator configuration section to configuration documentation
- [ ] 11.3 Document predator metrics in metrics documentation
- [ ] 11.4 Add example usage and training tips for predator-enabled simulations
- [ ] 11.5 Document configuration migration path from flat to nested structure
- [ ] 11.6 Add troubleshooting section for predator learning difficulties
- [ ] 11.7 Document future enhancements (health system, additional predator types, pursuit behavior)
- [ ] 11.8 Update CHANGELOG.md with feature addition and breaking changes (none)

## 12. Validation and Quality Assurance

- [ ] 12.1 Run full test suite and ensure all tests pass
- [ ] 12.2 Run type checking with Pyright and resolve any errors
- [ ] 12.3 Run linting with Ruff and resolve any issues
- [ ] 12.4 Verify backward compatibility: run existing configs unchanged
- [ ] 12.5 Verify predator-enabled configs run successfully
- [ ] 12.6 Validate OpenSpec change with `openspec validate add-predator-evasion --strict`
- [ ] 12.7 Performance benchmark: confirm <100ms step time on 100×100 grid with 5 predators
- [ ] 12.8 End-to-end test: train agent with predators for 100 episodes, verify metrics tracking
- [ ] 12.9 Visual inspection: verify predator rendering in both emoji and ASCII themes
- [ ] 12.10 Configuration validation: test invalid movement patterns produce clear errors
