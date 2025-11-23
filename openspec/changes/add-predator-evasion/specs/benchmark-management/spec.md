# benchmark-management Delta

## ADDED Requirements

### Requirement: Predator-Enabled Benchmark Categories
The system SHALL provide separate benchmark categories for predator-enabled simulations to enable tracking learning performance on survival-foraging multi-objective tasks.

#### Scenario: Dynamic Predator Quantum Small Category
- **GIVEN** a simulation with quantum brain (ModularBrain or QModularBrain), dynamic environment with `predators.enabled: true`, and grid size ≤ 20×20
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `dynamic_predator_quantum_small`
- **AND** this SHALL be distinct from `dynamic_small_quantum` (non-predator category)
- **AND** benchmarks SHALL track predator-specific metrics

#### Scenario: Dynamic Predator Quantum Medium Category
- **GIVEN** a simulation with quantum brain, `predators.enabled: true`, and 20×20 < grid size ≤ 50×50
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `dynamic_predator_quantum_medium`
- **AND** this SHALL use same grid size threshold as non-predator medium benchmarks

#### Scenario: Dynamic Predator Quantum Large Category
- **GIVEN** a simulation with quantum brain, `predators.enabled: true`, and grid size > 50×50
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `dynamic_predator_quantum_large`
- **AND** this SHALL represent the most challenging predator-enabled quantum scenarios

#### Scenario: Dynamic Predator Classical Small Category
- **GIVEN** a simulation with classical brain (MLPBrain, QMLPBrain, or SpikingBrain), `predators.enabled: true`, and grid size ≤ 20×20
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `dynamic_predator_classical_small`
- **AND** this SHALL enable comparison of classical vs quantum approaches on predator tasks

#### Scenario: Dynamic Predator Classical Medium Category
- **GIVEN** a simulation with classical brain, `predators.enabled: true`, and 20×20 < grid size ≤ 50×50
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `dynamic_predator_classical_medium`

#### Scenario: Dynamic Predator Classical Large Category
- **GIVEN** a simulation with classical brain, `predators.enabled: true`, and grid size > 50×50
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `dynamic_predator_classical_large`

#### Scenario: Predators Disabled Uses Non-Predator Categories
- **GIVEN** a simulation with `predators.enabled: false` or predators not configured
- **WHEN** benchmark category is determined
- **THEN** existing non-predator categories SHALL be used
- **AND** categories SHALL be `dynamic_small_quantum`, `dynamic_medium_classical`, etc.
- **AND** backward compatibility SHALL be maintained

### Requirement: Predator Benchmark Metrics Tracking
Benchmark submissions for predator-enabled categories SHALL include predator-specific metrics in addition to standard foraging metrics.

#### Scenario: Predator Metrics in Benchmark Submission
- **GIVEN** a benchmark submission for category `dynamic_predator_quantum_small`
- **WHEN** the benchmark result is recorded
- **THEN** the submission SHALL include standard metrics:
  - success_rate
  - average_steps
  - average_reward
  - foods_collected
  - foraging_efficiency
- **AND** SHALL additionally include predator metrics:
  - predator_encounters (average per episode)
  - successful_evasions (average per episode)
  - predator_death_rate (percentage of episodes ending in predator death)
  - average_survival_time (steps before predator death or success)

#### Scenario: Benchmark Comparison with Predator Metrics
- **GIVEN** multiple benchmark submissions in category `dynamic_predator_classical_medium`
- **WHEN** benchmarks are compared
- **THEN** the system SHALL rank submissions by primary metric (success_rate or total reward)
- **AND** SHALL display predator-specific metrics for detailed analysis
- **AND** SHALL enable filtering by predator death rate or evasion success

#### Scenario: Non-Predator Benchmark Metrics Unchanged
- **GIVEN** a benchmark submission for non-predator category
- **WHEN** the benchmark result is recorded
- **THEN** predator metrics SHALL be absent or null
- **AND** standard foraging metrics only SHALL be recorded
- **AND** backward compatibility with existing benchmarks SHALL be maintained

### Requirement: Predator Benchmark Category Naming Convention
Benchmark categories SHALL follow consistent naming convention indicating environment type, feature set, brain type, and size.

#### Scenario: Category Naming Pattern
- **GIVEN** the benchmark categorization system
- **WHEN** a predator-enabled benchmark category is created
- **THEN** the naming pattern SHALL be: `{environment}_{feature}_{brain}_{size}`
  - environment: "dynamic" (static not supported for predators)
  - feature: "predator" (indicates predator-enabled)
  - brain: "quantum" or "classical"
  - size: "small", "medium", or "large"
- **AND** this SHALL produce categories like `dynamic_predator_quantum_small`

#### Scenario: Category Name Consistency
- **GIVEN** all benchmark categories in the system
- **WHEN** category names are reviewed
- **THEN** predator categories SHALL clearly indicate the "predator" feature
- **AND** SHALL be easily distinguishable from non-predator equivalents
- **AND** SHALL sort logically in alphabetical listings (dynamic_predator_* grouped together)

### Requirement: Benchmark Metadata for Predator Configuration
Benchmark submissions SHALL capture predator configuration parameters to enable reproducibility and analysis of difficulty variations.

#### Scenario: Predator Configuration in Benchmark Metadata
- **GIVEN** a benchmark submission for a predator-enabled category
- **WHEN** the submission metadata is recorded
- **THEN** the following predator configuration SHALL be captured:
  - num_predators
  - predator_speed
  - detection_radius
  - kill_radius
  - proximity_penalty
- **AND** this SHALL enable analysis of performance across different predator difficulties

#### Scenario: Benchmark Reproducibility with Predator Settings
- **GIVEN** a historical benchmark submission with recorded predator configuration
- **WHEN** attempting to reproduce the benchmark
- **THEN** the exact predator parameters SHALL be available
- **AND** SHALL enable recreation of identical difficulty conditions
- **AND** SHALL support analysis of how parameter changes affect learning

#### Scenario: Benchmark Filtering by Predator Difficulty
- **GIVEN** multiple benchmark submissions with varying predator counts (2, 3, 5)
- **WHEN** benchmarks are filtered
- **THEN** the system SHALL support filtering by num_predators
- **AND** SHALL enable comparison of agent performance across difficulty levels
- **AND** SHALL show how additional predators affect success metrics
