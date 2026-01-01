## ADDED Requirements

### Requirement: Ablation Study Framework

The system SHALL provide a framework for systematic ablation studies that remove components from brain architectures and measure performance degradation.

#### Scenario: Ablation Study Configuration

- **GIVEN** an ablation study configuration specifying modules to ablate
- **WHEN** the ablation study is initialized
- **THEN** the framework SHALL accept:
  - Base brain configuration path
  - List of modules/components to ablate
  - Number of runs per condition for statistical significance
  - Output directory for results

#### Scenario: Baseline Run

- **GIVEN** an ablation study with a base configuration
- **WHEN** `run_baseline()` is called
- **THEN** the framework SHALL run the full brain (no ablations)
- **AND** SHALL return convergence metrics for comparison

#### Scenario: Ablated Run

- **GIVEN** an ablation study with module "chemotaxis" specified for ablation
- **WHEN** `run_ablated(ModuleAblation("chemotaxis"))` is called
- **THEN** the framework SHALL run the brain with chemotaxis module disabled
- **AND** SHALL return convergence metrics for the ablated condition

### Requirement: ModularBrain Ablation Support

The ModularBrain architecture SHALL support ablation of individual modules, entanglement layers, and circuit depth.

#### Scenario: Module Ablation

- **GIVEN** a ModularBrain with modules [chemotaxis, thermotaxis, nociception]
- **WHEN** chemotaxis module is ablated
- **THEN** the rotation angles for chemotaxis qubits SHALL be set to 0
- **AND** the module SHALL not contribute signal to decision-making
- **AND** other modules SHALL function normally

#### Scenario: Entanglement Ablation

- **GIVEN** a ModularBrain with CZ entanglement gates
- **WHEN** entanglement is ablated
- **THEN** no CZ gates SHALL be applied between qubits
- **AND** each qubit SHALL operate independently
- **AND** this tests whether quantum correlations are necessary

#### Scenario: Layer Ablation

- **GIVEN** a ModularBrain with 3 layers
- **WHEN** layer 2 is ablated
- **THEN** the circuit SHALL skip layer 2 during construction
- **AND** effective circuit depth SHALL be reduced
- **AND** this tests whether deeper circuits improve performance

### Requirement: PPOBrain Ablation Support

The PPOBrain architecture SHALL support ablation of hidden layers, critic network, and input features.

#### Scenario: Hidden Layer Bypass

- **GIVEN** a PPOBrain with 2 hidden layers
- **WHEN** hidden layers are ablated
- **THEN** the actor SHALL use direct linear projection from input to output
- **AND** this tests whether non-linear transformations are necessary

#### Scenario: Critic Ablation

- **GIVEN** a PPOBrain with actor-critic architecture
- **WHEN** critic is ablated
- **THEN** a fixed baseline value SHALL be used instead of learned value function
- **AND** this tests the importance of value estimation for PPO

#### Scenario: Input Feature Masking

- **GIVEN** a PPOBrain receiving unified feature vector
- **WHEN** "thermotaxis" features are masked
- **THEN** thermotaxis feature values SHALL be set to 0
- **AND** other features SHALL pass through normally
- **AND** this tests individual sensory modality importance

### Requirement: PPOBrain Sensory Module Ablation

The PPOBrain architecture SHALL support ablation of sensory module configurations to compare unified modular features vs legacy preprocessing.

#### Scenario: Legacy vs Unified Mode Comparison

- **GIVEN** a PPOBrain configuration with sensory_modules specified
- **WHEN** sensory module ablation switches to legacy mode
- **THEN** the brain SHALL use 2-feature legacy preprocessing (gradient_strength, rel_angle_norm)
- **AND** this tests whether unified modular features improve over legacy

#### Scenario: Sensory Module Subset Ablation

- **GIVEN** a PPOBrain with sensory_modules [food_chemotaxis, nociception, mechanosensation]
- **WHEN** nociception module is ablated
- **THEN** the brain SHALL use only [food_chemotaxis, mechanosensation] (6 features)
- **AND** nociception features SHALL not be included in input vector
- **AND** this tests individual sensory modality importance

#### Scenario: Module Combination Study

- **GIVEN** a PPOBrain ablation study
- **WHEN** module combinations are enumerated
- **THEN** the framework SHALL test:
  - Single module conditions (food_chemotaxis only, nociception only, mechanosensation only)
  - Paired module conditions (food_chemotaxis + nociception, etc.)
  - Full module configuration
- **AND** this identifies minimal sufficient module sets

### Requirement: Gradient Mode Ablation

The system SHALL support ablation of gradient computation modes to compare combined vs separated gradient sensing.

#### Scenario: Combined vs Separated Gradient Comparison

- **GIVEN** a PPOBrain with separated gradients (use_separated_gradients: true)
- **WHEN** gradient mode is ablated to combined mode
- **THEN** the brain SHALL use chemotaxis module (combined gradient) instead of food_chemotaxis + nociception
- **AND** environment SHALL compute combined gradient instead of separated food/predator gradients
- **AND** this tests whether gradient separation improves predator avoidance

#### Scenario: Gradient Mode with Module Mapping

- **GIVEN** gradient mode ablation configuration
- **WHEN** switching from separated to combined mode
- **THEN** sensory modules SHALL be remapped:
  - food_chemotaxis + nociception → chemotaxis
  - Input dimension changes from 6 to 3 (for gradient-related modules)
- **AND** results SHALL be comparable across modes

### Requirement: Feature Normalization Ablation

The system SHALL support ablation of feature normalization modes to validate correct scaling for different architectures.

#### Scenario: Classical vs Quantum Normalization

- **GIVEN** a PPOBrain with normalize_for_classical=True (default for PPO)
- **WHEN** normalization mode is ablated to quantum scaling
- **THEN** features SHALL be in range [-π/2, π/2] instead of [-1, 1]
- **AND** this tests whether classical normalization is necessary for PPO performance
- **NOTE** quantum normalization for PPO is expected to degrade performance significantly

#### Scenario: Normalization Impact Measurement

- **GIVEN** ablation results for classical vs quantum normalization
- **WHEN** performance is compared
- **THEN** the report SHALL include:
  - Success rate difference
  - Convergence rate difference
  - Feature value distribution statistics
- **AND** this validates the normalization design decision

### Requirement: Feature Importance Calculation

The system SHALL compute feature importance scores based on performance degradation when components are removed.

#### Scenario: Importance Score Calculation

- **GIVEN** baseline composite score of 0.80 and ablated score of 0.50
- **WHEN** feature importance is calculated
- **THEN** importance SHALL equal `(0.80 - 0.50) / 0.80 = 0.375`
- **AND** higher importance indicates more critical component

#### Scenario: Importance Ranking

- **GIVEN** importance scores for multiple modules
- **WHEN** ranking is generated
- **THEN** modules SHALL be sorted by importance (descending)
- **AND** critical modules (importance > 0.1) SHALL be flagged
- **AND** redundant modules (importance < 0.01) SHALL be flagged

#### Scenario: Cross-Architecture Comparison

- **GIVEN** ablation results for both ModularBrain and PPOBrain
- **WHEN** cross-architecture comparison is performed
- **THEN** module importance SHALL be compared between architectures
- **AND** architecture-specific critical components SHALL be identified

### Requirement: Automated Ablation Reporting

The system SHALL generate automated markdown reports summarizing ablation study results.

#### Scenario: Report Generation

- **GIVEN** completed ablation study with all conditions run
- **WHEN** `generate_report()` is called
- **THEN** a markdown report SHALL be generated containing:
  - Configuration summary
  - Results table with all conditions
  - Feature importance ranking
  - Key findings summary

#### Scenario: Report Results Table

- **GIVEN** ablation results for baseline and 3 ablated conditions
- **WHEN** the report is generated
- **THEN** the results table SHALL include:
  - Condition name
  - Composite score
  - Success rate
  - Importance score (for ablated conditions)

#### Scenario: Key Findings Extraction

- **GIVEN** completed ablation analysis
- **WHEN** key findings are extracted
- **THEN** the report SHALL identify:
  - Most critical component(s)
  - Redundant component(s) that can be removed
  - Notable architecture differences

### Requirement: Ablation CLI Interface

The system SHALL provide a command-line interface for running ablation studies.

#### Scenario: CLI Ablation Execution

- **GIVEN** a command: `python run_ablation.py --config config.yml --ablate-modules chemotaxis thermotaxis`
- **WHEN** the command is executed
- **THEN** an ablation study SHALL run with specified modules ablated
- **AND** results SHALL be saved to output directory
- **AND** a report SHALL be generated

#### Scenario: CLI Output

- **GIVEN** a completed ablation study via CLI
- **WHEN** execution finishes
- **THEN** the following files SHALL be created:
  - `results.json` - Raw ablation results
  - `importance.json` - Feature importance scores
  - `report.md` - Human-readable report
