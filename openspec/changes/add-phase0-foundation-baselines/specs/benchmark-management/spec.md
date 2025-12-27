# benchmark-management Specification Delta

## ADDED Requirements

### Requirement: NematodeBench Public Documentation
The system SHALL provide comprehensive public documentation enabling external researchers to submit, reproduce, and validate benchmarks.

#### Scenario: Submission Guide Access
- **WHEN** an external researcher accesses `docs/nematodebench/SUBMISSION_GUIDE.md`
- **THEN** the document SHALL explain prerequisites (50+ runs, clean git, config tracked)
- **AND** SHALL provide step-by-step submission process
- **AND** SHALL explain the PR and verification workflow
- **AND** SHALL include example commands with expected output

#### Scenario: Evaluation Methodology Documentation
- **WHEN** a researcher accesses `docs/nematodebench/EVALUATION.md`
- **THEN** the document SHALL explain composite score formula with weights
- **AND** SHALL explain convergence detection algorithm
- **AND** SHALL explain ranking criteria
- **AND** SHALL provide examples of score calculation

#### Scenario: Reproducibility Requirements Documentation
- **WHEN** a researcher accesses `docs/nematodebench/REPRODUCIBILITY.md`
- **THEN** the document SHALL explain config file requirements
- **AND** SHALL explain git state requirements
- **AND** SHALL explain version tracking requirements
- **AND** SHALL explain what makes a submission verifiable

#### Scenario: Documentation Discoverability
- **WHEN** a researcher views BENCHMARKS.md
- **THEN** the document SHALL link to NematodeBench documentation
- **AND** SHALL include a call-to-action for external submissions
- **AND** SHALL explain the public benchmark program

### Requirement: Benchmark Submission Evaluation Script
The system SHALL provide an automated script for validating benchmark submissions before PR review.

#### Scenario: Structure Validation
- **WHEN** `scripts/evaluate_submission.py` is run on a benchmark JSON file
- **THEN** the script SHALL validate JSON structure matches schema
- **AND** SHALL check all required fields are present
- **AND** SHALL validate field types and ranges
- **AND** SHALL report specific validation errors

#### Scenario: Minimum Runs Check
- **WHEN** evaluation is performed on a benchmark JSON with results
- **THEN** the script SHALL verify num_runs >= 50
- **AND** SHALL reject submissions with fewer runs
- **AND** SHALL display clear error message with requirement

#### Scenario: Config Existence Check
- **WHEN** evaluation is performed on a benchmark JSON with config_file path
- **THEN** the script SHALL verify the config file exists
- **AND** SHALL verify config is tracked in git
- **AND** SHALL warn if config has been modified since submission

#### Scenario: Optional Reproduction
- **WHEN** evaluation is performed with `--reproduce` flag on a benchmark JSON
- **THEN** the script SHALL attempt to reproduce results
- **AND** SHALL run specified number of validation episodes
- **AND** SHALL compare reproduced results to claimed results
- **AND** SHALL report if results match within tolerance (10%)
- **AND** SHALL flag significant discrepancies

#### Scenario: Evaluation Report
- **WHEN** evaluation completes
- **THEN** the script SHALL show PASS/FAIL status
- **AND** SHALL list all checks performed
- **AND** SHALL show composite score
- **AND** SHALL show category assignment
- **AND** SHALL provide actionable feedback for failures

### Requirement: Reproducibility Through Seeding
The system SHALL ensure all experiments are reproducible by automatically generating and tracking random seeds.

#### Scenario: Automatic Seed Generation
- **WHEN** an experiment is started without a seed parameter
- **THEN** the system SHALL generate a cryptographically random seed using `secrets.randbelow(2**32)`
- **AND** SHALL store the generated seed for the experiment
- **AND** SHALL use this seed consistently for all random number generation
- **AND** the seed SHALL be included in the experiment output

#### Scenario: Explicit Seed Configuration
- **WHEN** an experiment is started with a seed parameter in the config
- **THEN** the system SHALL use the provided seed
- **AND** SHALL NOT generate a new random seed
- **AND** SHALL record the provided seed in experiment output

#### Scenario: Environment Reproducibility
- **WHEN** an environment is initialized with a seed
- **THEN** all random operations (food spawning, predator movement, initial positions) SHALL be deterministic
- **AND** running the same seed twice SHALL produce identical episode results
- **AND** the system SHALL NOT use `secrets` module for seeded operations (only for seed generation)

#### Scenario: Brain Reproducibility
- **WHEN** a brain is initialized with a seed
- **THEN** weight initialization SHALL be deterministic
- **AND** action selection (for stochastic policies) SHALL be reproducible
- **AND** PyTorch and NumPy random states SHALL be seeded consistently

#### Scenario: Per-Run Seed Tracking
- **WHEN** multiple runs are executed in an experiment
- **THEN** each run SHALL have its own seed derived from the base seed
- **AND** per-run seeds SHALL be recorded in the submission JSON
- **AND** any individual run SHALL be reproducible using its recorded seed

### Requirement: Enhanced Metrics for Benchmarks
The system SHALL compute additional metrics required for comprehensive benchmark evaluation.

#### Scenario: Learning Speed Calculation
- **WHEN** convergence analysis is performed on experiment results
- **THEN** the system SHALL calculate episodes to reach 80% rolling success rate
- **AND** SHALL compute learning_speed = 1.0 - (episodes_to_80 / max_episodes)
- **AND** learning_speed SHALL be in range [0, 1] where 1 = instant learning
- **AND** SHALL handle cases where 80% is never reached (learning_speed = 0)

#### Scenario: Stability Metric Calculation
- **WHEN** metrics are aggregated across multiple runs
- **THEN** the system SHALL compute stability from coefficient of variation
- **AND** stability = 1.0 - (std / mean) for success rates, clamped to [0, 1]
- **AND** higher stability indicates more consistent results
- **AND** SHALL handle edge cases (zero mean, single run)

#### Scenario: Statistical Aggregation
- **WHEN** per-run metrics are collected
- **THEN** the system SHALL compute mean, std, min, max for each metric
- **AND** SHALL store these as StatValue objects in NematodeBench format
- **AND** metrics requiring aggregation: success_rate, composite_score, distance_efficiency, learning_speed, stability

### Requirement: Unified NematodeBench Format
The system SHALL use NematodeBench format as the single benchmark submission schema.

#### Scenario: Submission Schema
- **WHEN** a benchmark submission is created
- **THEN** the JSON SHALL use NematodeBench schema with StatValue objects
- **AND** StatValue SHALL contain: mean, std, min, max
- **AND** SHALL include individual_runs array with per-run data
- **AND** each individual run SHALL include: seed, success, steps, reward

#### Scenario: Experiment Metadata Migration
- **WHEN** experiment tracking saves results
- **THEN** the output SHALL conform to NematodeBench schema directly
- **AND** SHALL NOT require post-processing conversion
- **AND** legacy internal format SHALL be deprecated

#### Scenario: Benchmark Storage
- **WHEN** a benchmark is submitted to the benchmarks/ directory
- **THEN** the JSON SHALL be in NematodeBench format
- **AND** evaluate_submission.py SHALL validate against NematodeBench schema
- **AND** all new benchmarks SHALL include per-run seeds

## MODIFIED Requirements

### Requirement: Documentation Integration (MODIFIED)
The system SHALL integrate benchmark leaderboards into project documentation with clear reproduction instructions AND public submission guidance.

#### Scenario: BENCHMARKS.md External Submissions Section
- **WHEN** BENCHMARKS.md is updated
- **THEN** the file SHALL include an "External Submissions" section
- **AND** SHALL link to docs/nematodebench/ documentation
- **AND** SHALL explain how external researchers can contribute
- **AND** SHALL highlight successfully verified external submissions
