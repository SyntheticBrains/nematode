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

### Requirement: NematodeBench Benchmark Architecture

The system SHALL implement a multi-session benchmark system for official NematodeBench submissions.

#### Scenario: Session Experiments (Development)

- **WHEN** a developer runs `scripts/run_simulation.py --track-experiment`
- **THEN** experiment results SHALL be saved to `experiments/<experiment_id>/`
- **AND** the experiment JSON SHALL conform to ExperimentMetadata schema
- **AND** the original config file SHALL be automatically copied to the experiment folder
- **AND** the experiments/ directory SHALL be gitignored (temporary local storage)

#### Scenario: NematodeBench Submissions (Official Benchmarks)

- **WHEN** `scripts/benchmark_submit.py --experiments` is run with 10+ experiment folders
- **THEN** experiment folders SHALL be moved from `experiments/` to `artifacts/experiments/`
- **AND** metrics SHALL be aggregated using StatValue across sessions
- **AND** a fresh submission timestamp SHALL be generated
- **AND** the submission JSON SHALL be saved to `benchmarks/<category>/<timestamp>.json`
- **AND** the submission SHALL appear on the leaderboard

#### Scenario: NematodeBench Submission Schema

- **WHEN** a benchmark submission is created
- **THEN** the JSON SHALL use NematodeBenchSubmission schema
- **AND** SHALL include submission_id, brain_type, brain_config, environment, category
- **AND** SHALL include sessions array with SessionReference objects (experiment_id, file_path, session_seed, num_runs)
- **AND** SHALL include total_sessions (minimum 10), total_runs
- **AND** SHALL include AggregateMetrics with StatValue objects (success_rate, composite_score, learning_speed, stability)
- **AND** SHALL include all_seeds_unique validation flag
- **AND** SHALL include contributor attribution

#### Scenario: Seed Uniqueness Validation

- **WHEN** multiple experiments are aggregated for NematodeBench submission
- **THEN** the system SHALL validate that ALL seeds are unique across ALL runs in ALL sessions
- **AND** duplicate seeds SHALL cause submission rejection with specific error messages
- **AND** all_seeds_unique flag SHALL be false if any duplicates exist

#### Scenario: Minimum Session Requirements

- **WHEN** a NematodeBench submission is validated
- **THEN** the system SHALL require minimum 10 independent sessions
- **AND** each session SHALL require minimum 50 runs (MIN_RUNS_PER_SESSION)
- **AND** sessions with fewer runs SHALL generate warnings

#### Scenario: Configuration Consistency Validation

- **WHEN** multiple experiments are aggregated for NematodeBench submission
- **THEN** the system SHALL validate brain_type consistency across all sessions
- **AND** SHALL validate environment_type consistency
- **AND** SHALL validate grid_size consistency
- **AND** minor config differences (like seeds) SHALL be allowed

### Requirement: Experiment Storage and Tracking

The system SHALL persist experiment data in a structured folder hierarchy with automatic config preservation.

#### Scenario: Experiment Folder Structure

- **WHEN** an experiment is tracked
- **THEN** results SHALL be saved to `experiments/<experiment_id>/<experiment_id>.json`
- **AND** the config file used SHALL be copied to `experiments/<experiment_id>/<config_name>.yml`
- **AND** the folder SHALL be self-contained for reproducibility

#### Scenario: Artifact Storage for Submissions

- **WHEN** experiments are submitted as a NematodeBench benchmark
- **THEN** experiment folders SHALL be moved to `artifacts/experiments/`
- **AND** the artifacts/ directory SHALL be committed to the repository
- **AND** session references in submissions SHALL point to artifacts/experiments/ paths

#### Scenario: Benchmark Submission Output

- **WHEN** a benchmark submission is created
- **THEN** the JSON SHALL be saved to `benchmarks/<category>/<submission_timestamp>.json`
- **AND** the submission timestamp SHALL be generated at submission time (not from experiments)
- **AND** the benchmarks/ directory SHALL be committed to the repository
