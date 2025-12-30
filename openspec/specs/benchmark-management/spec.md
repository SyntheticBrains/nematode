# benchmark-management Specification

## Purpose

This specification defines the curated benchmark submission, validation, leaderboard generation, and verification system. It enables tracking and showcasing high-quality performance results with proper contributor attribution and reproducibility.

## Requirements

### Requirement: Benchmark Submission Workflow

The system SHALL provide a curated benchmark submission workflow with validation, quality checks, and contributor attribution.

#### Scenario: Benchmark Submission with Valid Metadata

- **GIVEN** a user runs a simulation with `--save-benchmark` flag
- **WHEN** the simulation completes successfully
- **THEN** the user SHALL be prompted for contributor name (required)
- **AND** the user SHALL be prompted for GitHub username (optional)
- **AND** the user SHALL be prompted for optimization notes (optional)
- **AND** a benchmark JSON file SHALL be created in `benchmarks/{category}/{timestamp}.json`
- **AND** the benchmark SHALL include all experiment metadata plus benchmark-specific fields
- **AND** the category SHALL be determined automatically from environment and brain types

#### Scenario: Benchmark Validation - Minimum Runs

- **GIVEN** a user attempts to save a benchmark with fewer than 50 runs
- **WHEN** validation is performed
- **THEN** the benchmark submission SHALL be rejected
- **AND** an error message SHALL inform the user of the minimum run requirement
- **AND** no benchmark file SHALL be created

#### Scenario: Benchmark Validation - Git Clean State

- **GIVEN** a user attempts to save a benchmark with uncommitted git changes
- **WHEN** validation is performed
- **THEN** a warning SHALL be displayed
- **AND** the user SHALL be asked to confirm submission despite dirty state
- **AND** if confirmed, the benchmark SHALL be saved with git_dirty=true flag

#### Scenario: Benchmark Validation - Config File Exists

- **GIVEN** a user attempts to save a benchmark
- **WHEN** validation is performed
- **THEN** the system SHALL verify the config file exists in the repository
- **AND** if the config file is not tracked in git, a warning SHALL be shown
- **AND** the config file path SHALL be validated as relative to repository root

### Requirement: Benchmark Categorization

The system SHALL automatically categorize benchmarks hierarchically by environment type (static/dynamic size) and brain architecture (quantum/classical).

#### Scenario: Foraging Medium Quantum Category

- **GIVEN** a benchmark using modular brain (quantum) with foraging medium environment
- **WHEN** the benchmark is saved
- **THEN** the category SHALL be "foraging_medium/quantum"
- **AND** the file SHALL be stored in `benchmarks/foraging_medium/quantum/{timestamp}.json`

#### Scenario: Static Maze Classical Category

- **GIVEN** a benchmark using MLP brain (classical) with static maze environment
- **WHEN** the benchmark is saved
- **THEN** the category SHALL be "static_maze/classical"
- **AND** the file SHALL be stored in `benchmarks/static_maze/classical/{timestamp}.json`

#### Scenario: Foraging Large Quantum Category

- **GIVEN** a benchmark using quantum brain with foraging large environment (100×100)
- **WHEN** the benchmark is saved
- **THEN** the category SHALL be "foraging_large/quantum"
- **AND** the file SHALL be stored in `benchmarks/foraging_large/quantum/{timestamp}.json`

### Requirement: Benchmark Leaderboard Generation

The system SHALL generate formatted leaderboards for README.md and BENCHMARKS.md from stored benchmark data.

#### Scenario: README Summary Table Generation

- **GIVEN** benchmarks exist for multiple categories
- **WHEN** leaderboard generation is triggered
- **THEN** a summary table SHALL be generated for README.md
- **AND** the table SHALL show top 3 results per category
- **AND** the table SHALL include: brain type, success rate, avg steps, foods/run, dist efficiency, contributor, date
- **AND** quantum and classical brains SHALL be grouped separately
- **AND** results SHALL be sorted by a composite score (success rate, then efficiency)

#### Scenario: BENCHMARKS.md Detailed Table

- **GIVEN** benchmarks exist for a category
- **WHEN** detailed leaderboard generation is triggered
- **THEN** a full table SHALL be generated for BENCHMARKS.md
- **AND** the table SHALL include all benchmarks (not just top 3)
- **AND** the table SHALL include additional columns: config file, commit hash, notes
- **AND** a "Reproduction" section SHALL provide instructions for each benchmark
- **AND** links to config files and commit hashes SHALL be formatted as GitHub URLs

#### Scenario: Empty Category Handling

- **GIVEN** a category has no submitted benchmarks
- **WHEN** leaderboard generation is triggered
- **THEN** the category section SHALL show "No benchmarks submitted yet"
- **AND** the category SHALL still appear in the structure
- **AND** a call-to-action SHALL encourage benchmark submissions

### Requirement: Benchmark Quality Metrics

The system SHALL define and validate quality criteria for benchmark submissions to ensure leaderboard integrity.

#### Scenario: Success Rate Threshold for Foraging

- **GIVEN** a benchmark for dynamic foraging environment
- **WHEN** validation is performed
- **THEN** a warning SHALL be shown if success rate < 50%
- **AND** the user SHALL confirm the submission is intentional
- **AND** low success rate benchmarks SHALL be marked with a flag for review

#### Scenario: Consistency Check

- **GIVEN** a benchmark submission
- **WHEN** validation is performed
- **THEN** results SHALL be checked for statistical consistency
- **AND** warnings SHALL be shown if standard deviation is unusually high
- **AND** warnings SHALL be shown if results seem anomalous compared to typical ranges

#### Scenario: Configuration Validation

- **GIVEN** a benchmark submission
- **WHEN** validation is performed
- **THEN** the configuration SHALL be validated against the config schema
- **AND** the environment and brain types SHALL match the declared category
- **AND** key hyperparameters SHALL be within expected ranges

### Requirement: Benchmark Comparison Tools

The system SHALL provide tools for comparing benchmark submissions and identifying best performers.

#### Scenario: Category Leaderboard Query

- **GIVEN** multiple benchmarks in the "foraging_medium/quantum" category
- **WHEN** a user runs `scripts/benchmark_submit.py leaderboard foraging_medium/quantum`
- **THEN** benchmarks SHALL be displayed in ranked order
- **AND** rankings SHALL be based on composite scoring (success rate primary, efficiency secondary)
- **AND** the table SHALL show relative performance vs. top performer

#### Scenario: Cross-Architecture Comparison

- **GIVEN** benchmarks for both quantum and classical brains in the same environment
- **WHEN** a user requests comparison
- **THEN** side-by-side performance SHALL be displayed
- **AND** quantum advantage metrics SHALL be highlighted (if quantum performs better)
- **AND** statistical significance SHALL be considered when comparing

#### Scenario: Personal Best Tracking

- **GIVEN** a contributor has submitted multiple benchmarks
- **WHEN** querying for contributor's benchmarks
- **THEN** all submissions by that contributor SHALL be listed
- **AND** personal best per category SHALL be highlighted
- **AND** improvement trends over time SHALL be shown

### Requirement: Benchmark Verification

The system SHALL support verification workflows for maintainers to validate benchmark submissions.

#### Scenario: Unverified Benchmark Submission

- **GIVEN** a new benchmark is submitted via PR
- **WHEN** the benchmark is saved
- **THEN** the verified field SHALL be set to false by default
- **AND** the benchmark SHALL appear in leaderboards but marked as "unverified"
- **AND** verification SHALL be required before the benchmark is considered official

#### Scenario: Maintainer Verification

- **GIVEN** a maintainer reviews a benchmark PR
- **WHEN** the maintainer runs verification workflow
- **THEN** the system SHALL attempt to reproduce results using stored config and commit
- **AND** if results match within tolerance, verified SHALL be set to true
- **AND** if results don't match, the maintainer SHALL be notified with discrepancies
- **AND** verification status SHALL be updated in the JSON file

#### Scenario: Re-verification After Code Changes

- **GIVEN** a verified benchmark that references an old commit
- **WHEN** major code changes occur
- **THEN** benchmarks MAY be marked for re-verification
- **AND** outdated benchmarks SHALL be clearly marked in leaderboards
- **AND** instructions for re-submission with newer commits SHALL be provided

### Requirement: Benchmark CLI Tools

The system SHALL provide command-line tools for managing benchmark submissions and querying leaderboards.

#### Scenario: Interactive Benchmark Submission

- **GIVEN** a user wants to submit a benchmark interactively
- **WHEN** the user runs `scripts/benchmark_submit.py submit`
- **THEN** the user SHALL be prompted for experiment ID to promote to benchmark
- **AND** the user SHALL be prompted for required metadata
- **AND** validation SHALL be performed before submission
- **AND** the benchmark file SHALL be created if validation passes

#### Scenario: Leaderboard Display

- **GIVEN** a user wants to view current leaderboards
- **WHEN** the user runs `scripts/benchmark_submit.py leaderboard`
- **THEN** all categories SHALL be listed with top performers
- **AND** the display SHALL match the README.md format
- **AND** filtering by category SHALL be supported

#### Scenario: Benchmark Regeneration

- **GIVEN** benchmarks have been updated or added
- **WHEN** a maintainer runs `scripts/benchmark_submit.py regenerate`
- **THEN** README.md and BENCHMARKS.md SHALL be updated with latest data
- **AND** existing content SHALL be preserved except benchmark tables
- **AND** generation timestamp SHALL be added as a comment

### Requirement: Documentation Integration

The system SHALL integrate benchmark leaderboards into project documentation with clear reproduction instructions.

#### Scenario: README Benchmark Section

- **GIVEN** benchmarks exist
- **WHEN** README.md is generated
- **THEN** a "🏆 Top Benchmarks" section SHALL be added after "Research Applications"
- **AND** summary tables SHALL show top 3 per major category
- **AND** a link to BENCHMARKS.md SHALL be provided for full details
- **AND** the section SHALL explain how to submit benchmarks

#### Scenario: BENCHMARKS.md Structure

- **GIVEN** benchmarks exist
- **WHEN** BENCHMARKS.md is generated
- **THEN** the file SHALL start with submission guidelines
- **AND** each category SHALL have its own section
- **AND** each section SHALL include full leaderboard table
- **AND** reproduction instructions SHALL be provided for each benchmark
- **AND** code examples SHALL show exact commands to reproduce results

#### Scenario: Config File Links

- **GIVEN** a benchmark references a config file
- **WHEN** leaderboard is generated
- **THEN** config file path SHALL be hyperlinked to GitHub repository
- **AND** the link SHALL point to the specific commit referenced in the benchmark
- **AND** clicking the link SHALL show the exact config used

### Requirement: Contributor Attribution

The system SHALL properly attribute benchmark submissions to contributors and support optional GitHub profile linking.

#### Scenario: Contributor Name Display

- **GIVEN** a benchmark with contributor name "John Doe"
- **WHEN** leaderboard is generated
- **THEN** the contributor name SHALL be displayed in the table
- **AND** if GitHub username is provided, name SHALL link to GitHub profile
- **AND** if GitHub username is not provided, name SHALL be displayed as plain text

#### Scenario: Anonymous Contributions

- **GIVEN** a user wants to submit a benchmark anonymously
- **WHEN** prompted for contributor name
- **THEN** the user MAY provide a pseudonym or "Anonymous"
- **AND** GitHub username SHALL remain optional
- **AND** the submission SHALL still be accepted

#### Scenario: Contributor History

- **GIVEN** a contributor has submitted multiple benchmarks
- **WHEN** querying contributor activity
- **THEN** all benchmarks by that contributor SHALL be listed
- **AND** total contribution count SHALL be displayed
- **AND** categories contributed to SHALL be shown

### Requirement: Predator-Enabled Benchmark Categories

The system SHALL provide separate benchmark categories for predator-enabled simulations to enable tracking learning performance on survival-foraging multi-objective tasks.

#### Scenario: Predator Quantum Small Category

- **GIVEN** a simulation with quantum brain (ModularBrain or QModularBrain), dynamic environment with `predators.enabled: true`, and grid size ≤ 20×20
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `predator_small/quantum`
- **AND** this SHALL be distinct from `foraging_small/quantum` (non-predator category)
- **AND** benchmarks SHALL track predator-specific metrics

#### Scenario: Predator Quantum Medium Category

- **GIVEN** a simulation with quantum brain, `predators.enabled: true`, and 20×20 < grid size ≤ 50×50
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `predator_medium/quantum`
- **AND** this SHALL use same grid size threshold as non-predator medium benchmarks

#### Scenario: Predator Quantum Large Category

- **GIVEN** a simulation with quantum brain, `predators.enabled: true`, and grid size > 50×50
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `predator_large/quantum`
- **AND** this SHALL represent the most challenging predator-enabled quantum scenarios

#### Scenario: Predator Classical Small Category

- **GIVEN** a simulation with classical brain (MLPBrain, QMLPBrain, or SpikingBrain), `predators.enabled: true`, and grid size ≤ 20×20
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `predator_small/classical`
- **AND** this SHALL enable comparison of classical vs quantum approaches on predator tasks

#### Scenario: Predator Classical Medium Category

- **GIVEN** a simulation with classical brain, `predators.enabled: true`, and 20×20 < grid size ≤ 50×50
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `predator_medium/classical`

#### Scenario: Predator Classical Large Category

- **GIVEN** a simulation with classical brain, `predators.enabled: true`, and grid size > 50×50
- **WHEN** benchmark category is determined
- **THEN** the category SHALL be `predator_large/classical`

#### Scenario: Predators Disabled Uses Foraging Categories

- **GIVEN** a simulation with `predators.enabled: false` or predators not configured
- **WHEN** benchmark category is determined
- **THEN** existing foraging categories SHALL be used
- **AND** categories SHALL be `foraging_small/quantum`, `foraging_medium/classical`, etc.
- **AND** backward compatibility SHALL be maintained

### Requirement: Predator Benchmark Metrics Tracking

Benchmark submissions for predator-enabled categories SHALL include predator-specific metrics in addition to standard foraging metrics.

#### Scenario: Predator Metrics in Benchmark Submission

- **GIVEN** a benchmark submission for category `predator_small/quantum`
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

- **GIVEN** multiple benchmark submissions in category `predator_medium/classical`
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

### Requirement: NematodeBench Public Documentation

The system SHALL provide comprehensive public documentation enabling external researchers to submit, reproduce, and validate benchmarks.

#### Scenario: Submission Guide Access

- **WHEN** an external researcher accesses `docs/nematodebench/SUBMISSION_GUIDE.md`
- **THEN** the document SHALL explain prerequisites (10+ sessions, 50+ runs per session, unique seeds)
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
- **AND** SHALL explain seed tracking requirements
- **AND** SHALL explain what makes a submission verifiable

### Requirement: Benchmark Submission Evaluation Script

The system SHALL provide an automated script for validating benchmark submissions before PR review.

#### Scenario: Structure Validation

- **WHEN** `scripts/evaluate_submission.py` is run on a benchmark JSON file
- **THEN** the script SHALL validate JSON structure matches NematodeBench schema
- **AND** SHALL check all required fields are present
- **AND** SHALL validate field types and ranges
- **AND** SHALL report specific validation errors

#### Scenario: Minimum Sessions Check

- **WHEN** evaluation is performed on a NematodeBench submission
- **THEN** the script SHALL verify total_sessions >= 10
- **AND** SHALL verify each session has num_runs >= 50
- **AND** SHALL reject submissions with fewer sessions/runs
- **AND** SHALL display clear error message with requirements

#### Scenario: Seed Uniqueness Check

- **WHEN** evaluation is performed on a NematodeBench submission
- **THEN** the script SHALL verify all_seeds_unique is true
- **AND** SHALL reject submissions with duplicate seeds
- **AND** SHALL display error message if seeds are not unique

### Requirement: Reproducibility Through Seeding

The system SHALL ensure all experiments are reproducible by automatically generating and tracking random seeds.

#### Scenario: Automatic Seed Generation

- **WHEN** an experiment is started without a seed parameter
- **THEN** the system SHALL generate a cryptographically random seed using `secrets.randbelow(2**32)`
- **AND** SHALL store the generated seed for the experiment
- **AND** SHALL use this seed consistently for all random number generation
- **AND** the seed SHALL be included in the experiment output

#### Scenario: Environment Reproducibility

- **WHEN** an environment is initialized with a seed
- **THEN** all random operations (food spawning, predator movement, initial positions) SHALL be deterministic
- **AND** running the same seed twice SHALL produce identical episode results

#### Scenario: Brain Reproducibility

- **WHEN** a brain is initialized with a seed
- **THEN** weight initialization SHALL be deterministic
- **AND** action selection (for stochastic policies) SHALL be reproducible
- **AND** PyTorch and NumPy random states SHALL be seeded consistently

#### Scenario: Per-Run Seed Tracking

- **WHEN** multiple runs are executed in an experiment
- **THEN** each run SHALL have its own seed
- **AND** per-run seeds SHALL be recorded in the experiment JSON
- **AND** any individual run SHALL be reproducible using its recorded seed

### Requirement: Enhanced Metrics for Benchmarks

The system SHALL compute additional metrics required for comprehensive benchmark evaluation.

#### Scenario: Learning Speed Calculation

- **WHEN** convergence analysis is performed on experiment results
- **THEN** the system SHALL calculate episodes to reach 80% rolling success rate
- **AND** SHALL compute learning_speed = 1.0 - (episodes_to_80 / max_episodes)
- **AND** learning_speed SHALL be in range [0, 1] where 1 = instant learning

#### Scenario: Stability Metric Calculation

- **WHEN** metrics are aggregated across multiple sessions
- **THEN** the system SHALL compute stability from coefficient of variation
- **AND** stability = 1.0 - (std / mean) for success rates, clamped to [0, 1]
- **AND** higher stability indicates more consistent results

#### Scenario: Statistical Aggregation

- **WHEN** per-session metrics are collected for NematodeBench submission
- **THEN** the system SHALL compute mean, std, min, max for each metric
- **AND** SHALL store these as StatValue objects
- **AND** metrics requiring aggregation: success_rate, composite_score, learning_speed, stability

### Requirement: NematodeBench Multi-Session Architecture

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

#### Scenario: NematodeBench Submission Schema

- **WHEN** a benchmark submission is created
- **THEN** the JSON SHALL use NematodeBenchSubmission schema
- **AND** SHALL include submission_id, brain_type, brain_config, environment, category
- **AND** SHALL include sessions array with SessionReference objects
- **AND** SHALL include total_sessions (minimum 10), total_runs
- **AND** SHALL include AggregateMetrics with StatValue objects
- **AND** SHALL include all_seeds_unique validation flag
- **AND** SHALL include contributor attribution

#### Scenario: Seed Uniqueness Validation

- **WHEN** multiple experiments are aggregated for NematodeBench submission
- **THEN** the system SHALL validate that ALL seeds are unique across ALL runs in ALL sessions
- **AND** duplicate seeds SHALL cause submission rejection
- **AND** all_seeds_unique flag SHALL be false if any duplicates exist

### Requirement: Experiment Storage and Tracking

The system SHALL persist experiment data in a structured folder hierarchy with automatic config preservation.

#### Scenario: Experiment Folder Structure

- **WHEN** an experiment is tracked
- **THEN** results SHALL be saved to `experiments/<experiment_id>/<experiment_id>.json`
- **AND** the config file used SHALL be copied to `experiments/<experiment_id>/<config_name>.yml`
- **AND** the folder SHALL be self-contained for reproducibility

#### Scenario: Benchmark Artifact Storage

- **WHEN** experiments are submitted as a NematodeBench benchmark
- **THEN** experiment JSONs SHALL be copied to `artifacts/benchmarks/<submission_id>/`
- **AND** each experiment JSON SHALL keep its original experiment_id as filename
- **AND** a single config.yml SHALL be copied from the first session
- **AND** the artifacts/benchmarks/ directory SHALL be committed to the repository
- **AND** session references in submissions SHALL point to individual JSON files

#### Scenario: Ad-hoc Experiment Storage

- **WHEN** experiments are referenced in logbooks or other documentation
- **THEN** experiment folders MAY be stored in `artifacts/experiments/`
- **AND** this storage is separate from official benchmark submissions
