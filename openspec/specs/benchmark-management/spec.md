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

#### Scenario: Dynamic Medium Quantum Category
- **GIVEN** a benchmark using modular brain (quantum) with dynamic medium environment
- **WHEN** the benchmark is saved
- **THEN** the category SHALL be "dynamic_medium_quantum"
- **AND** the file SHALL be stored in `benchmarks/dynamic_medium/quantum/{timestamp}.json`

#### Scenario: Static Maze Classical Category
- **GIVEN** a benchmark using MLP brain (classical) with static maze environment
- **WHEN** the benchmark is saved
- **THEN** the category SHALL be "static_maze_classical"
- **AND** the file SHALL be stored in `benchmarks/static_maze/classical/{timestamp}.json`

#### Scenario: Dynamic Large Quantum Category
- **GIVEN** a benchmark using quantum brain with dynamic large environment (100×100)
- **WHEN** the benchmark is saved
- **THEN** the category SHALL be "dynamic_large_quantum"
- **AND** the file SHALL be stored in `benchmarks/dynamic_large/quantum/{timestamp}.json`

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
- **GIVEN** multiple benchmarks in the "dynamic_medium_quantum" category
- **WHEN** a user runs `scripts/benchmark_submit.py leaderboard dynamic_medium_quantum`
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
