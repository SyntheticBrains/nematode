# cli-interface Specification

## Purpose

This specification defines the command-line interfaces for running simulations, managing experiments, and submitting benchmarks. It ensures consistent CLI argument parsing, help text, error handling, and output formatting across all CLI tools.

## Requirements

### Requirement: Brain Type Argument Extension

The CLI argument parser SHALL accept "spiking" as a valid brain type option.

#### Scenario: Spiking Brain Selection

- **GIVEN** a user wants to run simulation with spiking neural network
- **WHEN** they execute `python scripts/run_simulation.py --brain spiking`
- **THEN** the CLI SHALL parse "spiking" as a valid brain type
- **AND** SHALL pass the selection to the brain factory
- **AND** SHALL not raise validation errors

#### Scenario: Brain Type Help Text

- **GIVEN** a user requests help for brain type options
- **WHEN** they execute `python scripts/run_simulation.py --help`
- **THEN** the help text SHALL list "spiking" among valid brain types
- **AND** SHALL provide brief description of spiking neural network approach

### Requirement: Configuration Compatibility

The CLI SHALL support loading spiking brain configurations through existing configuration mechanisms.

#### Scenario: Spiking Configuration Loading

- **GIVEN** a YAML configuration file with spiking brain parameters
- **WHEN** loaded via `--config spikingreinforce_foraging_medium.yml`
- **THEN** the CLI SHALL parse the configuration
- **AND** SHALL validate spiking-specific parameters
- **AND** SHALL initialize the spiking brain with specified parameters

#### Scenario: Brain Type Override

- **GIVEN** a configuration file specifies a different brain type
- **WHEN** user provides `--brain spiking` CLI argument
- **THEN** the CLI argument SHALL override the configuration file
- **AND** SHALL use spiking brain regardless of config file brain type

### Requirement: Error Handling

The CLI SHALL provide clear error messages for spiking brain configuration issues.

#### Scenario: Invalid Spiking Parameters

- **GIVEN** a configuration with invalid spiking brain parameters
- **WHEN** the CLI attempts to initialize the brain
- **THEN** SHALL provide specific error message about invalid parameters
- **AND** SHALL suggest valid parameter ranges
- **AND** SHALL exit gracefully with appropriate error code

#### Scenario: Missing Dependencies

- **GIVEN** spiking brain is selected but required dependencies are missing
- **WHEN** the CLI attempts initialization
- **THEN** SHALL provide clear error about missing dependencies
- **AND** SHALL suggest installation commands

### Requirement: Experiment Tracking CLI Flags

The run_simulation.py script SHALL accept new command-line flags for experiment tracking and benchmark submission.

#### Scenario: Track Experiment Flag

- **GIVEN** a user wants to track experiment metadata
- **WHEN** the user runs `scripts/run_simulation.py --track-experiment ...`
- **THEN** experiment metadata SHALL be captured and saved after simulation completes
- **AND** the experiment ID SHALL be displayed in the output
- **AND** the path to the saved metadata file SHALL be shown

#### Scenario: Save Benchmark Flag

- **GIVEN** a user wants to submit a benchmark
- **WHEN** the user runs `scripts/run_simulation.py --save-benchmark ...`
- **THEN** experiment tracking SHALL be automatically enabled
- **AND** the user SHALL be prompted for benchmark metadata
- **AND** a benchmark file SHALL be created after validation
- **AND** instructions for creating a PR SHALL be displayed

#### Scenario: Benchmark Notes Flag

- **GIVEN** a user wants to provide optimization notes
- **WHEN** the user runs `scripts/run_simulation.py --save-benchmark --benchmark-notes "optimized LR schedule"`
- **THEN** the notes SHALL be included in the benchmark metadata
- **AND** the user SHALL not be prompted for notes interactively

#### Scenario: Help Text for Tracking Flags

- **GIVEN** a user runs `scripts/run_simulation.py --help`
- **WHEN** help text is displayed
- **THEN** the `--track-experiment` flag SHALL be documented as "Save experiment metadata for reproducibility and comparison"
- **AND** the `--save-benchmark` flag SHALL be documented as "Save experiment as a benchmark submission (implies --track-experiment)"
- **AND** the `--benchmark-notes` flag SHALL be documented as "Optional notes about optimization approach (requires --save-benchmark)"

### Requirement: Experiment Query CLI

The system SHALL provide a dedicated CLI tool for querying and analyzing stored experiments.

#### Scenario: Experiment List Command

- **GIVEN** experiments are stored
- **WHEN** a user runs `scripts/experiment_query.py list`
- **THEN** a formatted table of experiments SHALL be displayed
- **AND** the table SHALL show: ID, config, env type, brain type, success rate, date
- **AND** experiments SHALL be sorted by date (most recent first)

#### Scenario: Experiment Filter Options

- **GIVEN** a user runs `scripts/experiment_query.py list --help`
- **WHEN** help text is displayed
- **THEN** filter options SHALL include: --env-type, --brain-type, --min-success-rate, --since, --limit
- **AND** each filter option SHALL have clear documentation

#### Scenario: Experiment Show Command

- **GIVEN** an experiment ID
- **WHEN** a user runs `scripts/experiment_query.py show {experiment_id}`
- **THEN** detailed metadata SHALL be displayed in a readable format
- **AND** configuration, results, and system info SHALL be organized in sections
- **AND** the path to detailed exports SHALL be shown if available

#### Scenario: Experiment Compare Command

- **GIVEN** two experiment IDs
- **WHEN** a user runs `scripts/experiment_query.py compare {id1} {id2}`
- **THEN** a side-by-side comparison SHALL be displayed
- **AND** differences SHALL be highlighted with color coding (if terminal supports it)
- **AND** performance delta SHALL be shown for key metrics

### Requirement: Benchmark Management CLI

The system SHALL provide a dedicated CLI tool for managing benchmark submissions and viewing leaderboards.

#### Scenario: Benchmark Submit Command

- **GIVEN** an experiment ID
- **WHEN** a user runs `scripts/benchmark_submit.py submit {experiment_id}`
- **THEN** the experiment SHALL be promoted to a benchmark
- **AND** the user SHALL be prompted for required metadata
- **AND** validation SHALL be performed
- **AND** the benchmark file SHALL be created if valid

#### Scenario: Benchmark Leaderboard Command

- **GIVEN** benchmarks exist
- **WHEN** a user runs `scripts/benchmark_submit.py leaderboard`
- **THEN** all category leaderboards SHALL be displayed
- **AND** top performers SHALL be highlighted
- **AND** optional category filter SHALL be supported

#### Scenario: Benchmark Regenerate Command

- **GIVEN** benchmark data has changed
- **WHEN** a maintainer runs `scripts/benchmark_submit.py regenerate`
- **THEN** README.md and BENCHMARKS.md SHALL be updated
- **AND** git diff SHALL show what changed
- **AND** a summary of updates SHALL be displayed

#### Scenario: Benchmark Verify Command

- **GIVEN** an unverified benchmark
- **WHEN** a maintainer runs `scripts/benchmark_submit.py verify {benchmark_id}`
- **THEN** the system SHALL attempt to reproduce results
- **AND** verification status SHALL be updated based on results
- **AND** a verification report SHALL be generated

### Requirement: CLI Output Formatting

The experiment and benchmark CLI tools SHALL provide well-formatted, readable output for terminal display.

#### Scenario: Table Formatting

- **GIVEN** query results to display
- **WHEN** output is generated
- **THEN** tables SHALL be formatted with aligned columns
- **AND** headers SHALL be clearly distinguished
- **AND** terminal width SHALL be respected (no wrapping)

#### Scenario: Color Coding

- **GIVEN** a terminal that supports ANSI colors
- **WHEN** output is displayed
- **THEN** success indicators SHALL be green
- **AND** warnings SHALL be yellow
- **AND** errors SHALL be red
- **AND** important values SHALL be bold

#### Scenario: JSON Output Option

- **GIVEN** a user wants machine-readable output
- **WHEN** the user adds `--json` flag to any command
- **THEN** output SHALL be formatted as JSON
- **AND** all data SHALL be included (not truncated for display)
- **AND** JSON SHALL be pretty-printed for readability
