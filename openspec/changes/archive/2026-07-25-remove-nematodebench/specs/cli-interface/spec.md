# cli-interface Specification

## MODIFIED Requirements

### Requirement: Experiment Tracking CLI Flags

The run_simulation.py script SHALL accept a command-line flag for experiment tracking.

The `--save-benchmark` and `--benchmark-notes` scenarios are removed. Note these flags **were never implemented** — `grep` for either string over `scripts/*.py` returns zero hits, and `parse_arguments()` in `scripts/run_simulation.py` defines neither. So this is drift correction as much as removal: the requirement described a CLI surface that never existed, and the code it would have driven is deleted under this change.

#### Scenario: Track Experiment Flag

- **GIVEN** a user wants to track experiment metadata
- **WHEN** the user runs `scripts/run_simulation.py --track-experiment ...`
- **THEN** experiment metadata SHALL be captured and saved after simulation completes
- **AND** the experiment ID SHALL be displayed in the output
- **AND** the path to the saved metadata file SHALL be shown

#### Scenario: Help Text for Tracking Flags

- **GIVEN** a user runs `scripts/run_simulation.py --help`
- **WHEN** help text is displayed
- **THEN** the `--track-experiment` flag SHALL be documented as "Save experiment metadata for reproducibility and comparison"

## REMOVED Requirements

### Requirement: Benchmark Management CLI

**Reason**: `scripts/benchmark_submit.py` is deleted with the NematodeBench submission and leaderboard system. All four scenarios (`submit`, `leaderboard`, `regenerate`, `verify`) were subcommands of that script; `regenerate` additionally targeted `BENCHMARKS.md`, which is also deleted.

**Migration**: None. `scripts/experiment_query.py` remains for listing, filtering, viewing and comparing tracked experiments, and is specified by the `Experiment Query CLI` requirement in this same capability.
