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

## MODIFIED Requirements

### Requirement: Documentation Integration (MODIFIED)
The system SHALL integrate benchmark leaderboards into project documentation with clear reproduction instructions AND public submission guidance.

#### Scenario: BENCHMARKS.md External Submissions Section
- **WHEN** BENCHMARKS.md is updated
- **THEN** the file SHALL include an "External Submissions" section
- **AND** SHALL link to docs/nematodebench/ documentation
- **AND** SHALL explain how external researchers can contribute
- **AND** SHALL highlight successfully verified external submissions
