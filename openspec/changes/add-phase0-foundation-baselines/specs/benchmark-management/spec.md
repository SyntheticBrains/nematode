# benchmark-management Specification Delta

## ADDED Requirements

### Requirement: NematodeBench Public Documentation
The system SHALL provide comprehensive public documentation enabling external researchers to submit, reproduce, and validate benchmarks.

#### Scenario: Submission Guide Access
**Given** an external researcher wants to submit a benchmark
**When** they access `docs/nematodebench/SUBMISSION_GUIDE.md`
**Then** the document SHALL explain prerequisites (50+ runs, clean git, config tracked)
**And** SHALL provide step-by-step submission process
**And** SHALL explain the PR and verification workflow
**And** SHALL include example commands with expected output

#### Scenario: Evaluation Methodology Documentation
**Given** a researcher wants to understand benchmark scoring
**When** they access `docs/nematodebench/EVALUATION.md`
**Then** the document SHALL explain composite score formula with weights
**And** SHALL explain convergence detection algorithm
**And** SHALL explain ranking criteria
**And** SHALL provide examples of score calculation

#### Scenario: Reproducibility Requirements Documentation
**Given** a researcher wants to ensure reproducible submissions
**When** they access `docs/nematodebench/REPRODUCIBILITY.md`
**Then** the document SHALL explain config file requirements
**And** SHALL explain git state requirements
**And** SHALL explain version tracking requirements
**And** SHALL explain what makes a submission verifiable

#### Scenario: Documentation Discoverability
**Given** a researcher is exploring the project
**When** they view BENCHMARKS.md
**Then** the document SHALL link to NematodeBench documentation
**And** SHALL include a call-to-action for external submissions
**And** SHALL explain the public benchmark program

### Requirement: Benchmark Submission Evaluation Script
The system SHALL provide an automated script for validating benchmark submissions before PR review.

#### Scenario: Structure Validation
**Given** a benchmark JSON file
**When** `scripts/evaluate_submission.py` is run
**Then** the script SHALL validate JSON structure matches schema
**And** SHALL check all required fields are present
**And** SHALL validate field types and ranges
**And** SHALL report specific validation errors

#### Scenario: Minimum Runs Check
**Given** a benchmark JSON with results
**When** evaluation is performed
**Then** the script SHALL verify num_runs >= 50
**And** SHALL reject submissions with fewer runs
**And** SHALL display clear error message with requirement

#### Scenario: Config Existence Check
**Given** a benchmark JSON with config_file path
**When** evaluation is performed
**Then** the script SHALL verify the config file exists
**And** SHALL verify config is tracked in git
**And** SHALL warn if config has been modified since submission

#### Scenario: Optional Reproduction
**Given** a benchmark JSON and `--reproduce` flag
**When** evaluation is performed
**Then** the script SHALL attempt to reproduce results
**And** SHALL run specified number of validation episodes
**And** SHALL compare reproduced results to claimed results
**And** SHALL report if results match within tolerance (10%)
**And** SHALL flag significant discrepancies

#### Scenario: Evaluation Report
**Given** a completed evaluation
**When** results are displayed
**Then** the script SHALL show PASS/FAIL status
**And** SHALL list all checks performed
**And** SHALL show composite score
**And** SHALL show category assignment
**And** SHALL provide actionable feedback for failures

## MODIFIED Requirements

### Requirement: Documentation Integration (MODIFIED)
The system SHALL integrate benchmark leaderboards into project documentation with clear reproduction instructions AND public submission guidance.

#### Scenario: BENCHMARKS.md External Submissions Section
**Given** benchmarks documentation is generated
**When** BENCHMARKS.md is updated
**Then** the file SHALL include an "External Submissions" section
**And** SHALL link to docs/nematodebench/ documentation
**And** SHALL explain how external researchers can contribute
**And** SHALL highlight successfully verified external submissions
