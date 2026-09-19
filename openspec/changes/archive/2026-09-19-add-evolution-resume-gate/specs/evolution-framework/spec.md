## MODIFIED Requirements

### Requirement: Evolution Loop Checkpoint and Resume

The `EvolutionLoop` SHALL pickle optimiser state at a configurable interval and SHALL support resuming
from a checkpoint without altering the deterministic behaviour of the run. Because a checkpoint is a
Python pickle, resuming SHALL require an explicit per-invocation opt-in from the caller.

#### Scenario: Checkpoint contains optimiser, generation, RNG state, and version

- **GIVEN** an evolution run with `checkpoint_every: 5` configured
- **WHEN** generation 5 completes
- **THEN** `output_dir/checkpoint.pkl` SHALL exist
- **AND** the pickled object SHALL include keys `optimizer`, `generation`, `rng_state`, `lineage_path`, `checkpoint_version`

#### Scenario: Resume continues from last checkpoint

- **GIVEN** a run was killed after writing a checkpoint at generation 5
- **WHEN** the run is invoked with `--resume <path>` **and** `--allow-unsafe-resume`
- **THEN** the loop SHALL resume from generation 6
- **AND** the optimizer's internal state (CMA-ES covariance matrix or GA population) SHALL be restored from the checkpoint

#### Scenario: Incompatible checkpoint version is rejected

- **GIVEN** a checkpoint pickle whose `checkpoint_version` does not match the current loop's expected version
- **WHEN** the loop attempts to resume
- **THEN** an error SHALL be raised with both the expected and found version numbers
- **AND** the loop SHALL NOT silently continue

## ADDED Requirements

### Requirement: Resuming from a pickled checkpoint requires an explicit opt-in

Where a driver resumes a run by unpickling a checkpoint, the driver SHALL require an explicit
per-invocation flag before any part of the checkpoint is read, because `pickle.load` executes code
contained in the file and only the caller knows the file's provenance. A warning in help text SHALL NOT
be treated as satisfying this requirement.

#### Scenario: Resume without the opt-in is refused before anything is unpickled

- **GIVEN** a resume invocation naming a checkpoint path
- **WHEN** the opt-in flag is absent
- **THEN** the driver SHALL exit non-zero without reading the checkpoint
- **AND** `pickle.load` SHALL NOT be reached
- **AND** the refusal SHALL name the mechanism — that the file is a pickle and that loading it executes
  arbitrary code from it — rather than only naming the flag

#### Scenario: The gate precedes file access, so a missing and a hostile checkpoint are indistinguishable

- **GIVEN** a resume invocation whose checkpoint path does not exist
- **WHEN** the opt-in flag is absent
- **THEN** the refusal SHALL be the opt-in refusal, not a missing-file error
- **AND** the driver SHALL NOT report whether the path was found

#### Scenario: The refusal is visible without a configured log handler

- **GIVEN** a driver whose logging handlers are installed later in its entry point than this gate
- **WHEN** the gate refuses
- **THEN** the refusal SHALL be written to standard error directly rather than through a logger whose
  handler does not yet exist
- **AND** a test asserting this SHALL run with the test harness's environment markers removed, so it
  exercises the handler-less path a user meets rather than the test-mode path

#### Scenario: The opt-in changes only permission, not behaviour

- **GIVEN** a resume invocation carrying the opt-in flag
- **WHEN** the run resumes
- **THEN** the checkpoint format, the version check, the inheritance-mismatch rejection and any
  torn-save detection SHALL behave exactly as they did before the gate existed
