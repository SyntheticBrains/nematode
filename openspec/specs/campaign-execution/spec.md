# campaign-execution Specification

## Purpose

This capability specifies how a **campaign** — a set of simulation runs forming one experiment — is executed across the machine's cores.

The project's statistical protocol is paired-seed: a result is a set of runs over the same seed list, compared arm by arm. Those runs are independent by construction, so the work is embarrassingly parallel; before this capability existed it was nonetheless executed one run at a time, leaving most of a multi-core machine idle for the duration of every campaign.

The central guarantee is narrow and deliberate: parallel execution changes **when** runs happen and never **what** they compute. Each run is a separate process invoking the standard single-run entry point with the command line a person would otherwise type by hand, so a campaign's results remain comparable with every result measured before this capability existed. That is why the requirements here concern isolation, bounded concurrency, per-run logging and failure reporting rather than anything about simulation behaviour — the moment a campaign runner starts reimplementing what a run does, the guarantee is gone.

## Requirements

### Requirement: Parallel campaign execution across configs and seeds

The project SHALL provide a campaign runner that executes a set of simulation runs — the **cross product** of one or more configs with a set of seeds — concurrently under a bounded worker limit, so that paired-seed protocols and multi-arm panels use available cores without changing the numerical result of any individual run.

Each run SHALL execute as a **separate operating-system process invoking the standard single-run entry point** with an explicit `--seed`, rather than in-process within a shared interpreter. This is the requirement's central guarantee: a run launched by the campaign runner SHALL be numerically indistinguishable from the same run launched by hand with the equivalent command line. The runner SHALL NOT alter, wrap, or reimplement any simulation, brain, environment, or learning-rule code path.

The runner SHALL:

- accept configs and seeds such that N configs × M seeds yields N·M runs, and forward arbitrary extra arguments to every child unchanged;
- limit concurrency to a configurable worker count, defaulting to a value derived from the host's CPU count that reserves headroom for the operating system rather than saturating every core;
- pin numerical-library thread counts to 1 in children, so that W workers do not each spawn a full thread pool and oversubscribe the machine;
- write each run's output to its own log file and rely on the existing collision-free session-identifier scheme for run artefacts, so concurrent runs SHALL NOT contend for output paths;
- report progress while running and a per-run status summary on completion;
- exit non-zero if any run fails, having still attempted every other run;
- terminate its children on interrupt rather than orphaning them;
- support a dry-run mode that prints the exact commands without executing them.

Thread pinning SHALL be numerically inert: it is applied because oversubscription is slow, and it is permitted only because results are bit-identical across thread counts.

#### Scenario: Cross product of configs and seeds

- **WHEN** the runner is invoked with 2 configs and 4 seeds
- **THEN** it SHALL plan exactly 8 runs, one per (config, seed) pair
- **AND** each planned command SHALL name its own config and its own seed

#### Scenario: A campaign run equals the equivalent hand-run

- **WHEN** a run is executed through the campaign runner and the same config/seed is executed directly through the single-run entry point
- **THEN** the reported per-episode results SHALL be identical
- **AND** the campaign runner SHALL contribute no simulation, brain, environment, or learning-rule code to either path

#### Scenario: Thread pinning does not change results

- **WHEN** identical work is executed with numerical-library threads set to 1 and to the host default
- **THEN** forward outputs at batch size 1 and at minibatch size, gradients after a backward pass, and a connectome-scale matmul SHALL each be bit-identical between the two settings

#### Scenario: Concurrency is bounded

- **WHEN** a campaign of many runs is executed with a worker limit of W
- **THEN** at most W child processes SHALL be running simultaneously
- **AND** every run SHALL eventually be executed

#### Scenario: Failures are reported without aborting the campaign

- **GIVEN** a campaign in which one run exits non-zero
- **WHEN** the campaign completes
- **THEN** every other run SHALL still have been attempted
- **AND** the summary SHALL identify which runs failed
- **AND** the runner SHALL exit non-zero

#### Scenario: Interrupt terminates children

- **WHEN** the runner receives an interrupt while children are running
- **THEN** it SHALL terminate its child processes
- **AND** SHALL NOT leave orphaned simulation processes behind

#### Scenario: Dry run executes nothing

- **WHEN** the runner is invoked in dry-run mode
- **THEN** it SHALL print one command per planned run
- **AND** SHALL start no child process
