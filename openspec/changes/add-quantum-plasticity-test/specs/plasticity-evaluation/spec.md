# plasticity-evaluation Specification

## Purpose

This specification defines the sequential multi-objective training evaluation protocol for testing catastrophic forgetting across quantum and classical brain architectures. The protocol trains a brain on a sequence of objectives (A → B → C → A'), measuring backward forgetting, forward transfer, and plasticity retention at each transition.

## ADDED Requirements

### Requirement: Sequential Multi-Objective Training Protocol

The system SHALL provide a CLI script (`scripts/run_plasticity_test.py`) that executes a sequential training protocol across multiple environment objectives while preserving brain weights between phases.

#### Scenario: Full four-phase sequential training

- **WHEN** a user runs `scripts/run_plasticity_test.py --config configs/studies/plasticity/<arch>_plasticity.yml`
- **THEN** the system SHALL train the brain on four phases in sequence: A (foraging) → B (pursuit predators) → C (thermotaxis+pursuit) → A' (foraging return)
- **AND** all phases SHALL use the same grid size (100×100) to eliminate grid-size confounds in eval comparisons
- **AND** each phase SHALL run for the number of training episodes specified in the config's `plasticity.training_episodes_per_phase` field
- **AND** the brain's learned weights SHALL be preserved across all phase transitions without resetting
- **AND** evaluation blocks SHALL be executed at each transition point
- **AND** the script SHALL process a single architecture per invocation

#### Scenario: Brain weight preservation across environment switch

- **WHEN** the training transitions from phase A to phase B
- **THEN** the brain instance SHALL be the same object with the same parameter values as at the end of phase A
- **AND** only the environment, reward config, and agent components (satiety manager, reward calculator) SHALL be reconstructed for the new objective
- **AND** the brain's optimizer state (momentum, learning rate schedule progress) SHALL be preserved

#### Scenario: Multi-seed execution

- **WHEN** the config specifies multiple seeds in `plasticity.seeds`
- **THEN** the system SHALL execute the full four-phase protocol independently for each seed
- **AND** each seed SHALL produce an independent brain initialisation
- **AND** results SHALL be aggregated across seeds with mean and standard deviation

### Requirement: Evaluation Blocks at Transition Points

The system SHALL execute fixed-length evaluation episodes at each phase transition to measure task performance without training interference.

#### Scenario: Evaluation block execution

- **WHEN** a phase transition occurs (e.g., end of phase A, before phase B begins)
- **THEN** the system SHALL run the number of episodes specified in `plasticity.eval_episodes` on each relevant objective per the evaluation matrix
- **AND** the brain SHALL NOT update its weights during evaluation episodes (the eval runner SHALL skip all `learn()` and `post_process_episode()` calls on the brain)
- **AND** the brain's optimizer `state_dict()` SHALL be saved before evaluation and restored after
- **AND** PPO experience buffers SHALL be cleared via `buffer.reset()` before and after evaluation

#### Scenario: Full evaluation matrix

- **WHEN** the protocol runs evaluation blocks at each transition point
- **THEN** the system SHALL evaluate the following objectives at each transition point:

| Transition Point | Eval on A (foraging) | Eval on B (pursuit) | Eval on C (thermo+pursuit) |
|---|---|---|---|
| Pre-training (random baseline) | Yes | Yes | No |
| Post-A (after foraging training) | Yes | Yes | No |
| Post-B (after pursuit training) | Yes | Yes | No |
| Post-C (after thermo+pursuit training) | Yes | No | Yes |
| Post-A' (after foraging retraining) | Yes | No | No |

- **AND** each evaluation block SHALL record mean success rate, mean reward, and mean steps across all eval episodes

#### Scenario: Evaluation on current phase objective

- **WHEN** a training phase completes
- **THEN** the system SHALL evaluate the brain on that phase's objective as part of the evaluation matrix above
- **AND** this provides the "task competence" metric for the phase just completed

### Requirement: Plasticity Metrics Computation

The system SHALL compute backward forgetting, forward transfer, and plasticity retention metrics from the evaluation block results.

#### Scenario: Backward forgetting computation

- **WHEN** evaluation results are available for objective A at post-A, post-B, and post-C transition points
- **THEN** the system SHALL compute backward forgetting as: `BF = post_A_score - post_C_score_on_A`
- **AND** a positive BF value indicates forgetting (performance degraded)
- **AND** BF SHALL be computed per-seed and aggregated as mean ± std

#### Scenario: Forward transfer computation

- **WHEN** evaluation results are available for objective B at pre-training (random baseline) and at the post-A transition point (before B training)
- **THEN** the system SHALL compute forward transfer as: `FT = post_A_eval_on_B - random_baseline_on_B`
- **AND** a positive FT value indicates beneficial transfer (A-training helped B)

#### Scenario: Plasticity retention computation

- **WHEN** evaluation results are available for objective A at post-A and post-A' transition points
- **THEN** the system SHALL compute plasticity retention by comparing the convergence rate during phase A' (retraining on A) vs the original phase A
- **AND** plasticity retention SHALL be expressed as: `PR = convergence_episodes_A / convergence_episodes_A'`
- **AND** PR > 1.0 indicates the brain relearns faster than it originally learned (positive plasticity)

#### Scenario: Quantum vs classical forgetting comparison

- **WHEN** results are available for a quantum architecture and its classical control
- **THEN** the system SHALL compute the forgetting ratio: `FR = mean_BF_quantum / mean_BF_classical`
- **AND** the system SHALL perform a two-sample t-test on BF values across seeds
- **AND** FR ≤ 0.5 with p < 0.05 SHALL be reported as confirming the quantum plasticity hypothesis

### Requirement: Plasticity Test Configuration

The system SHALL accept YAML configuration files that define brain architecture, phase environments, and protocol parameters.

#### Scenario: Valid plasticity config loading

- **WHEN** a plasticity config file is provided with brain config, plasticity protocol parameters, and per-phase environment configs
- **THEN** the system SHALL validate that all required fields are present: `brain`, `plasticity.training_episodes_per_phase`, `plasticity.eval_episodes`, `plasticity.seeds`, and `plasticity.phases` (with at least 3 phases)
- **AND** each phase entry SHALL contain `name`, `environment`, and `reward` fields

#### Scenario: Per-phase environment and reward config

- **WHEN** a phase defines its environment configuration
- **THEN** that phase SHALL use the specified grid size, foraging params, predator params, health params, thermotaxis params, and reward config
- **AND** the brain architecture config SHALL remain constant across all phases

### Requirement: Results Export

The system SHALL export plasticity test results to CSV files for post-hoc analysis.

#### Scenario: Per-seed phase results CSV

- **WHEN** a plasticity test completes for a single seed
- **THEN** the system SHALL write a CSV file containing: seed, phase name, training episode metrics (per-episode success, reward, steps), and eval block results (mean success rate, mean reward)
- **AND** the CSV SHALL be written to `exports/{session_id}/plasticity/seed_{seed}/phase_results.csv`

#### Scenario: Aggregate metrics CSV

- **WHEN** all seeds complete for an architecture
- **THEN** the system SHALL write an aggregate CSV containing: architecture name, per-metric mean ± std across seeds for BF, FT, PR, and per-phase eval scores
- **AND** the CSV SHALL be written to `exports/{session_id}/plasticity/aggregate_metrics.csv`

#### Scenario: Cross-architecture comparison

- **WHEN** the user has completed plasticity tests for multiple architectures (separate invocations)
- **THEN** the system SHALL provide a post-hoc comparison script or function that reads aggregate CSVs from multiple runs and produces a combined comparison table
- **AND** the comparison SHALL include forgetting ratios and t-test p-values for quantum vs classical pairs (QRH vs CRH, HybridQuantum vs HybridClassical)

### Requirement: Brain Checkpoint Persistence

The system SHALL save brain weight checkpoints at each phase transition for reproducibility and debugging.

#### Scenario: Checkpoint save at phase transition

- **WHEN** a training phase completes and before evaluation begins
- **THEN** the system SHALL save the brain's current weights to disk at `exports/{session_id}/plasticity/seed_{seed}/checkpoint_post_{phase_name}.pt`
- **AND** for architectures with multiple components (e.g., HybridQuantum with reflex + cortex), all component weights SHALL be saved

#### Scenario: Checkpoint includes optimizer state

- **WHEN** a checkpoint is saved
- **THEN** the checkpoint file SHALL include both model parameters and optimizer state dictionaries
- **AND** loading a checkpoint SHALL restore the brain to the exact state at the point of saving
