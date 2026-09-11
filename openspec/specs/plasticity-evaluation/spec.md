# plasticity-evaluation Specification

## Purpose

This specification defines the sequential multi-objective training evaluation protocol for testing catastrophic forgetting across quantum and classical brain architectures. The protocol trains a brain on a sequence of objectives (A → B → C → A'), measuring backward forgetting, forward transfer, and plasticity retention at each transition.

## Requirements

### Requirement: Sequential Multi-Objective Training Protocol

The system SHALL provide a CLI script (`scripts/run_plasticity_test.py`) that executes a sequential training protocol across multiple environment objectives while preserving brain weights between phases.

#### Scenario: Full four-phase sequential training

- **WHEN** a user runs `scripts/run_plasticity_test.py --config artifacts/logbooks/008/plasticity/<arch>_plasticity.yml`
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
- **AND** each seed SHALL construct a fresh brain instance via `setup_brain_model()` with the seed set in the brain config, producing an independent weight initialisation
- **AND** results SHALL be aggregated across seeds with mean and standard deviation

### Requirement: Evaluation Blocks at Transition Points

The system SHALL execute fixed-length evaluation episodes at each phase transition to measure task performance without training interference.

#### Scenario: Evaluation block execution

- **WHEN** a phase transition occurs (e.g., end of phase A, before phase B begins)
- **THEN** the system SHALL run the number of episodes specified in `plasticity.eval_episodes` on each relevant objective per the evaluation matrix
- **AND** before the eval block, the system SHALL save all brain `state_dict()`s (model parameters, optimizer state, normalisation layers) into an in-memory snapshot
- **AND** eval episodes SHALL run through the standard agent episode loop (learning may occur internally)
- **AND** after the eval block, the system SHALL restore all `state_dict()`s from the snapshot via `load_state_dict()` and clear any PPO buffers via `buffer.reset()`
- **AND** after restore, the brain SHALL be in exactly the same state as before the eval block

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

- **WHEN** training episode metrics are available for phase A and phase A'
- **THEN** the system SHALL compute plasticity retention by comparing convergence speed during phase A' (retraining) vs the original phase A
- **AND** convergence SHALL be defined as the first episode where the trailing-20-episode mean success rate exceeds a configurable threshold (default: 60%, configurable via `plasticity.convergence_threshold`)
- **AND** plasticity retention SHALL be expressed as: `PR = convergence_episodes_A / convergence_episodes_A'`
- **AND** PR > 1.0 indicates the brain relearns faster than it originally learned (positive plasticity)
- **AND** if a phase does not converge within `training_episodes_per_phase`, PR SHALL be reported as `N/A` for that seed and excluded from cross-seed aggregation

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
- **AND** `plasticity.convergence_threshold` SHALL default to 0.6 if not specified
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
- **THEN** the system SHALL provide a post-hoc comparison script (`scripts/compare_plasticity_results.py`) that accepts aggregate CSV paths via `--results path1.csv path2.csv ...` CLI arguments
- **AND** the script SHALL read the aggregate CSVs, match quantum/classical pairs by architecture name convention, and produce a combined comparison table
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

### Requirement: A substrate result is read against the rule's positive control

A registered result about whether a substrate's wiring is legible to a local plasticity rule SHALL
be read against that rule's positive control. Where the control has not been run, or has not
passed, the record SHALL state that a null is consistent both with the wiring carrying no signal
and with the rule not learning, and SHALL NOT attribute it to the wiring alone.

The control's pass rule SHALL be fixed before it runs: a stated margin over the computed
cue-blind floor on a stated number of seeds, with the reference and floor arms deciding validity.

#### Scenario: A null is not attributed to the wiring while the instrument is unverified

- **GIVEN** a registered substrate result that does not confirm its primary test
- **WHEN** it is written up and the rule's positive control has not passed
- **THEN** the record SHALL state that the null is consistent with the rule not learning

#### Scenario: The pass rule precedes the run

- **WHEN** the control is launched
- **THEN** its margin, seed count and budget SHALL already be recorded

### Requirement: A panel's contrast reads both components of its outcome

Where a panel's outcome is bimodal — a seed reaching a competent policy or a dead one — the
registered contrast SHALL read both the frequency with which an arm reaches competence and the
level it reaches when it does, because a test of either alone reports no effect when only the other
moves. The registered family SHALL comprise a paired competent-fraction discordance at the
committed competence threshold, a paired contrast on the level among competent seeds, and the
all-seeds paired rank test the committed record was scored with, corrected together.

The competence threshold SHALL be the committed one and SHALL NOT be re-chosen with the outcome known. The level contrast SHALL compare each arm's mean over its **own** competent seeds, so that a seed competent in one arm only contributes to that arm's level and a frequency difference is not read as a level difference; where either arm has no competent seed the contrast SHALL be reported as undefined rather than as a null result. Every member SHALL be one-sided in the arm-improves direction at the committed significance level, corrected together.

#### Scenario: A level-only effect is not reported as no effect

- **GIVEN** two arms reaching competence about equally often, where one arm's competent seeds score
  higher than the other's
- **WHEN** the family is computed
- **THEN** the level contrast SHALL report the difference, and the verdict SHALL NOT be the
  no-effect branch

#### Scenario: A frequency difference is not read as a level difference

- **GIVEN** two arms whose competent seeds score the same, where one arm reaches competence on
  more seeds than the other
- **WHEN** the family is computed
- **THEN** the frequency contrast SHALL report the difference and the level contrast SHALL NOT,
  and the verdict SHALL be the frequency-only branch

#### Scenario: A seed competent in one arm counts toward that arm's level

- **GIVEN** a seed competent in one arm and not the other
- **WHEN** the level contrast is computed
- **THEN** its value SHALL enter that arm's level and SHALL NOT enter the other's

#### Scenario: An undefined level contrast is not a null

- **GIVEN** a panel in which one arm has no competent seed
- **WHEN** the family is computed
- **THEN** the level contrast SHALL be reported as undefined, and SHALL NOT contribute a passing or
  failing result to the family

### Requirement: A graded metric is read beside the full-clear metric

A panel SHALL be read on a graded measure of task progress in addition to the full-clear rate, so
that learning short of a full clear is visible. The graded reading SHALL use the level and all-seeds members of the family over the competence the primary metric defines, corrected within itself, SHALL NOT choose a competence threshold of its own, and the full-clear metric SHALL remain the primary one in which registered verdicts are expressed.

#### Scenario: Progress short of a clear is visible

- **GIVEN** an arm whose full-clear rate is at its floor while its graded measure exceeds the
  comparator's
- **WHEN** both readings are computed
- **THEN** the graded reading SHALL report the difference, and the record SHALL carry both

### Requirement: A bimodal outcome has a name that licenses nothing

The outcome map SHALL name every combination of the frequency and level contrasts' directions, and
SHALL include a branch for a two-directional result: the two contrasts significant against each
other. That branch SHALL license no follow-on work and SHALL require its own registration to act
on.

A per-seed spread — some seeds improved and some degraded — SHALL NOT decide that branch, and a
panel with neither contrast significant SHALL be the no-effect branch whatever its spread. The
counts MAY be recorded descriptively. The reason is that the null of such a panel is itself
bimodal, so two arms drawn from one law routinely place a seed high in one and low in the other;
no threshold on that statistic distinguishes a mixed response from noise at these panel sizes.

#### Scenario: A split outcome is named rather than discovered

- **GIVEN** a panel whose frequency and level contrasts are significant in opposite directions
- **WHEN** the verdict is assigned
- **THEN** it SHALL be the split branch, and the record SHALL state that it licenses nothing

#### Scenario: A per-seed spread is not a mixed response

- **GIVEN** a panel with neither contrast significant, some seeds improved and some degraded
- **WHEN** the verdict is assigned
- **THEN** it SHALL be the no-effect branch

#### Scenario: Opposed significant contrasts are the split branch, not degradation

- **GIVEN** a panel where fewer seeds reach competence and those that do score higher, both
  significant
- **WHEN** the verdict is assigned
- **THEN** it SHALL be the split branch

#### Scenario: Missing significance alone is not the split branch

- **GIVEN** a panel with neither contrast significant and no seed improved above the comparator
- **WHEN** the verdict is assigned
- **THEN** it SHALL be the no-effect branch

### Requirement: The re-read of committed tables cannot change a committed verdict

Committed results SHALL be re-read under the registered family from their committed per-seed
tables. Each committed verdict SHALL stand as registered, in the units and under the rule it was
registered with; the re-read SHALL be reported beside it as a second, pre-specified reading and
SHALL be an input to the ladder re-read alone.

#### Scenario: A disagreement leaves the verdict standing

- **GIVEN** a committed result whose re-read points the other way
- **WHEN** the re-read is recorded
- **THEN** both readings SHALL be reported, the committed verdict SHALL be unchanged, and the record
  SHALL state which is the verdict

#### Scenario: An assay is re-read in its own protocol

- **GIVEN** a committed result whose protocol is an assay against a per-seed comparator rather than
  a panel against a paired arm
- **WHEN** it is re-read
- **THEN** the contrasts SHALL be computed against each seed's own committed comparator, and results
  SHALL NOT be pooled across protocols

### Requirement: An annealed perturbation clears the control before the assay

A variant that schedules its perturbation scale SHALL clear the rule's positive control under that
schedule before it is run through the clone assay, and the order SHALL be that one. The control run
SHALL use the same three validity arms, the same seeds and the same pass rule as the control the
constant-scale variant cleared, with the schedule compressed to the control's episode budget, and
SHALL report the update's alignment to the analytic policy gradient separately over the decay
and over the floor rather than as one mean. The bounds and length SHALL be the registered ones:
initial 0.2, final 0.02, decay over the first half of the budget.

A failure at the control SHALL stop the sequence, and SHALL be reported as a property of the
registered schedule rather than resolved by re-tuning its bounds or its length.

#### Scenario: The control gates the assay

- **GIVEN** a variant scheduling its perturbation scale
- **WHEN** it has not cleared the positive control under that schedule
- **THEN** its clone-assay result SHALL NOT be reported as evidence about retention

#### Scenario: The alignment is reported across the schedule

- **WHEN** an annealed arm's control run is recorded
- **THEN** the record SHALL carry the gradient alignment over the decay and over the floor
  separately, and the score over the floor, so that a schedule which anneals away its own signal —
  a decay-phase alignment that does not rise and a floor-phase score below the bar — is
  distinguishable from one whose floor-phase alignment is low only because the estimator is nearly
  silent there by construction

### Requirement: A scheduled arm's frozen control runs the same schedule

Where the clone assay screens an arm whose perturbation scale follows a schedule, the frozen control
required of a perturbing variant SHALL run that identical schedule with updates frozen. Its score
SHALL be read as a trajectory over the schedule rather than as a single endpoint, since a frozen
arm under a decaying scale recovers as the scale falls. Both arms' curves SHALL be binned in eight
equal parts of the budget with the scheduled scale stated per bin, and the learning arm SHALL be
read against the frozen arm bin by bin.

#### Scenario: The control anneals too

- **GIVEN** an annealed screening arm
- **WHEN** its frozen control is configured
- **THEN** the control SHALL carry the same initial scale, final scale and anneal length, and SHALL
  differ from the screening arm only in that updates are frozen

#### Scenario: The comparison is against the trajectory

- **WHEN** an annealed arm's assay result is reported
- **THEN** it SHALL be reported beside the frozen control's binned trajectory over the same
  schedule, and a claim that the rule damaged the policy SHALL require the learning arm to fall
  below the frozen arm in the floor-phase bins rather than below the committed comparator alone

### Requirement: A perturbing rule's endpoint is evaluated with the perturbation off

Where a plastic rule perturbs its units during training, its endpoint weights SHALL be evaluated
with updates frozen and the perturbation removed, under the clone assay's comparator protocol —
the comparator's own configuration with only the weights changed — so that what the rule learned is
measured apart from the exploration noise it learned under. The evaluation SHALL use the assay's
registered pass rule and comparator unchanged, and SHALL be reported beside the under-perturbation
score from the assay as a descriptive annotation.

The outcomes and what each licenses SHALL be recorded before the evaluation runs, and the
registration that follows SHALL be the one the outcome selects.

#### Scenario: The endpoint is the registered arm's

- **GIVEN** a perturbing arm the clone assay has scored
- **WHEN** its endpoint is staged for evaluation
- **THEN** the staged weights SHALL be that arm's auto-saved final weights, per seed, with the
  source of each recorded, and SHALL NOT be re-trained or selected

#### Scenario: The evaluation runs the comparator's condition

- **WHEN** the endpoint is evaluated
- **THEN** updates SHALL be frozen, the perturbation scale SHALL be zero, and every other key SHALL
  equal the comparator's, so that the configuration differs from the comparator's in the weights
  alone

#### Scenario: The loaded weights are verified to be the endpoint

- **GIVEN** the per-seed cosine between the perturbing arm's endpoint and its clone, as the assay
  recorded it
- **WHEN** the evaluation's own final weights are compared to the clone
- **THEN** each seed's cosine SHALL reproduce the recorded value within 0.01, and a seed that does
  not SHALL be void and SHALL void the verdict, since a cosine near one means the clone was
  evaluated rather than the endpoint

#### Scenario: The rule is the assay's

- **WHEN** the endpoint's scores are assessed
- **THEN** the pass rule, comparator values, budget and metric SHALL be those registered for the
  clone assay, and the verdict SHALL be one of the assay's own

#### Scenario: The outcome selects the next registration

- **GIVEN** the outcomes and their consequences recorded before the run
- **WHEN** the verdict is known
- **THEN** the change authored next SHALL be the one that verdict licenses, and the other SHALL NOT
  be authored on the same evidence
