## ADDED Requirements

### Requirement: A batched evaluator for frozen controllers

The package SHALL provide `BatchedEvaluator`. It SHALL run N animals of one config in lockstep under a
frozen controller, holding every per-animal state as arrays, with no per-animal Python on the step path
except random draws and bout starts. Within a step it SHALL keep the per-step runner's order:

- assay phase;
- sensing;
- controller;
- primitive move;
- lawn intake and satiety restoration;
- satiety decay.

#### Scenario: One call runs many animals

- **WHEN** the evaluator runs 256 seeds of an assay config
- **THEN** it SHALL return 256 animals' captures, each the length of the assay

### Requirement: A declared surface

The evaluator SHALL accept only:

- the continuous point worm under locomotor primitives, with a step of at most one second;
- the lawn food model with a centred lawn or an assay protocol;
- klinotaxis chemosensing with STAM and the fold-change adaptive sensor;
- a satiety that never ends the episode;
- a frozen MLP-PPO or settling connectome brain without traces, plasticity, node noise or
  state-dependent action noise, or a non-learning hazard controller.

Any other config SHALL be refused at construction, naming the setting.

#### Scenario: Random lawn placement is refused

- **WHEN** the evaluator is built on a config with `lawns.placement: random`
- **THEN** construction SHALL fail, naming the placement

### Requirement: Per-animal random streams

Animal *i*, with seed `s_i`, SHALL draw its environment's randomness from `default_rng(s_i)` and its
policy noise from a torch generator seeded `s_i`, in the per-step runner's order. Its behaviour SHALL
therefore not depend on the other animals or on N.

#### Scenario: Batch size does not change an animal

- **WHEN** one seed runs alone and inside a batch of 64
- **THEN** its bouts SHALL be the same in both for the first 300 steps
- **AND** its positions and headings SHALL agree within 1e-3 for the first 150

### Requirement: Layered parity with the per-step runner

On seeds from 5001–5032, for each supported controller and each assay:

- the evaluator's features, action means, modulators and STAM derivatives SHALL match the per-step
  runner's within 1e-5 over 50 steps;
- over 300 steps its bouts' kinds and start steps SHALL match exactly; over the first 150 its positions
  and headings SHALL match within 1e-3, and its satiety and lawn densities within 1e-6;
- over full Scheer and Flavell assays, each graded statistic's paired difference SHALL have a 95%
  bootstrap interval containing zero.

#### Scenario: A deliberately broken evaluator fails

- **WHEN** the evaluator is run with the derivative scale left at its written value, or with the
  head-sweep sides swapped
- **THEN** the state-level parity check SHALL fail, and with the sides swapped the trajectory check
  SHALL fail too

### Requirement: Compact trajectories, readable like the per-step runner's

Every run SHALL return compact arrays over animals and steps: position, heading, concentration, the
derivative feature, satiety, intake, on-lawn, primitive and phase. On request it SHALL build an
animal's steps as `BehaviourStep` records with the per-step runner's fields, its primitive and phase,
and the protocol's record.

#### Scenario: The classifier reads a batched animal

- **WHEN** a batched Flavell animal's capture is classified
- **THEN** it SHALL be read by `episode_windows` and the calibrated model unchanged

### Requirement: The connectome's modulator release is batch-safe

The connectome brain's modulator release SHALL average each animal's release cells separately on a
batched input. Its single-sample result SHALL be unchanged.

#### Scenario: Two animals release independently

- **WHEN** release is computed on a batch of two hidden states
- **THEN** each row SHALL equal the release computed on that state alone

### Requirement: A cost pilot sizes the fits

A pilot on seeds 5001–5032, with torch and BLAS each pinned to one thread, SHALL record, for each
controller and assay at N = 32, 128 and 256:

- µs per animal-step;
- the speedup per core over the per-step runner;
- the wall time of a 200-animal evaluation;
- W's projected budget.

It SHALL be committed with Logbook 104. The target is at least 20× per core. Missing it SHALL be
recorded, not hidden.

#### Scenario: The pilot reports the target

- **WHEN** the pilot runs
- **THEN** its record SHALL say, per controller, whether 25 µs per animal-step at N = 256 was met
