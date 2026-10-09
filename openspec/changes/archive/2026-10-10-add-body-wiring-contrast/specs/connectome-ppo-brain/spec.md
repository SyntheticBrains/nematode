## ADDED Requirements

### Requirement: A frozen-wiring learner

The connectome brain SHALL offer `freeze_wiring`, default false. When true under PPO, the chemical
weights SHALL stay at their initial draw while every other learnable parameter trains. It SHALL be
refused under any learning rule but PPO, and alongside `freeze_updates`. When false every output,
optimiser state and random draw SHALL be identical to the brain before this option existed.

#### Scenario: The wiring is read, not written

- **WHEN** a `freeze_wiring` brain takes a PPO update
- **THEN** its chemical weights SHALL be unchanged and its other learnable parameters SHALL be free to
  change

#### Scenario: Off is byte-identical

- **WHEN** `freeze_wiring` is false
- **THEN** the learnable parameters and their order SHALL equal the brain's before this option existed

### Requirement: The boundary null's body-drive boundary

Under `action_space: body_drive` the boundary-preserving null's motor boundary SHALL be every cell the
body drive reads: the 162 cells with neuromuscular junctions, which include every neuron of the four
motor classes.

#### Scenario: The boundary covers the cells the body reads

- **WHEN** a boundary-preserving null is drawn under the body drive
- **THEN** every chemical edge into a cell with a neuromuscular junction SHALL be the wild type's
