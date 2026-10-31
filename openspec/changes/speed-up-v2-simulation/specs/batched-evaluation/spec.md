## ADDED Requirements

### Requirement: The record conversion and C.1's draws are faster with identical output

`BatchedTrajectories.to_behaviour_steps` SHALL return, for every animal and step, the same `BehaviourStep` field values as the field-by-field conversion it replaces. `BatchedMechanistic` SHALL draw from each animal's torch generator the same uniforms in the same order as one draw of three per step, and every simulation's graded statistics and D SHALL be byte-identical to before.

#### Scenario: A converted animal is the same

- **WHEN** an animal from a short Scheer or Chen run is converted by both the new and the reference conversion
- **THEN** every field of every step SHALL be equal

#### Scenario: Block draws are the per-step draws

- **WHEN** an animal's generator is drawn in one block of 3·k uniforms and, from the same seed, k times three
- **THEN** the numbers SHALL be equal, and so SHALL the generator's next draw

#### Scenario: A new run starts a new buffer

- **WHEN** an evaluator runs twice with the same controller
- **THEN** each run SHALL draw from its own fresh generators, as before, with no uniform skipped or repeated
