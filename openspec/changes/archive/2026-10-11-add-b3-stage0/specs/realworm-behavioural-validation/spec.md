## ADDED Requirements

### Requirement: The lawn readout reads any learner

The lawn analysis SHALL read a named learner arm against a named floor arm. That covers the learning
gate, the evaluation and the registered dwelling reading, so a learner other than the MLP is read by
the same code.

#### Scenario: A connectome arm is read against its floor

- **WHEN** the lawn analysis is given the connectome's learner and floor arms
- **THEN** it SHALL pair their intake plateaus and their dwelling shares by seed, as it does the MLP's
