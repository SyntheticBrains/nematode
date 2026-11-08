## ADDED Requirements

### Requirement: Parity with the encoder procedure's options on

`BatchedConnectome` SHALL match the per-step runner on an encoded brain with every option of the encoder procedure on: the soft cap, the pool centres, the graded box, the inhibition floor, the receptor signs, and slow speed noise.

#### Scenario: The two runners agree

- **WHEN** the same encoded brain, with every option on, runs one animal in each runner on the same seed
- **THEN** the two trajectories SHALL agree to the parity suite's existing tolerance
