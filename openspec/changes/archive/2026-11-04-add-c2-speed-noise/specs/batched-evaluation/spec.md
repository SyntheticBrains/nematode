## ADDED Requirements

### Requirement: Parity under slow speed noise

With `encoder_speed_noise: ou`, `BatchedConnectome` SHALL keep each animal's speed-noise state, reset at
`reset(n)` and carried across assay phases, and SHALL meet the layered parity with the per-step runner
on seeds 5001–5032 for the encoded wild type on Scheer's small lawn.

#### Scenario: The two runners agree under slow noise

- **WHEN** an encoded brain with slow noise runs in the per-step runner and in the batched evaluator on
  the same seed
- **THEN** their features, action means and modulators SHALL agree within 1e-5 over 50 steps, and their
  bouts and positions SHALL meet the trajectory-level parity

#### Scenario: An animal alone matches the animal in a batch

- **WHEN** one seed runs alone and inside a batch of 64 with slow noise on
- **THEN** its bouts SHALL be the same in both for the first 300 steps

#### Scenario: A phase start keeps the noise state

- **WHEN** a run crosses an assay's phase start with slow noise on
- **THEN** both runners SHALL carry the noise state across it and agree
- **AND** a batched controller tampered to reset the state at the phase, or to use ρ = 0, SHALL fail
  the check
