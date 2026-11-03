## ADDED Requirements

### Requirement: The encoded connectome under the batched evaluator

`BatchedConnectome` SHALL support a brain built with `encoder: compact`, feeding it food contact as the per-step runner does. It SHALL meet the layered parity with the per-step runner on seeds 5001–5032. With the encoder unset, every existing connectome record SHALL be byte-identical to a baseline recorded before the encoder was added.

#### Scenario: The encoded brain matches the per-step runner

- **WHEN** an encoded brain runs 50 steps in the per-step runner and in the batched evaluator on the same seed
- **THEN** its features, action means and modulator concentrations SHALL agree within 1e-5

#### Scenario: Existing connectome arms are unchanged

- **WHEN** the hand-placed and data-layer connectome arms are run in the batched evaluator on the baseline's discarded seeds
- **THEN** every record SHALL be byte-identical to the baseline's
