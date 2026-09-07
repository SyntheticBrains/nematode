## ADDED Requirements

### Requirement: The continuous action mean is reported beside the sample

`ActionData` SHALL carry `continuous_mean`, the tanh-squashed Gaussian mean rescaled to the
action bounds exactly as the sampled action is — the action the policy takes with its noise
removed — or `None`. The MLP-PPO and connectome brains SHALL set it on the continuous path from
the same forward pass that produced the sample. No sampled action, log-probability or update
SHALL change as a result.

#### Scenario: The mean is the noiseless action

- **GIVEN** a continuous MLP-PPO or connectome brain
- **WHEN** it runs one step
- **THEN** `continuous_mean` SHALL equal the squashed, rescaled mean recomputed from the same
  forward, and the sampled action and its probability SHALL be what they were before this
  requirement

#### Scenario: Discrete brains report none

- **WHEN** a discrete-action brain runs one step
- **THEN** `continuous_mean` SHALL be `None`
