## ADDED Requirements

### Requirement: Across-step leaky-integrator dynamics

The brain SHALL offer `dynamics: leaky` beside the default `dynamics: settling`. Under `leaky`, each
neuron SHALL carry a membrane potential across environment steps, reset to zero at episode start, and
each step SHALL integrate leak, ohmic gap-junction coupling on the potentials, chemical drive through
`tanh` and a sustained sensor current over `forward_pass_depth` sub-steps with time constant
`membrane_tau_steps`, treating leak and gap coupling implicitly. Under `leaky`, PPO SHALL replay
experience in contiguous chunks of `bptt_chunk_length` steps from each step's stored starting state.
Under `settling` every output, buffer and random draw SHALL be identical to the brain before this option
existed.

#### Scenario: Settling is byte-identical

- **WHEN** a run uses the default `dynamics: settling`
- **THEN** its actions, losses, weights and random draws SHALL equal those of the code before this
  option existed

#### Scenario: The step is stable on the raw gap weights

- **WHEN** the leaky substrate runs on the wild type's gap junctions with no rescaling, at any
  `membrane_tau_steps` above zero and any depth
- **THEN** the linear part of each sub-step SHALL contract, and the potentials SHALL stay bounded over an
  arbitrarily long episode under bounded input

#### Scenario: State carries across steps and resets at episode start

- **WHEN** two consecutive steps receive the same input
- **THEN** the second step SHALL start from the first step's final potentials
- **AND** the first step of a new episode SHALL start from zero

#### Scenario: Replay reproduces the rollout

- **WHEN** PPO replays a buffer at the parameters that collected it
- **THEN** every step's log-probability and value SHALL equal the rollout's within float tolerance,
  including chunks that cross an episode boundary and a partial final chunk

#### Scenario: Unsupported combinations are refused

- **WHEN** a configuration sets `dynamics: leaky` with a plasticity rule other than PPO, activity
  traces, e-prop or node noise
- **THEN** validation SHALL raise
