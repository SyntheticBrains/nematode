## ADDED Requirements

### Requirement: Shared action-policy module

The system SHALL provide a single shared action-policy module that all PPO-family brains use for action sampling, log-probability, entropy, and PPO surrogate terms, supporting both a discrete mode (categorical over a fixed action set) and a continuous mode (tanh-squashed Gaussian). Each brain selects the mode appropriate to the active environment substrate.

#### Scenario: Discrete mode parity

- **WHEN** a PPO brain runs in discrete mode through the shared module
- **THEN** it produces a categorical action, log-probability, and entropy equivalent to the pre-refactor per-brain implementation (subject to the migration-regression bar)

#### Scenario: Continuous mode available

- **WHEN** a PPO brain runs in continuous mode through the shared module
- **THEN** it produces a continuous action vector with a corresponding log-probability and entropy from a tanh-squashed Gaussian

### Requirement: Tanh-squashed Gaussian continuous policy

The continuous mode SHALL parameterise a diagonal Gaussian over a 2-dimensional action, squash samples through `tanh`, and affine-rescale to the action ranges `speed ∈ [0, max]` and `turn ∈ [−π, π]`, applying the log-det-Jacobian correction to the log-probability.

#### Scenario: Bounded sampled actions

- **WHEN** a continuous action is sampled
- **THEN** `speed` lies within `[0, max]` and `turn` lies within `[−π, π]`

#### Scenario: Jacobian-corrected log-probability

- **WHEN** the log-probability of a continuous action is computed
- **THEN** it includes the tanh change-of-variables (log-det-Jacobian) correction, so the reported log-probability is that of the squashed, rescaled action

#### Scenario: Finite, non-degenerate distribution

- **WHEN** the policy produces a Gaussian for any valid observation
- **THEN** the standard deviation is strictly positive and bounded (log-std clamped), and sampling/log-prob/entropy are finite (no NaN/Inf)

### Requirement: Generic continuous action contract

The action carrier SHALL be extensible to a continuous action vector, and the agent/simulation loop SHALL consume discrete or continuous actions based on the active environment's action mode, without branching on the brain architecture.

#### Scenario: Continuous action carried end-to-end

- **WHEN** a brain emits a continuous action in the continuous-2D environment
- **THEN** the action carrier conveys the `(speed, turn)` vector from the brain through the runner to the environment's movement update

#### Scenario: No per-architecture branch in the loop

- **WHEN** the agent applies an action
- **THEN** the dispatch between discrete and continuous handling depends on the environment's action mode, and the agent/simulation/training loop contains no branch on the concrete brain class
