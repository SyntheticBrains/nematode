# LSTMPPO Brain — Delta for `add-transgenerational-memory-redesign` (M6.9+ PR-A)

## MODIFIED Requirements

### Requirement: TEI Logit-Bias Prior Attribute

`LSTMPPOBrain.tei_prior` SHALL accept either a 1-D `torch.Tensor` of shape `(num_actions,)` (legacy M6 form, preserved byte-equivalent) OR a callable bias-network that takes a sensory-input tensor and returns a 1-D logit-bias tensor of shape `(num_actions,)`. When `tei_prior is None`, the brain SHALL behave as if no prior is applied (M6 byte-equivalent default).

#### Scenario: tensor-form tei_prior (M6 byte-equivalent)

- **WHEN** `brain.tei_prior` is set to a 1-D `Tensor[num_actions]` and `run_brain` executes a step
- **THEN** the actor logits SHALL be biased by `logits + self.tei_prior` before softmax
- **AND** behaviour SHALL be byte-equivalent to M6 PR #166

#### Scenario: callable-form tei_prior accepts sensory_input

- **WHEN** `brain.tei_prior` is set to a callable (e.g. an `nn.Sequential` MLP) and `run_brain` executes a step
- **THEN** the brain SHALL invoke `self.tei_prior(sensory_input)` where `sensory_input` is the slice of `BrainParams` named in the substrate's `input_features`
- **AND** the returned 1-D tensor SHALL be added to actor logits before softmax
- **AND** the brain SHALL NOT crash if the callable returns a tensor with shape mismatching `num_actions` — instead it SHALL raise `RuntimeError` with the expected/actual shapes

#### Scenario: tei_prior=None preserves no-bias path

- **WHEN** `brain.tei_prior = None`
- **AND** `run_brain` executes
- **THEN** the actor logits SHALL be unmodified
- **AND** behaviour SHALL be byte-equivalent to a brain with no TEI substrate (legacy default)

### Requirement: PPO Sampling/Update Distribution Consistency

When `tei_prior` is a callable bias-network, the PPO rollout-buffer's recorded log-prob SHALL be computed under the SAME bias function that produced the action — i.e. the bias-network output for the step's sensory_input is applied identically at sampling time and at update time. The brain SHALL NOT recompute the bias under a different sensory_input snapshot.

#### Scenario: callable bias-network output is captured per-step

- **WHEN** the rollout buffer records a transition at step t
- **AND** the bias-network is callable with sensory_input s_t
- **THEN** the buffer SHALL store the per-step bias output (or sufficient information to reproduce it deterministically at update time)
- **AND** the PPO ratio computed at update time SHALL equal 1.0 on the first update (no off-policy correction needed)

#### Scenario: shape/dtype mismatch raises at learn()

- **WHEN** a rollout step's recorded bias output has shape mismatching the current bias-network's output shape (e.g. due to mid-window mutation)
- **AND** `brain.learn()` is called
- **THEN** the method SHALL raise `RuntimeError` with a clear diagnostic naming the mismatch
- **AND** the run SHALL fail-fast rather than silently produce off-policy gradients
