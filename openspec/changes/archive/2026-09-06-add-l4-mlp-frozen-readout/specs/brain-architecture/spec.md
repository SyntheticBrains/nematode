## MODIFIED Requirements

### Requirement: Plastic learning-rule selection on the MLP-PPO brain

`MLPPPOBrainConfig` SHALL offer the same `learning_rule` selection as the connectome brain — the PPO update, the reward-modulated three-factor rule, and its unmodulated ablation — defaulting to PPO, with the plasticity fields inherited from the shared definition rather than redeclared.

Under a plastic rule the brain SHALL update once per environment step as rewards arrive, SHALL NOT fill the rollout buffer or compute advantages, and SHALL skip the per-step critic call on both the discrete and the continuous action paths rather than satisfying it with a stub — leaving the per-step and bootstrap value unset. The critic and optimiser SHALL still be constructed, because construction order fixes the random-number stream that the PPO path's reproducibility depends on; they SHALL simply go unused.

The brain SHALL hold an MLP topology that **wraps the actor's existing linear layers by reference** rather than rebuilding them, registering its eligibility traces on the topology and not on the actor. The actor SHALL therefore be the same object under either rule, its state-dict keys unchanged, so weight persistence and the PPO path are byte-identical to the pre-change brain.

The plastic set SHALL be selected by `plastic_layers`: `all` (the default) makes every linear weight matrix of the actor plastic; `hidden` makes every linear weight but the output layer's plastic, and the output layer SHALL NOT appear on the plastic-topology seam — no eligibility trace, no mask, no fan-in axis, no homeostatic target, and no plastic update. Biases, the action-noise parameters, and any feature-gating weights SHALL stay at initialisation under either setting. A plastic output layer takes its own output as its post-synaptic factor, and under the three-factor rule its rows rotate toward the hidden-activity direction that maximises the action mean until the actions saturate; the arm that is compared against the connectome's frozen anatomical readout SHALL therefore run with `hidden`, learning behind a fixed decoder as the connectome does, and the setting SHALL be stated wherever the arm's result is reported. The option SHALL have no effect on the gradient rule, which trains every actor parameter through the optimiser. With the default the build SHALL be bit-identical to the brain without this requirement.

Eligibility for a feedforward layer SHALL be the same-step product of the layer's output and its input, oriented to the weight's shape, accumulated with the shared trace decay under `torch.no_grad()` once per environment step. The traced forward SHALL run the whole actor and be bitwise-equal to the untraced actor forward on the same input, crediting eligibility to the plastic layers only.

Traces SHALL accrue from **exactly one** forward per environment step. The discrete action path evaluates the actor twice per step — once to select the action and once to record probabilities — and only the action-selecting evaluation SHALL update traces; the second SHALL not. Accruing from both would credit every synapse twice for one action and break the alignment between the eligibility and the reward that gates it.

#### Scenario: PPO path is byte-identical

- **WHEN** an MLP brain is constructed and trained with `learning_rule` unset
- **THEN** its parameters SHALL be bit-identical to a frozen pre-change reference after the same update sequence
- **AND** its weight components SHALL carry the same keys as before

#### Scenario: The actor is wrapped, not rebuilt

- **WHEN** a plastic MLP brain is constructed
- **THEN** the topology's layers SHALL be the same module objects as the actor's
- **AND** the actor's state dict SHALL contain no trace buffers
- **AND** after a deep copy of the brain, the copied topology's layers SHALL be the copied actor's layer objects

#### Scenario: Traced forward equals the actor forward

- **WHEN** the same input is passed through the traced forward and through the actor directly
- **THEN** the outputs SHALL be bitwise-equal
- **AND** under `hidden` no trace SHALL exist for the output layer while every hidden layer's trace accumulates

#### Scenario: Per-layer eligibility follows the closed form

- **WHEN** a scripted sequence of inputs is run with traces enabled
- **THEN** each plastic layer's trace SHALL equal the decayed sum of that layer's same-step output-by-input products
- **AND** each trace SHALL have the shape of its layer's weight

#### Scenario: One discrete step accrues one outer product per layer

- **GIVEN** a plastic MLP brain in discrete action mode with traces enabled
- **WHEN** exactly one environment step is taken from a reset
- **THEN** each plastic layer's trace SHALL equal that layer's output-by-input product from the action-selecting forward
- **AND** it SHALL NOT be twice that

#### Scenario: Plastic rule updates once per step without the PPO machinery

- **GIVEN** an MLP brain configured with a plastic rule
- **WHEN** a sequence of steps with rewards is run
- **THEN** one update SHALL occur per step
- **AND** the rollout buffer SHALL remain empty and the per-step value unset
- **AND** every plastic weight SHALL have changed

#### Scenario: Only linear weights are plastic on the MLP

- **WHEN** a plastic MLP brain trains
- **THEN** every bias, the action-noise parameters, and any gating weights SHALL be bit-identical to initialisation

#### Scenario: Frozen and unmodulated selections behave as on the connectome

- **GIVEN** an MLP brain with updates frozen
- **WHEN** an episode runs
- **THEN** every actor weight SHALL be bit-identical to initialisation
- **AND** an unmodulated MLP brain SHALL produce identical weights under different reward streams

#### Scenario: The magnitude bound clears the MLP's initialisation

- **WHEN** plastic MLP brains are constructed across several seeds
- **THEN** the largest initial linear weight SHALL be below the shared magnitude bound

#### Scenario: Default plastic depth is bit-identical

- **WHEN** an MLP brain is built with the default `plastic_layers`
- **THEN** its seam lists, trace buffers and forward SHALL be exactly today's

#### Scenario: A hidden-only substrate keeps its readout fixed

- **GIVEN** an MLP brain with `plastic_layers: hidden` under a plastic rule
- **WHEN** the rule steps with non-trivial traces and modulators
- **THEN** the output layer's weight and bias SHALL be bit-identical to their initial values
- **AND** at least one hidden weight SHALL have changed
- **AND** the seam SHALL expose one plastic tensor per hidden layer and none for the output layer

#### Scenario: The gradient rule ignores the plastic depth

- **WHEN** an MLP brain with `plastic_layers: hidden` trains under the gradient rule
- **THEN** every actor parameter SHALL remain trainable and the update SHALL be bit-identical to the frozen reference
