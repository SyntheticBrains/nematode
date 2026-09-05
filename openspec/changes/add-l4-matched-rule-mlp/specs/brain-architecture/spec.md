# Spec: brain-architecture

## ADDED Requirements

### Requirement: Plastic learning-rule selection on the MLP-PPO brain

`MLPPPOBrainConfig` SHALL offer the same `learning_rule` selection as the connectome brain — the PPO update, the reward-modulated three-factor rule, and its unmodulated ablation — defaulting to PPO, with the plasticity fields inherited from the shared definition rather than redeclared.

Under a plastic rule the brain SHALL update once per environment step as rewards arrive, SHALL NOT fill the rollout buffer or compute advantages, and SHALL skip the per-step critic call on both the discrete and the continuous action paths rather than satisfying it with a stub — leaving the per-step and bootstrap value unset. The critic and optimiser SHALL still be constructed, because construction order fixes the random-number stream that the PPO path's reproducibility depends on; they SHALL simply go unused.

The brain SHALL hold an MLP topology that **wraps the actor's existing linear layers by reference** rather than rebuilding them, registering its eligibility traces on the topology and not on the actor. The actor SHALL therefore be the same object under either rule, its state-dict keys unchanged, so weight persistence and the PPO path are byte-identical to the pre-change brain.

The plastic set SHALL be every linear weight matrix of the actor. Biases, the action-noise parameters, and any feature-gating weights SHALL stay at initialisation. The output layer learns: the asymmetry against the connectome's frozen anatomical readout is deliberate and makes the MLP a conservative yardstick, and it SHALL be stated wherever the arm's result is reported.

Eligibility for a feedforward layer SHALL be the same-step product of the layer's output and its input, oriented to the weight's shape, accumulated with the shared trace decay under `torch.no_grad()` once per environment step. The traced forward SHALL be bitwise-equal to the untraced actor forward on the same input.

#### Scenario: PPO path is byte-identical

- **WHEN** an MLP brain is constructed and trained with `learning_rule` unset
- **THEN** its parameters SHALL be bit-identical to a frozen pre-change reference after the same update sequence
- **AND** its weight components SHALL carry the same keys as before

#### Scenario: The actor is wrapped, not rebuilt

- **WHEN** a plastic MLP brain is constructed
- **THEN** the topology's layers SHALL be the same module objects as the actor's
- **AND** the actor's state dict SHALL contain no trace buffers

#### Scenario: Traced forward equals the actor forward

- **WHEN** the same input is passed through the traced forward and through the actor directly
- **THEN** the outputs SHALL be bitwise-equal

#### Scenario: Per-layer eligibility follows the closed form

- **WHEN** a scripted sequence of inputs is run with traces enabled
- **THEN** each layer's trace SHALL equal the decayed sum of that layer's same-step output-by-input products
- **AND** each trace SHALL have the shape of its layer's weight

#### Scenario: Plastic rule updates once per step without the PPO machinery

- **GIVEN** an MLP brain configured with a plastic rule
- **WHEN** a sequence of steps with rewards is run
- **THEN** one update SHALL occur per step
- **AND** the rollout buffer SHALL remain empty and the per-step value unset
- **AND** every linear weight SHALL have changed

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
