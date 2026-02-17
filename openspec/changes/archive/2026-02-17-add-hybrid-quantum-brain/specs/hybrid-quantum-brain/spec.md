# hybrid-quantum-brain Specification

## Purpose

Define the requirements for the Hierarchical Hybrid Quantum Brain architecture that combines a QSNN reflex layer (QLIF quantum circuits with surrogate gradient REINFORCE) with a classical cortex MLP (PPO training) for multi-objective decision-making. The architecture uses mode-gated fusion to let the cortex modulate trust in the quantum reflex, enabling both reactive foraging and strategic predator evasion.

## ADDED Requirements

### Requirement: Hierarchical Architecture

The HybridQuantumBrain SHALL implement a two-component hierarchical architecture combining a QSNN reflex layer with a classical cortex MLP.

#### Scenario: QSNN Reflex Layer Initialization

- **WHEN** HybridQuantumBrain is instantiated
- **THEN** the system SHALL create a QLIFNetwork from `_qlif_layers.py` with configurable sensory, hidden, and motor neuron counts
- **AND** SHALL initialize QSNN weights using the same scale as QSNNReinforceBrain (`WEIGHT_INIT_SCALE = 0.15`)
- **AND** SHALL configure surrogate gradient parameters (alpha, shots, backend)

#### Scenario: Classical Cortex Initialization

- **WHEN** HybridQuantumBrain is instantiated
- **THEN** the system SHALL create a cortex actor MLP with input dimension equal to `num_sensory_neurons`
- **AND** the cortex actor SHALL have `cortex_num_layers` hidden layers of `cortex_hidden_dim` units each with ReLU activation
- **AND** the cortex actor output dimension SHALL be `num_motor_neurons + num_modes` (action biases + mode logits)
- **AND** SHALL initialize cortex weights with orthogonal initialization (gain=sqrt(2)) and zero biases

#### Scenario: Critic Initialization

- **WHEN** HybridQuantumBrain is instantiated
- **THEN** the system SHALL create a critic MLP with input dimension equal to `num_sensory_neurons` (sensory-only, no hidden spikes)
- **AND** the critic SHALL have `cortex_num_layers` hidden layers of `cortex_hidden_dim` units each with ReLU activation
- **AND** the critic output dimension SHALL be 1 (state value estimate)
- **AND** SHALL initialize critic weights with orthogonal initialization (gain=sqrt(2)) and zero biases

### Requirement: Mode-Gated Fusion

The HybridQuantumBrain SHALL combine QSNN reflex logits with cortex action biases using a learned mode gate.

#### Scenario: Cortex Output Splitting

- **WHEN** the cortex actor produces output
- **THEN** the system SHALL split the output into action biases (first `num_motor_neurons` values) and mode logits (remaining `num_modes` values)

#### Scenario: Mode Gate Computation

- **WHEN** mode logits are produced
- **THEN** the system SHALL apply softmax to obtain mode probabilities
- **AND** SHALL use the first mode probability (forage mode) as `qsnn_trust`
- **AND** `qsnn_trust` SHALL represent the weight given to QSNN reflex logits in the fusion

#### Scenario: Logit Fusion

- **WHEN** both QSNN reflex logits and cortex outputs are available
- **THEN** the system SHALL compute fused logits as: `final_logits = reflex_logits * qsnn_trust + action_biases`
- **AND** SHALL apply softmax with temperature to obtain action probabilities
- **AND** SHALL sample an action from the categorical distribution

#### Scenario: Stage 1 Bypass

- **WHEN** `training_stage` is 1
- **THEN** the system SHALL use QSNN reflex logits directly for action selection (no cortex fusion)
- **AND** SHALL NOT compute cortex forward pass

### Requirement: Stage-Aware Training

The HybridQuantumBrain SHALL support three training stages that control which components are trainable.

#### Scenario: Stage 1 — QSNN Reflex Only

- **WHEN** `training_stage` is 1
- **THEN** the system SHALL train the QSNN reflex via REINFORCE policy gradient
- **AND** SHALL NOT train or use the cortex actor, critic, or PPO buffer
- **AND** SHALL support multi-epoch REINFORCE with quantum caching (reusing `_qlif_layers.py` cached execution)
- **AND** SHALL support adaptive entropy regulation

#### Scenario: Stage 2 — Cortex PPO Only

- **WHEN** `training_stage` is 2
- **THEN** the system SHALL freeze all QSNN parameters (no gradient updates)
- **AND** SHALL run the QSNN forward pass to produce reflex logits (read-only)
- **AND** SHALL train the cortex actor and critic via PPO
- **AND** SHALL use the PPO rollout buffer for experience collection
- **AND** SHALL compute GAE advantages using the critic

#### Scenario: Stage 3 — Joint Fine-Tune

- **WHEN** `training_stage` is 3
- **THEN** the system SHALL train the QSNN reflex via REINFORCE with a reduced learning rate (`qsnn_lr * joint_finetune_lr_factor`)
- **AND** SHALL train the cortex actor and critic via PPO at the full cortex learning rate
- **AND** SHALL run both REINFORCE and PPO updates each episode

### Requirement: QSNN REINFORCE Training

The HybridQuantumBrain SHALL implement REINFORCE policy gradient for the QSNN reflex component, reusing patterns from QSNNReinforceBrain.

#### Scenario: REINFORCE Update Window

- **WHEN** the REINFORCE update window size is reached or an episode ends (during stage 1 or 3)
- **THEN** the system SHALL compute normalized discounted returns
- **AND** SHALL compute policy loss with entropy bonus
- **AND** SHALL clip gradients and update QSNN weights via Adam optimizer

#### Scenario: Multi-Epoch REINFORCE

- **WHEN** `num_reinforce_epochs` > 1
- **THEN** epoch 0 SHALL run quantum circuits and cache spike probabilities
- **AND** subsequent epochs SHALL reuse cached spike probs but recompute RY angles from updated weights

#### Scenario: Adaptive Entropy Regulation

- **WHEN** QSNN entropy drops below 0.5 nats
- **THEN** the system SHALL scale entropy_coef up to 20x to prevent policy collapse
- **WHEN** QSNN entropy exceeds 95% of maximum
- **THEN** the system SHALL suppress entropy bonus

### Requirement: Cortex PPO Training

The HybridQuantumBrain SHALL implement PPO for the cortex actor and critic, following patterns from MLPPPOBrain.

#### Scenario: Rollout Buffer Collection

- **WHEN** the cortex PPO is active (stage 2 or 3)
- **THEN** the system SHALL store per-step data in a rollout buffer: sensory features, fused action, log probability, reward, value estimate, done flag
- **AND** SHALL trigger a PPO update when the buffer reaches `ppo_buffer_size` steps

#### Scenario: GAE Advantage Computation

- **WHEN** a PPO update is triggered
- **THEN** the system SHALL compute GAE advantages with `gamma` and `gae_lambda`
- **AND** SHALL compute target returns as `advantages + values`
- **AND** SHALL normalize advantages to zero mean and unit variance

#### Scenario: PPO Clipped Surrogate Update

- **WHEN** performing a PPO update
- **THEN** the system SHALL split the buffer into `ppo_minibatches` minibatches
- **AND** SHALL perform `ppo_epochs` gradient steps per update
- **AND** SHALL compute the clipped surrogate objective: `min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)`
- **AND** SHALL compute value loss using Huber loss (smooth L1) against target returns
- **AND** SHALL add entropy bonus scaled by `entropy_coeff`
- **AND** SHALL update cortex actor and critic parameters via their respective optimizers

#### Scenario: Gradient Clipping

- **WHEN** performing PPO gradient updates
- **THEN** the system SHALL clip cortex actor gradients to `max_grad_norm`
- **AND** SHALL clip cortex critic gradients to `max_grad_norm`

### Requirement: Separate Optimizers

The HybridQuantumBrain SHALL maintain separate optimizers for each trainable component.

#### Scenario: QSNN Optimizer

- **WHEN** HybridQuantumBrain is instantiated
- **THEN** the system SHALL create an Adam optimizer for QSNN parameters (raw tensors: W_sh, W_hm, theta_hidden, theta_motor) with learning rate `qsnn_lr`

#### Scenario: Cortex Actor Optimizer

- **WHEN** HybridQuantumBrain is instantiated
- **THEN** the system SHALL create an Adam optimizer for cortex actor parameters with learning rate `cortex_actor_lr`

#### Scenario: Cortex Critic Optimizer

- **WHEN** HybridQuantumBrain is instantiated
- **THEN** the system SHALL create an Adam optimizer for cortex critic parameters with learning rate `cortex_critic_lr`

#### Scenario: Stage-Dependent Optimizer Activity

- **WHEN** `training_stage` is 1
- **THEN** only the QSNN optimizer SHALL be active
- **WHEN** `training_stage` is 2
- **THEN** only the cortex actor and critic optimizers SHALL be active
- **WHEN** `training_stage` is 3
- **THEN** all three optimizers SHALL be active

### Requirement: QSNN Weight Persistence

The HybridQuantumBrain SHALL support saving and loading QSNN weights for stage transitions.

#### Scenario: Auto-Save After Stage 1 Training

- **WHEN** training completes (final episode ends) and `training_stage` is 1
- **THEN** the system SHALL save QSNN weights (W_sh, W_hm, theta_hidden, theta_motor) to `exports/<session_id>/qsnn_weights.pt` via `torch.save`
- **AND** SHALL log the save path for the user to reference in stage 2/3 configs

#### Scenario: Load Pre-Trained QSNN Weights

- **WHEN** HybridQuantumBrain is instantiated with `qsnn_weights_path` set to a valid file path
- **THEN** the system SHALL load the weight dict via `torch.load`
- **AND** SHALL assign loaded tensors to the QSNN's W_sh, W_hm, theta_hidden, theta_motor
- **AND** SHALL validate that loaded tensor shapes match the current configuration

#### Scenario: Shape Mismatch on Load

- **WHEN** loaded QSNN weights have shapes that don't match the current config (num_sensory, num_hidden, num_motor)
- **THEN** the system SHALL raise a ValueError with a descriptive error message listing expected vs actual shapes

#### Scenario: Missing Weights File

- **WHEN** `qsnn_weights_path` is set but the file does not exist
- **THEN** the system SHALL raise a FileNotFoundError with the path

#### Scenario: Stage 2 Without Pre-Trained Weights

- **WHEN** `training_stage` is 2 or 3 and `qsnn_weights_path` is not set
- **THEN** the system SHALL log a warning that QSNN weights are randomly initialized (not pre-trained)
- **AND** SHALL proceed with random initialization

### Requirement: Hybrid Quantum Brain Configuration Schema

The configuration system SHALL support HybridQuantumBrain-specific parameters via Pydantic BaseModel.

#### Scenario: QSNN Reflex Parameters

- **WHEN** parsing a HybridQuantumBrain configuration
- **THEN** the system SHALL accept `num_sensory_neurons` (int, default 8)
- **AND** SHALL accept `num_hidden_neurons` (int, default 16)
- **AND** SHALL accept `num_motor_neurons` (int, default 4)
- **AND** SHALL accept `num_qsnn_timesteps` (int, default 4) for multi-timestep integration
- **AND** SHALL accept `backend` (str, default "pennylane") for quantum circuit execution
- **AND** SHALL accept `shots` (int, default 100) for quantum measurement shots
- **AND** SHALL accept `surrogate_alpha` (float, default 1.0)
- **AND** SHALL accept `logit_scale` (float, default 5.0) for spike-to-logit conversion
- **AND** SHALL accept `weight_clip` (float, default 3.0) for weight clamping bounds
- **AND** SHALL accept `theta_motor_max_norm` (float, default 2.0) for theta motor L2 norm clamping

#### Scenario: Cortex Parameters

- **WHEN** parsing a HybridQuantumBrain configuration
- **THEN** the system SHALL accept `cortex_hidden_dim` (int, default 64)
- **AND** SHALL accept `cortex_num_layers` (int, default 2) for number of hidden layers
- **AND** SHALL accept `num_modes` (int, default 3) for mode gate output dimension

#### Scenario: Training Stage Parameter

- **WHEN** parsing a HybridQuantumBrain configuration
- **THEN** the system SHALL accept `training_stage` (int, default 1, valid values: 1, 2, 3)

#### Scenario: QSNN Learning Parameters

- **WHEN** parsing a HybridQuantumBrain configuration
- **THEN** the system SHALL accept `qsnn_lr` (float, default 0.01)
- **AND** SHALL accept `qsnn_lr_decay_episodes` (int, default 400)
- **AND** SHALL accept `num_reinforce_epochs` (int, default 2)
- **AND** SHALL accept `reinforce_window_size` (int, default 20)
- **AND** SHALL accept `gamma` (float, default 0.99)

#### Scenario: Cortex PPO Learning Parameters

- **WHEN** parsing a HybridQuantumBrain configuration
- **THEN** the system SHALL accept `cortex_actor_lr` (float, default 0.001)
- **AND** SHALL accept `cortex_critic_lr` (float, default 0.001)
- **AND** SHALL accept `ppo_clip_epsilon` (float, default 0.2)
- **AND** SHALL accept `ppo_epochs` (int, default 4)
- **AND** SHALL accept `ppo_minibatches` (int, default 4)
- **AND** SHALL accept `ppo_buffer_size` (int, default 512)
- **AND** SHALL accept `gae_lambda` (float, default 0.95)
- **AND** SHALL accept `entropy_coeff` (float, default 0.01)
- **AND** SHALL accept `max_grad_norm` (float, default 0.5)

#### Scenario: Joint Fine-Tune Parameter

- **WHEN** parsing a HybridQuantumBrain configuration
- **THEN** the system SHALL accept `joint_finetune_lr_factor` (float, default 0.1) for QSNN LR reduction in stage 3

#### Scenario: Weight Persistence Parameters

- **WHEN** parsing a HybridQuantumBrain configuration
- **THEN** the system SHALL accept `qsnn_weights_path` (str, optional, default None) for loading pre-trained QSNN weights

#### Scenario: Sensory Module Parameters

- **WHEN** parsing a HybridQuantumBrain configuration
- **THEN** the system SHALL accept `sensory_modules` (list of ModuleName, optional) for unified sensory mode

#### Scenario: Configuration Validation

- **WHEN** validating HybridQuantumBrain configuration
- **THEN** the system SHALL require `num_sensory_neurons` >= 1
- **AND** SHALL require `num_hidden_neurons` >= 1
- **AND** SHALL require `num_motor_neurons` >= 2
- **AND** SHALL require `training_stage` in {1, 2, 3}
- **AND** SHALL require `shots` >= 100
- **AND** SHALL require `cortex_hidden_dim` >= 1
- **AND** SHALL require `cortex_num_layers` >= 1
- **AND** SHALL require `num_modes` >= 2
- **AND** SHALL require `ppo_buffer_size` >= `ppo_minibatches`

### Requirement: Brain Registration

The HybridQuantumBrain SHALL be registered in the brain plugin system.

#### Scenario: BrainType Enum

- **WHEN** the BrainType enum is defined
- **THEN** it SHALL include `HYBRID_QUANTUM = "hybridquantum"`

#### Scenario: Config Loader

- **WHEN** a configuration file specifies brain type as `"hybridquantum"`
- **THEN** the system SHALL resolve this to `HybridQuantumBrainConfig`

#### Scenario: Brain Factory

- **WHEN** `setup_brain_model` is called with `BrainType.HYBRID_QUANTUM`
- **THEN** it SHALL instantiate and return a `HybridQuantumBrain`

#### Scenario: Module Export

- **WHEN** `brain.arch` package is imported
- **THEN** `HybridQuantumBrain` and `HybridQuantumBrainConfig` SHALL be available in `__all__`

### Requirement: Diagnostics Logging

The HybridQuantumBrain SHALL log training diagnostics for both components.

#### Scenario: QSNN Diagnostics (Stage 1, 3)

- **WHEN** a QSNN REINFORCE update is performed
- **THEN** the system SHALL log `qsnn_policy_loss`, `qsnn_entropy`, `qsnn_grad_norm`
- **AND** SHALL log QSNN weight norms: `W_sh_norm`, `W_hm_norm`, `theta_h_norm`, `theta_m_norm`

#### Scenario: Cortex PPO Diagnostics (Stage 2, 3)

- **WHEN** a cortex PPO update is performed
- **THEN** the system SHALL log `cortex_policy_loss`, `cortex_value_loss`, `cortex_entropy`
- **AND** SHALL log `explained_variance` for the critic
- **AND** SHALL log `approx_kl` for early stopping diagnostics

#### Scenario: Fusion Diagnostics

- **WHEN** an episode completes (stage 2 or 3)
- **THEN** the system SHALL log `qsnn_trust_mean` (average mode_probs[0] across episode steps)
- **AND** SHALL log `mode_distribution` (average mode probabilities across episode steps)

### Requirement: Brain Interface Compatibility

The HybridQuantumBrain SHALL implement the `ClassicalBrain` protocol for compatibility with the simulation runner.

#### Scenario: run_brain Interface

- **WHEN** `run_brain()` is called with `BrainParams` and sensory data
- **THEN** the system SHALL extract features using sensory modules (if configured) or legacy mode
- **AND** SHALL compute QSNN reflex logits via QLIFNetwork forward pass
- **AND** SHALL compute cortex output (if stage >= 2) and fuse with reflex logits
- **AND** SHALL return a list containing one ActionData

#### Scenario: learn Interface

- **WHEN** `learn()` is called with reward and episode_done flag
- **THEN** the system SHALL accumulate reward in the appropriate buffer(s)
- **AND** SHALL trigger REINFORCE update when window is reached (stage 1, 3)
- **AND** SHALL store step in PPO rollout buffer and trigger PPO update when buffer is full (stage 2, 3)
- **AND** SHALL handle episode boundaries (reset QSNN state, clear buffers)

#### Scenario: Episode Reset

- **WHEN** a new episode begins
- **THEN** the system SHALL reset QSNN membrane potentials and refractory states
- **AND** SHALL clear REINFORCE episode buffers (stage 1, 3)
- **AND** SHALL NOT clear the PPO rollout buffer (it spans episodes)
