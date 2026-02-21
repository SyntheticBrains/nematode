# hybrid-quantum-cortex-brain Specification

## Purpose

Define the requirements for the HybridQuantumCortex brain architecture that combines a QSNN reflex layer (QLIF quantum circuits with surrogate gradient REINFORCE) with a QSNN cortex layer (grouped sensory QLIF neurons with shared hidden layer, REINFORCE with critic-provided GAE advantages) and a classical critic (PPO GAE) for multi-objective decision-making. The architecture uses mode-gated fusion identical to HybridQuantumBrain but increases the quantum parameter fraction from ~1% to ~11%.

## ADDED Requirements

### Requirement: Shared Hybrid Brain Infrastructure

The HybridQuantumCortexBrain SHALL reuse shared hybrid brain infrastructure from `_hybrid_common.py` rather than duplicating code from existing hybrid brain implementations.

#### Scenario: Rollout Buffer Reuse

- **WHEN** HybridQuantumCortexBrain collects experience during training (stage 2, 3, 4)
- **THEN** it SHALL import and use `_CortexRolloutBuffer` from `_hybrid_common.py`
- **AND** SHALL use the same GAE computation implementation as HybridQuantumBrain

#### Scenario: Fusion Logic Reuse

- **WHEN** HybridQuantumCortexBrain fuses reflex and cortex outputs
- **THEN** it SHALL use the shared mode-gated fusion function from `_hybrid_common.py`

#### Scenario: LR Scheduling Reuse

- **WHEN** HybridQuantumCortexBrain adjusts cortex learning rates during training
- **THEN** it SHALL use the shared LR scheduling functions from `_hybrid_common.py`

#### Scenario: Shared Constants

- **WHEN** HybridQuantumCortexBrain references default training hyperparameters
- **THEN** it SHALL import shared constants from `_hybrid_common.py` (e.g., `DEFAULT_GAMMA`, `DEFAULT_GAE_LAMBDA`, `DEFAULT_PPO_BUFFER_SIZE`)

### Requirement: Hierarchical Dual-QSNN Architecture

The HybridQuantumCortexBrain SHALL implement a two-QSNN hierarchical architecture combining a QSNN reflex layer with a QSNN cortex layer and a classical critic.

#### Scenario: QSNN Reflex Layer Initialization

- **WHEN** HybridQuantumCortexBrain is instantiated
- **THEN** the system SHALL create a QSNN reflex network using `_qlif_layers.py` infrastructure with configurable sensory, hidden, and motor neuron counts
- **AND** SHALL initialize reflex weights using `WEIGHT_INIT_SCALE = 0.15`
- **AND** SHALL configure surrogate gradient parameters (alpha, shots, backend)
- **AND** the reflex architecture SHALL be identical to HybridQuantumBrain's QSNN reflex

#### Scenario: QSNN Cortex Layer Initialization

- **WHEN** HybridQuantumCortexBrain is instantiated
- **THEN** the system SHALL create a QSNN cortex network with three layers: modality-grouped sensory QLIF neurons, shared hidden QLIF neurons, and output QLIF neurons
- **AND** SHALL initialize cortex weights using `WEIGHT_INIT_SCALE = 0.15`
- **AND** the cortex output dimension SHALL be `num_motor_neurons + num_modes + 1` (action biases + mode logits + trust modulation)

#### Scenario: Classical Critic Initialization

- **WHEN** HybridQuantumCortexBrain is instantiated
- **THEN** the system SHALL create a classical critic MLP with input dimension equal to the cortex sensory feature dimension
- **AND** the critic SHALL have configurable hidden layers with ReLU activation
- **AND** the critic output dimension SHALL be 1 (state value estimate)
- **AND** SHALL initialize critic weights with orthogonal initialization (gain=sqrt(2)) and zero biases

### Requirement: Grouped Sensory QLIF Processing

The QSNN cortex SHALL process multi-sensory input using modality-specific QLIF neuron groups with block-diagonal connectivity at the sensory layer.

#### Scenario: Modality Group Configuration

- **WHEN** the QSNN cortex is configured with `cortex_sensory_modules`
- **THEN** the system SHALL create one QLIF neuron group per sensory module
- **AND** each group SHALL have `cortex_neurons_per_group` QLIF neurons (default 4)
- **AND** the total sensory layer size SHALL be `num_groups * cortex_neurons_per_group`

#### Scenario: Block-Diagonal Sensory Weights

- **WHEN** cortex sensory weights are initialized
- **THEN** the system SHALL create per-group weight matrices `W_group[i]` with shape `(module_feature_dim, cortex_neurons_per_group)`
- **AND** each group's weights SHALL connect only to its corresponding sensory module features
- **AND** no cross-modality connections SHALL exist at the sensory layer

#### Scenario: Per-Group QLIF Execution

- **WHEN** the cortex sensory layer processes input
- **THEN** the system SHALL execute `execute_qlif_layer_differentiable` separately for each modality group
- **AND** SHALL pass only the group's corresponding sensory features as input
- **AND** SHALL concatenate group spike outputs into a single sensory spike vector
- **AND** SHALL apply fan-in-aware scaling within each group: `tanh(w·x / sqrt(group_fan_in))`

#### Scenario: Shared Hidden Layer Integration

- **WHEN** cortex sensory group outputs are concatenated
- **THEN** the system SHALL pass the concatenated spike vector through a fully-connected hidden QLIF layer using `execute_qlif_layer_differentiable`
- **AND** the hidden layer SHALL have `cortex_hidden_neurons` QLIF neurons (default 12)
- **AND** SHALL apply fan-in-aware scaling: `tanh(w·x / sqrt(total_sensory_neurons))`

#### Scenario: Adding a New Sensory Modality

- **WHEN** a new sensory module is added to `cortex_sensory_modules` configuration
- **THEN** the system SHALL create an additional QLIF group for the new module
- **AND** SHALL expand the shared hidden layer input dimension accordingly
- **AND** SHALL NOT require code changes (only configuration changes)

### Requirement: Mode-Gated Fusion

The HybridQuantumCortexBrain SHALL combine QSNN reflex logits with QSNN cortex outputs using a learned mode gate identical to HybridQuantumBrain.

#### Scenario: Cortex Output Splitting

- **WHEN** the QSNN cortex output layer produces spike probabilities
- **THEN** the system SHALL map output neurons as follows:
  - Neurons 0 to `num_motor_neurons - 1`: action bias logits via `(spike_prob - 0.5) * logit_scale`
  - Neurons `num_motor_neurons` to `num_motor_neurons + num_modes - 1`: mode logits via `(spike_prob - 0.5) * mode_logit_scale`
  - Last neuron (`num_motor_neurons + num_modes`): trust modulation via raw `spike_prob` (range [0, 1])
- **AND** the trust modulation neuron SHALL provide a direct continuous-valued scaling factor for the fusion mechanism

#### Scenario: Mode Gate Computation

- **WHEN** mode logits are produced
- **THEN** the system SHALL apply softmax to obtain mode probabilities
- **AND** SHALL use the first mode probability (forage mode) as `qsnn_trust`

#### Scenario: Logit Fusion

- **WHEN** both QSNN reflex logits and cortex outputs are available
- **THEN** the system SHALL compute fused logits as: `final_logits = reflex_logits * qsnn_trust + action_biases`
- **AND** SHALL apply softmax with temperature to obtain action probabilities
- **AND** SHALL sample an action from the categorical distribution

#### Scenario: Stage 1 Bypass

- **WHEN** `training_stage` is 1
- **THEN** the system SHALL use QSNN reflex logits directly for action selection (no cortex fusion)
- **AND** SHALL NOT compute cortex forward pass

### Requirement: Four-Stage Curriculum Training

The HybridQuantumCortexBrain SHALL support four training stages that control which components are trainable.

#### Scenario: Stage 1 — QSNN Reflex Only

- **WHEN** `training_stage` is 1
- **THEN** the system SHALL train the QSNN reflex via REINFORCE policy gradient
- **AND** SHALL NOT train or use the cortex QSNN, critic, or rollout buffer
- **AND** SHALL support multi-epoch REINFORCE with quantum caching
- **AND** SHALL support adaptive entropy regulation

#### Scenario: Stage 2 — QSNN Cortex with Critic

- **WHEN** `training_stage` is 2
- **THEN** the system SHALL freeze all QSNN reflex parameters (no gradient updates)
- **AND** SHALL run the QSNN reflex forward pass to produce reflex logits (read-only)
- **AND** SHALL train the QSNN cortex via REINFORCE with GAE advantages from the classical critic
- **AND** SHALL train the classical critic via Huber loss against observed returns
- **AND** SHALL use the rollout buffer for experience collection

#### Scenario: Stage 3 — Joint Fine-Tune

- **WHEN** `training_stage` is 3
- **THEN** the system SHALL train the QSNN reflex via REINFORCE with a reduced learning rate (`qsnn_lr * joint_finetune_lr_factor`)
- **AND** SHALL train the QSNN cortex via REINFORCE with GAE advantages at the full cortex learning rate
- **AND** SHALL train the classical critic via Huber loss
- **AND** SHALL run both reflex REINFORCE and cortex REINFORCE updates each episode

#### Scenario: Stage 4 — Multi-Sensory Scaling

- **WHEN** `training_stage` is 4
- **THEN** the system SHALL behave identically to stage 3
- **AND** SHALL support additional sensory modules in `cortex_sensory_modules` (e.g., thermotaxis)
- **AND** the QSNN cortex SHALL automatically include QLIF groups for all configured modules

### Requirement: QSNN Cortex REINFORCE with GAE Advantages

The HybridQuantumCortexBrain SHALL implement REINFORCE policy gradient for the QSNN cortex component using GAE advantages from the classical critic for variance reduction.

#### Scenario: Cortex REINFORCE Update

- **WHEN** the rollout buffer reaches `ppo_buffer_size` steps or an episode ends (during stage 2, 3, or 4)
- **THEN** the system SHALL compute GAE advantages using the classical critic's value estimates
- **AND** SHALL normalize advantages to zero mean and unit variance
- **AND** SHALL compute cortex policy loss as: `loss = -log_prob * gae_advantage.detach() - entropy_coef * entropy`
- **AND** the GAE advantages SHALL be detached from the cortex autograd graph (no gradient flow from critic through cortex)

#### Scenario: Cortex Multi-Epoch REINFORCE

- **WHEN** `num_cortex_reinforce_epochs` > 1
- **THEN** epoch 0 SHALL run quantum circuits and cache cortex spike probabilities
- **AND** subsequent epochs SHALL reuse cached spike probs but recompute RY angles from updated weights

#### Scenario: Cortex Adaptive Entropy Regulation

- **WHEN** cortex entropy drops below `entropy_floor` (0.5 nats)
- **THEN** the system SHALL scale entropy_coef up to `entropy_boost_max` to prevent policy collapse
- **WHEN** cortex entropy exceeds `entropy_ceiling_fraction` of maximum entropy
- **THEN** the system SHALL suppress entropy bonus

#### Scenario: Fallback to Pure REINFORCE

- **WHEN** `use_gae_advantages` is set to false
- **THEN** the system SHALL train the cortex QSNN with self-computed normalized discounted returns instead of critic-provided GAE advantages
- **AND** the critic SHALL NOT be used for advantage computation (but may still be trained for diagnostics)

### Requirement: Classical Critic Training

The HybridQuantumCortexBrain SHALL train the classical critic independently from the QSNN components using the rollout buffer.

#### Scenario: Critic Value Estimation

- **WHEN** a step is added to the rollout buffer
- **THEN** the system SHALL compute the critic value estimate from raw sensory features (not cortex QLIF hidden spikes)
- **AND** SHALL store the value estimate in the buffer alongside state, action, reward, log_prob, and done flag

#### Scenario: Critic Training Update

- **WHEN** the rollout buffer triggers an update
- **THEN** the system SHALL compute target returns from rewards and GAE advantages
- **AND** SHALL update critic parameters via Huber loss (smooth L1) against target returns
- **AND** SHALL clip critic gradients to `max_grad_norm`

#### Scenario: Critic Explained Variance Logging

- **WHEN** a critic update is performed
- **THEN** the system SHALL compute and log explained variance: `1 - var(returns - values) / var(returns)`
- **AND** SHALL log the explained variance for monitoring critic learning quality

### Requirement: Separate Optimizers

The HybridQuantumCortexBrain SHALL maintain separate optimizers for each trainable component.

#### Scenario: QSNN Reflex Optimizer

- **WHEN** HybridQuantumCortexBrain is instantiated
- **THEN** the system SHALL create an Adam optimizer for QSNN reflex parameters (W_sh, W_hm, theta_hidden, theta_motor) with learning rate `qsnn_lr`

#### Scenario: QSNN Cortex Optimizer

- **WHEN** HybridQuantumCortexBrain is instantiated
- **THEN** the system SHALL create an Adam optimizer for QSNN cortex parameters (all cortex weight matrices and theta parameters) with learning rate `cortex_lr`

#### Scenario: Critic Optimizer

- **WHEN** HybridQuantumCortexBrain is instantiated
- **THEN** the system SHALL create an Adam optimizer for critic parameters with learning rate `critic_lr`

#### Scenario: Stage-Dependent Optimizer Activity

- **WHEN** `training_stage` is 1
- **THEN** only the QSNN reflex optimizer SHALL be active
- **WHEN** `training_stage` is 2
- **THEN** only the cortex and critic optimizers SHALL be active
- **WHEN** `training_stage` is 3 or 4
- **THEN** all three optimizers SHALL be active

### Requirement: Weight Persistence

The HybridQuantumCortexBrain SHALL support saving and loading weights for both QSNN components across stage transitions.

#### Scenario: Auto-Save Reflex Weights After Stage 1

- **WHEN** training completes and `training_stage` is 1
- **THEN** the system SHALL save QSNN reflex weights (W_sh, W_hm, theta_hidden, theta_motor) to `exports/<session_id>/reflex_weights.pt`

#### Scenario: Auto-Save Cortex Weights After Stage 2

- **WHEN** training completes and `training_stage` is 2
- **THEN** the system SHALL save QSNN cortex weights (all group weights, hidden weights, output weights, theta parameters) to `exports/<session_id>/cortex_weights.pt`
- **AND** SHALL save critic weights to `exports/<session_id>/critic_weights.pt`

#### Scenario: Auto-Save All Weights After Stage 3 or 4

- **WHEN** training completes and `training_stage` is 3 or 4
- **THEN** the system SHALL save QSNN reflex weights to `exports/<session_id>/reflex_weights.pt`
- **AND** SHALL save QSNN cortex weights to `exports/<session_id>/cortex_weights.pt`
- **AND** SHALL save critic weights to `exports/<session_id>/critic_weights.pt`

#### Scenario: Load Pre-Trained Weights

- **WHEN** HybridQuantumCortexBrain is instantiated with `reflex_weights_path`, `cortex_weights_path`, or `critic_weights_path`
- **THEN** the system SHALL load the corresponding weight dict via `torch.load`
- **AND** SHALL validate that loaded tensor shapes match the current configuration
- **AND** SHALL raise a ValueError on shape mismatch with descriptive error message

#### Scenario: Stage 2 Without Pre-Trained Reflex

- **WHEN** `training_stage` >= 2 and `reflex_weights_path` is not set
- **THEN** the system SHALL log a warning that reflex weights are randomly initialized

### Requirement: HybridQuantumCortex Configuration Schema

The configuration system SHALL support HybridQuantumCortexBrain-specific parameters via Pydantic BaseModel.

#### Scenario: QSNN Reflex Parameters

- **WHEN** parsing a HybridQuantumCortexBrain configuration
- **THEN** the system SHALL accept the same QSNN reflex parameters as HybridQuantumBrainConfig: `num_sensory_neurons`, `num_hidden_neurons`, `num_motor_neurons`, `num_qsnn_timesteps`, `shots`, `surrogate_alpha`, `logit_scale`, `weight_clip`, `theta_motor_max_norm`

#### Scenario: QSNN Cortex Parameters

- **WHEN** parsing a HybridQuantumCortexBrain configuration
- **THEN** the system SHALL accept `cortex_sensory_modules` (list of ModuleName) for modality-specific QLIF groups
- **AND** SHALL accept `cortex_neurons_per_group` (int, default 4) for neurons per sensory modality group
- **AND** SHALL accept `cortex_hidden_neurons` (int, default 12) for shared hidden layer size
- **AND** SHALL accept `cortex_output_neurons` (int, default 8) for output layer size
- **AND** SHALL accept `num_cortex_timesteps` (int, default 4) for cortex multi-timestep integration
- **AND** SHALL accept `cortex_shots` (int, default 100) for cortex quantum measurement shots
- **AND** SHALL accept `num_modes` (int, default 3) for mode gate output dimension

#### Scenario: Training Parameters

- **WHEN** parsing a HybridQuantumCortexBrain configuration
- **THEN** the system SHALL accept `training_stage` (int, default 1, valid values: 1, 2, 3, 4)
- **AND** SHALL accept `qsnn_lr` (float, default 0.01) for reflex learning rate
- **AND** SHALL accept `cortex_lr` (float, default 0.01) for cortex QSNN learning rate
- **AND** SHALL accept `critic_lr` (float, default 0.001) for critic learning rate
- **AND** SHALL accept `gamma` (float, default 0.99)
- **AND** SHALL accept `gae_lambda` (float, default 0.95)
- **AND** SHALL accept `ppo_buffer_size` (int, default 512) for rollout buffer size
- **AND** SHALL accept `num_cortex_reinforce_epochs` (int, default 2)
- **AND** SHALL accept `use_gae_advantages` (bool, default true) to toggle critic-provided vs self-computed advantages
- **AND** SHALL accept `joint_finetune_lr_factor` (float, default 0.1)
- **AND** SHALL accept `entropy_coeff` (float, default 0.02)
- **AND** SHALL accept `max_grad_norm` (float, default 0.5)

#### Scenario: Weight Persistence Parameters

- **WHEN** parsing a HybridQuantumCortexBrain configuration
- **THEN** the system SHALL accept `reflex_weights_path` (str, optional) for loading pre-trained reflex weights
- **AND** SHALL accept `cortex_weights_path` (str, optional) for loading pre-trained cortex weights
- **AND** SHALL accept `critic_weights_path` (str, optional) for loading pre-trained critic weights

#### Scenario: Configuration Validation

- **WHEN** validating HybridQuantumCortexBrain configuration
- **THEN** the system SHALL require `training_stage` in {1, 2, 3, 4}
- **AND** SHALL require `cortex_sensory_modules` to be non-empty when `training_stage` >= 2
- **AND** SHALL require `cortex_neurons_per_group` >= 2
- **AND** SHALL require `cortex_hidden_neurons` >= 4
- **AND** SHALL require `num_modes` >= 2

### Requirement: Brain Registration

The HybridQuantumCortexBrain SHALL be registered in the brain plugin system.

#### Scenario: BrainType Enum

- **WHEN** the BrainType enum is defined
- **THEN** it SHALL include `HYBRID_QUANTUM_CORTEX = "hybridquantumcortex"`

#### Scenario: Config Loader

- **WHEN** a configuration file specifies brain type as `"hybridquantumcortex"`
- **THEN** the system SHALL resolve this to `HybridQuantumCortexBrainConfig`

#### Scenario: Brain Factory

- **WHEN** `setup_brain_model` is called with `BrainType.HYBRID_QUANTUM_CORTEX`
- **THEN** it SHALL instantiate and return a `HybridQuantumCortexBrain`

#### Scenario: Module Export

- **WHEN** `brain.arch` package is imported
- **THEN** `HybridQuantumCortexBrain` and `HybridQuantumCortexBrainConfig` SHALL be available in `__all__`

### Requirement: Diagnostics Logging

The HybridQuantumCortexBrain SHALL log training diagnostics for all components.

#### Scenario: QSNN Reflex Diagnostics

- **WHEN** a QSNN reflex REINFORCE update is performed (stage 1, 3, 4)
- **THEN** the system SHALL log `reflex_policy_loss`, `reflex_entropy`, `reflex_grad_norm`
- **AND** SHALL log reflex weight norms: `W_sh_norm`, `W_hm_norm`, `theta_h_norm`, `theta_m_norm`

#### Scenario: QSNN Cortex Diagnostics

- **WHEN** a QSNN cortex REINFORCE update is performed (stage 2, 3, 4)
- **THEN** the system SHALL log `cortex_policy_loss`, `cortex_entropy`, `cortex_grad_norm`
- **AND** SHALL log cortex weight norms per group and for hidden/output layers

#### Scenario: Critic Diagnostics

- **WHEN** a critic update is performed (stage 2, 3, 4)
- **THEN** the system SHALL log `critic_value_loss`, `critic_explained_variance`

#### Scenario: Fusion Diagnostics

- **WHEN** an episode completes (stage 2, 3, 4)
- **THEN** the system SHALL log `qsnn_trust_mean` (average trust across episode steps)
- **AND** SHALL log `mode_distribution` (average mode probabilities across episode steps)

### Requirement: Brain Interface Compatibility

The HybridQuantumCortexBrain SHALL implement the `ClassicalBrain` protocol for compatibility with the simulation runner.

#### Scenario: run_brain Interface

- **WHEN** `run_brain()` is called with `BrainParams` and sensory data
- **THEN** the system SHALL extract reflex features using legacy 2-feature preprocessing
- **AND** SHALL extract cortex features via `extract_classical_features` with `cortex_sensory_modules` (if stage >= 2)
- **AND** SHALL compute QSNN reflex logits via reflex network forward pass
- **AND** SHALL compute QSNN cortex output via grouped sensory QLIF forward pass (if stage >= 2)
- **AND** SHALL fuse reflex and cortex outputs via mode-gated fusion
- **AND** SHALL return a list containing one ActionData

#### Scenario: learn Interface

- **WHEN** `learn()` is called with reward and episode_done flag
- **THEN** the system SHALL accumulate reward in the appropriate buffer(s)
- **AND** SHALL trigger reflex REINFORCE update when window is reached (stage 1, 3, 4)
- **AND** SHALL store step in rollout buffer and trigger cortex REINFORCE + critic update when buffer is full (stage 2, 3, 4)
- **AND** SHALL handle episode boundaries (reset both QSNN states, clear buffers)

#### Scenario: Episode Reset

- **WHEN** a new episode begins
- **THEN** the system SHALL reset both QSNN reflex and cortex membrane potentials and refractory states
- **AND** SHALL clear REINFORCE episode buffers (stage 1, 3, 4)
- **AND** SHALL NOT clear the rollout buffer (it spans episodes)
