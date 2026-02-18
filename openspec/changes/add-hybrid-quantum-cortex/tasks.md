## 0. Extract Shared Hybrid Brain Infrastructure (Prerequisite)

- [x] 0.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/_hybrid_common.py` with shared infrastructure extracted from `hybridquantum.py`: `_CortexRolloutBuffer` class, `_fuse()` mode-gated fusion, `_cortex_forward()` and `_cortex_value()` classical cortex forward passes, `_init_cortex()` MLP initialization with orthogonal init, `_get_cortex_lr()` and `_update_cortex_learning_rate()` LR scheduling, cortex weight persistence helpers, PPO update logic, shared constants/defaults
- [x] 0.2 Refactor `hybridquantum.py` to import shared code from `_hybrid_common.py` instead of defining it inline — verify all existing tests pass after refactor
- [x] 0.3 Refactor `hybridclassical.py` to import shared code from `_hybrid_common.py` instead of its duplicated copies — verify all existing tests pass after refactor
- [x] 0.4 Run full test suite (`uv run pytest`) and lint (`uv run pre-commit run -a`) to confirm no regressions from the extraction refactor

## 1. Brain Type Registration

- [x] 1.1 Add `HYBRID_QUANTUM_CORTEX = "hybridquantumcortex"` to the `BrainType` enum in `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py`
- [x] 1.2 Add `BrainType.HYBRID_QUANTUM_CORTEX` to the `QUANTUM_BRAIN_TYPES` set in `dtypes.py`
- [x] 1.3 Add `HybridQuantumCortexBrain` and `HybridQuantumCortexBrainConfig` to `__all__` in `packages/quantum-nematode/quantumnematode/brain/arch/__init__.py`
- [x] 1.4 Add factory dispatch for `BrainType.HYBRID_QUANTUM_CORTEX` in `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` — follow the `HYBRID_QUANTUM` pattern (import, validate config type, instantiate brain)
- [x] 1.5 Add `HybridQuantumCortexBrainConfig` to `packages/quantum-nematode/quantumnematode/utils/config_loader.py`: import it, add to the `BrainConfigType` union, and add `"hybridquantumcortex": HybridQuantumCortexBrainConfig` entry to `BRAIN_CONFIG_MAP`

## 2. Configuration Schema

- [x] 2.1 Create `HybridQuantumCortexBrainConfig` Pydantic model in the new brain file with all fields from spec: reflex params (`num_sensory_neurons`, `num_hidden_neurons`, `num_motor_neurons`, `num_qsnn_timesteps`, `shots`, `surrogate_alpha`, `logit_scale`, `weight_clip`, `theta_motor_max_norm`), cortex params (`cortex_sensory_modules`, `cortex_neurons_per_group`, `cortex_hidden_neurons`, `cortex_output_neurons`, `num_cortex_timesteps`, `cortex_shots`, `num_modes`), training params (`training_stage`, `qsnn_lr`, `cortex_lr`, `critic_lr`, `gamma`, `gae_lambda`, `ppo_buffer_size`, `num_cortex_reinforce_epochs`, `use_gae_advantages`, `joint_finetune_lr_factor`, `entropy_coeff`, `max_grad_norm`), and weight persistence params (`reflex_weights_path`, `cortex_weights_path`, `critic_weights_path`)
- [x] 2.2 Add Pydantic validators: `training_stage` in {1,2,3,4}, `cortex_neurons_per_group` >= 2, `cortex_hidden_neurons` >= 4, `num_modes` >= 2, `cortex_sensory_modules` non-empty when `training_stage` >= 2

## 3. Core Brain Implementation

- [x] 3.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/hybridquantumcortex.py` with `HybridQuantumCortexBrain` class extending `ClassicalBrain`, importing shared infrastructure from `_hybrid_common.py` (rollout buffer, fusion, LR scheduling, weight persistence helpers)
- [x] 3.2 Implement QSNN reflex initialization (`_init_reflex_weights`) — identical to `HybridQuantumBrain._init_qsnn_weights()`: W_sh, W_hm, theta_hidden, theta_motor with `WEIGHT_INIT_SCALE`, `requires_grad_(True)`
- [x] 3.3 Implement grouped sensory QLIF cortex initialization (`_init_cortex_qsnn`): create per-group weight matrices `W_group[i]` with shape `(module_feature_dim, cortex_neurons_per_group)`, hidden weights `W_cortex_sh` with shape `(total_sensory_neurons, cortex_hidden_neurons)`, output weights `W_cortex_ho` with shape `(cortex_hidden_neurons, cortex_output_neurons)`, and theta parameters for hidden and output layers
- [x] 3.4 Implement classical critic initialization (`_init_critic`) — import the critic MLP initialization from `_hybrid_common.py` (sensory_dim → hidden → hidden → 1, orthogonal init with gain=sqrt(2), zero biases)
- [x] 3.5 Implement separate optimizer initialization (`_init_optimizers`): Adam for reflex params, Adam for cortex QSNN params (all group weights + hidden weights + output weights + thetas), Adam for critic params, with stage-dependent activity

## 4. Cortex QSNN Forward Pass

- [x] 4.1 Implement per-group sensory layer execution (`_cortex_sensory_forward`): for each modality group, call `execute_qlif_layer_differentiable` with the group's features and weights, then concatenate all group spike outputs into a single vector
- [x] 4.2 Implement shared hidden layer execution (`_cortex_hidden_forward`): pass concatenated sensory spikes through `execute_qlif_layer_differentiable` with `W_cortex_sh` and `theta_cortex_hidden`, applying fan-in-aware scaling
- [x] 4.3 Implement output layer execution (`_cortex_output_forward`): pass hidden spikes through `execute_qlif_layer_differentiable` with `W_cortex_ho` and `theta_cortex_output`
- [x] 4.4 Implement cortex multi-timestep integration (`_cortex_multi_timestep`): average cortex output spikes across `num_cortex_timesteps`, analogous to reflex `_multi_timestep`
- [x] 4.5 Implement cortex output mapping: convert output spike probs to action biases (neurons 0-3 via `(prob-0.5)*logit_scale`), mode logits (neurons 4-6 via `(prob-0.5)*mode_logit_scale`), and trust modulation (neuron 7 via raw prob)
- [x] 4.6 Implement differentiable and cached variants of cortex forward pass (for multi-epoch REINFORCE training), following the `_multi_timestep_differentiable` / `_multi_timestep_differentiable_cached` pattern from hybridquantum.py

## 5. Fusion and Action Selection

- [x] 5.1 Implement `_fuse()` — import from `_hybrid_common.py` mode-gated fusion: `final_logits = reflex_logits * qsnn_trust + action_biases`, with trust from `softmax(mode_logits)[0]`
- [x] 5.2 Implement `run_brain()`: preprocess reflex features (legacy 2-feature), preprocess cortex features (via `extract_classical_features` with `cortex_sensory_modules`), run reflex forward, run cortex forward (stage >= 2), fuse, select action with epsilon-greedy exploration
- [x] 5.3 Implement stage 1 bypass: use reflex logits directly, skip cortex forward pass entirely

## 6. Training: REINFORCE with GAE Advantages

- [x] 6.1 Implement reflex REINFORCE training (`_reflex_reinforce_update`) — reuse the REINFORCE update pattern from `hybridquantum.py` with window-based intra-episode updates, multi-epoch quantum caching, adaptive entropy regulation, weight clamping
- [x] 6.2 Implement rollout buffer — import `_CortexRolloutBuffer` from `_hybrid_common.py` for collecting (state, action, log_prob, reward, value, done) tuples
- [x] 6.3 Implement GAE advantage computation — use `_CortexRolloutBuffer.compute_returns_and_advantages()` from `_hybrid_common.py`
- [x] 6.4 Implement cortex REINFORCE+GAE update (`_cortex_reinforce_update`): re-run cortex QSNN forward pass (differentiable) for buffered states, compute log_probs and entropy, compute loss as `-log_prob * gae_advantage.detach() - entropy_coef * entropy`, backprop through surrogate gradients, clip gradients, step cortex optimizer
- [x] 6.5 Implement critic training update (`_critic_update`): Huber loss against target returns, gradient clipping, log explained variance
- [x] 6.6 Implement `use_gae_advantages=false` fallback: when disabled, use self-computed normalized discounted returns instead of critic GAE advantages for the cortex REINFORCE loss
- [x] 6.7 Implement `learn()` method: dispatch to reflex REINFORCE (stage 1,3,4), rollout buffer storage + cortex/critic updates (stage 2,3,4), episode boundary handling (reset both QSNN states, clear buffers, trigger final updates)

## 7. Curriculum and Weight Persistence

- [x] 7.1 Implement four-stage training logic: stage 1 (reflex only), stage 2 (cortex+critic, reflex frozen), stage 3 (all trainable, reflex at reduced LR), stage 4 (same as 3 with extended sensory modules)
- [x] 7.2 Implement reflex weight save/load (`_save_reflex_weights`, `_load_reflex_weights`): save W_sh, W_hm, theta_hidden, theta_motor to `reflex_weights.pt`, validate shapes on load
- [x] 7.3 Implement cortex weight save/load (`_save_cortex_weights`, `_load_cortex_weights`): save all group weights, hidden weights, output weights, and theta parameters to `cortex_weights.pt`, validate shapes on load
- [x] 7.4 Implement critic weight save/load: save/load critic state_dict to `critic_weights.pt`
- [x] 7.5 Implement auto-save on training completion: save reflex weights after stage 1, cortex+critic weights after stage 2, all weights after stage 3/4

## 8. Diagnostics and Logging

- [x] 8.1 Implement reflex diagnostics logging: `reflex_policy_loss`, `reflex_entropy`, `reflex_grad_norm`, weight norms (W_sh, W_hm, theta_h, theta_m)
- [x] 8.2 Implement cortex diagnostics logging: `cortex_policy_loss`, `cortex_entropy`, `cortex_grad_norm`, per-group and hidden/output weight norms
- [x] 8.3 Implement critic diagnostics logging: `critic_value_loss`, `critic_explained_variance`
- [x] 8.4 Implement fusion diagnostics logging: `qsnn_trust_mean`, `mode_distribution` (averaged across episode steps)

## 9. Configuration Files

- [x] 9.1 Create `configs/examples/hybridquantumcortex_foraging_small.yml` — stage 1 config: reflex-only training on foraging, legacy 2-feature mode, matching hybridquantum stage 1 environment settings
- [x] 9.2 Create `configs/examples/hybridquantumcortex_pursuit_predators_small.yml` — stage 2 config: cortex REINFORCE+GAE on pursuit predators with frozen reflex, `cortex_sensory_modules: [food_chemotaxis, nociception, mechanosensation]`, load reflex weights from stage 1
- [x] 9.3 Create `configs/examples/hybridquantumcortex_pursuit_predators_small_finetune.yml` — stage 3 config: joint fine-tune, load both reflex and cortex weights, `joint_finetune_lr_factor: 0.1`

## 10. Tests

- [x] 10.1 Create `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_hybridquantumcortex.py` with unit tests for: brain instantiation, config validation, grouped sensory QLIF forward pass, cortex output shape, mode-gated fusion, action selection
- [x] 10.2 Add tests for stage-aware training: verify reflex-only in stage 1, cortex+critic only in stage 2, all in stage 3/4, correct optimizer activity per stage
- [x] 10.3 Add tests for weight persistence: save/load roundtrip for reflex and cortex weights, shape mismatch error handling
- [x] 10.4 Add tests for REINFORCE+GAE: verify loss computation with detached advantages, verify fallback to pure REINFORCE when `use_gae_advantages=false`
- [x] 10.5 Add smoke test (mark with `@pytest.mark.smoke`): run 3-episode training session end-to-end on foraging config, verify brain produces valid actions and training step completes without error
- [x] 10.6 Run full test suite (`uv run pytest`) and lint (`uv run pre-commit run -a`) to verify no regressions
