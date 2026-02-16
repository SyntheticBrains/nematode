# Tasks: add-hybrid-quantum-brain

## Configuration and Registration

- [ ] Add `HYBRID_QUANTUM = "hybridquantum"` to `BrainType` enum in `dtypes.py` and add to `BRAIN_TYPES` literal
- [ ] Add `HybridQuantumBrainConfig` to `BRAIN_CONFIG_MAP` in `config_loader.py` with key `"hybridquantum"`
- [ ] Add `HybridQuantumBrain` elif branch in `brain_factory.py` `setup_brain_model()` following existing QSNN pattern
- [ ] Add `HybridQuantumBrain`, `HybridQuantumBrainConfig` imports and exports in `brain/arch/__init__.py`

## Brain Implementation — Config

- [ ] Create `hybridquantum.py` with `HybridQuantumBrainConfig(BrainConfig)` Pydantic model containing: QSNN reflex params (num_sensory_neurons, num_hidden_neurons, num_motor_neurons, num_qsnn_timesteps, backend, shots, surrogate_alpha, logit_scale, weight_clip, theta_motor_max_norm), cortex params (cortex_hidden_dim, cortex_num_layers, num_modes), training stage (training_stage), QSNN REINFORCE params (qsnn_lr, qsnn_lr_decay_episodes, num_reinforce_epochs, reinforce_window_size, gamma), cortex PPO params (cortex_actor_lr, cortex_critic_lr, ppo_clip_epsilon, ppo_epochs, ppo_minibatches, ppo_buffer_size, gae_lambda, entropy_coeff, max_grad_norm), joint fine-tune param (joint_finetune_lr_factor), sensory_modules, with Pydantic validators for all constraints

## Brain Implementation — Core Architecture

- [ ] Implement `HybridQuantumBrain(ClassicalBrain)` `__init__` with: QLIFNetwork creation (from `_qlif_layers.py`), cortex actor MLP (input=num_sensory, hidden=cortex_hidden_dim x cortex_num_layers, output=num_motor+num_modes), cortex critic MLP (input=num_sensory, hidden=cortex_hidden_dim x cortex_num_layers, output=1), three separate Adam optimizers (QSNN, cortex actor, cortex critic), orthogonal weight init for cortex, PPO rollout buffer, REINFORCE episode buffers
- [ ] Implement `run_brain()` method: extract sensory features (unified or legacy mode), run QSNN forward pass (QLIFNetwork multi-timestep), compute reflex logits via logit scaling, conditionally run cortex forward pass (stage >= 2), apply mode-gated fusion, sample action, store step data in appropriate buffers, return ActionData
- [ ] Implement `learn()` method: accumulate reward, handle stage-dependent training (stage 1: REINFORCE only, stage 2: PPO only, stage 3: both), trigger updates at window/buffer boundaries, handle episode reset (clear QSNN state, clear REINFORCE buffers, keep PPO buffer)

## Brain Implementation — QSNN REINFORCE Training

- [ ] Implement QSNN REINFORCE update: compute normalized discounted returns, compute policy loss with entropy bonus, clip gradients, update QSNN weights via Adam, support multi-epoch with quantum caching (epoch 0 runs circuits + caches, subsequent epochs reuse cache), adaptive entropy regulation (scale up when < 0.5 nats, suppress when > 95% max)
- [ ] Implement QSNN weight management: weight clamping to `[-weight_clip, weight_clip]` after updates, theta motor L2 norm clamping to `theta_motor_max_norm`, learning rate decay schedule

## Brain Implementation — Cortex PPO Training

- [ ] Implement PPO rollout buffer: store per-step (sensory features, action, log_prob, reward, value estimate, done), trigger update when buffer reaches `ppo_buffer_size`
- [ ] Implement PPO update: compute GAE advantages (gamma, gae_lambda), normalize advantages, split into minibatches, run `ppo_epochs` gradient steps with clipped surrogate objective, Huber value loss, entropy bonus, gradient clipping for both actor and critic
- [ ] Implement cortex diagnostics logging: cortex_policy_loss, cortex_value_loss, cortex_entropy, explained_variance, approx_kl per PPO update

## Brain Implementation — Weight Persistence

- [ ] Implement `_save_qsnn_weights(session_id)` method: save QSNN tensors (W_sh, W_hm, theta_hidden, theta_motor) to `exports/<session_id>/qsnn_weights.pt` via `torch.save`, log the save path
- [ ] Implement `_load_qsnn_weights()` in `__init__`: when `qsnn_weights_path` is set, load via `torch.load`, validate shapes match config, assign to QSNN tensors. Raise FileNotFoundError if missing, ValueError if shape mismatch
- [ ] Wire auto-save into `learn()`: when episode_done and training is complete (or at configurable intervals), call `_save_qsnn_weights`. Log warning if stage >= 2 and no `qsnn_weights_path` provided

## Brain Implementation — Fusion and Diagnostics

- [ ] Implement fusion diagnostics: track qsnn_trust per step, log qsnn_trust_mean and mode_distribution per episode
- [ ] Implement QSNN diagnostics logging: qsnn_policy_loss, qsnn_entropy, qsnn_grad_norm, weight norms (W_sh, W_hm, theta_h, theta_m) per REINFORCE update
- [ ] Implement `update_memory()`, `build_brain()` stub (return empty QuantumCircuit), `inspect_circuit()` stub, and remaining `Brain` protocol methods

## Example Configs

- [ ] Create `hybridquantum_foraging_small.yml` for stage 1 (QSNN reflex training on foraging environment, based on existing `qsnnreinforce_foraging_small.yml` params)
- [ ] Create `hybridquantum_pursuit_predators_small.yml` for stage 2 (cortex PPO training on pursuit-predator environment with frozen QSNN, using environment settings from existing `qsnnreinforce_a2c_pursuit_predators_small.yml`)

## Tests

- [ ] Create `test_hybridquantum.py` with tests for: config validation (valid defaults, invalid training_stage, invalid shots, etc.), brain instantiation (all components created, correct dimensions), QSNN forward pass (reflex logits shape and range), cortex forward pass (output splitting into action_biases + mode_logits), fusion mechanism (mode gating math, stage 1 bypass), stage-aware optimizer activity (correct optimizers active per stage), REINFORCE update (loss decreases, weights change), PPO buffer (fills and triggers update correctly), PPO update (loss computation, gradient clipping), episode reset (QSNN state cleared, buffers cleared correctly), brain registration (factory creates correct type, config loader resolves), weight save/load (round-trip save then load produces identical weights, shape mismatch raises ValueError, missing file raises FileNotFoundError, stage 2 without weights logs warning)
- [ ] Run `uv run pre-commit run -a` and fix any ruff/pyright violations
- [ ] Run `uv run pytest tests/quantumnematode_tests/brain/arch/test_hybridquantum.py -v` — all tests pass
- [ ] Run `uv run pytest` — full test suite passes with no regressions
