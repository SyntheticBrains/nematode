## 1. Configuration & Registration

- [x] 1.1 Add `QSNN_PPO = "qsnnppo"` to `BrainType` enum in `brain/arch/dtypes.py`
- [x] 1.2 Add `BrainType.QSNN_PPO` to `QUANTUM_BRAIN_TYPES` and `BRAIN_TYPES`
- [x] 1.3 Create `QSNNPPOBrainConfig` Pydantic model with QLIF network, PPO, optimizer, critic, and schedule parameters
- [x] 1.4 Add config field validators (neurons >= 1, membrane_tau in (0,1\], threshold in (0,1), shots >= 100)
- [x] 1.5 Import and export `QSNNPPOBrain`, `QSNNPPOBrainConfig` in `brain/arch/__init__.py`
- [x] 1.6 Add `"qsnnppo": QSNNPPOBrainConfig` to `BRAIN_CONFIG_MAP` in `utils/config_loader.py`
- [x] 1.7 Add `QSNNPPOBrainConfig` to `BrainConfigType` union in `utils/config_loader.py`
- [x] 1.8 Add `BrainType.QSNN_PPO` case to `setup_brain_model()` in `utils/brain_factory.py`

## 2. Rollout Buffer (`QSNNRolloutBuffer`)

- [x] 2.1 Implement buffer storage for features, actions, log_probs, values, rewards, dones, hidden_spike_rates
- [x] 2.2 Implement `add()` method with all required fields
- [x] 2.3 Implement `is_full()` capacity check
- [x] 2.4 Implement `reset()` to clear all stored data and spike caches
- [x] 2.5 Implement `compute_returns_and_advantages()` with GAE (gamma, gae_lambda)
- [x] 2.6 Implement `get_minibatches()` with random permutation, advantage normalization, and seeded RNG

## 3. Classical Critic (`QSNNPPOCritic`)

- [x] 3.1 Implement MLP critic with configurable hidden layers and dimensions
- [x] 3.2 Use ReLU activations between hidden layers
- [x] 3.3 Apply orthogonal weight initialization for stable early training
- [x] 3.4 Input: raw sensory features + hidden spike rates (detached from actor graph)
- [x] 3.5 Output: scalar value estimate V(s)

## 4. QSNN-PPO Brain (`QSNNPPOBrain`)

### 4a. Initialization

- [x] 4.1 Initialize QSNN actor weights (W_sh, W_hm, theta_hidden, theta_motor) with requires_grad
- [x] 4.2 Initialize classical critic MLP
- [x] 4.3 Create separate Adam optimizers for actor (with weight decay) and critic
- [x] 4.4 Create optional cosine annealing LR scheduler for actor
- [x] 4.5 Initialize rollout buffer with configured size and seeded RNG

### 4b. Forward Pass

- [x] 4.6 Implement `_multi_timestep()` for non-differentiable forward pass (action selection during rollout)
- [x] 4.7 Implement `_multi_timestep_differentiable_caching()` for epoch 0 (run circuits + cache spike probs)
- [x] 4.8 Implement `_multi_timestep_differentiable_cached()` for epochs 1+ (reuse cached spike probs)
- [x] 4.9 Implement `_get_critic_input()` to concatenate features and detached hidden spike rates

### 4c. Feature Extraction

- [x] 4.10 Implement legacy mode preprocessing (gradient_strength, relative_angle)
- [x] 4.11 Implement unified sensory mode using `extract_classical_features()`

### 4d. Brain Protocol

- [x] 4.12 Implement `run_brain()` with action selection, value estimation, and buffer storage
- [x] 4.13 Implement `learn()` with buffer fill check and PPO update trigger
- [x] 4.14 Implement `prepare_episode()` to reset refractory and pending state
- [x] 4.15 Implement `post_process_episode()` with episode counter and LR scheduler step
- [x] 4.16 Implement `copy()` returning independent clone
- [x] 4.17 Implement `update_memory()` (no-op)

### 4e. PPO Update

- [x] 4.18 Implement `_perform_ppo_update()` with multi-epoch quantum caching
- [x] 4.19 Implement epoch 0 pre-caching pass for all buffer steps
- [x] 4.20 Implement per-step forward passes within minibatches (not batched)
- [x] 4.21 Implement PPO clipped surrogate loss computation
- [x] 4.22 Implement Huber value loss computation
- [x] 4.23 Implement separate actor/critic backward passes with gradient clipping
- [x] 4.24 Implement weight clamping and theta motor norm clamping after update
- [x] 4.25 Implement update logging (policy_loss, value_loss, entropy, gradient norms)

## 5. Example Config

- [x] 5.1 Create `configs/examples/qsnnppo_pursuit_predators_small.yml` with pursuit predators, health system, and sensory modules

## 6. Unit Tests (65 tests)

- [x] 6.1 Config validation tests (defaults, custom values, boundary errors)
- [x] 6.2 Rollout buffer tests (init, add, full, reset, GAE, minibatches, normalization)
- [x] 6.3 Critic MLP tests (init, shapes, finite values, orthogonal init)
- [x] 6.4 Brain initialization tests (shapes, gradients, theta values, optimizers, scheduler)
- [x] 6.5 Preprocessing tests (legacy mode, sensory modules, critic input)
- [x] 6.6 Forward pass tests (action selection, pending data, probabilities, history)
- [x] 6.7 PPO learning tests (buffer fill, update triggers, weight changes, clamping)
- [x] 6.8 Episode lifecycle tests (prepare, post_process, LR scheduler, multi-episode)
- [x] 6.9 Reproducibility tests (seeded determinism)
- [x] 6.10 Integration tests (full episode, multi-episode, varying rewards, sensory modules)
- [x] 6.11 Error handling tests (copy, build_brain, action_set validation)
- [x] 6.12 Registration tests (enum, quantum type, config map, init exports)

## 7. Verification

- [x] 7.1 All 65 unit tests passing
- [x] 7.2 `uv run pre-commit run -a` passes (ruff check, ruff format, pyright)
- [x] 7.3 Committed as `9490cac` on `feature/add-qsnn-brain`

## 8. Experiment Evaluation

- [x] 8.1 Run initial QSNN-PPO sessions on pursuit predators config (PPO-0: 4 sessions, 200 episodes each)
- [x] 8.2 Analyse results and compare against QSNN Reinforce baseline (4 rounds, 16 sessions total)
- [x] 8.3 Tune hyperparameters based on training metrics (PPO-1 through PPO-3: cross-ep buffer, entropy decay, theta init)
- [x] 8.4 Update logbook with QSNN-PPO results (008-quantum-brain-evaluation.md + supporting/008/qsnnppo-optimization.md)

**Outcome**: HALTED after 4 rounds. PPO fundamentally incompatible with surrogate gradient spiking networks (policy_loss=0 in 100% of updates). Development pivoted to QSNNReinforce A2C.
