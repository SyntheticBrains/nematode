## 1. Configuration & Registration

- [x] 1.1 Add `QSNN_REINFORCE` to `BrainType` enum in `brain/arch/dtypes.py` (with `QSNN` as deprecated alias)
- [x] 1.2 Create `QSNNReinforceBrainConfig` Pydantic model with fields: `num_sensory_neurons`, `num_hidden_neurons`, `num_motor_neurons`, `membrane_tau`, `threshold`, `refractory_period`, `use_local_learning`, `shots`, `gamma`, `learning_rate`, `entropy_coef`, `weight_clip`, `update_interval`, `num_integration_steps`, `num_reinforce_epochs`, `exploration_decay_episodes`, `lr_decay_episodes`, `lr_min_factor`, `logit_scale`, `advantage_clip`, `theta_motor_max_norm`, `use_critic`, `sensory_modules`
- [x] 1.3 Add config validation (neurons >= 1, membrane_tau in (0,1\], threshold in (0,1), shots >= 100)
- [x] 1.4 Import and export `QSNNReinforceBrain`, `QSNNReinforceBrainConfig` in `brain/arch/__init__.py`
- [x] 1.5 Add `"qsnnreinforce": QSNNReinforceBrainConfig` to `BRAIN_CONFIG_MAP` in `utils/config_loader.py` (with `"qsnn"` backward-compatible alias)
- [x] 1.6 Add `QSNNReinforceBrainConfig` to `BrainConfigType` union in `utils/config_loader.py`
- [x] 1.7 Add `BrainType.QSNN_REINFORCE` case to `setup_brain_model()` in `utils/brain_factory.py`

## 2. Shared QLIF Module (`_qlif_layers.py`)

- [x] 2.1 Create `brain/arch/_qlif_layers.py` with module docstring and imports
- [x] 2.2 Implement `build_qlif_circuit()`: `|0> -> RY(theta + tanh(w*x/sqrt(fan_in))*pi) -> RX(theta_leak) -> Measure`
- [x] 2.3 Implement `QLIFSurrogateSpike(torch.autograd.Function)` with sigmoid surrogate centered at pi/2
- [x] 2.4 Implement `encode_sensory_spikes()` to convert features to spike probabilities via sigmoid
- [x] 2.5 Implement `execute_qlif_layer()` for numpy forward pass (no gradients)
- [x] 2.6 Implement `execute_qlif_layer_differentiable()` with `QLIFSurrogateSpike` wrapping
- [x] 2.7 Implement `execute_qlif_layer_differentiable_cached()` for multi-epoch reuse of cached spike probs
- [x] 2.8 Implement `get_qiskit_backend()` for backend initialization
- [x] 2.9 Export constants: `DEFAULT_SURROGATE_ALPHA`, `WEIGHT_INIT_SCALE`, `LOGIT_SCALE`

## 3. Core QSNN Reinforce Implementation (`qsnnreinforce.py`)

- [x] 3.1 Create `brain/arch/qsnnreinforce.py` with module docstring and imports from `_qlif_layers`
- [x] 3.2 Implement weight matrix initialization (W_sh, W_hm with Gaussian scale 0.15, theta_hidden=pi/4, theta_motor=0)
- [x] 3.3 Implement refractory period tracking per neuron
- [x] 3.4 Implement `_timestep()` and `_timestep_differentiable()` for full forward pass
- [x] 3.5 Implement `_multi_timestep()` and variants (differentiable, cached, caching) for noise-averaged integration
- [x] 3.6 Implement action selection: logit scaling, temperature softmax, epsilon-greedy exploration floor

## 4. Surrogate Gradient Learning (Primary Mode)

- [x] 4.1 Implement `_reinforce_update()`: discounted returns, advantage normalization/clipping, policy loss + entropy bonus
- [x] 4.2 Implement multi-epoch REINFORCE with quantum output caching (epoch 0 runs circuits, epochs 1+ reuse cached probs)
- [x] 4.3 Implement `_adaptive_entropy_coef()`: two-sided regulation (floor boost 20x, ceiling suppression)
- [x] 4.4 Implement `_apply_gradients_and_log()`: gradient clipping, optimizer step, weight clamping, theta motor norm clamping, diagnostics
- [x] 4.5 Configure Adam optimizer with cosine annealing LR scheduler
- [x] 4.6 Implement exploration decay schedule (epsilon + temperature decay)
- [x] 4.7 Implement reward normalization (EMA-based)
- [x] 4.8 Implement degenerate batch skipping (returns_std = 0)

## 5. Hebbian Learning (Legacy Mode)

- [x] 5.1 Implement eligibility trace storage (per synapse, gamma-decayed each step)
- [x] 5.2 Implement `_accumulate_eligibility()` with centered spikes and action-specific credit assignment
- [x] 5.3 Implement `_normalize_trace()` for eligibility norm capping
- [x] 5.4 Implement `_local_learning_update()`: `delta_w = lr * normalize(eligibility) * advantage - weight_decay * w`
- [x] 5.5 Implement intra-episode updates every `update_interval` steps

## 6. ClassicalBrain Protocol

- [x] 6.1 Implement `__init__()` with config, num_actions, device parameters
- [x] 6.2 Implement `run_brain()` returning ActionData with action and probability
- [x] 6.3 Implement `learn()` to accumulate rewards and trigger learning at episode end
- [x] 6.4 Implement `update_memory()` (no-op)
- [x] 6.5 Implement `prepare_episode()` to reset episode state
- [x] 6.6 Implement `post_process_episode()` (no-op, learning happens in learn())
- [x] 6.7 Implement `copy()` returning independent clone with copied weights, optimizer, and scheduler state

## 7. Multi-Sensory Support

- [x] 7.1 Add `sensory_modules` config option for unified feature extraction
- [x] 7.2 Implement legacy mode (2 features: gradient_strength, relative_angle) when `sensory_modules=None`
- [x] 7.3 Implement unified mode using `extract_classical_features()` when `sensory_modules` is set

## 8. Example Configs

- [x] 8.1 Create `configs/examples/qsnnreinforce_foraging_small.yml` (surrogate gradient mode, num_integration_steps=10)
- [x] 8.2 Create `configs/examples/qsnnreinforce_predators_small.yml` (with sensory_modules: food_chemotaxis, nociception)
- [x] 8.3 Create `configs/examples/qsnnreinforce_pursuit_predators_small.yml` (pursuit predators, health system)

## 9. Unit Tests (175 tests)

- [x] 9.1 Configuration validation tests (valid/invalid configs)
- [x] 9.2 QLIF circuit construction and measurement tests
- [x] 9.3 Sensory spike encoding tests
- [x] 9.4 Forward pass and action probability tests
- [x] 9.5 Eligibility trace computation tests (Hebbian mode)
- [x] 9.6 Local learning weight update tests
- [x] 9.7 Refractory period behavior tests
- [x] 9.8 Reproducibility with same seed tests
- [x] 9.9 `copy()` independent clone tests
- [x] 9.10 Surrogate gradient mode tests (QLIFSurrogateSpike, differentiable layers, optimizer creation)
- [x] 9.11 Multi-timestep integration tests
- [x] 9.12 Adaptive entropy regulation tests (floor, ceiling, formulas)
- [x] 9.13 Weight initialization tests (Gaussian scale, theta warm-start, column diversity)
- [x] 9.14 Exploration schedule tests
- [x] 9.15 Multi-epoch REINFORCE with caching tests
- [x] 9.16 Theta motor norm clamping tests
- [x] 9.17 Reward normalization tests

## 10. Documentation & Verification

- [x] 10.1 Update `docs/experiments/logbooks/008-quantum-brain-evaluation.md` with QSNN foraging and predator results
- [x] 10.2 Create `docs/experiments/logbooks/008-appendix-qsnn-foraging-optimization.md` with foraging optimization history
- [x] 10.3 Create `docs/experiments/logbooks/008-appendix-qsnn-predator-optimization.md` with predator optimization history (16 rounds)
- [x] 10.4 Run all 175 unit tests passing
- [x] 10.5 Benchmark: 67% success on foraging (4x200 episodes)

## 11. Predator Evaluation

- [x] 11.1 Run QSNN Reinforce on predator evasion with multi-sensory config (food_chemotaxis + nociception)
- [x] 11.2 Run 16 rounds of predator optimization (P0-P3a, PP0-PP9)
- [x] 11.3 Evaluate pursuit predators — first 5% pursuit success achieved (PP8)
- [x] 11.4 Conclude: per-encounter evasion does not improve through training; standalone QSNN Reinforce halted on predators
- [x] 11.5 Update logbook with predator results and decision to create QSNN-PPO as separate change

## 12. Code Cleanup

- [x] 12.1 Fix 8 ruff violations in qsnnreinforce.py (extract helpers to reduce complexity)
- [x] 12.2 Rename `QSNNBrain` -> `QSNNReinforceBrain` for consistency with MLP naming convention
- [x] 12.3 Rename `qsnn.py` -> `qsnnreinforce.py`, `test_qsnn.py` -> `test_qsnnreinforce.py`
- [x] 12.4 Rename config files: `qsnn_*.yml` -> `qsnnreinforce_*.yml`
- [x] 12.5 Add backward compatibility aliases (`QSNN`, `QSNNBrain`, `QSNNBrainConfig`)
- [x] 12.6 Extract shared QLIF components into `_qlif_layers.py` (523 lines)
