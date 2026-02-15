## 1. Configuration & Registration

- [x] 1.1 Add `QSNN` to `BrainType` enum in `brain/arch/dtypes.py`
- [x] 1.2 Create `QSNNBrainConfig` Pydantic model with fields: `num_sensory_neurons`, `num_hidden_neurons`, `num_motor_neurons`, `membrane_tau`, `threshold`, `refractory_period`, `use_local_learning`, `shots`, `gamma`, `learning_rate`, `entropy_coef`, `weight_clip`, `update_interval`, `num_integration_steps`, `sensory_modules`
- [x] 1.3 Add config validation (neurons >= 1, membrane_tau in (0,1\], threshold in (0,1), shots >= 100)
- [x] 1.4 Import and export `QSNNBrain`, `QSNNBrainConfig` in `brain/arch/__init__.py`
- [x] 1.5 Add `"qsnn": QSNNBrainConfig` to `BRAIN_CONFIG_MAP` in `utils/config_loader.py`
- [x] 1.6 Add `QSNNBrainConfig` to `BrainConfigType` union in `utils/config_loader.py`
- [x] 1.7 Add `BrainType.QSNN` case to `setup_brain_model()` in `utils/brain_factory.py`

## 2. Core QSNN Implementation

- [x] 2.1 Create `brain/arch/qsnn.py` with module docstring and imports (Qiskit, PyTorch, numpy)
- [x] 2.2 Implement `_build_qlif_circuit()`: `|0> -> RY(theta + tanh(w*x)*pi) -> RX(theta_leak) -> Measure`
- [x] 2.3 Implement weight matrix initialization (W_sh, W_hm with Gaussian scale 0.15, theta_hidden=pi/4, theta_motor=0)
- [x] 2.4 Implement `_encode_sensory_spikes()` to convert features to spike probabilities via sigmoid
- [x] 2.5 Implement refractory period tracking per neuron
- [x] 2.6 Implement `_execute_qlif_layer()` for numpy forward pass (no gradients)
- [x] 2.7 Implement `_execute_qlif_layer_differentiable()` with `QLIFSurrogateSpike` wrapping
- [x] 2.8 Implement `_timestep()` and `_timestep_differentiable()` for full forward pass
- [x] 2.9 Implement `_multi_timestep()` and `_multi_timestep_differentiable()` for noise-averaged integration
- [x] 2.10 Implement action selection: logit scaling, temperature softmax, epsilon-greedy exploration floor

## 3. Surrogate Gradient Learning (Primary Mode)

- [x] 3.1 Implement `QLIFSurrogateSpike` autograd function (forward: quantum spike prob, backward: sigmoid surrogate centered at pi/2)
- [x] 3.2 Implement `_reinforce_update()`: discounted returns, advantage normalization/clipping, policy loss + entropy bonus
- [x] 3.3 Implement `_adaptive_entropy_coef()`: two-sided regulation (floor boost 20x, ceiling suppression)
- [x] 3.4 Implement `_apply_gradients_and_log()`: gradient clipping, optimizer step, weight clamping, diagnostics
- [x] 3.5 Configure Adam optimizer with cosine annealing LR scheduler (0.01 -> 0.001 over 200 episodes)
- [x] 3.6 Implement exploration decay schedule (epsilon + temperature decay over 80 episodes)

## 4. Hebbian Learning (Legacy Mode)

- [x] 4.1 Implement eligibility trace storage (per synapse, gamma-decayed each step)
- [x] 4.2 Implement `_accumulate_eligibility()` with centered spikes and action-specific credit assignment
- [x] 4.3 Implement `_normalize_trace()` for eligibility norm capping
- [x] 4.4 Implement `_local_learning_update()`: `delta_w = lr * normalize(eligibility) * advantage - weight_decay * w`
- [x] 4.5 Implement intra-episode updates every `update_interval` steps

## 5. ClassicalBrain Protocol

- [x] 5.1 Implement `__init__()` with config, num_actions, device parameters
- [x] 5.2 Implement `run_brain()` returning ActionData with action and probability
- [x] 5.3 Implement `learn()` to accumulate rewards and trigger learning at episode end
- [x] 5.4 Implement `update_memory()` (no-op)
- [x] 5.5 Implement `prepare_episode()` to reset episode state
- [x] 5.6 Implement `post_process_episode()` (no-op, learning happens in learn())
- [x] 5.7 Implement `copy()` returning independent clone with copied weights, optimizer, and scheduler state

## 6. Multi-Sensory Support

- [x] 6.1 Add `sensory_modules` config option for unified feature extraction
- [x] 6.2 Implement legacy mode (2 features: gradient_strength, relative_angle) when `sensory_modules=None`
- [x] 6.3 Implement unified mode using `extract_classical_features()` when `sensory_modules` is set

## 7. Example Configs

- [x] 7.1 Create `configs/examples/qsnn_foraging_small.yml` (surrogate gradient mode, num_integration_steps=10)
- [x] 7.2 Create `configs/examples/qsnn_predators_small.yml` (with sensory_modules: food_chemotaxis, nociception)

## 8. Unit Tests (100 tests)

- [x] 8.1 Configuration validation tests (valid/invalid configs)
- [x] 8.2 QLIF circuit construction and measurement tests
- [x] 8.3 Sensory spike encoding tests
- [x] 8.4 Forward pass and action probability tests
- [x] 8.5 Eligibility trace computation tests (Hebbian mode)
- [x] 8.6 Local learning weight update tests
- [x] 8.7 Refractory period behavior tests
- [x] 8.8 Reproducibility with same seed tests
- [x] 8.9 `copy()` independent clone tests
- [x] 8.10 Surrogate gradient mode tests (QLIFSurrogateSpike, differentiable layers, optimizer creation)
- [x] 8.11 Multi-timestep integration tests
- [x] 8.12 Adaptive entropy regulation tests (floor, ceiling, formulas)
- [x] 8.13 Weight initialization tests (Gaussian scale, theta warm-start, column diversity)
- [x] 8.14 Exploration schedule tests

## 9. Documentation & Verification

- [x] 9.1 Update `docs/experiments/logbooks/008-quantum-brain-evaluation.md` with QSNN foraging results
- [x] 9.2 Create `docs/experiments/logbooks/008-appendix-qsnn-foraging-optimization.md` with optimization history
- [x] 9.3 Run smoke test with foraging config
- [x] 9.4 Run all 100 unit tests passing
- [x] 9.5 Benchmark: 73.9% success on foraging (4x200 episodes), matches SpikingReinforceBrain (73.3%)

## 10. Predator Evaluation (Pending)

- [ ] 10.1 Run QSNN on predator evasion with multi-sensory config (food_chemotaxis + nociception)
- [ ] 10.2 Compare QSNN vs SpikingReinforceBrain on predator tasks
- [ ] 10.3 Tune hyperparameters for predator scenario if needed
- [ ] 10.4 Update logbook with predator results
