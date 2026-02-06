## 1. Configuration & Registration

- [ ] 1.1 Add `QSNN` to `BrainType` enum in `brain/arch/dtypes.py`
- [ ] 1.2 Create `QSNNBrainConfig` Pydantic model in `brain/arch/qsnn.py` with fields: `num_sensory_neurons`, `num_hidden_neurons`, `num_motor_neurons`, `membrane_tau`, `threshold`, `refractory_period`, `use_local_learning`, `shots`, `gamma`, `learning_rate`
- [ ] 1.3 Add config validation (neurons >= 1, membrane_tau in (0,1\], threshold in (0,1), shots >= 100)
- [ ] 1.4 Import and export `QSNNBrain`, `QSNNBrainConfig` in `brain/arch/__init__.py`
- [ ] 1.5 Add `"qsnn": QSNNBrainConfig` to `BRAIN_CONFIG_MAP` in `utils/config_loader.py`
- [ ] 1.6 Add `QSNNBrainConfig` to `BrainConfigType` union in `utils/config_loader.py`
- [ ] 1.7 Add `BrainType.QSNN` case to `setup_brain_model()` in `utils/brain_factory.py`

## 2. Core QSNN Implementation

- [ ] 2.1 Create `brain/arch/qsnn.py` with module docstring and imports (Qiskit, PyTorch, numpy)
- [ ] 2.2 Implement `_build_qlif_circuit()` method: `|0⟩ → RY(θ_membrane + input) → RX(θ_leak) → Measure`
- [ ] 2.3 Implement weight matrix initialization (W_sh for sensory→hidden, W_hm for hidden→motor)
- [ ] 2.4 Implement `_encode_sensory_spikes()` to convert BrainParams to spike probabilities via sigmoid
- [ ] 2.5 Implement refractory period tracking per neuron
- [ ] 2.6 Implement `_forward_layer()` to propagate spikes through a layer with QLIF circuits
- [ ] 2.7 Implement `_timestep()` for full sensory→hidden→motor forward pass
- [ ] 2.8 Implement action selection from motor neuron firing probabilities (softmax + sampling)

## 3. Learning Implementation

- [ ] 3.1 Implement eligibility trace storage (per synapse, updated each timestep)
- [ ] 3.2 Implement `_compute_eligibility()`: `pre_spike × post_spike`
- [ ] 3.3 Implement `_local_learning_update()`: `Δw = lr × eligibility × reward`
- [ ] 3.4 Add optional weight clipping for stability
- [ ] 3.5 Implement episode-level learning in `post_process_episode()` (apply accumulated eligibility × total reward)
- [ ] 3.6 Add `use_local_learning=False` fallback to REINFORCE with surrogate gradients

## 4. ClassicalBrain Protocol

- [ ] 4.1 Implement `__init__()` with config, num_actions, device parameters
- [ ] 4.2 Implement `run_brain()` returning ActionData with action and probability
- [ ] 4.3 Implement `learn()` to accumulate eligibility traces
- [ ] 4.4 Implement `update_memory()` for reward tracking
- [ ] 4.5 Implement `prepare_episode()` to reset episode state
- [ ] 4.6 Implement `post_process_episode()` to trigger learning update
- [ ] 4.7 Implement `copy()` returning independent clone with copied weights

## 5. Multi-Sensory Support

- [ ] 5.1 Add `sensory_modules` config option for unified feature extraction
- [ ] 5.2 Implement legacy mode (2 features: gradient_strength, relative_angle) when `sensory_modules=None`
- [ ] 5.3 Implement unified mode using `extract_classical_features()` when `sensory_modules` is set

## 6. Example Configs

- [ ] 6.1 Create `configs/examples/qsnn_foraging_small.yml` for basic foraging task
- [ ] 6.2 Create `configs/examples/qsnn_predators_small.yml` for predator evasion task

## 7. Unit Tests

- [ ] 7.1 Create `tests/quantumnematode_tests/brain/arch/test_qsnn.py`
- [ ] 7.2 Test QSNNBrainConfig validation (valid/invalid configs)
- [ ] 7.3 Test QLIF circuit construction and measurement
- [ ] 7.4 Test sensory spike encoding (sigmoid scaling)
- [ ] 7.5 Test forward pass produces valid action probabilities
- [ ] 7.6 Test eligibility trace computation
- [ ] 7.7 Test local learning weight updates
- [ ] 7.8 Test refractory period behavior
- [ ] 7.9 Test reproducibility with same seed
- [ ] 7.10 Test `copy()` produces independent clone

## 8. Documentation & Verification

- [ ] 8.1 Update `docs/experiments/logbooks/008-quantum-brain-evaluation.md` with QSNN section
- [ ] 8.2 Run smoke test: `python scripts/run_simulation.py --brain qsnn --config configs/examples/qsnn_foraging_small.yml`
- [ ] 8.3 Run unit tests: `pytest tests/quantumnematode_tests/brain/arch/test_qsnn.py -v`
- [ ] 8.4 Run benchmark: 200 episodes on foraging, record success rate and chemotaxis index
