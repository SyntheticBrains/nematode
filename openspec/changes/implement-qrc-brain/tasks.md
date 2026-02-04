## 1. Configuration Schema

- [ ] 1.1 Create `QRCBrainConfig` Pydantic model in `brain/config.py` with fields: `num_reservoir_qubits` (default 8), `reservoir_depth` (default 3), `reservoir_seed` (default 42), `readout_hidden` (default 32), `readout_type` (literal "mlp" | "linear", default "mlp"), `shots` (default 1024), `gamma`, `learning_rate`
- [ ] 1.2 Add validators: `num_reservoir_qubits` >= 2, `reservoir_depth` >= 1, `readout_hidden` >= 1, `shots` >= 100

## 2. Core QRCBrain Implementation

- [ ] 2.1 Create `brain/arch/qrc.py` with `QRCBrain` class implementing `ClassicalBrain` protocol
- [ ] 2.2 Implement reservoir circuit construction: Hadamard gates → random RX/RY/RZ rotations (seeded) → circular CZ entanglement, repeated for `reservoir_depth` layers
- [ ] 2.3 Implement sensory input encoding: compute gradient strength [0,1] and relative angle [-1,1] from BrainParams, encode as RY rotations (θ = v × π) on reservoir qubits (cycling via i % num_qubits)
- [ ] 2.4 Implement reservoir state extraction: execute circuit with `shots` measurements, return probability distribution as numpy array of shape (2^num_qubits,)

## 3. Readout Network

- [ ] 3.1 Implement MLP readout: two-layer network with input dim = 2^num_qubits, hidden dim = `readout_hidden`, output dim = 5, ReLU activation
- [ ] 3.2 Implement linear readout: single linear layer with input dim = 2^num_qubits, output dim = 5
- [ ] 3.3 Add readout type selection based on `readout_type` config parameter

## 4. REINFORCE Learning

- [ ] 4.1 Implement action selection: pass reservoir state through readout, apply softmax, sample from categorical distribution, store log probability
- [ ] 4.2 Implement episode-level learning: compute discounted returns (G_t = r_t + γ·G\_{t+1}), normalize for variance reduction
- [ ] 4.3 Implement policy loss computation: L = -Σ log_prob(a_t) · G_t, backpropagate through readout only
- [ ] 4.4 Add Adam optimizer for readout parameter updates

## 5. Protocol Compliance

- [ ] 5.1 Implement `run_brain(params, reward, input_data, top_only, top_randomize)` returning ActionData
- [ ] 5.2 Implement `learn(params, reward, episode_done)` for REINFORCE updates
- [ ] 5.3 Implement `update_memory(reward)` for reward tracking
- [ ] 5.4 Implement `prepare_episode()` and `post_process_episode(episode_success)` for episode lifecycle
- [ ] 5.5 Implement `copy()` returning independent clone with same reservoir but independent readout weights

## 6. Brain Factory Integration

- [ ] 6.1 Add "qrc" case to brain factory in `brain/brain_factory.py`
- [ ] 6.2 Validate QRCBrainConfig schema when brain type is "qrc"
- [ ] 6.3 Add "qrc" to CLI brain type choices in argument parser

## 7. Example Configurations

- [ ] 7.1 Create `configs/examples/qrc_foraging_small.yml` for basic foraging task
- [ ] 7.2 Create `configs/examples/qrc_predators_small.yml` for predator avoidance task

## 8. Testing

- [ ] 8.1 Add unit tests for reservoir circuit generation (verify Hadamard + rotation + CZ structure)
- [ ] 8.2 Add unit tests for reservoir reproducibility (same seed → same circuit → same output)
- [ ] 8.3 Add unit tests for input encoding (verify RY angle calculations)
- [ ] 8.4 Add unit tests for readout network architecture (verify layer dimensions)
- [ ] 8.5 Add integration test for full episode with QRCBrain
- [ ] 8.6 Add test for brain copy independence (modify copy, verify original unchanged)
