## Context

The quantum nematode project currently has two quantum brain architectures:

1. **QVarCircuitBrain**: Parameterized VQC trained via parameter-shift gradients or CMA-ES
2. **QQLearningBrain**: Hybrid quantum feature extractor with classical DQN (incomplete)

Both suffer from gradient-related challenges. QVarCircuitBrain achieves only 22.5% success with gradient-based learning vs 88% with CMA-ES, indicating barren plateau issues. The project roadmap identifies QRC as a priority for Phase 2 architecture comparison because it inherently avoids barren plateaus.

The existing brain architecture follows a Protocol pattern with `Brain`, `QuantumBrain`, and `ClassicalBrain` interfaces. All brains must implement `run_brain()`, `update_memory()`, `prepare_episode()`, `post_process_episode()`, and `copy()`.

## Goals / Non-Goals

**Goals:**

- Implement QRCBrain as a new quantum brain architecture
- Fixed quantum reservoir that generates rich feature representations
- Trainable classical readout using standard PyTorch gradient descent
- Full integration with brain factory, CLI, and benchmarking system
- Configuration parity with existing brain architectures

**Non-Goals:**

- Optimizing reservoir parameters (the key insight is they stay fixed)
- QPU deployment in this change (focus on simulation first)
- Evolutionary optimization support (QRC uses gradient descent on readout only)
- Data re-uploading or multi-reservoir ensembles (future work)

## Decisions

### Decision 1: Implement ClassicalBrain Protocol

**Choice**: QRCBrain implements `ClassicalBrain` protocol, not `QuantumBrain`

**Rationale**: Despite using quantum circuits internally, QRCBrain's trainable parameters are entirely classical (the readout network). The `QuantumBrain` protocol requires `build_brain()` and `inspect_circuit()` which imply parameterized circuits. QRC's reservoir is fixed, so the `ClassicalBrain` interface is more appropriate—it focuses on `learn()` which maps to readout training.

**Alternatives considered**:

- `QuantumBrain` protocol: Would require exposing circuit-building methods that don't make sense for fixed reservoirs
- New `ReservoirBrain` protocol: Over-engineering for a single implementation

### Decision 2: Reservoir Architecture

**Choice**: Hadamard initialization → random RX/RY/RZ rotations → circular CZ entanglement, repeated for N layers

**Rationale**: This follows the standard QRC architecture from literature (arXiv:2602.03522). Hadamards create initial superposition, random rotations provide feature diversity, and CZ gates create entanglement for complex correlations. The circular topology ensures all qubits interact.

**Configuration parameters**:

- `num_reservoir_qubits`: 8 (default) - balances expressivity vs simulation cost
- `reservoir_depth`: 3 (default) - number of entangling layers
- `reservoir_seed`: 42 (default) - for reproducibility

### Decision 3: Readout Architecture

**Choice**: Two-layer MLP with configurable hidden size, trained via REINFORCE

**Rationale**: Matches the learning approach used by MLPReinforceBrain, enabling direct comparison. The readout receives the 2^n probability distribution from quantum measurements and outputs action logits.

**Configuration parameters**:

- `readout_hidden`: 32 (default) - hidden layer size
- `readout_type`: "mlp" | "linear" - for ablation studies

### Decision 4: Input Encoding

**Choice**: Encode sensory features as RY rotations on reservoir qubits before the fixed reservoir circuit

**Rationale**: RY rotations are standard for amplitude encoding and work well with the sensory module system already in place. Features are mapped to qubits cyclically (qubit_idx = feature_idx % num_qubits).

**Input features** (from BrainParams):

- Gradient strength (normalized 0-1)
- Relative angle (normalized -1 to 1)
- Additional sensory modules if enabled

### Decision 5: Reservoir State Extraction

**Choice**: Full measurement probability distribution (2^n dimensional vector)

**Rationale**: Provides maximum information from the reservoir. The readout network can learn which bitstring probabilities are informative. This is more expressive than single-qubit expectation values.

**Trade-off**: Memory scales as O(2^n), limiting practical qubit counts to ~10-12 on CPU.

### Decision 6: Training Algorithm

**Choice**: REINFORCE with baseline subtraction (same as MLPReinforceBrain)

**Rationale**: Consistency with existing classical brains enables fair benchmarking. The quantum reservoir is never trained—only the classical readout parameters update via standard backpropagation.

## Risks / Trade-offs

**[Risk] Reservoir expressivity may be insufficient for complex tasks**
→ Mitigation: Configurable depth and qubit count; can increase if needed. Start with 8 qubits, 3 layers.

**[Risk] Memory usage scales exponentially with qubit count**
→ Mitigation: Default to 8 qubits (256-dim state vector). Document memory requirements. Consider expectation-value-based extraction as future optimization.

**[Risk] Fixed reservoir may not generalize across different environments**
→ Mitigation: Configurable seed allows testing multiple random reservoirs. This is actually a research question we want to answer.

**[Risk] Performance comparison unfair due to different learning algorithms**
→ Mitigation: Use identical REINFORCE parameters as MLPReinforceBrain. Both use gradient descent, just on different architectures.

**[Trade-off] No evolutionary optimization support**
→ Acceptable: QRC's value proposition is avoiding gradient issues entirely. If gradients work well (expected for classical readout), evolution is unnecessary.
