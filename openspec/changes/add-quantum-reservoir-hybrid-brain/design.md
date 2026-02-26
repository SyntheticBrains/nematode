## Context

The QRC brain (added 2026-02-06) achieved 0% success across 1,600+ runs because its **random** reservoir topology produces non-discriminative representations. The QRH (Quantum Reservoir Hybrid) is the highest-priority next-generation architecture (proposal H.1 in `docs/research/quantum-architectures.md`), designed to fix QRC's three failure modes:

1. **Random topology → non-discriminative features**: Replace with C. elegans connectome-inspired structured topology
2. **Raw 2^N probability distribution → uninformative features**: Replace with X/Y/Z-expectations + ZZ-correlations (O(N²) features)
3. **REINFORCE training → high variance**: Replace with PPO (actor-critic with GAE)

The existing brain architecture follows a Protocol pattern (`Brain`, `QuantumBrain`, `ClassicalBrain`). QRC implements `ClassicalBrain` since only its readout is trained. Several utility functions are currently duplicated or misplaced — `get_qiskit_backend()` lives in `_qlif_layers.py` but is used by 5 brains and duplicated inline in QRC.

## Goals / Non-Goals

**Goals:**

- Implement QRHBrain as a structured quantum reservoir with PPO-trained classical readout
- C. elegans-inspired reservoir topology mapped from the sensory-interneuron subnetwork
- X/Y/Z-expectation and ZZ-correlation feature extraction (richer than raw probability distributions)
- PPO training on the classical readout (actor-critic with GAE advantages)
- Mutual information decision gate script for Week 1 go/no-go validation
- Extract shared quantum utilities (`get_qiskit_backend`, readout network builder) into common modules
- Full integration with brain factory, CLI, benchmarking, and documentation

**Non-Goals:**

- Training the quantum reservoir parameters (the reservoir is intentionally fixed)
- QPU deployment in this change (focus on AerSimulator statevector simulation)
- Evolutionary optimization support (QRH uses standard PPO gradient descent)
- Multi-stage curriculum (QRH has a single training stage — no reflex/cortex separation)
- Implementing proposals H.2-H.4 (SQS-QLIF, Entangled QLIF, QKAN-QLIF are separate future changes)

## Decisions

### Decision 1: ClassicalBrain Protocol

**Choice**: QRHBrain implements `ClassicalBrain`, classified in `QUANTUM_BRAIN_TYPES`

**Rationale**: Like QRC, only the classical readout is trained — no quantum gradients. The `ClassicalBrain` interface is appropriate. However, QRH is placed in `QUANTUM_BRAIN_TYPES` (not `CLASSICAL_BRAIN_TYPES` like QRC) because it runs quantum circuits for feature extraction and should benchmark against other quantum brains.

**Alternatives considered**:

- `CLASSICAL_BRAIN_TYPES` (like QRC): Would benchmark against MLP/PPO, not the scientifically interesting comparison
- New `ReservoirBrain` protocol: Over-engineering for two implementations

### Decision 2: Structured Reservoir Topology

**Choice**: C. elegans sensory-interneuron subnetwork mapped to 8 qubits

**Rationale**: The C. elegans 302-neuron connectome is fully characterized (White et al. 1986, Cook et al. 2019). The 8-qubit mapping uses the core navigation circuit:

| Qubit | Neuron | Role |
|-------|--------|------|
| 0, 1 | ASEL, ASER | Amphid sensory (salt/food chemotaxis) |
| 2, 3 | AIYL, AIYR | First-layer interneurons (sensory integration) |
| 4, 5 | AIAL, AIAR | Second-layer interneurons (navigation command) |
| 6, 7 | AVAL, AVAR | Command interneurons (forward/reverse decision) |

**Gate mapping**:

- Gap junctions (bidirectional electrical coupling) → CZ gates: bilateral pairs (0-1, 2-3, 4-5, 6-7) + feedforward (2-4, 3-5)
- Chemical synapses (directed signaling) → controlled rotations (CRY/CRZ) where the postsynaptic response is conditioned on the presynaptic qubit state, with angles normalized from published synaptic weight ratios (Cook et al. 2019)

**Configuration**: `use_random_topology=True` flag generates a random reservoir with identical CZ density and rotation count, enabling controlled MI comparison.

### Decision 3: Feature Extraction — X/Y/Z-expectations + ZZ-correlations

**Choice**: Per-qubit ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩ (3N features) + pairwise ⟨Z_i Z_j⟩ (N(N-1)/2 features) = 52 features for 8 qubits

**Rationale**: QRC's 2^N probability distribution (256-dim for 8 qubits) is both high-dimensional and uninformative. X/Y/Z-expectations capture the full single-qubit Bloch sphere state, while ZZ-correlations capture entanglement-induced pairwise structure — the signature of quantum processing. This scales as O(N²) instead of O(2^N), and each feature has a clear physical interpretation.

**Computation**: From the statevector |ψ⟩:

- ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩ computed from probability amplitudes (full Bloch sphere per qubit)
- ⟨Z_i Z_j⟩ = Σ_k (-1)^(bit(k,i)+bit(k,j)) |ψ_k|²

All features lie in [-1, 1]. A LayerNorm layer normalizes the heterogeneous X/Y/Z/ZZ feature scales before the readout MLPs.

**Feature dimension formula**: `3*N + N*(N-1)//2` — 52 for 8 qubits, 75 for 10 qubits.

**Alternatives considered**:

- Raw 2^N probabilities (QRC approach): Failed — non-discriminative, doesn't scale
- Per-qubit Z-expectations only (N features): Misses X/Y phase information and entanglement correlations
- Z-expectations + ZZ-correlations only (N + N(N-1)/2 = 36 for 8q): Originally designed, but X/Y expectations provide richer phase-space coverage at minimal additional cost

### Decision 4: Statevector Simulation

**Choice**: Use `Statevector.from_instruction(circuit)` for exact feature computation

**Rationale**: Z-expectations and ZZ-correlations are computed from probability amplitudes. Statevector simulation gives exact values in one execution, whereas shot-based measurement would require O(shots × observables) circuit executions. The `shots` config parameter is retained for future QPU deployment where statevector is unavailable.

**Trade-off**: Statevector scales as O(2^N) memory, limiting to ~20 qubits. For 8-12 qubits this is negligible (~4KB-16KB).

### Decision 5: PPO Training on Readout

**Choice**: Self-contained PPO implementation following the `mlpppo.py` pattern for structure (rollout buffer, GAE, clipped surrogate loss)

**Rationale**: QRH's architecture is simpler than the hybrid brains — no reflex/cortex separation, no mode-gating, no curriculum stages. Using `mlpppo.py`'s self-contained PPO loop (MSE value loss, standard GAE) is cleaner than importing from `_hybrid_common.py` which carries mode-gating coupling.

**Key parameters** (matching proven HybridQuantum cortex training):

- Buffer: 512 steps, 4 epochs, 4 minibatches
- GAE: γ=0.99, λ=0.95
- PPO: clip_ε=0.2, entropy_coeff=0.01
- Optimizer: Adam, LR=3e-4 (single combined optimizer for actor + critic + feature_norm)

**NOTE**: The config retains separate `actor_lr` and `critic_lr` fields for flexibility, but the implementation uses a **single combined Adam optimizer** (matching the MLPPPO pattern) with `actor_lr` as the base rate. This proved more stable during training than the originally planned separate optimizers.

**Alternatives considered**:

- REINFORCE (QRC approach): Higher variance, no value function — QRC's failure was partly due to REINFORCE's inefficiency
- Import `_hybrid_common.py`'s `perform_ppo_update()`: Tightly coupled to cortex actor/mode-gating pattern (expects `num_motor` + mode logits)

### Decision 6: Shared Utility Extraction

**Choice**: Create `_quantum_utils.py` and `_quantum_reservoir.py`

**`_quantum_utils.py`** — Move `get_qiskit_backend()` from `_qlif_layers.py`:

- Currently imported by HybridQuantum, HybridQuantumCortex, QSNNReinforce, QSNNPPO
- Duplicated inline in QRC (lines 370-384)
- Re-export from `_qlif_layers.py` for backward compatibility

**`_quantum_reservoir.py`** — Extract shared reservoir components:

- `build_readout_network(input_dim, hidden_dim, output_dim, readout_type, num_layers)`: From QRC's `_build_readout_network()` (lines 331-368). Orthogonal-initialized MLP or linear readout.
- Input encoding helpers shared between QRC and QRH

**Guard**: Only extract if QRC's existing tests continue to pass without modification. If coupling is too tight, accept duplication and note for future cleanup.

### Decision 7: Readout Architecture

**Choice**: Two separate MLPs — actor (features → action logits) and critic (features → value scalar), both 2-layer with 64 hidden units, orthogonal initialization

**Rationale**: Standard PPO actor-critic pattern. Using the shared `build_readout_network()` for both. Separate optimizers allow independent learning rates if needed.

**Feature input dimension**: 52 for 8 qubits (24 X/Y/Z-expectations + 28 ZZ-correlations), 75 for 10 qubits (30 + 45).

## Risks / Trade-offs

**[Risk] Structured reservoir may still produce non-discriminative features**
→ Mitigation: MI decision gate script (Week 1) compares structured reservoir, random reservoir, and classical MLP (raw input) feature quality using mutual information. If MI(structured) ≤ MI(random) or MI(structured) ≤ MI(classical), the structured topology adds no value. Three-way comparison prevents false confidence from both reservoir types being equally poor.

**[Risk] C. elegans topology mapping may not transfer to the simulation's sensory space**
→ Mitigation: The simulation's sensory modules (food chemotaxis, nociception) map naturally to the C. elegans chemosensory circuit. The topology is biologically grounded, not arbitrary.

**[Risk] 8 qubits may be insufficient for multi-objective tasks**
→ Mitigation: Design allows scaling to 12 qubits (78 features). Start with 8, increase if foraging succeeds but predator evasion fails.

**[Risk] Statevector simulation is not QPU-compatible**
→ Mitigation: The `shots` parameter is retained. A future change can add shot-based observable estimation for QPU deployment.

**[Risk] Shared utility extraction may break existing brains**
→ Mitigation: Re-export from original locations for backward compatibility. Run full test suite after refactoring. If coupling is too tight, accept duplication.

**[Trade-off] No multi-stage curriculum**
→ Acceptable: QRH has no trainable quantum layer to pre-train. The fixed reservoir + PPO readout trains end-to-end in a single stage.

**[Trade-off] Self-contained PPO duplicates patterns from mlpppo.py**
→ Acceptable: The alternative (importing from `_hybrid_common.py`) would add coupling to mode-gating infrastructure QRH doesn't use. Clean duplication is preferred over inappropriate abstraction.
