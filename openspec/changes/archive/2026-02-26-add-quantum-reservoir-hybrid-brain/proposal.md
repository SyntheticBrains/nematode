## Why

The QRC brain (0% success across 1,600+ runs) failed because its **random** reservoir topology produces non-discriminative representations — different sensory inputs yield similar measurement distributions, causing the readout to learn the wrong behavior. Meanwhile, recent QRC research (Llodrà et al. 2025, Ivaki et al. 2025) demonstrates that **structured** reservoir topologies consistently outperform random circuits. The Quantum Reservoir Hybrid (QRH) is proposal H.1 from `docs/research/quantum-architectures.md` — the highest-priority next-generation architecture — addressing QRC's failure with three key changes: biologically-structured topology from the C. elegans connectome, richer feature extraction (Z-expectations + ZZ-correlations instead of raw probability distributions), and PPO training on the readout instead of REINFORCE.

## What Changes

- Add new `QRHBrain` architecture implementing a structured quantum reservoir with C. elegans-inspired topology and PPO-trained classical readout
- Structured quantum reservoir circuit: gap junctions → CZ gates (symmetric), chemical synapses → fixed RY/RZ rotations, mapped from the C. elegans sensory-interneuron subnetwork (ASEL/ASER → AIY → AIA → AVA) onto 8 qubits
- Novel feature extraction: per-qubit X/Y/Z Pauli expectations (⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩) + pairwise ZZ-correlations (⟨Z_i Z_j⟩) using statevector simulation, producing 52 features for 8 qubits (vs QRC's 256-dim raw probability distribution)
- Classical actor-critic readout (MLP) trained with PPO (clipped surrogate objective, GAE advantages) — replacing QRC's REINFORCE training
- Mutual information decision gate script for Week 1 go/no-go validation: structured vs random reservoir feature quality comparison
- Extract shared quantum utilities from `_qlif_layers.py` into a new `_quantum_utils.py` module (`get_qiskit_backend()` used by 5 brains + duplicated in QRC)
- Extract shared reservoir utilities (readout network builder, input encoding) into `_quantum_reservoir.py` for QRC/QRH reuse
- Integration with brain factory, configuration system, benchmarking infrastructure, and documentation (README, CONTRIBUTING, AGENTS)
- Example YAML configs for foraging and multi-objective pursuit predator environments

## Capabilities

### New Capabilities

- `qrh-brain`: Quantum Reservoir Hybrid brain architecture with C. elegans-structured quantum reservoir, X/Y/Z+ZZ feature extraction, and PPO-trained classical readout

### Modified Capabilities

- `brain-architecture`: Add QRH to the brain factory, CLI brain selection, and brain type classification (extending existing spec with new brain type in `QUANTUM_BRAIN_TYPES`)
- `qrc-brain`: Refactor shared reservoir utilities (readout network builder, backend initialization) into common modules for QRC/QRH reuse — no behavioral changes to QRC

## Impact

- **Code**: New `qrh.py` in `brain/arch/`, new shared modules `_quantum_utils.py` and `_quantum_reservoir.py`, updates to `brain_factory.py`, `config_loader.py`, `dtypes.py`, `__init__.py`; refactor `qrc.py` and `_qlif_layers.py` to use shared modules
- **APIs**: Brain protocol compliance required (ClassicalBrain interface, ActionData/BrainData compatibility); no breaking changes to existing brains
- **Dependencies**: Uses existing Qiskit infrastructure (QuantumCircuit, Statevector, AerSimulator) and PyTorch (actor-critic MLPs, PPO training); new optional dependency on sklearn for MI analysis script
- **Configs**: New `qrh_foraging_small.yml` and `qrh_pursuit_predators_small.yml` example configs
- **Tests**: Unit tests for structured reservoir circuit, Z/ZZ feature extraction, PPO readout training, shared utility modules; smoke test for CLI end-to-end; tests for shared module refactoring (ensure QRC still passes)
- **Benchmarks**: New benchmark entries for QRH in quantum category (foraging + predator environments)
- **Documentation**: Update README.md (brain architecture list), CONTRIBUTING.md (architecture overview, example commands), AGENTS.md (brain list)
