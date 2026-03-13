## Why

QRH is our only architecture with genuine quantum advantage (+9.4pp over classical CRH on pursuit predators). However, QRH uses a random reservoir with no purposeful structure in its entanglement. Recent research (arXiv:2603.10289) demonstrates that entangled circuits develop structurally distinct features for modeling interacting variables — exactly the predator-prey dynamics in our environments. QA-5 (QEF) extends the QRH paradigm by replacing random reservoir entanglement with purposeful cross-modal entanglement between sensory-modality qubit pairs, while keeping the proven classical PPO readout. This is our highest-priority next candidate based on the post-QA-4 deep research investigation.

## What Changes

- New brain architecture: `QEFBrain` (Quantum Entangled Features) — entangled PQC feature extractor with classical PPO readout
- New config: `QEFBrainConfig` extending `ReservoirHybridBaseConfig` with entanglement topology selection, circuit depth, and separable ablation flag
- Registration in `BrainType` enum, `BRAIN_CONFIG_MAP`, `brain_factory`, and `arch/__init__`
- Example YAML configs matching QRH evaluation environments: foraging small, pursuit predators small, thermotaxis + pursuit predators large, thermotaxis + stationary predators large, and separable ablation control
- Unit tests following `test_qrh.py` structure
- Smoke test integration

## Capabilities

### New Capabilities

- `qef-brain`: Quantum Entangled Features brain architecture — entangled PQC feature extractor (8 qubits, configurable entanglement topology, Z+ZZ+cos_sin features) with classical PPO actor-critic readout. Includes separable ablation mode and three topology options (modality-paired, ring, random).

### Modified Capabilities

- `brain-architecture`: Add QEF to the brain type registry, config loader mapping, and factory instantiation

## Impact

- **New files**: `brain/arch/qef.py`, `test_qef.py`, 5 example configs
- **Modified files**: `dtypes.py` (BrainType enum + type sets), `__init__.py` (exports), `config_loader.py` (BRAIN_CONFIG_MAP + BrainConfigType union), `brain_factory.py` (factory case)
- **Dependencies**: No new dependencies — uses existing Qiskit, numpy, PyTorch, Pydantic
- **Risk**: Low — extends proven ReservoirHybridBase; no changes to existing brain architectures
