## Why

QRH is our only architecture with genuine quantum advantage (+9.4pp over classical CRH on pursuit predators). However, QRH uses a random reservoir with no purposeful structure in its entanglement. Recent research (arXiv:2603.10289) demonstrates that entangled circuits develop structurally distinct features for modeling interacting variables — exactly the predator-prey dynamics in our environments. QA-5 (QEF) extends the QRH paradigm by replacing random reservoir entanglement with purposeful cross-modal entanglement between sensory-modality qubit pairs, while keeping the proven classical PPO readout. This is our highest-priority next candidate based on the post-QA-4 deep research investigation.

## What Changes

- New brain architecture: `QEFBrain` (Quantum Entangled Features) — entangled PQC feature extractor with classical PPO readout
- New config: `QEFBrainConfig` extending `ReservoirHybridBaseConfig` with entanglement topology, circuit depth, encoding/gate/feature modes, hybrid input, feature gating, curated feature subsets
- Registration in `BrainType` enum, `BRAIN_CONFIG_MAP`, `brain_factory`, and `arch/__init__`
- Example YAML configs for all evaluation environments with task-specific optimizations
- Classical ablation infrastructure in MLP PPO (polynomial/random projection feature expansion + gating)
- Comprehensive unit tests (78 QEF tests + 18 MLP PPO ablation tests)
- 12-seed statistical validation across 3 tasks

## Capabilities

### New Capabilities

- `qef-brain`: Quantum Entangled Features brain architecture — entangled PQC feature extractor (8 qubits, configurable entanglement topology, hybrid input, learnable feature gating, curated feature subsets) with classical PPO actor-critic readout. Includes separable ablation mode and three topology options (modality-paired, ring, random).
- `mlpppo-ablation`: Feature expansion (polynomial, polynomial3, random projection) and feature gating support in MLPPPOBrain for rigorous classical ablation testing.

### Modified Capabilities

- `brain-architecture`: Add QEF to the brain type registry, config loader mapping, and factory instantiation
- `mlpppo-brain`: Extended with feature_expansion, feature_gating, and related preprocessing

## Impact

- **New files**: `brain/arch/qef.py`, `test_qef.py`, example configs
- **Modified files**: `dtypes.py` (BrainType enum), `__init__.py` (exports), `config_loader.py` (BRAIN_CONFIG_MAP), `brain_factory.py` (factory case), `mlpppo.py` (feature expansion/gating)
- **Dependencies**: No new dependencies
- **Risk**: Low — extends proven ReservoirHybridBase; MLP PPO changes are additive config fields

## Evaluation Outcome

12-seed validation (~500+ runs) concluded: QEF is quantum-competitive but not quantum-advantageous. It matches classical approaches within ~1-3pp across all tasks but does not demonstrate statistically significant superiority. See `build/brains/qef/qef_scratchpad.md` for complete evaluation history and analysis.
