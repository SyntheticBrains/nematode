## Why

QRH (Quantum Reservoir Hybrid) has converged to 41.2% average success (68% post-convergence) on the large pursuit predator environment after 9 rounds of iteration. To determine whether the quantum reservoir provides value over the architecture itself, we need a **classical control**: a brain with the same overall structure (fixed reservoir + PPO readout) but replacing the quantum reservoir with a classical Echo State Network (ESN). This isolates the quantum contribution: `quantum_benefit = QRH - CRH`. Additionally, comparing CRH against the current SOTA (MLPPPO at 97.5% post-convergence) measures the reservoir architecture's standalone value: `reservoir_benefit = CRH - MLPPPO`.

Beyond ablation, CRH fills a unique cell in the benchmark suite — classical dynamics with zero trainable reservoir parameters and a PPO readout. No existing brain occupies this niche. Classical reservoirs have properties (catastrophic forgetting immunity, fixed computation cost, potential separation property) that make CRH scientifically interesting as a permanent architecture, not a throwaway comparison.

## What Changes

- Extract shared PPO readout infrastructure from `QRHBrain` into a new `ReservoirHybridBase` base class (~800 lines: actor/critic MLPs, rollout buffer, PPO training loop, LR scheduling, episode tracking, feature normalization)
- Refactor `QRHBrain` to inherit from `ReservoirHybridBase`, implementing only quantum-specific methods (`_get_reservoir_features()`, `_encode_and_run()`, `_extract_features()`, reservoir construction)
- Add new `CRHBrain` architecture implementing a classical ESN reservoir with configurable feature channels, inheriting the PPO readout from `ReservoirHybridBase`
- CRH reservoir: fixed random weight matrices (W_in, W_res), tanh nonlinearity, configurable spectral radius, sparse or dense input connectivity
- Configurable feature extraction channels: `raw` (activations h), `cos_sin` (cos(pi*h), sin(pi*h)), `squared` (h^2), `pairwise` (h_i\*h_j for i\<j)
- Two config profiles: ablation mode (N=10, sparse input, [raw, cos_sin, pairwise] -> 75 features matching QRH) and standalone mode (tunable N, channels, connectivity)
- Integration with brain factory, configuration system, benchmarking infrastructure, and documentation
- Example YAML configs for large pursuit predator environment (ablation + standalone)

## Capabilities

### New Capabilities

- `crh-brain`: Classical Reservoir Hybrid brain architecture with ESN reservoir, configurable feature channels, and PPO-trained classical readout

### Modified Capabilities

- `brain-architecture`: Add CRH to the brain factory, CLI brain selection, and brain type classification (extending existing spec with new brain type in `CLASSICAL_BRAIN_TYPES`)
- `qrh-brain`: Refactor to inherit from `ReservoirHybridBase` — no behavioral changes to QRH

## Impact

- **Code**: New `crh.py` in `brain/arch/`, new `_reservoir_hybrid_base.py` base class, refactored `qrh.py` to inherit from base, updates to `brain_factory.py`, `config_loader.py`, `dtypes.py`, `__init__.py`
- **APIs**: Brain protocol compliance required (ClassicalBrain interface, ActionData/BrainData compatibility); no breaking changes to existing brains including QRH
- **Dependencies**: Uses existing PyTorch (ESN matrices, actor-critic MLPs, PPO training); no new external dependencies
- **Configs**: New `crh_thermotaxis_pursuit_predators_large.yml` (ablation) and `crh_thermotaxis_pursuit_predators_large_standalone.yml` (standalone) example configs
- **Tests**: Unit tests for ESN reservoir, classical feature extraction, base class PPO, QRH regression; smoke test for CLI end-to-end
- **Benchmarks**: New benchmark entries for CRH in classical category (large pursuit predator environment)
- **Documentation**: Update README.md, CONTRIBUTING.md, AGENTS.md (brain architecture list, count update)
