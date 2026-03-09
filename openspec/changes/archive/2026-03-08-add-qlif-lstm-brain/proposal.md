## Why

The codebase has zero temporal architectures — every brain treats each simulation step independently, discarding all temporal context. This is the single largest capability gap blocking Phase 3 (Learning & Memory) of the roadmap. H.4 (QLIF-LSTM) introduces within-episode temporal memory by replacing LSTM forget and input gates with QLIF quantum neuron measurements, directly testing whether quantum gate dynamics provide richer temporal gating than classical sigmoid. After 200+ evaluation sessions across six architectures, this is the highest-impact next step because it simultaneously advances Phase 2 (novel quantum architecture) and Phase 3 (temporal infrastructure).

## What Changes

- Add a new `QLIFLSTMBrain` architecture: a custom LSTM cell where forget gate (f_t) and input gate (i_t) use QLIF quantum circuits (via `QLIFSurrogateSpike`) instead of sigmoid, with classical tanh cell candidate and sigmoid output gate
- Implement chunk-based truncated BPTT for recurrent PPO training — the rollout buffer stores LSTM hidden states at chunk boundaries so chunks can be re-run during multi-epoch PPO updates
- Add `use_quantum_gates: bool` config flag for classical ablation control (when False, uses torch.sigmoid instead of QLIF circuits)
- LSTM hidden state (h_t, c_t) resets per episode via `prepare_episode()` — food and predators fully reset each episode, so cross-episode memory would be harmful
- Register `BrainType.QLIF_LSTM` in dtypes, `__init__.py`, and config_loader
- Create 4 evaluation config YAMLs covering all H.4 stages: foraging (4a), pursuit predators small (4b), thermotaxis pursuit predators large (4c), thermotaxis stationary predators large (4c)
- Add unit tests for cell, brain, rollout buffer, config, and hidden state reset

## Capabilities

### New Capabilities

- `qlif-lstm-brain`: QLIF-LSTM brain architecture — custom LSTM cell with quantum QLIF gates, recurrent PPO training via chunk-based truncated BPTT, classical ablation support, and episode-scoped temporal memory

### Modified Capabilities

- `brain-architecture`: Add QLIF_LSTM brain type registration (BrainType enum, QUANTUM_BRAIN_TYPES, BRAIN_TYPES, module exports, config loader mapping)

## Impact

- **New file:** `packages/quantum-nematode/quantumnematode/brain/arch/qliflstm.py` — QLIFLSTMBrainConfig, QLIFLSTMCell, QLIFLSTMRolloutBuffer, QLIFLSTMBrain
- **Modified:** `brain/arch/dtypes.py` (BrainType enum, QUANTUM_BRAIN_TYPES, BRAIN_TYPES)
- **Modified:** `brain/arch/__init__.py` (imports + `__all__`)
- **Modified:** `utils/config_loader.py` (BrainConfigType union, BRAIN_CONFIG_MAP, import)
- **Modified:** `utils/brain_factory.py` (BrainConfigType union, QLIF_LSTM handler in setup_brain_model)
- **New configs:** 4 YAML files in `configs/examples/`
- **New test file:** `tests/quantumnematode_tests/brain/arch/test_qliflstm.py`
- **Dependencies:** Reuses existing `_qlif_layers.py` (QLIFSurrogateSpike, build_qlif_circuit, get_qiskit_backend, encode_sensory_spikes) — no new external dependencies
