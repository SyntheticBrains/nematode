## Why

QRH (Quantum Reservoir Hybrid) excels at pursuit predator evasion (+11pp over CRH) but plateaus at ~45% last-100 SR on stationary predators — matching CRH with no quantum advantage. The stationary predator task requires spatial memory (remembering fixed toxic zone locations), which QRH's feedforward MLP readout cannot provide. Meanwhile, QLIF-LSTM (H.4) demonstrated strong temporal memory for pursuit predators (82% last-100) but also plateaus at ~37% on stationary predators due to limited sensory features (9-dim raw input). Composing QRH's rich 52-dim quantum reservoir features with QLIF-LSTM's temporal readout could address both weaknesses — the reservoir provides expressive spatial features while the LSTM provides temporal integration to remember zone locations across timesteps.

## What Changes

- Add `QRHQLSTMBrain` class: a new brain that uses QRH's quantum reservoir for feature extraction and QLIF-LSTM cell (with optional quantum gates) as the temporal readout, replacing the feedforward MLP
- Add `QRHQLSTMBrainConfig`: config extending reservoir hybrid base with LSTM-specific parameters (hidden dim, BPTT chunk length, QLIF gate settings)
- Add `QRHQLSTM` brain type registration in dtypes, config loader, and brain arch init
- Replace the base class's minibatch PPO with truncated BPTT recurrent PPO (reusing patterns from QLIF-LSTM brain) to properly train the recurrent readout
- Add evaluation configs for stationary predators (quantum + classical ablation) and pursuit predators (regression test)
- Add corresponding CRH-LSTM variant (`CRHQLSTMBrain`) for classical reservoir ablation comparison

## Capabilities

### New Capabilities

- `qrh-qlstm-brain`: QRH quantum reservoir with QLIF-LSTM temporal readout — brain registration, config, reservoir-to-LSTM pipeline, recurrent PPO training, and QLIF gate quantum/classical ablation mode
- `crh-qlstm-brain`: CRH classical reservoir with QLIF-LSTM temporal readout — classical reservoir ablation companion to QRH-QLSTM, enabling the 2×2 comparison matrix (quantum/classical reservoir × quantum/classical LSTM gates)

### Modified Capabilities

- `qrh-brain`: No requirement changes — QRH reservoir code is reused unchanged
- `crh-brain`: No requirement changes — CRH reservoir code is reused unchanged

## Impact

- **New files**: `brain/arch/qrhqlstm.py` (main implementation), evaluation config YAMLs, test file
- **Modified files**: `brain/arch/dtypes.py` (new brain type), `brain/arch/__init__.py` (exports), `utils/config_loader.py` (config registration), `utils/brain_factory.py` (brain instantiation)
- **Dependencies**: Reuses `QLIFLSTMCell` from `qliflstm.py`, quantum reservoir from `qrh.py`, `ReservoirHybridBase` patterns from `_reservoir_hybrid_base.py`
- **No breaking changes**: Existing QRH and QLIF-LSTM brains are unchanged
- **Risk**: Recurrent PPO over quantum reservoir features is untested — the 52-dim reservoir output may be too high-dimensional for a 64-dim LSTM hidden state. May need to tune LSTM hidden dim upward or add a projection layer.

## Evaluation Results

Implementation complete. Extensive evaluation across 58 training sessions (~15,400 episodes) covering 4 stages:

### Key Findings

1. **QRH-QLSTM quantum gates provide no advantage over classical sigmoid gates** — performance within noise across all environments
2. **CRH-QLSTM matches or slightly exceeds QRH-QLSTM** on most tasks, confirming that the quantum reservoir provides no advantage when paired with LSTM readout
3. **QRH-LSTM (classical gates) performs WORSE than QRH standalone MLP** on every task tested — LSTM readout hurts QRH
4. **Stage 4d hypothesis REJECTED**: QRH-LSTM does NOT beat QRH-MLP on stationary predators (−4.2pp worse, not the hypothesized ≥5pp improvement)

### Summary Table

| Environment | QRH-QLSTM | CRH-QLSTM | QRH-LSTM | QRH-MLP (baseline) |
|---|---|---|---|---|
| Foraging small | 100% | 100% | — | 100% |
| Pursuit small | 25.3% | 32.0% | 17.0% | 36.5% |
| Thermo+pursuit large | 20.5% | 25.5% | 16.4% | 41.3% |
| Thermo+stationary large | 40.8% | 49.8% | 41.0% | 45.2% |

### Conclusion

The reservoir-LSTM composition does not improve over simpler architectures. The bottleneck is the fixed (non-trainable) quantum reservoir, not the readout architecture. The reservoir's feature expansion (7 sensory → 75 features) helps local evasion but hurts path efficiency in large grids. The QLIF quantum gates add overhead without measurable benefit when the input is already quantum-processed reservoir features.
