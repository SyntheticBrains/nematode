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
