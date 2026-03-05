## 1. Brain Type Registration

- [ ] 1.1 Add `QLIF_LSTM = "qliflstm"` to `BrainType` enum in `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py`
- [ ] 1.2 Add `BrainType.QLIF_LSTM` to `BRAIN_TYPES` Literal and `QUANTUM_BRAIN_TYPES` set in `dtypes.py`

## 2. Core Implementation — QLIFLSTMCell

- [ ] 2.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/qliflstm.py` with module docstring, imports, and default constants
- [ ] 2.2 Implement `QLIFLSTMBrainConfig` (Pydantic BaseModel) with all config fields: lstm_hidden_dim, shots, membrane_tau, refractory_period, PPO params, actor/critic LRs, bptt_chunk_length, use_quantum_gates, sensory_modules, device_type
- [ ] 2.3 Implement `QLIFLSTMCell(nn.Module)` with 4 linear projections (W_f, W_i, W_c, W_o), QLIF quantum gate execution for forget/input gates, classical tanh/sigmoid for candidate/output gates
- [ ] 2.4 Implement classical ablation path in QLIFLSTMCell — when `use_quantum_gates=False`, use `torch.sigmoid()` instead of QLIF circuits

## 3. Core Implementation — Rollout Buffer

- [ ] 3.1 Implement `QLIFLSTMRolloutBuffer` with storage for features, actions, log_probs, values, rewards, dones, and LSTM hidden states (h_t, c_t)
- [ ] 3.2 Implement `compute_returns_and_advantages()` with GAE computation
- [ ] 3.3 Implement `get_sequential_chunks()` — split buffer into bptt_chunk_length chunks with initial (h_0, c_0) per chunk, episode boundary detection for hidden state reset

## 4. Core Implementation — QLIFLSTMBrain

- [ ] 4.1 Implement `QLIFLSTMBrain.__init__()` — create QLIFLSTMCell, actor head (Linear), critic MLP, optimizers, Qiskit backend, sensory module config, zero-init h_t/c_t
- [ ] 4.2 Implement `run_brain()` — extract features, encode sensory spikes, run LSTM cell forward, actor sampling, critic value, store transition in buffer
- [ ] 4.3 Implement `learn()` — store reward/done, trigger PPO update when buffer full or episode done, chunk-based truncated BPTT, PPO clipped surrogate loss, critic Huber loss, gradient clipping
- [ ] 4.4 Implement `prepare_episode()` — reset h_t/c_t to zeros, clear pending transition
- [ ] 4.5 Implement `post_process_episode()`, `copy()`, `update_memory()`, and remaining Brain protocol methods

## 5. Module Registration

- [ ] 5.1 Add `from .qliflstm import QLIFLSTMBrain, QLIFLSTMBrainConfig` to `brain/arch/__init__.py` and update `__all__`
- [ ] 5.2 Add `QLIFLSTMBrainConfig` to imports, `BrainConfigType` union, and `BRAIN_CONFIG_MAP` in `utils/config_loader.py`
- [ ] 5.3 Add `QLIFLSTMBrainConfig` to `BrainConfigType` union and QLIF_LSTM handler in `utils/brain_factory.py`

## 6. Configuration Files

- [ ] 6.1 Create `configs/examples/qliflstm_foraging_small.yml` — Stage 4a foraging baseline
- [ ] 6.2 Create `configs/examples/qliflstm_pursuit_predators_small.yml` — Stage 4b pursuit predators (small grid)
- [ ] 6.3 Create `configs/examples/qliflstm_thermotaxis_pursuit_predators_large.yml` — Stage 4c pursuit + thermotaxis (large grid)
- [ ] 6.4 Create `configs/examples/qliflstm_thermotaxis_stationary_predators_large.yml` — Stage 4c stationary + thermotaxis (large grid)

## 7. Tests

- [ ] 7.1 Create `tests/quantumnematode_tests/brain/arch/test_qliflstm.py` with test_qlif_lstm_cell_forward_shape, test_qlif_lstm_cell_classical_ablation
- [ ] 7.2 Add test_qlif_lstm_brain_run_brain, test_qlif_lstm_brain_learn, test_qlif_lstm_brain_hidden_state_reset
- [ ] 7.3 Add test_qlif_lstm_rollout_buffer_chunks, test_qlif_lstm_brain_config_defaults

## 8. Verification

- [ ] 8.1 Run `uv run pre-commit run -a` — lint and type check pass
- [ ] 8.2 Run `uv run pytest tests/quantumnematode_tests/brain/arch/test_qliflstm.py -v` — all new tests pass
- [ ] 8.3 Run `uv run pytest -m "not nightly"` — no regressions in existing tests
- [ ] 8.4 Run `uv run pytest -m smoke -v` — smoke tests pass
