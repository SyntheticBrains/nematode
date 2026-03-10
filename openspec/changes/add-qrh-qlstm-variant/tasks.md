## Tasks

### Step 1: Register brain types in dtypes.py

**File:** `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py`

- [x] Add `QRH_QLSTM = "qrhqlstm"` to `BrainType` enum
- [x] Add `CRH_QLSTM = "crhqlstm"` to `BrainType` enum
- [x] Add `BrainType.QRH_QLSTM` to `QUANTUM_BRAIN_TYPES` set
- [x] Add `BrainType.CRH_QLSTM` to `CLASSICAL_BRAIN_TYPES` set
- [x] Add both to `BRAIN_TYPES` Literal

### Step 2: Create QRHQLSTMBrainConfig and CRHQLSTMBrainConfig

**File:** `packages/quantum-nematode/quantumnematode/brain/arch/qrhqlstm.py` (new)

- [x] Create `QRHQLSTMBrainConfig(BrainConfig)` with flat Pydantic fields:
  - Reservoir: `num_reservoir_qubits` (8), `reservoir_depth` (3), `reservoir_seed` (42), `use_random_topology` (False), `num_sensory_qubits` (None)
  - LSTM: `lstm_hidden_dim` (64), `bptt_chunk_length` (32)
  - QLIF gates: `shots` (1024), `membrane_tau` (0.9), `refractory_period` (0), `use_quantum_gates` (True)
  - PPO: `actor_lr` (0.0005), `critic_lr` (0.0005), `gamma` (0.99), `gae_lambda` (0.95), `clip_epsilon` (0.2), `entropy_coef` (0.02), `entropy_coef_end` (0.008), `entropy_decay_episodes` (500), `value_loss_coef` (0.5), `num_epochs` (6), `rollout_buffer_size` (1024), `max_grad_norm` (0.5)
  - LR schedule: `lr_warmup_episodes` (None), `lr_warmup_start` (None), `lr_decay_episodes` (None), `lr_decay_end` (None)
  - Critic: `critic_hidden_dim` (128), `critic_num_layers` (2)
  - Sensory: `sensory_modules` (list)
- [x] Create `CRHQLSTMBrainConfig(BrainConfig)` with CRH reservoir params:
  - Reservoir: `num_reservoir_neurons` (10), `reservoir_depth` (3), `spectral_radius` (0.9), `input_connectivity` ("sparse"), `input_scale` (1.0), `feature_channels` (["raw", "cos_sin", "pairwise"]), `input_encoding` ("linear")
  - All other fields identical to QRHQLSTMBrainConfig

### Step 3: Create ReservoirLSTMBase abstract class

**File:** same `qrhqlstm.py`

- [x] Create `ReservoirLSTMBase(ClassicalBrain)` abstract class with:
  - Abstract method: `_create_reservoir(config) -> ReservoirHybridBase`
  - Abstract method: `_compute_reservoir_feature_dim(config) -> int`
  - `__init__`: instantiate reservoir via `_create_reservoir()`, create `QLIFLSTMCell` (imported from `qliflstm.py`), create actor head (`nn.Linear(feature_dim + lstm_hidden_dim, num_actions)`), create critic MLP, create `LayerNorm(feature_dim)`, create optimizers (separate actor/critic), create rollout buffer, init LSTM hidden state
  - Sensory preprocessing: delegate to `reservoir.preprocess(params)` which internally uses `extract_classical_features()` from `brain/modules.py`
  - Reservoir feature extraction: call `reservoir._get_reservoir_features(sensory_features)` on the preprocessed output

### Step 4: Implement run_brain() in ReservoirLSTMBase

**File:** same `qrhqlstm.py`

- [x] Implement `run_brain(params, reward=None, input_data=None, *, top_only, top_randomize) -> list[ActionData]` matching `ClassicalBrain` protocol:
  1. Preprocess sensory input via `reservoir.preprocess(params)`
  2. Extract reservoir features via `reservoir._get_reservoir_features(sensory_features)`
  3. Apply LayerNorm to reservoir features
  4. Run QLIF-LSTM cell: `(normalized_features, h_t, c_t) -> (h_new, c_new)`
  5. Actor: `[reservoir_features, h_t] -> logits -> Categorical -> sample action`
  6. Critic: `[reservoir_features, h_t.detach()] -> value`
  7. Store transition in buffer (reservoir features, action, log_prob, value)
  8. Return `list[ActionData]`

### Step 5: Implement learn() and PPO update in ReservoirLSTMBase

**File:** same `qrhqlstm.py`

- [x] Create rollout buffer class (adapted from `QLIFLSTMRolloutBuffer`):
  - Store: reservoir features, actions, log_probs, values, rewards, dones, chunk-boundary h/c states
  - `get_sequential_chunks()`: split into BPTT chunks with initial hidden states
  - Note: reuses `QLIFLSTMRolloutBuffer` directly (import, not copy)
- [x] Implement `learn(params, reward, *, episode_done)` matching `ClassicalBrain` protocol:
  1. Store reward/done in buffer
  2. If buffer full or episode done: compute GAE, run PPO update
- [x] Implement PPO update with truncated BPTT:
  1. For each epoch: get sequential chunks, re-run LSTM forward per chunk from stored h/c
  2. Compute PPO clipped loss + entropy bonus + value loss
  3. Gradient step with max_grad_norm clipping
  4. Reset buffer after update

### Step 6: Implement episode lifecycle in ReservoirLSTMBase

**File:** same `qrhqlstm.py`

- [x] `prepare_episode()`: reset h_t, c_t to zeros, clear pending state
- [x] `post_process_episode(*, episode_success=None)`: increment episode count, update LR schedule
- [x] `update_memory(reward)`: no-op
- [x] `copy()`: deep copy with fresh hidden states and independent reservoir
- [x] LR scheduling: reuse `_get_current_lr()` and `_update_learning_rate()` pattern from QLIF-LSTM (warmup + decay)
- [x] Entropy decay: reuse `_get_entropy_coef()` pattern from QLIF-LSTM

### Step 7: Implement QRHQLSTMBrain and CRHQLSTMBrain subclasses

**File:** same `qrhqlstm.py`

- [x] `QRHQLSTMBrain(ReservoirLSTMBase)`:
  - `_create_reservoir()`: construct `QRHBrain(qrh_config)` from own config fields
  - `_compute_reservoir_feature_dim()`: `3N + N(N-1)/2` for N qubits
  - Build a `QRHBrainConfig` from own fields to pass to QRHBrain constructor (unused MLP readout params use defaults — accepted trade-off, see design decision 2)
  - Pass `sensory_modules` through to the inner `QRHBrainConfig` so `preprocess()` uses the correct sensory modules
- [x] `CRHQLSTMBrain(ReservoirLSTMBase)`:
  - `_create_reservoir()`: construct `CRHBrain(crh_config)` from own config fields
  - `_compute_reservoir_feature_dim()`: delegate to CRH's feature channel computation
  - Build a `CRHBrainConfig` from own fields to pass to CRHBrain constructor (same trade-off as QRH)
  - Pass `sensory_modules` through to the inner `CRHBrainConfig`

### Step 8: Register in `__init__.py`, config_loader.py, and brain_factory.py

**Files:**

- `packages/quantum-nematode/quantumnematode/brain/arch/__init__.py`

- `packages/quantum-nematode/quantumnematode/utils/config_loader.py`

- `packages/quantum-nematode/quantumnematode/utils/brain_factory.py`

- [ ] Add imports to `__init__.py`: `QRHQLSTMBrain`, `QRHQLSTMBrainConfig`, `CRHQLSTMBrain`, `CRHQLSTMBrainConfig`

- [ ] Add to `__all__`

- [ ] Add to `BRAIN_CONFIG_MAP` in config_loader.py: `"qrhqlstm": QRHQLSTMBrainConfig`, `"crhqlstm": CRHQLSTMBrainConfig`

- [ ] Add `BrainType.QRH_QLSTM` and `BrainType.CRH_QLSTM` branches to `brain_factory.py`'s `create_brain()` function, following the existing pattern (import, isinstance check, constructor call with `config=brain_config, num_actions=4, device=device`)

### Step 9: Create evaluation config YAMLs

**Directory:** `configs/examples/`

- [ ] `qrhqlstm_thermotaxis_stationary_predators_large.yml` — QRH-QLSTM on stationary predators (primary test, quantum gates enabled). Use same environment as `qrh_thermotaxis_stationary_predators_large.yml`.
- [ ] `qrhqlstm_thermotaxis_stationary_predators_large_classical.yml` — Same but `use_quantum_gates: false` (classical QLIF gate ablation)
- [ ] `crhqlstm_thermotaxis_stationary_predators_large.yml` — CRH-QLSTM on stationary predators (classical reservoir ablation)
- [ ] `qrhqlstm_pursuit_predators_small.yml` — QRH-QLSTM on pursuit predators (regression test). Use same environment as `qrh_pursuit_predators_small.yml`.
- [ ] `qrhqlstm_foraging_small.yml` — QRH-QLSTM on foraging (smoke test baseline)
- [ ] `crhqlstm_foraging_small.yml` — CRH-QLSTM on foraging (smoke test baseline for CRH variant)

### Step 10: Create unit tests

**File:** `tests/quantumnematode_tests/brain/arch/test_qrhqlstm.py` (new)

- [x] `test_qrhqlstm_config_defaults` — verify config field defaults
- [x] `test_qrhqlstm_brain_init` — verify brain initializes with default config (reservoir created, LSTM cell created, actor/critic built)
- [x] `test_qrhqlstm_brain_run_brain` — verify run_brain() with mock BrainParams produces valid list[ActionData]
- [x] `test_qrhqlstm_brain_hidden_state_reset` — verify prepare_episode() zeros h_t/c_t
- [x] `test_qrhqlstm_brain_learn` — verify learn() stores transitions and triggers PPO update when buffer full
- [x] `test_qrhqlstm_classical_ablation` — verify `use_quantum_gates=False` produces valid outputs
- [x] `test_crh_qlstm_brain_init` — verify CRH variant initializes correctly
- [x] `test_crh_qlstm_brain_run_brain` — verify CRH variant produces valid list[ActionData]

### Step 11: Verification

- [ ] Lint/type check: `uv run pre-commit run -a`
- [ ] Smoke tests: `uv run pytest -m smoke -v` (all pass, including new configs)
- [ ] Unit tests: `uv run pytest tests/quantumnematode_tests/brain/arch/test_qrhqlstm.py -v`
- [ ] Full test suite (skip nightly): `uv run pytest -m "not nightly"`
- [ ] Manual smoke run: `uv run ./scripts/run_simulation.py --config ./configs/examples/qrhqlstm_foraging_small.yml --episodes 3`
