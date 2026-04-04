## 1. Core Implementation

- [x] 1.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py` with `LSTMPPOBrainConfig` — Pydantic config with all LSTM PPO parameters, validators for lstm_hidden_dim >= 2, bptt_chunk_length >= 4, rollout_buffer_size >= bptt_chunk_length, sensory_modules must not be None (no legacy 2-feature mode)
- [x] 1.2 Implement `LSTMPPORolloutBuffer` in lstmppo.py — stores per-step (features, action, log_prob, value, reward, done, h_state, c_state), `compute_returns_and_advantages()` with standard GAE, `get_sequential_chunks()` yielding chunks with h_init/c_init and shuffled chunk order
- [x] 1.3 Implement `LSTMPPOBrain.__init__()` — build LayerNorm, nn.LSTM or nn.GRU (based on rnn_type), actor MLP, critic MLP, separate Adam optimizers (actor covers LSTM+LayerNorm+actor, critic covers critic only), initialize hidden state to zeros, create rollout buffer
- [x] 1.4 Implement `preprocess()` — use `extract_classical_features(params, self.sensory_modules)` identical to MLPPPOBrain
- [x] 1.5 Implement `run_brain()` — preprocess → LayerNorm → LSTM step (no_grad) → actor sample from h_t → critic value from h_t.detach() → store pending (features, action, log_prob, value, pre-step h, c). Follow `_reservoir_lstm_base.py:489-573` pattern
- [x] 1.6 Implement `learn()` — add pending data + hidden states to buffer, trigger PPO update when buffer full or episode done with sufficient data. Follow `_reservoir_lstm_base.py:575-603` pattern
- [x] 1.7 Implement `_perform_ppo_update()` — for each epoch, iterate shuffled chunks from `get_sequential_chunks()`, within each chunk re-run LSTM from h_init/c_init sequentially, reset hidden at episode boundaries (dones), compute PPO clipped loss + value loss + entropy bonus, separate backward passes for actor and critic optimizers with grad clipping. Follow `_reservoir_lstm_base.py:605-773` pattern
- [x] 1.8 Implement `prepare_episode()` — reset LSTM hidden state to zeros
- [x] 1.9 Implement `post_process_episode()` — increment episode count, update LR schedule, update entropy decay
- [x] 1.10 Implement entropy decay `_get_entropy_coef()` — linear decay from entropy_coef to entropy_coef_end over entropy_decay_episodes
- [x] 1.11 Implement LR scheduling `_get_current_lr()` and `_update_learning_rate()` — warmup + decay for both optimizers, critic LR scales proportionally (preserve configured actor_lr/critic_lr ratio)
- [x] 1.12 Implement weight persistence `get_weight_components()` and `load_weight_components()` — components: lstm, layer_norm, policy, value, actor_optimizer, critic_optimizer, training_state
- [x] 1.13 Implement remaining Brain/ClassicalBrain protocol methods — `action_set` property (getter/setter), `update_memory()` (no-op), `copy()` (raise NotImplementedError, same as MLPPPOBrain), `build_brain()` (raise NotImplementedError), `update_parameters()` (no-op)

## 2. Registration

- [x] 2.1 Add `LSTM_PPO = "lstmppo"` to `BrainType` enum in `brain/arch/dtypes.py`, add to `CLASSICAL_BRAIN_TYPES` set
- [x] 2.2 Add `LSTMPPOBrain` and `LSTMPPOBrainConfig` to `brain/arch/__init__.py` imports and `__all__`
- [x] 2.3 Add `"lstmppo": LSTMPPOBrainConfig` to `BRAIN_CONFIG_MAP` in `utils/config_loader.py`
- [x] 2.4 Add `BrainType.LSTM_PPO` handler in `utils/brain_factory.py` — import and instantiate LSTMPPOBrain

## 3. Tests

- [x] 3.1 Create `tests/quantumnematode_tests/brain/arch/test_lstmppo.py` with config validation tests (defaults, custom values, invalid values rejected)
- [x] 3.2 Add rollout buffer tests — add, full, reset, GAE computation, sequential chunk generation with correct h_init/c_init, episode boundary handling within chunks
- [x] 3.3 Add brain construction test — correct layer dimensions, parameter count, LSTM vs GRU variant
- [x] 3.4 Add single-step test — run_brain returns valid ActionData, hidden state updates
- [x] 3.5 Add multi-step + learn test — buffer fills, PPO update runs without error, loss is computed
- [x] 3.6 Add episode boundary test — prepare_episode resets hidden state, new episode starts fresh
- [x] 3.7 Add GRU end-to-end test — construct with `rnn_type: gru`, run multiple steps, learn, verify PPO update completes and loss changes
- [x] 3.8 Add weight persistence round-trip test — save and load produces same outputs
- [x] 3.9 Add sensory module compatibility test — works with temporal/derivative modules + STAM

## 4. Example Configurations

- [x] 4.1 Create `configs/examples/lstmppo_foraging_small_derivative.yml` — LSTM PPO with derivative chemotaxis foraging
- [x] 4.2 Create `configs/examples/lstmppo_foraging_small_temporal.yml` — LSTM PPO with temporal chemotaxis foraging
- [x] 4.3 Create `configs/examples/lstmppo_thermotaxis_pursuit_predators_large_derivative.yml` — derivative sensing on large triple-objective environment
- [x] 4.4 Create `configs/examples/lstmppo_thermotaxis_pursuit_predators_large_temporal.yml` — temporal sensing on large triple-objective environment
- [x] 4.5 Create `configs/examples/lstmppo_thermotaxis_stationary_predators_large_derivative.yml` — derivative sensing on large stationary predator environment
- [x] 4.6 Create `configs/examples/lstmppo_thermotaxis_stationary_predators_large_temporal.yml` — temporal sensing on large stationary predator environment

## 5. Verification

- [x] 5.1 Run `uv run pytest -m "not nightly"` — all existing + new tests pass
- [x] 5.2 Run `uv run pre-commit run -a` — lint, format, pyright pass
- [x] 5.3 Run foraging sanity check — agent learns (food count increases)
- [x] 5.4 Run predator sanity check — agent survives longer over episodes
- [x] 5.5 Add `lstmppo_foraging_small_derivative.yml` to smoke test suite in `test_smoke.py`
- [x] 5.6 Comprehensive evaluation across foraging, pursuit predators, stationary predators (small and large) with derivative and temporal sensing
- [x] 5.7 GRU vs LSTM ablation — GRU outperforms LSTM across all environments; configs updated to use GRU as default

## 6. Documentation

- [x] 6.1 Update `AGENTS.md` — add `lstmppo` to the brain architecture list in Key Directories section
- [x] 6.2 Update `openspec/config.yaml` — add `lstmppo` to brain architecture list if present
