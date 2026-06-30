# Tasks: Minimal-RNN (minGRU / minLSTM) Policy Arms

## 1. Enabling refactor (lstmppo, behaviour-preserving)

- [ ] 1.1 Extract the inline recurrent-core construction in `LSTMPPOBrain.__init__` (lines 588-603, the `nn.LSTM` / `nn.GRU` / `LayerNormRecurrent` selection) into an overridable `_build_recurrent_core() -> nn.Module` method. The hook MUST also absorb the `self._is_gru = config.rnn_type == "gru"` assignment (line 589) and set a `self._recurrent_core_label`, then replace the direct `config.rnn_type` reference in the init-log (line 744) with that label — these are the two base-`__init__` reads of `config.rnn_type` that would otherwise `AttributeError` for a subclass config. No behaviour change (same `_is_gru`, same cell, same effective log) for `lstm` / `gru` / LayerNorm paths.
- [ ] 1.2 Confirm `_init_recurrent_weights()` / `_rnn_forward()` / `_zero_hidden()` are override-friendly (single-state, `c=None` path already exists for GRU). The minimal-RNN subclass pins `self._is_gru = True` so these take the single-state path. Keep the existing arm byte-identical.
- [ ] 1.3 Run the existing `lstmppo` tests to assert the refactor is byte-identical (no regression).

## 2. Minimal-RNN recurrent core

- [ ] 2.1 Add `MinimalRNN(nn.Module)` (new module `brain/arch/minimal_rnn_ppo.py`): a single-state, input-only-gate cell with `variant in {"mingru", "minlstm"}`, `forward(x_seq, h) -> (output_seq, h_new)` mirroring `LayerNormRecurrent`'s signature.
- [ ] 2.2 Implement the minGRU recurrence: `z = sigmoid(W_z x)`, `h_tilde = W_h x`, `h = (1-z)*h_prev + z*h_tilde`.
- [ ] 2.3 Implement the minLSTM recurrence: `f = sigmoid(W_f x)`, `i = sigmoid(W_i x)`, normalized `f'/i' = f/(f+i), i/(f+i)` (with an `eps` floor), `h_tilde = W_h x`, `h = f'*h_prev + i'*h_tilde`.
- [ ] 2.4 Confirm the convex update keeps the state bounded (no `weight_hh`, no squashing nonlinearity); document the boundedness argument in the class docstring.

## 3. Registered arms

- [ ] 3.1 Add `MinGRUPPOBrainConfig` / `MinLSTMPPOBrainConfig` subclassing `LSTMPPOBrainConfig` (inherit the hyperparameter surface; `sensory_modules` required). Add a `model_validator` that **rejects explicitly setting** the inherited `rnn_type` / `recurrent_layernorm` (via `model_fields_set`) — they are not honoured by the minimal arms (D7), so a stray value must fail loudly, not silently select a two-state path.
- [ ] 3.2 Add `MinGRUPPOBrain` / `MinLSTMPPOBrain` subclassing `LSTMPPOBrain`, overriding `_build_recurrent_core` (pin `self._is_gru = True`, set the core label, build `MinimalRNN` with the right variant) and `_init_recurrent_weights` (Xavier input projections, zero biases; no `weight_hh` pass).
- [ ] 3.3 Decorate each with `@register_brain(name=…, config_cls=…, brain_type=…, families=("classical",))`; ensure `name == BrainType.value`.

## 4. Registration touchpoints (the plugin budget)

- [ ] 4.1 `brain/arch/dtypes.py`: add `BrainType.MINGRU_PPO = "mingruppo"` and `BrainType.MINLSTM_PPO = "minlstmppo"` to the enum **and** to the hand-maintained `BRAIN_TYPES = Literal[...]` (line ~92; NOT auto-derived). The `CLASSICAL_BRAIN_TYPES` family sets ARE registry-derived from the `families=("classical",)` tag — no edit there.
- [ ] 4.2 `brain/arch/__init__.py`: export the two brains + two configs.
- [ ] 4.3 `utils/config_loader.py`: add the two configs to the `BrainConfigType` union (`BRAIN_CONFIG_MAP` auto-derives from the registry — verify it picks them up).
- [ ] 4.4 Verify no `brain_factory.py` branch is needed (default infra shape `{num_actions, device}`) and no manual `_registry.py` edit (decorator self-registers).

## 5. Tests

- [ ] 5.1 Cell unit tests: minGRU + minLSTM step/sequence shapes, gates are input-only (state-independent gating), single-state roll, minLSTM normalization sums to 1, state stays bounded over a long input.
- [ ] 5.2 Config validation: required `sensory_modules`, hidden-dim/chunk guards inherited; the `rnn_type` / `recurrent_layernorm` rejection validator fires when either is explicitly set.
- [ ] 5.3 Registry round-trip: `mingruppo` / `minlstmppo` resolve name ↔ `BrainType` ↔ config; both present in `BRAIN_CONFIG_MAP`; the existing YAML-compat regression enumerates the new configs.
- [ ] 5.4 Forward/learn smoke (discrete + continuous): one `run_brain` → `learn` cycle for each arm in each action mode produces an action and triggers a PPO update without error.
- [ ] 5.5 Weight-persistence round-trip: `get_weight_components()` → `load_weight_components()` restores the core/heads/optimizers/training-state and resets the buffer (the recurrent core serializes under the inherited `"lstm"` component key — assert it round-trips).

## 6. Configs

- [ ] 6.1 Memory cell: `configs/scenarios/bit_memory/{mingruppo,minlstmppo}_small_bit_memory.yml` ([cue, go_signal], bit-memory task enabled, continuous head; mirror the merged per-arm bit-memory configs).
- [ ] 6.2 Reactive cell: the continuous-2D klinotaxis C3 cell configs for the stability A/B (mirror the 029 `lstmppo` continuous config, swapping the brain name).

## 7. Evaluation (two prongs)

- [ ] 7.1 Extend the memory-cell harness: add `mingruppo`, `minlstmppo` to `MEMORY_ARMS` in `scripts/analysis/bit_memory_separation.py` (line 42, hardcoded) and update the pinned-set assertion in `test_bit_memory_separation.py:57`.
- [ ] 7.2 Memory-cell separation: run the new arms (paired seeds) on the bit-memory control, analyze with the harness, and record whether they clear chance + beat the memoryless MLP (LSTM / CfC as the yardstick).
- [ ] 7.3 Extend the reactive-cell harness: add `mingruppo`, `minlstmppo` to `PPO_ARCHS` in `scripts/analysis/t7_continuous_ranking.py` (line 57, hardcoded) — or implement a bespoke pairwise A/B vs `lstmppo` reusing the shared `weight_search_architecture_ranking` stats functions.
- [ ] 7.4 Reactive-cell stability A/B: run the new arms paired-seed vs `lstmppo` on the 029 C3 cell; report return + seeds-converged vs the `lstmppo` baseline.
- [ ] 7.5 Write the logbook (objective / method / results / analysis / limitations), and persist committed supporting artefacts (per-seed CSV + summary JSON).

## 8. Docs + tracker

- [ ] 8.1 Tick the minGRU/minLSTM portion of `T7.separation.new_arch_candidates` with the two-prong verdict; note modified-S5 remains a separate follow-on.
- [ ] 8.2 Update the `AGENTS.md` brain-arch count + enumeration (line 33: 25 → 27, append `mingruppo`, `minlstmppo`); update any matching count/roster in `docs/architecture/plugin-developer-guide.md`.
- [ ] 8.3 Add the logbook row to `docs/experiments/README.md` and any canonical architecture-doc arm listing.

## 9. Gates

- [ ] 9.1 Targeted `pre-commit` (ruff / pyright / markdownlint) on changed files during iteration; full `pre-commit run -a` before push.
- [ ] 9.2 `openspec validate add-minimal-rnn-arms --strict`.
- [ ] 9.3 Full `uv run pytest -m "not nightly"` green (no regression; the `lstmppo` byte-identical assertion holds).
- [ ] 9.4 Archive the change after merge (`openspec archive add-minimal-rnn-arms -y`).
