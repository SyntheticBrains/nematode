# Tasks: Minimal-RNN (minGRU / minLSTM) Policy Arms

## 1. Enabling refactor (lstmppo, behaviour-preserving)

- [x] 1.1 Extract the inline recurrent-core construction in `LSTMPPOBrain.__init__` (lines 588-603, the `nn.LSTM` / `nn.GRU` / `LayerNormRecurrent` selection) into an overridable `_build_recurrent_core() -> nn.Module` method. The hook MUST also absorb the `self._is_gru = config.rnn_type == "gru"` assignment (line 589) and set a `self._recurrent_core_label`, then replace the direct `config.rnn_type` reference in the init-log (line 744) with that label — these are the two base-`__init__` reads of `config.rnn_type` that would otherwise `AttributeError` for a subclass config. No behaviour change (same `_is_gru`, same cell, same effective log) for `lstm` / `gru` / LayerNorm paths.
- [x] 1.2 Confirm `_init_recurrent_weights()` / `_rnn_forward()` / `_zero_hidden()` are override-friendly (single-state, `c=None` path already exists for GRU). The minimal-RNN subclass pins `self._is_gru = True` so these take the single-state path. Keep the existing arm byte-identical.
- [x] 1.3 Run the existing `lstmppo` tests to assert the refactor is byte-identical (no regression). — 77 lstmppo tests pass.

## 2. Minimal-RNN recurrent core

- [x] 2.1 Add `MinimalRNN(nn.Module)` (new module `brain/arch/minimal_rnn_ppo.py`): a single-state, input-only-gate cell with `variant in {"mingru", "minlstm"}`, `forward(x_seq, h) -> (output_seq, h_new)` mirroring `LayerNormRecurrent`'s signature.
- [x] 2.2 Implement the minGRU recurrence: `z = sigmoid(W_z x)`, `h_tilde = W_h x`, `h = (1-z)*h_prev + z*h_tilde`.
- [x] 2.3 Implement the minLSTM recurrence: `f = sigmoid(W_f x)`, `i = sigmoid(W_i x)`, normalized `f'/i' = f/(f+i), i/(f+i)` (with an `eps` floor), `h_tilde = W_h x`, `h = f'*h_prev + i'*h_tilde`.
- [x] 2.4 Confirm the convex update keeps the state bounded (no `weight_hh`, no squashing nonlinearity); document the boundedness argument in the class docstring. — covered by `test_cell_state_is_bounded_by_candidates`.

## 3. Registered arms

- [x] 3.1 Add `MinGRUPPOBrainConfig` / `MinLSTMPPOBrainConfig` subclassing `LSTMPPOBrainConfig` (inherit the hyperparameter surface; `sensory_modules` required). Add a `model_validator` that **rejects a non-default value** of the inherited `rnn_type` / `recurrent_layernorm` — they are not honoured by the minimal arms (D7), so a stray value must fail loudly. (Compares values, not `model_fields_set`: the config loader repopulates every field from its default, so a `model_fields_set` check is unreliable.)
- [x] 3.2 Add `MinGRUPPOBrain` / `MinLSTMPPOBrain` subclassing `LSTMPPOBrain`, overriding `_build_recurrent_core` (pin `self._is_gru = True`, set the core label, build `MinimalRNN` with the right variant) and `_init_recurrent_weights` (Xavier input projections, zero biases; no `weight_hh` pass).
- [x] 3.3 Decorate each with `@register_brain(name=…, config_cls=…, brain_type=…, families=("classical",))`; ensure `name == BrainType.value`.

## 4. Registration touchpoints (the plugin budget)

- [x] 4.1 `brain/arch/dtypes.py`: add `BrainType.MINGRU_PPO = "mingruppo"` and `BrainType.MINLSTM_PPO = "minlstmppo"` to the enum **and** to the hand-maintained `BRAIN_TYPES = Literal[...]` (NOT auto-derived). The `CLASSICAL_BRAIN_TYPES` family sets ARE registry-derived from the `families=("classical",)` tag — no edit there.
- [x] 4.2 `brain/arch/__init__.py`: export the two brains + two configs.
- [x] 4.3 `utils/config_loader.py`: add the two configs to the `BrainConfigType` union (`BRAIN_CONFIG_MAP` auto-derives from the registry — verified it picks them up).
- [x] 4.4 Verify no `brain_factory.py` branch is needed (default infra shape `{num_actions, device}`) and no manual `_registry.py` edit (decorator self-registers). — confirmed.

## 5. Tests

- [x] 5.1 Cell unit tests: minGRU + minLSTM step/sequence shapes, gates are input-only (affine `h0 -> h_T`), single-state roll, state stays bounded over a long input.
- [x] 5.2 Config validation: required `sensory_modules`; the `rnn_type` / `recurrent_layernorm` rejection validator fires on a non-default value; defaults accepted.
- [x] 5.3 Registry round-trip: `mingruppo` / `minlstmppo` resolve name ↔ `BrainType` ↔ config; both present in `BRAIN_CONFIG_MAP`; the existing YAML-compat regression enumerates the new configs.
- [x] 5.4 Forward/learn smoke (discrete + continuous): one `run_brain` → `learn` cycle for each arm in each action mode produces an action and triggers a PPO update without error.
- [x] 5.5 Weight-persistence round-trip: `get_weight_components()` → `load_weight_components()` restores the core (the recurrent core serializes under the inherited `"lstm"` component key).

## 6. Configs

- [x] 6.1 Memory cell: `configs/scenarios/bit_memory/{mingruppo,minlstmppo}_small_bit_memory.yml` ([cue, go_signal], bit-memory task enabled, continuous head; mirror the merged per-arm bit-memory configs, minus the rejected `rnn_type`).
- [x] 6.2 Reactive cell: `configs/scenarios/foraging_predator_thermal/{mingruppo,minlstmppo}_small_continuous2d_combined_klinotaxis.yml` — identical env/reward/satiety block + recipe to the 029 `lstmppo` C3 config, swapping only the brain (minus `rnn_type`).

## 7. Evaluation (two prongs)

- [x] 7.1 Extend the memory-cell harness: add `mingruppo`, `minlstmppo` to `MEMORY_ARMS` in `scripts/analysis/bit_memory_separation.py` (line 42, hardcoded). (The `test_bit_memory_separation.py:57` assertion needs no change — its synthetic table excludes the new arms, so they are not in `separating_arms`.)
- [x] 7.2 Memory-cell separation (n=8): SEPARATION CONFIRMED — minLSTM 0.966 / minGRU 0.956 clear chance + beat the memoryless MLP (BH-FDR q=0.007), on par with LSTM 0.939. (Needed a memory-friendly retention-gate init — without it both sat at chance; see logbook 031.)
- [x] 7.3 Extend the reactive-cell harness: add `mingruppo`, `minlstmppo` to `PPO_ARCHS` in `scripts/analysis/t7_continuous_ranking.py` (line 57, hardcoded).
- [x] 7.4 Reactive-cell stability A/B (n=8, plateau-tail full-clear success via the bespoke `scripts/analysis/minimal_rnn_reactive_ab.py`): both BEAT `lstmppo` — minLSTM 73.1 (+17.0, q=0.016 \*\*\*), minGRU 66.2 (+10.1, ns) vs lstmppo 56.1 — lower spread + higher floor; the hold-init costs nothing on the reactive cell.
- [x] 7.5 Logbook [031](../../../docs/experiments/logbooks/031-minimal-rnn-candidates.md) written + committed supporting artefacts (separation.json, cue-match-per-seed.csv, reactive_ab.json).

## 8. Docs + tracker

- [x] 8.1 Annotated `T7.separation.new_arch_candidates` with the two-prong verdict (box left open — modified-S5 remains the open Tier-1 item).
- [x] 8.2 Update the `AGENTS.md` brain-arch count + enumeration (25 → 27, append `mingruppo`, `minlstmppo`). (The plugin-developer-guide has no fixed arch count/roster to bump.)
- [x] 8.3 Added the 031 row to `docs/experiments/README.md`.

## 9. Gates

- [ ] 9.1 Targeted `pre-commit` (ruff / pyright / markdownlint) on changed files during iteration; full `pre-commit run -a` before push.
- [x] 9.2 `openspec validate add-minimal-rnn-arms --strict`.
- [x] 9.3 Full `uv run pytest -m "not nightly"` green — 4002 passed (no regression; the `lstmppo` byte-identical assertion holds).
- [ ] 9.4 Archive the change after merge (`openspec archive add-minimal-rnn-arms -y`).
