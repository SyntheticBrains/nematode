# Tasks

Line numbers are as of `0ae24375`. Re-verify before editing.

## 1. Extend `_policy.py`

- [x] 1.1 Add `categorical_logprob_entropy_from_probs(probs, action, *, device=None) -> (log_prob, entropy, probs)` to [`_policy.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_policy.py) (D2). Score an already-constructed probability vector via `Categorical(probs)`; document that `Categorical` normalises internally and that no epsilon floor is applied (D6).
- [x] 1.2 Refactor `categorical_logprob_entropy_torch` to delegate to 1.1 after its `softmax`, so there is one `Categorical` construction site. Must stay byte-exact for the seven already-migrated brains — assert with `torch.equal` in `test_policy.py`.
- [x] 1.3 Add `reinforce_policy_loss(log_probs, advantages) -> Tensor` returning `-(log_probs * advantages).mean()` (D5). Docstring notes it is the REINFORCE counterpart of `ppo_clip_policy_loss` and that callers add their own entropy bonus.
- [x] 1.4 Add both to `__all__`. Update the module docstring for the four-family split (D1). **D8 open question resolved: table.** The prose form named individual brains, which does not survive 20 brains across four families; the docstring now carries a four-row family table (distribution scored / rollout sampler / tolerance) plus a note that rollout and update scoring must always migrate together.
- [x] 1.5 Unit tests in `test_policy.py`: the probs-based scorer against an explicit ε-mixture reference; `reinforce_policy_loss` against the inline expression; the 1.2 delegation byte-equivalence.
- [x] 1.6 Tree green with no brain touched yet: **4071 passed, 1 skipped, 2 xfailed** (baseline 4062 + the 9 tests added here; skip/xfail unchanged), `pyright` **0 errors**, ruff clean. Commit.

## 2. Shared bases — Family A + B (5 brains)

- [ ] 2.1 `_reservoir_hybrid_base.py` **update** (`:678-693`): `probs`/`dist`/`log_prob`/`entropy` → `categorical_evaluate_torch`; `surr1`/`surr2`/`min` → `ppo_clip_policy_loss`. Family A, byte-exact.
- [ ] 2.2 `_reservoir_hybrid_base.py` **rollout** (`:454-457`): → `categorical_sample_torch`, passing **`device=self.device` explicitly** — the current code builds the action tensor on `self.device` (`:457`) while the helper defaults to `logits.device`; they coincide today, but relying on that would make byte-exactness depend on an unstated invariant. Confirm the returned `probs` still feeds `self.current_probabilities` and the `buffer.position % 50` diagnostic unchanged, and that the discarded `entropy` is the accepted cost recorded in D8.
- [ ] 2.3 Confirm `crh`, `qef`, `qrh` are covered with no per-brain edit (D0.2 — `qef` overrides neither `run()` nor the update). Re-verify by grep that none of the three defines a policy-scoring method.
- [ ] 2.4 `_reservoir_lstm_base.py` **rollout** (`:538-545`): keep `rng.choice` at `:541` **verbatim**; replace `np.log(action_probs[action_idx] + 1e-8)` with `categorical_logprob_entropy_torch(logits, int(action_idx))`. Family B.
- [ ] 2.5 `_reservoir_lstm_base.py` **update** (`:686-721`): per-step manual `torch.log`/`-Σ p log p` → `categorical_logprob_entropy_torch`; surrogate → `ppo_clip_policy_loss`, keeping `ratio` for the clip-fraction metric at `:726` (mirror the `lstmppo.py:1202-1216` comment style).
- [ ] 2.6 Family A + B migration tests per D7 (same-run inline reference; `torch.equal` for A, `allclose(rtol=0, atol=1e-7)` for B).
- [ ] 2.7 `test_reservoir_hybrid_base.py`, `test_qef.py`, `test_crh.py`, `test_qrh.py`, `test_crhqlstm.py`, `test_qrhqlstm.py` pass unchanged. Commit.

## 3. Hybrid family — `_hybrid_common` + the three brains (D4: one commit)

- [ ] 3.1 `_hybrid_common.py` `perform_ppo_update` (`:445-469`): → `categorical_evaluate_torch` + `ppo_clip_policy_loss`, keeping `log_ratio`/`ratio` for the `approx_kl` term at `:455-458`. Byte-exact in isolation.
- [ ] 3.2 `hybridquantum.py` cortex **rollout** (`:1072-1073`): `torch.log(cortex_probs + 1e-8)` → the shared scorer, so both halves of the cortex ratio use one formula (D4). Leave the `np.clip`/renormalise at `:1081-1082` and `rng.choice` at `:1083` **verbatim**.
- [ ] 3.3 Same for `hybridclassical.py` (`:777-778`, sampler at `:794` region) and `hybridquantumcortex.py` (`:1926`, `:1934` — note these two adjacent branches spell the same `1e-8` as a literal and as `NORM_EPS`; both go).
- [ ] 3.3b Where a Family-C rollout hands its **numpy** mixture to the shared scorer, convert with `torch.as_tensor(action_probs, dtype=torch.float32)` so the rollout matches the update's dtype (D2). Do **not** use `torch.from_numpy`, which carries float64 through and would leave a dtype-induced offset in the ratio after the formula was shared.
- [ ] 3.4 `hybridquantum.py` reflex **update** (`:1339-1360`): ε-mixed `action_probs` → `categorical_logprob_entropy_from_probs`; surrogate → `ppo_clip_policy_loss`, keeping `- effective_entropy_coef * mean_entropy` as a separate term. Leave `_exploration_schedule()` and the mixture construction untouched (D2).
- [ ] 3.5 Same for `hybridclassical.py` (`:1020-1042`) and `hybridquantumcortex.py` (`:2222-2244`).
- [ ] 3.6 Measure the pre/post ratio deviation at `ratio == 1` for the **cortex** and **reflex** paths **separately** (D4) and record both here. Expected: cortex reaches **exactly 1** (same-dtype torch on both sides — the formula mismatch is the whole defect); reflex lands at **~1e-7** (the numpy-float64 / torch-float32 boundary is pre-existing and survives). A single averaged figure would hide a half-done fix, so do not report one.
- [ ] 3.7 Family C migration test per D7 — reference expression must be the **ε-mixture**, not a plain softmax, so a helper that re-softmaxed would fail.
- [ ] 3.8 `test_hybridquantum.py`, `test_hybridclassical.py`, `test_hybridquantumcortex.py` pass unchanged. Commit.

## 4. Remaining direct copies

- [ ] 4.1 `qsnnppo.py` **rollout** (`:920-931`): keep the numpy softmax and `rng.choice` at `:927` verbatim; `np.log(...+1e-8)` at `:931` → shared scorer. **update** (`:1110-1149`): manual log/entropy → `categorical_logprob_entropy_torch`; surrogate → `ppo_clip_policy_loss`, keeping `ratio` for clip-frac and `approx_kl` at `:1152-1157`.
- [ ] 4.2 `qsnnreinforce.py` (`:1590-1615` update, `:1946` rollout): Family C — ε-mixed, same treatment as 3.4. Despite the brain's name this path is a PPO-clipped surrogate.
- [ ] 4.3 `qliflstm.py` (`:1099-1134` update, `:965` rollout): Family B, same treatment as 2.4/2.5; keep the clip-frac at `:1137-1139`.
- [ ] 4.4 `env/mlpppo_predator_brain.py`: **rollout** (`:318-322`) → `categorical_sample_torch`; **update** (`:438-449`) → `categorical_evaluate_torch` + `ppo_clip_policy_loss`. Family A, byte-exact. Check the `:339` cumulative-softmax inversion branch is unaffected.
- [ ] 4.5 `test_qsnnppo.py`, `test_qsnnreinforce.py`, `test_qliflstm.py` and the predator-brain tests pass unchanged. Commit.

## 5. Family D — REINFORCE partial reuse

- [ ] 5.1 `spikingreinforce.py` (`:652-661`, `:770-777`): already `Categorical` → `categorical_logprob_entropy_torch` + `reinforce_policy_loss`. **Byte-exact** (D3 exception). Leave the action-probability floor at `:448` untouched.
- [ ] 5.2 `hybridquantumcortex.py` cortex REINFORCE path (`:2369-2382`): plain softmax → shared scorer + `reinforce_policy_loss`. Byte-exact.
- [ ] 5.3 `qrc.py` (`:502-517` rollout, `:630-645` update): keep `rng.choice` at `:513` verbatim; loop-accumulated `-Σ lp·adv` → `reinforce_policy_loss`. **Not byte-exact** — summation reorder, ~1e-7 (D5).
- [ ] 5.4 `mlpreinforce.py` (`:279-291` rollout, `:422-437` update): keep the temperature-sampled `rng.choice` at `:287` verbatim (note the log-prob is scored on the **non**-temperature `probs` — preserve that). Decompose `(policy_loss + entropy_loss)/n` as `reinforce_policy_loss(...) - β·entropies.mean()`. Not byte-exact, ~1e-7.
- [ ] 5.5 Family D migration test per D7. `test_qrc.py`, `test_mlpreinforce.py`, `test_spikingreinforce.py` pass unchanged. Commit.

## 6. Sweep, spec, and close-out

- [ ] 6.1 Grep-audit with a **known expected answer**, not an open-ended sweep. Baseline at `0ae24375` is 10 clipped-surrogate modules and 23 manual `log(p + ε)` sites across 10 modules. After the migration: `surr1` / `surr2` / `clamp(ratio` SHALL appear only in `_policy.py` and its test, and exactly **two** manual scoring sites SHALL remain — both diagnostic-only, both pre-identified: `qrc.py:431` (f-string) and `spikingreinforce.py:490` (detached `.item()` for a log line). Any third survivor is a miss, not a judgement call.
- [ ] 6.2 Confirm the four non-PG brains (`qqlearning`, `mlpdqn`, `feedforwardga`, `qvarcircuit`) were correctly excluded, not missed — and that `feedforward_ga`'s `no_grad()` sampling site at `:186-190` was left inline for the reason recorded in the D-risks, not overlooked.
- [ ] 6.2b Confirm `mingruppo` / `minlstmppo` still inherit their scoring from `LSTMPPOBrain` unchanged (they are covered transitively and must need no edit). `uv run pytest -k "minimal_rnn"` passes.
- [ ] 6.3 Land the `brain-architecture` ADDED deltas; `openspec validate consolidate-ppo-policy-helpers --strict` passes.
- [ ] 6.4 Full `uv run pytest -m "not nightly"` against the baseline (**4062 passed, 1 skipped, 2 xfailed**): skipped and xfailed unchanged, no previously-passing test failing, and passed up by exactly the number of tests this change adds — record that number here. `uv run pyright` — must stay **0 errors**. `uv run pre-commit run -a` clean.
- [ ] 6.5 Close [#204](https://github.com/SyntheticBrains/nematode/issues/204) with the D0 corrections: seven already migrated (not six); `qef` **is** a candidate and is covered free via `ReservoirHybridBase`; `qrc` is REINFORCE, not a `ReservoirHybridBase` subclass; `env/mlpppo_predator_brain.py` added to scope.
- [ ] 6.6 Archive to `openspec/changes/archive/<YYYY-MM-DD>-consolidate-ppo-policy-helpers/`.
