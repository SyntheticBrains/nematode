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

- [x] 2.1 `_reservoir_hybrid_base.py` **update** (`:678-693`): `probs`/`dist`/`log_prob`/`entropy` → `categorical_evaluate_torch`; `surr1`/`surr2`/`min` → `ppo_clip_policy_loss`. Family A, byte-exact.
- [x] 2.2 `_reservoir_hybrid_base.py` **rollout** (`:454-457`): → `categorical_sample_torch`, passing **`device=self.device` explicitly** — the current code builds the action tensor on `self.device` (`:457`) while the helper defaults to `logits.device`; they coincide today, but relying on that would make byte-exactness depend on an unstated invariant. Confirm the returned `probs` still feeds `self.current_probabilities` and the `buffer.position % 50` diagnostic unchanged, and that the discarded `entropy` is the accepted cost recorded in D8.
- [x] 2.3 Confirm `crh`, `qef`, `qrh` are covered with no per-brain edit (D0.2 — `qef` overrides neither `run()` nor the update). **Verified:** grep for `Categorical` / `softmax` / `log_prob` / `surr1` / `rng.choice` across all three returns nothing but one docstring line in `qrh.py:24`. Same check on `crhqlstm` / `qrhqlstm`: nothing.
- [x] 2.4 `_reservoir_lstm_base.py` **rollout** (`:538-545`): keep `rng.choice` at `:541` **verbatim**; replace `np.log(action_probs[action_idx] + 1e-8)` with `categorical_logprob_entropy_torch(logits, int(action_idx))`. Family B.
- [x] 2.5 `_reservoir_lstm_base.py` **update** (`:686-721`): per-step manual `torch.log`/`-Σ p log p` → `categorical_logprob_entropy_torch`; surrogate → `ppo_clip_policy_loss`, keeping `ratio` for the clip-fraction metric at `:726` (mirror the `lstmppo.py:1202-1216` comment style).
- [x] 2.6 Family A + B migration tests per D7 in `test_reservoir_policy_migration.py` (7 tests). Family A asserts `torch.equal`. **Family B's bar was amended here — see D3.** The first version, written against the flat `atol=1e-7` D3 originally declared, *failed*: the log-prob deviation is the `+1e-8` floor being removed, which is `-log1p(ε/p)` — ~4e-8 at p=0.25 but ~1e-4 at p=1e-4, unbounded as p→0. The test now asserts that **model** (residual ≤2e-6 for all p ≥ 1e-6, measured over 240k samples) and self-checks that it exercised at least one action improbable enough for the floor to bite. Entropy keeps a constant bar at 5e-7 — its `+1e-10` floor is damped by each term's factor of p. A separate test pins the p < 1e-6 boundary where float32 softmax loses the value and neither form is closer to exact.
- [x] 2.7 `test_reservoir_hybrid_base.py`, `test_qef.py`, `test_crh.py`, `test_qrh.py`, `test_crhqlstm.py`, `test_qrhqlstm.py` — **248 passed, unchanged**.
- [x] 2.8 **End-to-end training validation (D3a).** Family A (`crh`, 300 runs, seed 1, 94.7% success): output byte-identical before/after apart from the session UUID. Family B (`crhqlstm`, 500 runs × 5 paired seeds): diverges at run 10–17 as expected, but success rate Δ = −0.08 pp (95% CI [−1.02, +0.86]), reward Δ = +0.26 (CI [−0.23, +0.74]), foods Δ = +0.004 (CI [−0.065, +0.073]) — none significant, all sign columns mixed. **Process note:** the harness swaps files in the shared working tree; two tail-probe measurements were silently taken against the pre-migration code while it ran. Use a git worktree for any future A/B, and verify `git status` is clean before trusting a measurement.
- [x] 2.9 **Float32-tail scope decision (D3).** Instrumented probe over real sessions: 267,638 calls at 200 runs (min p 1.34e-3) and 859,040 calls at 800 runs past full entropy decay (97.3% success, min p 1.20e-4) — **zero** calls below the 1e-6 floor in either. Family C is structurally incapable of reaching it (ε-mixture floors p at ε/n_actions = 0.0075). Out of scope; file a low-priority issue in WS5 rather than deferring work this change needs. Full suite **4078 passed, 1 skipped, 2 xfailed** (4071 + the 7 added here). pyright **0 errors**. ruff clean. Commit.

## 3. Hybrid family — `_hybrid_common` + the three brains (D4: one commit)

- [x] 3.1 `_hybrid_common.py` `perform_ppo_update` (`:445-469`): → `categorical_evaluate_torch` + `ppo_clip_policy_loss`, keeping `log_ratio`/`ratio` for the `approx_kl` term at `:455-458`. Byte-exact in isolation.
- [x] 3.2 `hybridquantum.py` cortex **rollout** (`:1072-1073`): `torch.log(cortex_probs + 1e-8)` → the shared scorer, so both halves of the cortex ratio use one formula (D4). Leave the `np.clip`/renormalise at `:1081-1082` and `rng.choice` at `:1083` **verbatim**.
- [x] 3.3 Same for `hybridclassical.py` (`:777-778`, sampler at `:794` region) and `hybridquantumcortex.py` (`:1926`, `:1934` — note these two adjacent branches spell the same `1e-8` as a literal and as `NORM_EPS`; both go).
- [x] 3.3b Where a Family-C rollout hands its **numpy** mixture to the shared scorer, convert with `torch.as_tensor(action_probs, dtype=torch.float32)`. **Justification corrected in Task 4:** this originally said `from_numpy` would leave a dtype-induced offset in the ratio — measured over 20k samples, false (the two swap order between sample sizes). The residual is dominated by the numpy-vs-torch **softmax backends**, which differ by ~3.8e-7 on `p` before any scoring. The cast stays for dtype discipline, not ratio tightness (D2 amended).
- [x] 3.4 `hybridquantum.py` reflex **update** (`:1339-1360`): ε-mixed `action_probs` → `categorical_logprob_entropy_from_probs`; surrogate → `ppo_clip_policy_loss`, keeping `- effective_entropy_coef * mean_entropy` as a separate term. Leave `_exploration_schedule()` and the mixture construction untouched (D2).
- [x] 3.5 Same for `hybridclassical.py` (`:1020-1042`) and `hybridquantumcortex.py` (`:2222-2244`).
- [x] 3.6 **Measured**, 50k samples, action sampled from the policy (as the real code does), `clip_epsilon = 0.2` for scale. **Cortex:** `|ratio-1|` before mean 5.7e-8 / max 8.3e-5 → after **exactly 0 in 100% of samples**. **Reflex:** before mean 5.5e-8 / max 4.3e-7 → after mean 4.1e-8 / max 4.8e-7, exactly 0 in 61%. Both D4 predictions hold; the reflex residual is the float64/float32 boundary and is explicitly *not* claimed exact. **Correction:** only `hybridquantum` and `hybridclassical` call `perform_ppo_update` — `hybridquantumcortex`'s cortex path is a REINFORCE loop with no ratio, so D4 covers two brains on the cortex path, not three (design D4 amended).
- [x] 3.7 Family C migration test per D7 — reference expression must be the **ε-mixture**, not a plain softmax, so a helper that re-softmaxed would fail.
- [x] 3.8 `test_hybridquantum.py`, `test_hybridclassical.py`, `test_hybridquantumcortex.py` — **157 passed, unchanged**.
- [x] 3.9 **End-to-end validation of the cortex fix (D3a).** Ran in a **git worktree**, not by swapping the shared tree (Task 2's lesson). `hybridclassical` **stage 2** — the default foraging configs are stage 1 and never reach `perform_ppo_update`, so a stage-2 config was built and its cortex-update call verified (40 in 40 runs) first. Both conditions load identical stage-1 weights. 300 runs × 5 paired seeds: success +0.20 pp (CI [−0.03, +0.43]), foods +0.016 (CI [−0.003, +0.035]), reward +0.12 (CI [−0.21, +0.45]) — none significant, but **no seed regressed** on success or foods. Main tree `git status` clean throughout. Full suite **4086 passed, 1 skipped, 2 xfailed** (4078 + the 8 added here). pyright **0 errors**. ruff clean. Commit.

## 4. Remaining direct copies

- [x] 4.1 `qsnnppo.py` **rollout** (`:920-931`): keep the numpy softmax and `rng.choice` at `:927` verbatim; `np.log(...+1e-8)` at `:931` → shared scorer. **update** (`:1110-1149`): manual log/entropy → `categorical_logprob_entropy_torch`; surrogate → `ppo_clip_policy_loss`, keeping `ratio` for clip-frac and `approx_kl` at `:1152-1157`.

- [x] 4.2 `qsnnreinforce.py` (`:1590-1615` update, `:1946` rollout): Family C — ε-mixed, same treatment as 3.4. Despite the brain's name this path is a PPO-clipped surrogate.

- [x] 4.3 `qliflstm.py` (`:1099-1134` update, `:965` rollout): Family B, same treatment as 2.4/2.5; keep the clip-frac at `:1137-1139`.

- [x] 4.4 `env/mlpppo_predator_brain.py`: **rollout** (`:318-322`) → `categorical_sample_torch`; **update** (`:438-449`) → `categorical_evaluate_torch` + `ppo_clip_policy_loss`. Family A, byte-exact. Check the `:339` cumulative-softmax inversion branch is unaffected.

- [x] 4.5 `test_qsnnppo.py`, `test_qsnnreinforce.py`, `test_qliflstm.py` + the three predator-brain suites — **374 passed, unchanged**. 10 migration tests added in `test_qsnn_qlif_policy_migration.py`. Full suite **4096 passed, 1 skipped, 2 xfailed** (4086 + 10). pyright **0 errors**. ruff clean. Commit.

  **Note on `qsnnppo` vs `qliflstm` (both Family B, treated differently on purpose):** `qliflstm` has a torch `logits` tensor at rollout, so it scores via `categorical_logprob_entropy_torch` — the same expression the update uses. `qsnnppo`'s rollout forward pass is NumPy, so no such tensor exists; re-deriving one would score a distribution the sampler never saw. It therefore scores its own probability vector via `categorical_logprob_entropy_from_probs`.

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
