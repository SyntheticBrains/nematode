## Context

T5 ([`add-continuous-2d-and-action-heads`](../archive/2026-06-07-add-continuous-2d-and-action-heads/design.md), D6) created [`brain/arch/_policy.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_policy.py) as the single reviewed home for the parity-sensitive policy code, and migrated seven brains onto it. The module's own docstring states the intent: *"factors the action sampling, log-probability, entropy, and PPO surrogate terms out of the per-brain copy-paste."* Thirteen brains never made the trip.

A full re-inventory at commit `0ae24375` (the brief's line numbers were taken at `234e2140` and have shifted) found **10 live clipped-surrogate copies** and **13 manual `log(probs + 1e-8)` / `-Σ p·log(p + 1e-10)` pairs**. The `surr1`/`surr2` strings in `lstmppo`, `cfc_ppo`, and `spiking_ppo` are comments describing their completed migration, not live code — confirmed.

The inventory changed the shape of the work in three ways that were not visible from issue #204: a fourth structural family the existing helpers cannot express (D2), a latent ratio-bias defect in the hybrid brains (D4), and two miscategorised brains in the issue's own scope list (D0).

## Goals / Non-Goals

**Goals:**

- `_policy.py` becomes the single source of truth for discrete policy scoring and the PPO clipped surrogate across **every** PPO-family brain, so an L4 plasticity change in Phase 7 has one place to land.
- Zero intended behavioural change. Sampled-action trajectories stay byte-identical everywhere; numerical divergence is bounded by a per-brain tolerance declared *before* the migration.
- The ε-mixed policy form gets first-class vocabulary in the shared module rather than being left inline as "the case the helper can't do."
- Correct the hybrid rollout/update ratio asymmetry (D4) as part of the same pass, since the inventory is what exposed it.

**Non-Goals:**

- No reward, env, config, hyperparameter, or `Brain` Protocol change. No re-tuning, no new architectures.
- Not a rewrite of the brains' *update loops* — buffer handling, BPTT chunking, epoch/minibatch structure, gradient clipping, and diagnostics all stay exactly as they are. Only the policy-scoring and surrogate expressions move.
- Not a fix for the `qqlearning` TODO backlog or the `DynamicForagingEnvironment` decomposition (WS5 issues; separate work).
- Continuous (tanh-Gaussian) helpers are untouched — none of the 13 brains has a continuous head.

## Decisions

### D0 — Issue #204's scope list has three errors; correct it on close

Verified against the tree at `0ae24375`:

1. **"six already migrated" → seven.** The issue predates `transformer_ppo`, which is migrated.
2. **`qef` is listed as "not a candidate" (variational, no categorical PG) — it is a candidate.** [`QEFBrain`](../../../packages/quantum-nematode/quantumnematode/brain/arch/qef.py) overrides only `_transform_features`, `_compute_critic_value`, `_get_all_trainable_params`, and `copy()`; it inherits `run()` and the PPO update from `ReservoirHybridBase` wholesale, so it reaches both the `Categorical` sampler (`_reservoir_hybrid_base.py:454`) and the clip (`:688`). The variational part is the *feature extractor*; the policy head is an ordinary categorical PG. Migrating the base covers `qef` at no extra cost.
3. **`qrc` is listed under `_reservoir_hybrid_base`'s coverage — it is not a subclass.** The base's subclasses are exactly `crh`, `qef`, `qrh`. [`QRCBrain`](../../../packages/quantum-nematode/quantumnematode/brain/arch/qrc.py) is a standalone `ClassicalBrain` with a **REINFORCE** loss and no PPO clip at all (`qrc.py:630-645`); it belongs to the partial-reuse group (D5), not the reservoir group.

`env/mlpppo_predator_brain.py` is absent from the issue's list and is added, per the brief.

### D1 — Four structural families, not two

The brains do not divide into "torch path" and "numpy path" as D6 assumed. The inventory yields four:

| Family | Distribution scored | Members | Bar |
|---|---|---|---|
| **A** — torch `Categorical` both sides | `softmax(logits)` | `_reservoir_hybrid_base` (crh, qef, qrh), `_hybrid_common` (cortex PPO ×3), `env/mlpppo_predator_brain` | **byte-exact** |
| **B** — numpy sampler, manual torch log | `softmax(logits)` | `_reservoir_lstm_base` (crhqlstm, qrhqlstm), `qsnnppo`, `qliflstm` | ~1e-7 |
| **C** — ε-greedy mixture | `(1-ε)·softmax(logits/T) + ε·uniform` | `hybridquantum`, `hybridclassical`, `hybridquantumcortex`, `qsnnreinforce` | ~1e-7 |
| **D** — REINFORCE, partial reuse | `softmax(logits)` | `qrc`, `mlpreinforce`, `spikingreinforce`, `hybridquantumcortex` (cortex path) | ~1e-7, except `spikingreinforce` |

Family A is a pure lift: the code already reads `probs = softmax(logits); dist = Categorical(probs); dist.log_prob(actions); dist.entropy().mean()`, which is `categorical_evaluate_torch` verbatim.

Family B is the LSTM-PPO/CfC-PPO case D6 already settled: keep the numpy sampler byte-for-byte, route log-prob/entropy/surrogate through torch.

Family C is new (D2). Family D needs a REINFORCE term (D5).

### D2 — Family C needs a probs-based scorer; add `categorical_logprob_entropy_from_probs`

The four Family-C brains build their action distribution as an explicit **ε-greedy mixture**:

```python
softmax_probs = torch.softmax(logits / temperature, dim=-1)
uniform = torch.ones_like(softmax_probs) / self.num_actions
action_probs = (1 - epsilon) * softmax_probs + epsilon * uniform
```

`categorical_logprob_entropy_torch` takes **logits** and applies softmax internally, so it cannot express this — a mixture of a tempered softmax and a uniform is not the softmax of any logits vector the brain has on hand. Passing `torch.log(action_probs)` back in as pseudo-logits would work arithmetically but is an obfuscation (a log-then-exp round trip, and `log(0)` is reachable at `ε = 0` with a saturated softmax).

**Decision:** add `categorical_logprob_entropy_from_probs(probs, action, *, device=None) -> (log_prob, entropy, probs)` — the probability-vector twin of the existing logits-based scorer, sharing its return contract, implemented as `dist = Categorical(probs); dist.log_prob(a), dist.entropy()`. The existing logits helper is refactored to delegate to it so there is one `Categorical` construction site, not two.

**Why this is the right level:** the ε-mixture is a *policy* decision (how much forced exploration), and scoring whatever distribution the policy produced is precisely the shared module's job. The alternative — leaving Family C inline because "the helper doesn't fit" — would leave the four largest hand-written copies in place and defeat the change's purpose.

**Numerical consequence:** `log(p[a] + 1e-8)` becomes `Categorical(p).log_prob(a)`, i.e. `log(p[a]) - log(Σp)`. The mixture sums to 1 up to float32 round-off, so the normalisation term is ~1e-8 and the `+1e-8` floor is removed; net deviation ~1e-7. Entropy moves from `-Σ p·log(p + 1e-10)` to torch's normalised equivalent, likewise ~1e-7. Both are strict stability improvements — the manual forms bias low-probability actions upward by exactly the epsilon they add.

*Alternative rejected:* a `mixture_epsilon` parameter on the existing logits helper, so the helper builds the mixture itself. Rejected because the four brains' mixtures differ in their temperature schedules (`_exploration_schedule()` is per-brain) and folding a scheduling concern into a scoring helper couples them; the helper should score a distribution, not construct one.

### D3 — Per-brain tolerance, declared here, before any code moves

Family A is **byte-exact** and is held to `torch.equal`. Families B, C, and D are held to `rtol=0, atol=1e-7`.

That `1e-7` is not a new number: the existing `brain-architecture` requirement *"Migration Regression Bar — Other 17 Architectures Numerical Equivalence"* already binds `{QRC, QRH, QEF, CRH, QSNN_REINFORCE, QSNN_PPO, HYBRID_QUANTUM, HYBRID_CLASSICAL, HYBRID_QUANTUM_CORTEX, QLIF_LSTM, QRH_QLSTM, CRH_QLSTM, MLP_REINFORCE, SPIKING_REINFORCE, …}` — the same set — to `torch.allclose(rtol=0, atol=1e-7)`. This change adopts the standing bar rather than inventing a second one, and D6's measured ~1e-7 float32 round-off for the identical LSTM/CfC transformation is the empirical basis.

One exception in the other direction: **`spikingreinforce` is byte-exact.** It already uses `torch.distributions.Categorical` at `spikingreinforce.py:655` and `:773`; its migration is a Family-A lift despite sitting in the REINFORCE group.

**Sampled-action trajectories are byte-identical for every brain**, including B/C/D — every `rng.choice` sampler is kept verbatim, untouched. The tolerance applies only to the scalar log-prob/entropy/loss values, never to which action was taken. This is the distinction D6 drew between an acceptable round-off tolerance and the forbidden "different-seed run."

### D4 — Migrate rollout and update together, per brain; this fixes a live ratio bias

PPO's importance ratio is `exp(new_log_prob − old_log_prob)`, where `old` is stored at rollout time and `new` recomputed at update time. **Both halves must use the same formula** or the ratio is biased even when the policy has not changed.

Today the three hybrid brains violate this on the cortex PPO path: the rollout stores `log(softmax(logits) + 1e-8)` ([`hybridquantum.py:1073`](../../../packages/quantum-nematode/quantumnematode/brain/arch/hybridquantum.py), [`hybridclassical.py:778`](../../../packages/quantum-nematode/quantumnematode/brain/arch/hybridclassical.py)) while `_hybrid_common.perform_ppo_update` re-scores with `Categorical.log_prob` ([`_hybrid_common.py:448`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_hybrid_common.py)). At `ratio == 1` the two differ by the `+1e-8` floor, so `ratio ≈ 1 ± 1e-7` systematically rather than exactly 1.

**Decision:** each brain's rollout and update scoring move in the **same commit**, and `_hybrid_common`'s migration is paired with the corresponding rollout migration in the three hybrids. The migration therefore *corrects* the asymmetry rather than preserving it — the two halves become the same `Categorical` call.

**This is a deliberate, declared deviation from "no behavioural change."** The bias it removes is ~1e-7 on the ratio, far below the `clip_epsilon = 0.2` clip band and well inside the D3 tolerance, so no training outcome can turn on it. It is recorded here rather than fixed silently, and it is the reason `_hybrid_common` alone cannot be migrated in isolation despite being a shared module.

*Alternative rejected:* preserve the asymmetry exactly by teaching the helper an epsilon-floor mode. That would encode a defect in the shared module to protect a difference no result depends on.

### D5 — Family D gets `reinforce_policy_loss`; two of three are not byte-exact

`qrc`, `mlpreinforce`, `spikingreinforce`, and the `hybridquantumcortex` cortex path all compute `-Σ log_prob(a_t)·A_t / n`, but in two different shapes. `spikingreinforce` and the cortex path already write the vectorised `-(log_probs * advantages).mean()`; `qrc` (`:633-641`) and `mlpreinforce` (`:425-437`) accumulate in a Python `for` loop and divide at the end.

**Decision:** add `reinforce_policy_loss(log_probs, advantages) -> Tensor` returning `-(log_probs * advantages).mean()`, and migrate all four onto it. `spikingreinforce` and the cortex path are byte-exact (identical expression). `qrc` and `mlpreinforce` are **not** — replacing a left-fold accumulation with torch's blocked `sum` reassociates the additions, which is a ~1e-7 float32 reorder, covered by D3.

`mlpreinforce` folds its entropy bonus into the same division (`total = (policy_loss + entropy_loss) / n`); this decomposes cleanly as `reinforce_policy_loss(...) − β · entropies.mean()`, so the helper composes without a special case.

*Note:* the entropy bonus stays per-brain. `qrc` and `mlpreinforce` use `+1e-8` inside the entropy log while every PPO brain uses `+1e-10`; both become torch's normalised entropy, so the inconsistency dissolves rather than needing a parameter.

### D6 — The `1e-8`/`1e-10` split is unexplained and is removed, not parameterised

All 13 copies pair `log(probs[a] + 1e-8)` for the chosen-action log-prob with `-Σ p·log(p + 1e-10)` for the entropy — two different floors in adjacent lines, with no comment anywhere justifying the difference. It reads as copy-paste drift from an original that had one value.

**Decision:** do not carry either constant into the shared module. `torch.distributions.Categorical` is numerically stable at small probabilities without a floor (it works in log-space internally), which is why D6 recorded the migrated brains' manual `log(softmax)` as *less* stable than the torch path. Removing the floors is part of the ~1e-7 deviation already budgeted in D3, and it removes the question of which constant is correct.

### D7 — Verification is same-run inline-reference, not stored goldens

The verification template is [`test_mlpppo_policy_migration.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_mlpppo_policy_migration.py), whose docstring records why: hard-coded absolute float constants *"drift at ~1e-8 across BLAS / torch builds (so a golden snapshot passes locally but fails CI) without indicating any change in computation."*

**Decision:** one migration test per family, each computing the pre-migration inline expression and the migrated brain path **in the same process, on the same logits, under the same pinned RNG state**, and asserting `torch.equal` (Family A, `spikingreinforce`) or `torch.allclose(rtol=0, atol=1e-7)` (Families B/C/D). No stored fixtures, no absolute float literals. Family C's test asserts against the ε-mixed reference expression specifically, so it would catch a helper that silently re-softmaxed.

The second, environment-robust half of the bar is that **every touched brain's existing suite passes unchanged** — `test_qef.py`, `test_crh.py`, `test_qrh.py`, `test_crhqlstm.py`, `test_qrhqlstm.py`, `test_qsnnppo.py`, `test_qsnnreinforce.py`, `test_qliflstm.py`, `test_hybridquantum.py`, `test_hybridclassical.py`, `test_hybridquantumcortex.py`, `test_qrc.py`, `test_mlpreinforce.py`, `test_spikingreinforce.py`, `test_reservoir_hybrid_base.py`.

### D8 — Order of work: helpers, then bases, then copies

1. `_policy.py` helpers + their unit tests, landing green before any brain is touched.
2. The three shared bases (Family A's `_reservoir_hybrid_base`, Family B's `_reservoir_lstm_base`) — highest leverage, 5 brains.
3. `_hybrid_common` **paired with** the three hybrid brains' rollout+reflex copies (D4 forbids splitting these).
4. The remaining direct copies: `qsnnppo`, `qsnnreinforce`, `qliflstm`, `env/mlpppo_predator_brain`.
5. Family D partial reuse: `spikingreinforce`, `qrc`, `mlpreinforce`, `hybridquantumcortex` cortex path.

Each numbered step is a commit that leaves the tree green, so a regression bisects to one family.

## Risks / Trade-offs

- **[A silent behavioural regression hides inside a "no-op" refactor]** → the D7 same-run inline-reference tests compare against the *exact* pre-migration expression, so any change of formula fails loudly; the per-brain existing suites are the second net; the verification baseline is a fully green `pytest` (4062 passed) + `pyright` (0 errors) on the pinned versions, so "no new failure, no new pyright error" is an exact bar, not a count comparison.
- **[Family C's helper is used with a probs vector that does not sum to 1]** → `Categorical` normalises internally, so a drifted vector is silently renormalised rather than erroring. `hybridquantum` already clips-and-renormalises before sampling (`:1081-1082`); the helper's docstring states the normalisation explicitly so a future caller is not surprised.
- **[D4's ratio-bias correction perturbs an existing tuned hybrid result]** → the correction is ~1e-7 against a `clip_epsilon` of 0.2; the hybrid suites pin behaviour; and the bias it removes is in the direction of *more* correct PPO, not a re-tune. Declared up front rather than discovered later.
- **[The 1e-7 tolerance masks a real error of the same size]** → the tolerance applies only to scalars, never to sampled actions (which stay byte-identical), so a policy-changing bug cannot hide inside it; and the same `rtol=0, atol=1e-7` bar already governs these exact architectures under the standing T2 requirement.
- **[13 modules in one PR is hard to review]** → D8's commit-per-family split makes the diff readable family by family, and Family A/B are mechanical enough to skim once the helper contract is agreed.
- **\[`qqlearning`, `mlpdqn`, `feedforward_ga`, `qvarcircuit` remain unmigrated\]** → they are not policy-gradient brains (Q-learning, DQN, evolutionary, and a variational circuit with its own update); they have no categorical PG surrogate to share. Out of scope by construction, not by omission.

## Migration Plan

1. Land the two `_policy.py` helpers with unit coverage; no brain touched (tree green).
2. Migrate the shared bases (`_reservoir_hybrid_base`, `_reservoir_lstm_base`), covering crh/qef/qrh/crhqlstm/qrhqlstm; run the Family A + B migration tests and those five brains' suites.
3. Migrate `_hybrid_common` together with the three hybrid brains' rollout and reflex copies (D4); run the Family C migration test and the three hybrid suites.
4. Migrate `qsnnppo`, `qsnnreinforce`, `qliflstm`, `env/mlpppo_predator_brain`.
5. Family D partial reuse (`spikingreinforce`, `qrc`, `mlpreinforce`, `hybridquantumcortex` cortex path).
6. Land the spec delta; full `uv run pytest -m "not nightly"` + `uv run pre-commit run -a` against the green baseline; close #204 with the D0 corrections.

## Open Questions

- Whether `_policy.py`'s module docstring should grow a short per-family table (it currently narrates the two T5 families in prose). Leaning yes — with four families and 20 brains, prose stops scaling. Resolve during implementation and record in tasks.
