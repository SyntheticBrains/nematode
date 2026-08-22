## Context

T5 ([`add-continuous-2d-and-action-heads`](../archive/2026-06-07-add-continuous-2d-and-action-heads/design.md), D6) created [`brain/arch/_policy.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_policy.py) as the single reviewed home for the parity-sensitive policy code, and migrated seven brains onto it. The module's own docstring states the intent: *"factors the action sampling, log-probability, entropy, and PPO surrogate terms out of the per-brain copy-paste."* Fourteen brains never made the trip.

The full 27-brain accounting, derived from the live `_REGISTRY` rather than by hand:

| Status | Count | Brains |
|---|---|---|
| Migrated directly | 7 | `mlpppo`, `connectomeppo`, `transformerppo`, `equivariantquantum`, `lstmppo`, `cfcppo`, `spikingppo` |
| Migrated transitively | 2 | `mingruppo`, `minlstmppo` — both subclass `LSTMPPOBrain` and inherit its migrated scoring, so they need no work here |
| **In scope for this change** | **14** | `crh`, `qef`, `qrh`, `crhqlstm`, `qrhqlstm`, `qsnnppo`, `qliflstm`, `hybridquantum`, `hybridclassical`, `hybridquantumcortex`, `qsnnreinforce`, `qrc`, `mlpreinforce`, `spikingreinforce` |
| Excluded (not policy-gradient) | 4 | `qqlearning`, `mlpdqn`, `qvarcircuit`, `feedforwardga` |

A full re-inventory at commit `0ae24375` (the brief's line numbers were taken at `234e2140` and have shifted) found **10 live clipped-surrogate copies** and **23 manual `log(probs + ε)` scoring sites across 10 modules** — 21 live in a loss or ratio path, plus two diagnostic-only survivors that will legitimately remain after the migration (`qrc.py:431`, an f-string; `spikingreinforce.py:490`, a detached `.item()` for a log line). The `surr1`/`surr2` strings in `lstmppo`, `cfc_ppo`, and `spiking_ppo` are comments describing their completed migration, not live code — confirmed.

Fourteen modules are touched: `_policy.py`, the three shared bases, nine per-brain copies, and `env/mlpppo_predator_brain.py` (a PPO predator controller, not a registered brain). The five brains behind the shared bases need no per-brain edit.

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
- Continuous (tanh-Gaussian) helpers are untouched — none of the 14 brains has a continuous head.

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

**Where the mixture is built differs by side.** All four Family-C brains construct the ε-mixture in **numpy (float64)** at rollout ([`qsnnreinforce.py:1907-1908`](../../../packages/quantum-nematode/quantumnematode/brain/arch/qsnnreinforce.py)) and in **torch (float32)** at update. The rollout therefore has to hand the shared scorer a converted tensor. Implementations MUST convert as `torch.as_tensor(action_probs, dtype=torch.float32)` — matching the update side's dtype — rather than letting `from_numpy` carry float64 through, or the two sides of the ratio would keep a dtype-induced offset even once the formula is shared (see D4).

*Alternative rejected:* a `mixture_epsilon` parameter on the existing logits helper, so the helper builds the mixture itself. Rejected because the four brains' mixtures differ in their temperature schedules (`_exploration_schedule()` is per-brain) and folding a scheduling concern into a scoring helper couples them; the helper should score a distribution, not construct one.

### D3 — Per-brain tolerance, declared here, before any code moves

Family A is **byte-exact** and is held to `torch.equal`.

**Amended during Task 2 (measured, not assumed).** This decision originally held Families B, C, and D to a flat `rtol=0, atol=1e-7`, borrowed from the standing *"Migration Regression Bar — Other 17 Architectures Numerical Equivalence"* requirement and from D6's report of "~1e-7 at all policy-confidence levels." The first Family-B test written against that bar **failed**, and the measurement shows the flat constant was wrong:

| Quantity | Declared bar | Basis |
|---|---|---|
| Sampled action | **byte-identical** | sampler untouched; no tolerance at all |
| Entropy | `atol = 5e-7` | the `+1e-10` floor is damped by each term's factor of `p`, so the residual is float32 round-off of the sum (measured worst 2.4e-7 over 2000 random policies) |
| Log-probability | `-log1p(ε / p)`, verified to `2e-6` | the deviation **is** the `+1e-8` floor being removed; it is a *model*, not a constant |
| Log-prob, `p < 1e-6` | no bar | float32 `softmax` has already lost the value; see below |

The log-prob deviation is `log(p) - log(p + ε) = -log1p(ε/p) ≈ ε/p`. That is ~4e-8 at `p = 0.25`, ~1e-6 at `p = 0.01`, and ~1e-4 at `p = 1e-4` — so a flat `1e-7` holds only for `p ≳ 0.1`. Measured over 240k samples, the residual after subtracting the model is ≤ 1.1e-6 for every `p ≥ 1e-6`.

**Why this is still a correction rather than a regression.** The excess belongs entirely to the expression being *removed*: the floor biased low-probability actions upward by exactly the epsilon it added, and it did so hardest precisely where PPO is most sensitive — an improbable action that was nonetheless taken. Both halves of the ratio migrate together (D4), so within a run the two sides stay consistent.

**Boundary, recorded rather than hidden.** Below `p ≈ 1e-6` the float32 `softmax` has already lost the probability to its own round-off, so *both* the old and the new expression sit far from the float64-exact value and neither is reliably closer (measured: at `p = 8.2e-8`, old is 0.115 from exact and new is 0.375, in the same direction). Both read the same float32 `probs`, so this is a property of the brains' float32 pipeline that the migration neither changes nor claims to fix. `test_below_the_float32_softmax_floor_the_model_stops_applying` pins the boundary so a future reader does not mistake it for migration drift.

**The boundary is never reached in practice — measured, not assumed.** Instrumenting the shared scorer and running real training sessions:

| Session | Scoring calls | Min `p` scored | Calls below `1e-6` |
|---|---|---|---|
| `crhqlstm`, 200 runs (90% success) | 267,638 | 1.34e-3 | **0** |
| `crhqlstm`, 800 runs (97.3% success, past full entropy decay) | 859,040 | 1.20e-4 | **0** |

The converged 800-run policy — the most confident state the config produces — still stays 120× above the floor, where the floor-removal deviation is 8.3e-5 log units, i.e. a ratio factor of 1.00008 against a `clip_epsilon` of 0.2. Two structural reasons, not luck: the scored action is the *sampled* action (so a `p = 1e-6` action is scored about once per million steps by construction), and the entropy bonus exists precisely to stop the policy saturating.

**Family C cannot reach it at all.** The ε-greedy mixture floors every action at `ε / n_actions`, and `exploration_schedule` ([`_hybrid_common.py:572`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_hybrid_common.py)) decays ε to 30% of its initial value, never to zero: `current_epsilon = exploration_epsilon * (1.0 - progress * 0.7)`. At the configured `exploration_epsilon: 0.1` with 4 actions that is a hard floor of `p ≥ 0.0075` — 7,500× above the float32 floor, worst deviation 1.3e-6 log units — for `hybridquantum`, `hybridclassical`, `hybridquantumcortex`, and `qsnnreinforce`, independent of task.

Addressing the float32 tail is therefore **out of scope and recorded as a low-priority issue**, not deferred work this change depends on. The condition that would make it matter — a config annealing `entropy_coef` to zero with no ε floor — does not exist in `configs/`.

This **does not** conflict with the standing T2 requirement, which binds `torch.allclose(rtol=0, atol=1e-7)` on *parameter tensors after 5-step smoke training* — a different quantity from a single-step log-probability, and one where a `1e-4` log-prob shift on a rarely-taken action does not propagate at that magnitude.

One exception in the other direction: **`spikingreinforce` is byte-exact.** It already uses `torch.distributions.Categorical` at `spikingreinforce.py:655` and `:773`; its migration is a Family-A lift despite sitting in the REINFORCE group.

**Sampled-action trajectories are byte-identical for every brain**, including B/C/D — every `rng.choice` sampler is kept verbatim, untouched. The tolerance applies only to the scalar log-prob/entropy/loss values, never to which action was taken. This is the distinction D6 drew between an acceptable round-off tolerance and the forbidden "different-seed run."

### D3a — End-to-end training validation (added during Task 2)

The unit-level bars in D3 bound the *scalar* deviation. They do not answer the question that actually matters — whether a migrated brain still trains the same. So each migrated family is also checked by running real training sessions before and after, at matched seeds.

Method: revert **only** the migrated module(s) to their pre-migration state, run `scripts/run_simulation.py` at a fixed seed, restore, re-run. Both conditions share identical `_policy.py`, so the migration is the only variable.

**Family A — byte-identical end to end.** `crh`, `foraging/crh_small_oracle`, 300 runs, seed 1 (94.7% success, so a genuinely trained policy): the *only* difference in the entire output is the session UUID. Every run line, reward, step count, and summary metric matches exactly. This confirms the byte-exactness claim on a real training run rather than only in a unit test — and validates the harness, since a method capable of manufacturing a difference would have shown one here.

**Family B — trajectories diverge, outcomes do not.** `crhqlstm`, `foraging/crhqlstm_small_classical_oracle`, 500 runs × 5 paired seeds. Divergence begins at run 10–17 of 500 — expected, since the log-prob feeds the gradient, so the two conditions become independent training runs rather than the same run with noise. The correct claim is therefore *distributional equivalence*, not identity:

| Metric | Before | After | Paired Δ | 95% CI | Signs |
|---|---|---|---|---|---|
| success rate | 95.36 ±0.57 | 95.28 ±0.52 | **−0.08 pp** | [−1.02, +0.86] | 3−/2+ |
| avg reward | 34.23 ±0.12 | 34.49 ±0.37 | **+0.26** | [−0.23, +0.74] | 1−/4+ |
| avg foods | 9.622 ±0.042 | 9.626 ±0.049 | **+0.004** | [−0.065, +0.073] | 2−/1=/2+ |

No metric is significant at p < 0.05 (paired *t*, df = 4), and every sign column is mixed. For scale, the Phase 6a headline separations this platform is built to resolve are 13–75 pp; the CI here bounds any migration effect on success rate at about ±1 pp.

**Honest limits.** n = 5 seeds on one config bounds the effect, it does not prove absence — a sub-1 pp shift would not be detected. The avg-reward column leans positive in 4 of 5 seeds, which is worth re-checking if the same lean appears in later families rather than dismissing now. Families C and D get the same treatment as they land.

### D4 — Migrate rollout and update together, per brain; this fixes a live ratio bias

PPO's importance ratio is `exp(new_log_prob − old_log_prob)`, where `old` is stored at rollout time and `new` recomputed at update time. **Both halves must use the same formula** or the ratio is biased even when the policy has not changed.

Two distinct pairs violate this today, and they are **not** fixed to the same degree. The distinction matters because it determines what the implementation may claim:

**Cortex path — a pure formula mismatch, fully corrected.** The rollout stores `log(softmax(logits) + 1e-8)` ([`hybridquantum.py:1073`](../../../packages/quantum-nematode/quantumnematode/brain/arch/hybridquantum.py), [`hybridclassical.py:778`](../../../packages/quantum-nematode/quantumnematode/brain/arch/hybridclassical.py)) while `_hybrid_common.perform_ppo_update` re-scores with `Categorical.log_prob` ([`_hybrid_common.py:448`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_hybrid_common.py)). Both sides are torch on the same dtype, so the `+1e-8` floor is the *only* difference: `ratio ≈ 1 ± 1e-7` systematically rather than exactly 1. Routing both through the same scorer removes it entirely. This is the real defect, and it is the reason `_hybrid_common` cannot be migrated in isolation despite being a shared module.

**Reflex path — formula *and* dtype, only the formula is closed.** Here the rollout is numpy float64 (`hybridquantum.py:1089`) and the update is torch float32 (`:1344`). The migration makes both use `Categorical`, which removes the formula half; the float64/float32 gap is pre-existing, is not what this change is about, and survives (mitigated by the D2 cast, which at least stops the *new* code from widening it). So the reflex ratio improves but does not become exact.

**Decision:** each brain's rollout and update scoring move in the **same commit**, and `_hybrid_common`'s migration is paired with the corresponding rollout migration in the three hybrids. Task 3.6 measures the two paths **separately** — the cortex pair is expected to reach exactly 1, the reflex pair to land at ~1e-7 — so a single averaged number cannot hide a half-done fix.

**This is a deliberate, declared deviation from "no behavioural change."** The bias removed is ~1e-7 on the ratio, far below the `clip_epsilon = 0.2` clip band and well inside the D3 tolerance, so no training outcome can turn on it. It is recorded here rather than fixed silently.

*Alternative rejected:* preserve the asymmetry exactly by teaching the helper an epsilon-floor mode. That would encode a defect in the shared module to protect a difference no result depends on.

### D5 — Family D gets `reinforce_policy_loss`; two of three are not byte-exact

`qrc`, `mlpreinforce`, `spikingreinforce`, and the `hybridquantumcortex` cortex path all compute `-Σ log_prob(a_t)·A_t / n`, but in two different shapes. `spikingreinforce` and the cortex path already write the vectorised `-(log_probs * advantages).mean()`; `qrc` (`:633-641`) and `mlpreinforce` (`:425-437`) accumulate in a Python `for` loop and divide at the end.

**Decision:** add `reinforce_policy_loss(log_probs, advantages) -> Tensor` returning `-(log_probs * advantages).mean()`, and migrate all four onto it. `spikingreinforce` and the cortex path are byte-exact (identical expression). `qrc` and `mlpreinforce` are **not** — replacing a left-fold accumulation with torch's blocked `sum` reassociates the additions, which is a ~1e-7 float32 reorder, covered by D3.

`mlpreinforce` folds its entropy bonus into the same division (`total = (policy_loss + entropy_loss) / n`); this decomposes cleanly as `reinforce_policy_loss(...) − β · entropies.mean()`, so the helper composes without a special case.

*Note:* the entropy bonus stays per-brain. `qrc` and `mlpreinforce` use `+1e-8` inside the entropy log while every PPO brain uses `+1e-10`; both become torch's normalised entropy, so the inconsistency dissolves rather than needing a parameter.

### D6 — The `1e-8`/`1e-10` split is unexplained and is removed, not parameterised

Every copy pairs `log(probs[a] + 1e-8)` for the chosen-action log-prob with `-Σ p·log(p + 1e-10)` for the entropy — two different floors in adjacent lines, with no comment anywhere justifying the difference. It reads as copy-paste drift from an original that had one value. The sharpest evidence is in `hybridquantumcortex`, which spells the *same* constant two ways in adjacent branches of one function: a literal `1e-8` at [`:1926`](../../../packages/quantum-nematode/quantumnematode/brain/arch/hybridquantumcortex.py) and the named `NORM_EPS` (defined as `1e-8` at `:172`) at `:1934`. Whatever the original intent, it is no longer legible from the code.

**Decision:** do not carry either constant into the shared module. `torch.distributions.Categorical` is numerically stable at small probabilities without a floor (it works in log-space internally), which is why D6 recorded the migrated brains' manual `log(softmax)` as *less* stable than the torch path. Removing the floors is part of the ~1e-7 deviation already budgeted in D3, and it removes the question of which constant is correct.

### D7 — Verification is same-run inline-reference, not stored goldens

The verification template is [`test_mlpppo_policy_migration.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_mlpppo_policy_migration.py), whose docstring records why: hard-coded absolute float constants *"drift at ~1e-8 across BLAS / torch builds (so a golden snapshot passes locally but fails CI) without indicating any change in computation."*

**Decision:** one migration test per family, each computing the pre-migration inline expression and the migrated brain path **in the same process, on the same logits, under the same pinned RNG state**, and asserting `torch.equal` (Family A, `spikingreinforce`) or `torch.allclose(rtol=0, atol=1e-7)` (Families B/C/D). No stored fixtures, no absolute float literals. Family C's test asserts against the ε-mixed reference expression specifically, so it would catch a helper that silently re-softmaxed.

The second, environment-robust half of the bar is that **every touched brain's existing suite passes unchanged** — `test_qef.py`, `test_crh.py`, `test_qrh.py`, `test_crhqlstm.py`, `test_qrhqlstm.py`, `test_qsnnppo.py`, `test_qsnnreinforce.py`, `test_qliflstm.py`, `test_hybridquantum.py`, `test_hybridclassical.py`, `test_hybridquantumcortex.py`, `test_qrc.py`, `test_mlpreinforce.py`, `test_spikingreinforce.py`, `test_reservoir_hybrid_base.py`.

### D8 — Order of work: helpers, then bases, then copies

1. `_policy.py` helpers + their unit tests, landing green before any brain is touched.
2. The three shared bases (Family A's `_reservoir_hybrid_base`, Family B's `_reservoir_lstm_base`) — highest leverage, 5 brains.

**Accepted cost at the two Family-A rollout sites.** `_reservoir_hybrid_base.py:454-457` and `env/mlpppo_predator_brain.py:318-322` compute only `probs`/`dist`/`action`/`log_prob` at rollout; `categorical_sample_torch` additionally returns `dist.entropy()`, which both discard. This is byte-exactness-neutral (entropy consumes no RNG and the value is dropped) but is a small extra computation per step in a hot loop. Accepted deliberately: one sampling contract across every brain is worth more than the saved call, and a separate no-entropy variant would fragment the very surface this change exists to unify.
3\. `_hybrid_common` **paired with** the three hybrid brains' rollout+reflex copies (D4 forbids splitting these).
4\. The remaining direct copies: `qsnnppo`, `qsnnreinforce`, `qliflstm`, `env/mlpppo_predator_brain`.
5\. Family D partial reuse: `spikingreinforce`, `qrc`, `mlpreinforce`, `hybridquantumcortex` cortex path.

Each numbered step is a commit that leaves the tree green, so a regression bisects to one family.

## Risks / Trade-offs

- **[A silent behavioural regression hides inside a "no-op" refactor]** → the D7 same-run inline-reference tests compare against the *exact* pre-migration expression, so any change of formula fails loudly; the per-brain existing suites are the second net; the verification baseline is a fully green `pytest` (4062 passed) + `pyright` (0 errors) on the pinned versions, so "no new failure, no new pyright error" is an exact bar, not a count comparison.
- **[Family C's helper is used with a probs vector that does not sum to 1]** → `Categorical` normalises internally, so a drifted vector is silently renormalised rather than erroring. `hybridquantum` already clips-and-renormalises before sampling (`:1081-1082`); the helper's docstring states the normalisation explicitly so a future caller is not surprised.
- **[D4's ratio-bias correction perturbs an existing tuned hybrid result]** → the correction is ~1e-7 against a `clip_epsilon` of 0.2; the hybrid suites pin behaviour; and the bias it removes is in the direction of *more* correct PPO, not a re-tune. Declared up front rather than discovered later.
- **[The 1e-7 tolerance masks a real error of the same size]** → the tolerance applies only to scalars, never to sampled actions (which stay byte-identical), so a policy-changing bug cannot hide inside it; and the same `rtol=0, atol=1e-7` bar already governs these exact architectures under the standing T2 requirement.
- **[14 modules in one PR is hard to review]** → D8's commit-per-family split makes the diff readable family by family, and Family A/B are mechanical enough to skim once the helper contract is agreed.
- **\[`qqlearning`, `mlpdqn`, `feedforward_ga`, `qvarcircuit` remain unmigrated\]** → they are not policy-gradient brains (Q-learning, DQN, evolutionary, and a variational circuit with its own update), so none has a categorical PG surrogate or a log-prob/entropy term to share. Out of scope by construction, not by omission. **One qualification:** `feedforward_ga` *does* have a shareable sampling site — `Categorical` under `no_grad()` at [`feedforward_ga.py:186-190`](../../../packages/quantum-nematode/quantumnematode/brain/arch/feedforward_ga.py) — but it is left inline deliberately, because `categorical_sample_torch` also computes a log-prob and an entropy that a GA-evolved brain never consumes; routing it through the helper would add work to a hot loop to remove three lines. Revisit only if the GA brain ever gains a gradient path.

## Migration Plan

1. Land the two `_policy.py` helpers with unit coverage; no brain touched (tree green).
2. Migrate the shared bases (`_reservoir_hybrid_base`, `_reservoir_lstm_base`), covering crh/qef/qrh/crhqlstm/qrhqlstm; run the Family A + B migration tests and those five brains' suites.
3. Migrate `_hybrid_common` together with the three hybrid brains' rollout and reflex copies (D4); run the Family C migration test and the three hybrid suites.
4. Migrate `qsnnppo`, `qsnnreinforce`, `qliflstm`, `env/mlpppo_predator_brain`.
5. Family D partial reuse (`spikingreinforce`, `qrc`, `mlpreinforce`, `hybridquantumcortex` cortex path).
6. Land the spec delta; full `uv run pytest -m "not nightly"` + `uv run pre-commit run -a` against the green baseline; close #204 with the D0 corrections.

## Open Questions

- Whether `_policy.py`'s module docstring should grow a short per-family table (it currently narrates the two T5 families in prose). Leaning yes — with four families and 20 brains, prose stops scaling. Resolve during implementation and record in tasks.
