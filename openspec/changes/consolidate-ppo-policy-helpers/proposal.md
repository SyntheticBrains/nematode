## Why

`brain/arch/_policy.py` was introduced by T5 (`add-continuous-2d-and-action-heads`, D6) to be the single reviewed home for the parity-sensitive discrete policy code — sampling, log-probability, entropy, and the PPO clipped surrogate. Seven brains were migrated onto it (`mlpppo`, `connectome_ppo`, `transformer_ppo`, `equivariant_quantum`, `lstmppo`, `cfc_ppo`, `spiking_ppo`). **Thirteen were not.** The clipped surrogate is still written out by hand in **10 more modules**, and the manual `log(probs + 1e-8)` / `-Σ p·log(p + 1e-10)` pair it replaces survives in **13**.

This is exactly the drift surface D6 was created to close, and it has already drifted:

- Four brains (`hybridquantum`, `hybridclassical`, `hybridquantumcortex`, `qsnnreinforce`) score an **ε-greedy mixture** `(1-ε)·softmax(logits/T) + ε·uniform` rather than a softmax, so they cannot use the existing logits-based helper at all — the shared module has no vocabulary for the form four brains actually use.
- The epsilon constant is **inconsistent within a single expression pair**: `+1e-8` for the chosen-action log-prob, `+1e-10` for the entropy sum, in all 13 copies. Nothing enforces or explains the difference.
- The three hybrid brains carry a **rollout/update asymmetry** on the cortex PPO path: the rollout stores `log(softmax(logits) + 1e-8)` ([`hybridquantum.py:1073`](../../../packages/quantum-nematode/quantumnematode/brain/arch/hybridquantum.py)) while the update re-scores with `Categorical.log_prob` ([`_hybrid_common.py:448`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_hybrid_common.py)). The two halves of the PPO importance ratio are computed by different formulas, so `ratio` carries a small systematic bias at `ratio == 1`. This is a latent defect that only became visible by inventorying the copies.

Phase 7 builds L4 plasticity on the L1 plugin layer. Every additional inline copy is a place where an L4 change has to be applied 13 times and can be applied 12 times.

## What Changes

- **Extend `_policy.py`** with the two forms the remaining brains need, alongside the existing logits-based helpers:
  - `categorical_logprob_entropy_from_probs(probs, action)` — scores an **already-constructed probability vector**, for the ε-mixed family whose distribution is not a softmax of any logits.
  - `reinforce_policy_loss(log_probs, advantages)` — the `-(log_probs · advantages).mean()` REINFORCE term, duplicated in `qrc`, `mlpreinforce`, `spikingreinforce`, and the `hybridquantumcortex` cortex path. (The PPO-family brains keep `ppo_clip_policy_loss`.)
- **Migrate the 10 remaining clipped-surrogate sites** onto `ppo_clip_policy_loss`, and the 13 manual log-prob/entropy sites onto the shared scorers, covering 13 brains: `crh`, `qef`, `qrh`, `crhqlstm`, `qrhqlstm`, `qsnnppo`, `qliflstm`, `hybridquantum`, `hybridclassical`, `hybridquantumcortex`, `qsnnreinforce`, plus the `env/mlpppo_predator_brain` PPO predator and partial reuse in `mlpreinforce`, `spikingreinforce`, `qrc`.
- **Fix the hybrid rollout/update asymmetry** as a consequence of the migration: both halves of the cortex ratio route through the same torch scorer.
- **Extend the migration-regression bar** to these brains with a per-brain, declared-up-front tolerance (byte-exact where the brain already uses `torch.distributions.Categorical`; `rtol=0, atol=1e-7` otherwise, matching the existing T2 bar for these same architectures).
- **No behavioural change is intended** for any brain: no reward, env, config, hyperparameter, or public API change. Numpy `rng.choice` samplers are kept verbatim so sampled-action trajectories stay byte-identical.

## Capabilities

### New Capabilities

<!-- none — this consolidates the implementation behind existing brain-architecture requirements -->

### Modified Capabilities

- `brain-architecture`: the shared action-policy module is specified as the **single source of truth** for discrete policy scoring and the PPO clipped surrogate across all PPO-family brains (not only the four T5 MUST brains), and the migration-regression bar is extended to the remaining 13 architectures with a per-brain declared tolerance.

## Impact

- **Code (13 modules):** `brain/arch/_policy.py` (two added helpers); the three shared bases `_reservoir_hybrid_base.py`, `_reservoir_lstm_base.py`, `_hybrid_common.py`; the per-brain copies `hybridquantum.py`, `hybridclassical.py`, `hybridquantumcortex.py`, `qsnnppo.py`, `qsnnreinforce.py`, `qliflstm.py`, `qrc.py`, `mlpreinforce.py`, `spikingreinforce.py`; and `env/mlpppo_predator_brain.py`.
- **Spec:** `openspec/specs/brain-architecture/spec.md` (two ADDED requirements).
- **Tests:** a per-family migration test on the [`test_mlpppo_policy_migration.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_mlpppo_policy_migration.py) template (same-run inline-reference comparison, no hard-coded float goldens, so it survives CI's BLAS), plus coverage for the two new helpers in `test_policy.py`. Every touched brain's existing suite must pass unchanged.
- **Issue:** closes [#204](https://github.com/SyntheticBrains/nematode/issues/204), whose body needs three corrections (see design D0).
- **No** change to reward, env, configs, saved weights, artifacts, or the `Brain` Protocol surface. `composite_benchmark_score` is untouched.
