## Why

`brain/arch/_policy.py` was introduced by T5 (`add-continuous-2d-and-action-heads`, D6) to be the single reviewed home for the parity-sensitive discrete policy code — sampling, log-probability, entropy, and the PPO clipped surrogate. Of the 27 registered brains, **9 reach it**: the seven migrated directly (`mlpppo`, `connectome_ppo`, `transformer_ppo`, `equivariant_quantum`, `lstmppo`, `cfc_ppo`, `spiking_ppo`) plus `mingruppo` and `minlstmppo`, which subclass `LSTMPPOBrain` and are covered transitively. **Fourteen are not.** The clipped surrogate is still written out by hand in **10 modules**, and the manual `log(probs + 1e-8)` / `-Σ p·log(p + 1e-10)` scoring it replaces survives at **23 sites across 10 modules** (21 of them live in a loss or ratio path; two are diagnostic-only).

This is exactly the drift surface D6 was created to close, and it has already drifted:

- Four brains (`hybridquantum`, `hybridclassical`, `hybridquantumcortex`, `qsnnreinforce`) score an **ε-greedy mixture** `(1-ε)·softmax(logits/T) + ε·uniform` rather than a softmax, so they cannot use the existing logits-based helper at all — the shared module has no vocabulary for the form four brains actually use.
- The epsilon constant is **inconsistent within a single expression pair**: `+1e-8` for the chosen-action log-prob, `+1e-10` for the entropy sum, in every copy. Nothing enforces or explains the difference — and `hybridquantumcortex` spells the same constant two ways in adjacent branches of one function (a literal `1e-8` at `:1926`, `NORM_EPS` at `:1934`).
- The three hybrid brains carry a **rollout/update asymmetry** on the cortex PPO path: the rollout stores `log(softmax(logits) + 1e-8)` ([`hybridquantum.py:1073`](../../../packages/quantum-nematode/quantumnematode/brain/arch/hybridquantum.py)) while the update re-scores with `Categorical.log_prob` ([`_hybrid_common.py:448`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_hybrid_common.py)). Both sides are torch, so the only difference is the formula — and it means the two halves of the PPO importance ratio disagree, giving `ratio` a small systematic bias at `ratio == 1`. This is a latent defect that only became visible by inventorying the copies.

Phase 7 builds L4 plasticity on the L1 plugin layer. Every additional inline copy is a place where an L4 change has to be applied 14 times and can be applied 13 times.

## What Changes

- **Extend `_policy.py`** with the two forms the remaining brains need, alongside the existing logits-based helpers:
  - `categorical_logprob_entropy_from_probs(probs, action)` — scores an **already-constructed probability vector**, for the ε-mixed family whose distribution is not a softmax of any logits.
  - `reinforce_policy_loss(log_probs, advantages)` — the `-(log_probs · advantages).mean()` REINFORCE term, duplicated in `qrc`, `mlpreinforce`, `spikingreinforce`, and the `hybridquantumcortex` cortex path. (The PPO-family brains keep `ppo_clip_policy_loss`.)
- **Migrate the 10 remaining clipped-surrogate sites** onto `ppo_clip_policy_loss`, and the live manual log-prob/entropy sites onto the shared scorers, covering **all 14 unmigrated brains**: `crh`, `qef`, `qrh`, `crhqlstm`, `qrhqlstm`, `qsnnppo`, `qliflstm`, `hybridquantum`, `hybridclassical`, `hybridquantumcortex`, `qsnnreinforce`, plus partial reuse in `mlpreinforce`, `spikingreinforce`, `qrc` — and the `env/mlpppo_predator_brain` PPO predator, which is not a registered brain.
- **Fix the hybrid cortex rollout/update asymmetry** as a consequence of the migration: both halves of the cortex ratio route through the same torch scorer (D4; the separate reflex path is a narrower, partial fix).
- **Extend the migration-regression bar** to these brains with a per-brain, declared-up-front tolerance (byte-exact where the brain already uses `torch.distributions.Categorical`; `rtol=0, atol=1e-7` otherwise, matching the existing T2 bar for these same architectures).
- **Leave four brains out by construction**: `qqlearning`, `mlpdqn`, `qvarcircuit`, and `feedforwardga` are not policy-gradient brains and have no categorical PG surrogate to share; `feedforwardga`'s `no_grad()` sampler is left inline deliberately (see design Risks).
- **No behavioural change is intended** for any brain: no reward, env, config, hyperparameter, or public API change. Numpy `rng.choice` samplers are kept verbatim so sampled-action trajectories stay byte-identical.

## Capabilities

### New Capabilities

<!-- none — this consolidates the implementation behind existing brain-architecture requirements -->

### Modified Capabilities

- `brain-architecture`: the shared action-policy module is specified as the **single source of truth** for discrete policy scoring and the PPO clipped surrogate across all PPO-family brains (not only the four T5 MUST brains), and the migration-regression bar is extended to the remaining 14 architectures with a per-brain declared tolerance.

## Impact

- **Code (14 modules):** `brain/arch/_policy.py` (two added helpers); the three shared bases `_reservoir_hybrid_base.py`, `_reservoir_lstm_base.py`, `_hybrid_common.py`; the per-brain copies `hybridquantum.py`, `hybridclassical.py`, `hybridquantumcortex.py`, `qsnnppo.py`, `qsnnreinforce.py`, `qliflstm.py`, `qrc.py`, `mlpreinforce.py`, `spikingreinforce.py`; and `env/mlpppo_predator_brain.py`. The five brains behind the shared bases (`crh`, `qef`, `qrh`, `crhqlstm`, `qrhqlstm`) need no per-brain edit.
- **Spec:** `openspec/specs/brain-architecture/spec.md` (three ADDED requirements).
- **Tests:** a per-family migration test on the [`test_mlpppo_policy_migration.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_mlpppo_policy_migration.py) template (same-run inline-reference comparison, no hard-coded float goldens, so it survives CI's BLAS), plus coverage for the two new helpers in `test_policy.py`. Every touched brain's existing suite must pass unchanged.
- **Issue:** closes [#204](https://github.com/SyntheticBrains/nematode/issues/204), whose body needs three corrections (see design D0).
- **No** change to reward, env, configs, saved weights, artifacts, or the `Brain` Protocol surface. `composite_benchmark_score` is untouched.
