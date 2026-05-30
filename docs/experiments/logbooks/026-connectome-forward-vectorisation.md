# 026: Connectome PPO Forward Vectorisation

**Date**: 2026-05-30
**Type**: Performance optimisation (infrastructure)
**Brain**: `ConnectomePPOBrain` (Cook 2019 connectome, focal architecture)
**Status**: ✅ shipped

## Objective

Cut the per-run wall-time of the connectome brain, which the Phase 2 pre-flight
for the `weight-search-architecture-ranking` change identified as the
architecture-comparison sweep's bottleneck (~10× the per-run wall-time of
MLPPPO, AND the slowest-converging architecture — so it dominates both the
sweep cost and every calibration cycle). A safe speed optimisation here shortens
the whole Phase 4 sweep + makes per-architecture tuning on the connectome
affordable.

## Background

The connectome PPO update ran an **un-vectorised per-sample Python loop**:

```python
for k in range(batch_size):
    food_k, ... = self._unpack_state(states[k])
    logits_k, hidden_k = self.topology.forward_with_hidden(food_k, ...)  # single 302-vec forward
    ...
```

With `rollout_buffer_size 256 × num_epochs 10`, that is ~2560 single-sample
302-neuron forwards per update — and roughly **90% of all connectome forward
passes happen here** (the per-env-step `run_brain` forward is inherently
single-sample and unavoidable; the update is not). Each sample ran the K=4
recurrence as separate `(302,302) @ (302,)` matvecs, which badly underutilise
the hardware vs a batched matmul.

## Method

### Code changes (`packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py`)

- **`forward_with_hidden_batched((B, n_food), ...) -> ((B, 4), (B, 302))`** — a
  batched analogue of `forward_with_hidden`: batched food/predator/thermotaxis
  injection + batched K-step recurrence (`h @ chem_mat` over `(B, 302)` instead
  of B separate `chem_mat.T @ h` matvecs) + batched motor-pool + readout.
  Predator zone routing is vectorised via the `(B, 4)` contact-zone one-hot
  mask — each sample receives exactly its zone's injection, and masked-out
  terms add `0.0` (exact in float).
- **`_unpack_state_batched`** — slices the buffered `(B, state_dim)` states
  along dim 1; returns the raw zone one-hot for the masked injection.
- **PPO update loop** — the per-sample Python loop replaced with ONE batched
  forward + batched critic per minibatch.
- **Micro-opt (bit-identical):** `_pool_motor` unified to pool over the last dim
  for 1D + 2D inputs using Python-int class slices cached at construction —
  removes the per-call `.item()` CPU syncs (each forced a CPU↔device
  synchronisation). Helps the single-sample `run_brain` path too.

`run_brain` (one action per env step) keeps the unchanged single-sample
`forward_with_hidden` path → the rollout stays **bit-identical** for given
weights. Only the UPDATE is batched.

### Validation approach

Batched matmul reorders float accumulation vs B separate matvecs, so the
optimisation is **numerically equivalent but NOT bit-identical**. The bar:

1. **Numerical-equivalence unit test** (`test_connectome_vectorisation.py`, 7
   tests): the batched forward must match the single-sample loop within float
   tolerance across all four projection configs + every ContactZone — a much
   tighter correctness guard than the end-to-end smoke.
2. **R2b regression** (Gate 1 klinotaxis baseline, logbook 023): learning must
   be preserved, not degraded.
3. **All 64 existing connectome tests** (PPO update + projections) still pass.

## Results

### Speedup — 8.1×

Connectome C1 foraging @200ep, seed 2026, isolated (`/usr/bin/time`):

| | Wall-time |
|---|---|
| Baseline (`main`, per-sample loop) | 232.89s |
| Optimised (batched update) | 28.67s |
| **Speedup** | **8.1×** |

### Numerical equivalence — to ~1e-6

Batched forward vs single-sample loop, max abs diff (B=16):

| Config | logits | hidden |
|---|---|---|
| foraging-only | 7e-8 | 4e-7 |
| predator | 6e-8 | 6e-6 |
| thermo | 7e-8 | 9e-7 |
| combined | 1e-7 | 2e-6 |

Well within float32 accumulation noise over 302 neurons × K=4 steps. Every
ContactZone's masked-batched injection matches its per-zone branch.

### Learning preserved — no regression

R2b config (`connectomeppo_small_low_entropy_klinotaxis.yml`, 500ep), optimised:

| Seed | last-25 | overall |
|---|---|---|
| 2026 | 100% | 85% |
| 42 | 100% | 89% |
| 43 | 100% | 94% |
| 44 | 100% | 71% |

All four seeds converge to 100% last-25 (R2b reference: 92% at seed 2026). The
optimised connectome learns the task fully — no degradation.

## Analysis

**Why it is safe.** The forward is mathematically equivalent (equivalence test
to ~1e-6) and `run_brain`'s rollout is bit-identical, so the only change is the
batched UPDATE forward. The masked-zone predator injection is exactly equal to
the per-zone branch (adding `0.0` is exact in float).

**Why R2b is not bit-reproduced (and why that is acceptable).** The vectorised
update's ulp-level gradient differences compound over PPO training, shifting the
trajectory the way a seed change would — PPO's well-known chaotic sensitivity to
small numerical perturbations (the same mechanism behind the entropy-drift in
logbook 023). The optimised connectome's last-25 distribution (100% across four
seeds) sits at/above R2b's 92% reference, so this is a trajectory shift, not a
degradation. The exact byte-identical 92% remains reproducible from the
pre-optimisation commit (preserved in git history + logbook 023).

**Why batching is the right lever.** ~90% of connectome forwards are in the
batchable update; the recurrence (K=4 × 302×302) is the dominant compute and
batches cleanly into one matmul per depth-step. The `.item()`-sync removal is a
free bit-identical bonus.

## Conclusions

- The connectome brain is **8.1× faster** with learning fully preserved and
  forward-pass correctness validated to ~1e-6.
- The optimisation is numerically equivalent, not bit-identical; R2b shifts
  within seed variance (no regression). This is documented as the honest framing
  (vs the earlier "byte-identical" wording).
- The whole `weight-search-architecture-ranking` Phase 4 sweep — which is
  connectome-dominated — benefits proportionally; per-architecture calibration
  cycles on the connectome are now ~8× cheaper.

## Next steps

- The Phase 4 sweep (the next major `weight-search-architecture-ranking` work)
  runs on this optimised connectome.
- The `forward_with_hidden_batched` path is also available to any future caller
  that needs batched connectome inference.

## Data references

- Code: `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py`
- Tests: `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_connectome_vectorisation.py`
- Prior brain-optimisation precedent: logbooks 001 (quantum), 003 (spiking),
  supporting/008 (QSNN-PPO).
