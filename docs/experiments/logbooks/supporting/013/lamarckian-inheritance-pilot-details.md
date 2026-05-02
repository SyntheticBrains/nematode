# 013 Supporting: Lamarckian Inheritance Pilot — Per-Seed Details + Sensitivity Analyses

Companion to [logbook 013](../../013-lamarckian-inheritance-pilot.md). Holds per-seed trajectories, evolved-hyperparameter tables, sensitivity analyses, and the save/load round-trip verification methodology.

## Per-seed `history.csv` trajectories (best fitness per generation)

### Lamarckian arm

| Gen | Seed 42 | Seed 43 | Seed 44 | Seed 45 |
|---|---|---|---|---|
| 1 | 0.84 | 0.56 | 0.64 | 0.84 |
| 2 | **1.00** | 0.88 | 0.88 | 0.84 |
| 3 | 1.00 | 0.92 | 0.96 | 0.88 |
| 4 | 1.00 | 0.92 | 0.96 | 0.88 |
| 5 | 1.00 | 0.92 | **1.00** | 0.88 |
| 6 | 1.00 | **1.00** | 1.00 | **1.00** |
| 7 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8 | 1.00 | 1.00 | 1.00 | 1.00 |
| 9 | 1.00 | 1.00 | 1.00 | 1.00 |
| 10-20 | 1.00 (sustained) | 1.00 (sustained) | 1.00 (sustained) | 1.00 (sustained) |

**Per-seed first-reach generation** (the core convergence-speed metric):

| Threshold | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Mean |
|---|---|---|---|---|---|
| ≥ 0.84 | gen 2 | gen 3 | gen 3 | gen 2 | 2.50 |
| ≥ 0.88 | gen 3 | gen 3 | gen 3 | gen 4 | 3.25 |
| ≥ 0.92 | gen 3 | gen 4 | gen 4 | gen 7 | **4.50** |
| ≥ 0.96 | gen 3 | gen 4 | gen 5 | gen 7 | 4.75 |
| = 1.00 | gen 2 | gen 6 | gen 5 | gen 6 | 4.75 |

### Control arm

| Gen | Seed 42 | Seed 43 | Seed 44 | Seed 45 |
|---|---|---|---|---|
| 1 | 0.84 | 0.56 | 0.64 | 0.84 |
| 2 | 0.84 | 0.80 | 0.88 | 0.84 |
| 3 | 0.84 | 0.80 | 0.88 | 0.84 |
| 4 | 0.84 | 0.96 | 0.96 | 0.88 |
| 5 | 0.84 | **0.96** | **0.96** | 0.88 |
| 6 | 0.84 | 0.96 | 0.96 | 0.88 |
| 7 | 0.84 | 0.96 | 0.96 | 0.96 |
| 8 | 0.84 | 0.96 | 0.96 | **0.96** |
| 9 | 0.84 | 0.96 | 0.96 | 0.96 |
| ... | (saturated 0.84-0.88) | 0.96 (sustained) | 0.96 (sustained) | 0.96 (sustained) |
| 18 | 0.88 | 0.96 | 0.96 | 0.96 |
| 19-20 | 0.88 | 0.96 | 0.96 | 0.96 |

**Per-seed first-reach generation:**

| Threshold | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Mean (with seed 42 → 21 fallback) |
|---|---|---|---|---|---|
| ≥ 0.84 | gen 2 | gen 3 | gen 3 | gen 2 | 2.50 |
| ≥ 0.88 | gen 19 | gen 5 | gen 3 | gen 5 | 8.00 |
| ≥ 0.92 | **never** | gen 5 | gen 5 | gen 8 | **9.75** |
| ≥ 0.96 | never | gen 5 | gen 5 | gen 8 | 9.75 |

Gen-1 best fitness is identical between arms because gen-0 is from-scratch in both arms (same seed → same TPE first sample → identical evaluations). The gen-0-from-scratch step is the same code path under either inheritance setting (the spec's "First generation runs from-scratch under any inheritance strategy" scenario, verified at scale here).

## Per-generation across-seed mean (population-mean column from `history.csv`)

This is the trajectory the corrected floor metric uses (lamarckian gen-2 mean vs control gen-3 mean). Reproduced in full because the convergence-shape difference is the headline finding.

| Gen | Lam mean | Ctrl mean | Δ (lam − ctrl) |
|---|---|---|---|
| 1 | 0.124 | 0.124 | +0.000 |
| 2 | **0.676** | 0.325 | **+0.351** |
| 3 | 0.772 | 0.257 | **+0.515** |
| 4 | 0.774 | 0.364 | +0.410 |
| 5 | 0.825 | 0.048 | **+0.778** |
| 6 | 0.858 | 0.437 | +0.421 |
| 7 | 0.867 | 0.314 | +0.553 |
| 8 | 0.867 | 0.440 | +0.427 |
| 9 | 0.845 | 0.432 | +0.413 |
| 10 | 0.838 | 0.318 | +0.520 |
| 11 | 0.834 | 0.411 | +0.423 |
| 12 | 0.838 | 0.428 | +0.409 |
| 13 | 0.841 | 0.404 | +0.437 |
| 14 | 0.886 | 0.487 | +0.398 |
| 15 | 0.836 | 0.371 | +0.465 |
| 16 | 0.886 | 0.437 | +0.449 |
| 17 | 0.872 | 0.395 | +0.477 |
| 18 | 0.888 | 0.502 | +0.386 |
| 19 | 0.898 | 0.455 | +0.443 |
| 20 | 0.899 | 0.486 | +0.413 |

**Two patterns visible in the table:**

1. **Inheritance kicks in at gen 2.** Gen-1 numbers are by-construction identical (same gen-0 evaluation). Gen-2 is the first generation where children warm-start from the gen-0 elite.
2. **The +0.40-0.55pp lift is sustained.** No drift back to parity, no late-stage collapse. The lamarckian population stays in the 0.83-0.90 band from gen-6 onwards while control bounces in 0.31-0.50.

The gen-5 control dip to 0.048 is striking — that's TPE exploring a region where most of the 12 children scored ~0 in their L=25 eval (almost certainly seed-42's TPE drift toward a hyperparameter region the brain can't learn from in K=50 episodes). Lamarckian's gen-5 mean (0.825) for the same seeds shows inheritance shielded the population from this exploratory excursion.

## Evolved hyperparameters (gen-20 best per seed)

### Lamarckian winners (all reach fitness 1.00)

| Seed | actor_lr | critic_lr | gamma | entropy_coef | Notes |
|---|---|---|---|---|---|
| 42 | 6.14e-04 | 5.89e-05 | 0.984 | 1.04e-04 | Low entropy → exploitative |
| 43 | 1.12e-04 | 6.24e-05 | 0.979 | 1.23e-02 | Low actor_lr + high entropy |
| 44 | 9.53e-05 | 2.87e-04 | 0.924 | 3.13e-04 | Lowest gamma (least long-horizon) |
| 45 | 5.38e-04 | 2.72e-05 | 0.967 | 2.79e-04 | High actor_lr + low critic_lr |

### Control winners

| Seed | Best fitness | actor_lr | critic_lr | gamma | entropy_coef |
|---|---|---|---|---|---|
| 42 | **0.88** (saturated) | 7.06e-04 | 2.22e-05 | 0.966 | 1.68e-02 |
| 43 | 0.96 | 5.74e-04 | 5.91e-04 | 0.920 | 2.90e-03 |
| 44 | 0.96 | 6.07e-04 | 1.55e-04 | 0.970 | 4.73e-03 |
| 45 | 0.96 | 8.45e-04 | 7.96e-05 | 0.946 | 3.14e-03 |

**Key observation**: control seed 42's hyperparams are *not pathological* — actor_lr 7.06e-4 is mid-range, critic_lr 2.22e-5 is low but within bounds, gamma 0.966 is reasonable, entropy 1.68e-2 is on the high side but valid. **Nothing clipped at bounds.** This is qualitatively different from M2.11's seed-43 dead-zone (where actor_lr was clipped at 1e-5 and entropy ≈ 3e-4 — the brain literally couldn't update or explore). M3 control seed 42 is "TPE converged on a region that's slightly suboptimal", not "TPE dead-zone-trapped".

This makes the lamarckian-rescue finding more interesting: inheritance can compensate for *suboptimal-but-not-broken* TPE convergence, not just for the catastrophic dead-zone case.

## Hyperparameter spread analysis

design.md Risk 2 asked: would inheritance cause TPE posterior collapse onto the elite parent's exact hyperparams? Looking at the lamarckian winners' hyperparam ranges:

| Field | Min | Max | Range | log10 spread |
|---|---|---|---|---|
| actor_lr | 9.53e-05 | 6.14e-04 | 6.4× | 0.81 |
| critic_lr | 2.72e-05 | 2.87e-04 | 10.6× | 1.03 |
| gamma | 0.924 | 0.984 | 1.06× | n/a |
| entropy_coef | 1.04e-04 | 1.23e-02 | 118× | 2.07 |

Seeds reach 1.00 best fitness with hyperparameters spanning an order of magnitude or more on the log-scale fields. This is **not a collapsed posterior** — TPE found multiple distinct hyperparameter regions that all converge to perfect performance once inheritance jump-starts the policy. Risk 2 mitigation (raise `inheritance_elite_count`) is not needed for M3.

## Speed-margin sensitivity (full table)

The aggregator's default treatment of "never reached" (control seed 42) is `run_length + 1 = 21`. Sensitivity to alternative treatments:

| Treatment | Lam mean | Ctrl mean | Margin | Speed gate (≥4) |
|---|---|---|---|---|
| Aggregator default (21) | 4.50 | 9.75 | +5.25 | **PASS** |
| Conservative (20) | 4.50 | 9.50 | +5.00 | **PASS** |
| Extrapolate (25) | 4.50 | 10.75 | +6.25 | **PASS** |
| Optimistic (12, ~halfway) | 4.50 | 7.50 | +3.00 | FAIL |
| Exclude seed 42 (n=3) | 5.00 | 6.00 | +1.00 | FAIL |

The strongest counter-test is "exclude seed 42 entirely" — n=3 with margin +1.0 gens. Under this treatment, inheritance still helps (positive margin), but doesn't clear the +4 threshold. This says: the +5.25 gen margin is partially driven by inheritance rescuing seed 42's TPE-unlucky trajectory, but the underlying speed-up exists across all 4 seeds.

**Threshold sensitivity** (which fitness threshold the speed metric measures):

| Threshold | Lam per-seed | Ctrl per-seed | Margin (with 21 fallback) |
|---|---|---|---|
| 0.80 | [2, 3, 3, 2] | [2, 3, 3, 2] | +0.00 |
| 0.84 | [2, 3, 3, 2] | [2, 5, 3, 2] | +0.50 |
| **0.88** | [3, 3, 3, 4] | **[19, 5, 3, 5]** | **+4.75** |
| 0.92 | [3, 4, 4, 7] | [—, 5, 5, 8] | +5.25 |

This table is the most striking finding in the supporting data. **Inheritance buys nothing at threshold 0.80** — both arms reach "decent" policies in 2-3 gens. The value is concentrated at thresholds 0.88+. TPE alone gets to "decent" quickly; inheritance gets to "near-perfect" quickly. For M4 (Baldwin) this means the relevant comparison threshold is 0.88-0.92, not 0.80.

## Cross-schema confounder check (M3 vs M2.12)

Read M2.12's archived per-seed history files at [`artifacts/logbooks/012/m2_hyperparam_lstmppo_klinotaxis_predator_tpe_pilot/seed-{42-45}/history.csv`](../../../../artifacts/logbooks/012/m2_hyperparam_lstmppo_klinotaxis_predator_tpe_pilot/) and computed the same speed metric:

| Seed | M3-lam (4-fld) | M3-ctrl (4-fld) | M2.12 (6-fld) |
|---|---|---|---|
| 42 | 3 | — (saturated 0.88) | 12 |
| 43 | 4 | 5 | 3 |
| 44 | 4 | 5 | 9 |
| 45 | 7 | 8 | 16 |
| **mean (with 21 fallback for "—")** | **4.50** | **9.75** | **10.00** |

Three pairwise margins:

- M3-lam vs M3-ctrl: **+5.25 gens** — the headline M3 result.
- M3-lam vs M2.12 (cross-schema): **+5.50 gens** — lamarckian advantage holds across the schema change.
- M3-ctrl vs M2.12 (4-fld vs 6-fld, both no inheritance): **+0.25 gens** — schema simplification buys ~nothing.

**The schema confounder is dispelled empirically.** Dropping `rnn_type` and `lstm_hidden_dim` from the schema (because they're architecture-changing and incompatible with per-genome warm-start) does not advantage the M3 control over M2.12. The full +5-gen lift is from inheritance.

## Save/load round-trip verification methodology

Before publishing the verdict, ran a verification to rule out partial-load bugs in the inheritance pipeline (the highest-likelihood remaining failure mode after the framework tests passed). Methodology:

1. Build an LSTMPPO brain via `HyperparameterEncoder.decode` on the M3 pilot YAML's deterministic initial genome.
2. Run `LearnedPerformanceFitness.evaluate` with K=2 train + L=1 eval and `weight_capture_path=tmp/captured.pt` to exercise the actual fitness-function save path.
3. Decode genome → fresh brain; `load_weights(brain, captured.pt)`; capture `brain.get_weight_components()`.
4. `save_weights(brain, tmp/roundtrip.pt)`; decode genome → fresh brain again; `load_weights(brain, roundtrip.pt)`; capture again.
5. Compare every tensor in every component between step 3 and step 4.

Result: **18 tensors round-trip bit-exact** — `lstm.state["weight_ih_l0"]`, `lstm.state["weight_hh_l0"]`, `lstm.state["bias_ih_l0"]`, `lstm.state["bias_hh_l0"]`, `layer_norm.state["weight"]`, `layer_norm.state["bias"]`, policy network's actor head + value network's critic head, `actor_optimizer.state` (Adam first/second moments), `critic_optimizer.state`, and `training_state.step_count`. All 18 also confirmed mutated by training (zero remained at fresh-init values). The lamarckian children inherit the parent's exact trained brain — including optimiser momentum, second moments, training counters — no silent truncation.

Captured checkpoint file size: 615 KB (a useful concrete number for design.md Risk 3 wall-time math; per-genome IO at this scale is negligible vs the K=50 train phase).

## Wall-time analysis

Per-seed wall (parallel=4 within each seed; seeds run sequentially):

| Arm | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Total |
|---|---|---|---|---|---|
| Lamarckian | 14m 50s | 15m 21s | 15m 16s | 14m 45s | ~60 min |
| Control | 13m 02s | 13m 08s | 13m 32s | 13m 36s | ~53 min |

The +2 min/seed lamarckian overhead is dominated by `save_weights` IO. Under M3's GC contract, only 1 file survives in `inheritance/gen-NNN/` between generations, so disk usage stays bounded — no IO regression risk for longer runs.

Pre-pilot smoke (3 gens × pop 6 × seed 42, single seed) wall:

| Arm | Wall |
|---|---|
| Lamarckian smoke | ~85s |
| Control smoke | ~65s |

Smoke wall-time per episode-unit: ~222 ms (lamarckian) and ~196 ms (control), within ~10% of M2.12's published ~187 ms/episode-unit at parallel=4. This confirmed the framework holds at real K=50/L=25 scale before launching the full pilots.

## Reproducibility notes

- All 4 seeds match M2.11/M2.12's published baseline (15-22% success on hand-tuned config) under the M3 revision — confirms `run_simulation.py` path is unchanged and reproducible.
- Lamarckian gen-1 best/mean is bit-equal to control gen-1 best/mean across all 4 seeds — confirms the gen-0-from-scratch code path is byte-equivalent to pre-M3 (the M3 spec's "First generation runs from-scratch under any inheritance strategy" scenario, verified at full pilot scale).
- All evolution unit tests (148 total, 30 new for M3) pass on this revision; pre-commit clean.
