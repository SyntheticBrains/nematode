# 012 Supporting: Hyperparameter Evolution — MLPPPO Pilot Details

Detailed data, full per-seed history tables, and bug investigation traces for [logbook 012](../../012-hyperparam-evolution-mlpppo-pilot.md).

## Per-seed best params (decoded)

After CMA-ES + decode, each seed's best genome resolves to these MLPPPO hyperparameters:

| Field | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Baseline |
|---|---|---|---|---|---|
| `actor_hidden_dim` | 224 | 188 | 32 (clip) | 128 | 64 |
| `critic_hidden_dim` | 125 | 32 (clip) | 139 | 165 | 64 |
| `num_hidden_layers` | 3 | 1 (clip) | 2 | 3 | 2 |
| `learning_rate` | 0.0071 | 0.0018 | 0.0013 | 0.0027 | 0.001 |
| `gamma` | 0.91 | 0.97 | 0.90 (clip) | 0.95 | 0.99 |
| `entropy_coef` | 0.065 | 0.00029 | 0.0052 | 0.0040 | 0.02 |
| `num_epochs` | 8 | 7 | 4 | 8 | 10 |

`(clip)` indicates the genome's pre-decode value fell outside the schema bounds and was clipped at the boundary by the decoder. Note the diversity: every dimension varies 4-10× across the 4 winners.

## Per-seed history — best vs mean fitness

### Seed 42

| Gen | Best | Mean | Std |
|---|---|---|---|
| 1 | 1.000 | 0.883 | 0.191 |
| 2 | 1.000 | 0.783 | 0.237 |
| 3 | 1.000 | 0.917 | 0.099 |
| 4 | 1.000 | 0.767 | 0.229 |
| 5 | 1.000 | 0.917 | 0.099 |
| 10 | 1.000 | 0.800 | 0.283 |
| 15 | 1.000 | 0.933 | 0.125 |
| 20 | 1.000 | 0.917 | 0.152 |

Mean fitness across 20 gens: **0.872**.

### Seed 43

| Gen | Best | Mean | Std |
|---|---|---|---|
| 1 | 1.000 | 0.850 | 0.218 |
| 5 | 1.000 | 0.850 | 0.202 |
| 8 | 1.000 | 0.550 | 0.348 |
| 10 | 1.000 | 0.817 | 0.276 |
| 15 | 1.000 | 0.867 | 0.149 |
| 20 | 1.000 | 0.767 | 0.281 |

Mean fitness across 20 gens: **0.792**. Notable dip at gen 8 (0.550) — CMA-ES's covariance update temporarily centred on a weaker region before recovering.

### Seed 44

| Gen | Best | Mean | Std |
|---|---|---|---|
| 1 | 1.000 | 0.950 | 0.087 |
| 2 | 1.000 | 0.567 | 0.373 |
| 5 | 1.000 | 0.650 | 0.366 |
| 9 | 1.000 | 0.600 | 0.400 |
| 15 | 1.000 | 0.867 | 0.149 |
| 20 | 1.000 | 0.833 | 0.229 |

Mean fitness across 20 gens: **0.815**. Largest population variance — multiple gens with std > 0.30, consistent with CMA-ES exploring widely.

### Seed 45

| Gen | Best | Mean | Std |
|---|---|---|---|
| 1 | 1.000 | 0.800 | 0.141 |
| 5 | 1.000 | 0.867 | 0.170 |
| 10 | 1.000 | 0.917 | 0.223 |
| 15 | 1.000 | 0.817 | 0.276 |
| 20 | 1.000 | 0.667 | 0.350 |

Mean fitness across 20 gens: **0.860**. Wall-time outlier (~18 min vs ~3 min for the other seeds) — likely sampled deeper-network + small-lr + many-epoch combinations that took longer per genome.

### Across-seed mean-fitness summary

| Seed | Mean across gens | Std across gens | Wall (min) |
|---|---|---|---|
| 42 | 0.872 | 0.061 | 4.6 |
| 43 | 0.792 | 0.099 | 2.3 |
| 44 | 0.815 | 0.105 | 2.3 |
| 45 | 0.860 | 0.080 | 18.0 |

Mean-fitness varies 0.79-0.87 across seeds; CMA-ES doesn't consistently push it upward.

## Bug investigation traces

### Bug 1: CMA-ES x0 = zeros for hyperparameter encoders

**Symptom**: Calibration produced `best_params` with log-scale values consistently saturated at one bound (e.g. log_lr = 2.39 → exp(clip(2.39, -11.5, -4.6)) = exp(-4.6) = 0.01, the upper bound).

**Trace**:

```text
genome[3] (log learning_rate) = 2.39
log-bounds = [log(1e-5), log(1e-2)] = [-11.51, -4.61]
x0[3] (CMA-ES initial mean) = 0.0  # default
sigma0 = π/2 ≈ 1.57
sample[3] ~ N(0, 1.57)  ≈ samples in [-4.7, +4.7] at ±3σ
ALL samples > -4.6 → clip to upper bound → lr = 0.01 (too aggressive)
```

**Root cause**: `scripts/run_evolution.py:_build_optimizer` constructed `CMAESOptimizer` without passing x0, so it defaulted to zeros. For weight-evolution this is OK (weights cluster around zero); for hyperparameter evolution where bounds sit far from origin, it's catastrophic.

**Fix**: pass `encoder.initial_genome().params` as x0 to the optimiser. The encoder's `initial_genome` already samples uniformly within bounds — perfect starting point. Plumbed through `_build_optimizer(... , x0=...)`.

### Bug 2: Single global sigma can't handle mixed-scale schemas

**Symptom**: Even with valid x0, samples saturated at bound extremes for tight-range params.

**Trace**:

```text
After x0 fix:
genome[4] (gamma) sample ≈ N(0.91, 1.57) → range [-3.8, +5.6]
gamma bounds = [0.9, 0.999], range 0.099
ALL samples outside bounds → clip
```

**Root cause**: log-lr range is ~7 in log-units; gamma range is ~0.1. A single sigma=1.57 can't be appropriate for both. CMA-ES library supports per-param scaling via `CMA_stds` but the framework wasn't using it.

**Fix**: `GenomeEncoder` protocol gains `genome_stds(sim_config)`. `HyperparameterEncoder` returns `std = bound-width / 6` per slot (so ±3 stds at sigma=1.0 spans the full bound range). Weight encoders return `None` (uniform sigma — matches M0 behaviour exactly). `CMAESOptimizer` accepts optional `stds: list[float]`, threaded through as cma's `CMA_stds`.

### Bug 3: max_body_length not passed in fitness env construction

**Symptom**: Even after bugs 1 + 2 fixed, fitness still produced 0% for the initial_genome (which has lr=0.001, matching the baseline exactly).

**Trace**:

```python
# fitness.py: create_env_from_config(sim_config.environment, seed=seed, theme=Theme.HEADLESS)
# config_loader.py:
def create_env_from_config(env_config, *, max_body_length=None, ...):
    return DynamicForagingEnvironment(
        max_body_length=max_body_length if max_body_length is not None else 6,  # ← default 6
        ...
    )
# pilot YAML has body_length: 2 — silently overridden by default 6
```

**Root cause**: pre-existing M0 bug. All three `create_env_from_config` call sites in `fitness.py` (EpisodicSuccessRate + LearnedPerformanceFitness train + LearnedPerformanceFitness eval) omitted the `max_body_length` argument, so the env always ran with body=6 regardless of YAML.

**Fix**: pass `max_body_length=sim_config.body_length` at all three sites.

### Bug 4: No agent.reset_environment() between episodes

**Symptom**: After bugs 1-3 fixed, training-loop episodes all kept terminating with `starved` reason. Brain weights were genuinely changing (verified by snapshotting before/after), but no episode succeeded.

**Trace**:

```text
ep 0: term=starved
ep 1: term=starved
...
ep 29: term=starved
```

vs. `run_simulation.py` with the same seed: first SUCCESS at episode 4.

**Root cause**: `run_simulation.py` calls `agent.reset_environment()` between runs (rebuilds the env). `fitness.py` never did. After episode 0 starves (320 steps, 3/10 food eaten, satiety=0, agent stuck in corner), episode 1 starts from that broken state — food positions reset by `prepare_episode` but the agent is still in the corner. Brain trains on increasingly-degraded states for 30 episodes and never recovers.

**Fix**: call `agent.reset_environment()` between train episodes and between eval episodes (matching `run_simulation.py`'s per-run pattern). Brain state persists (that's the point of training); env state does not.

This was THE biggest fix — without it, hyperparameter evolution couldn't produce any non-zero fitness regardless of how good the hyperparameters were.

## Calibration vs final results

Pre-fix calibration (1 gen × pop 4 × K=30 / L=5):

| Genome | Fitness |
|---|---|
| 0 | 0.000 |
| 1 | 0.000 |
| 2 | 0.000 |
| 3 | 0.000 |

Same command after all 4 fixes:

| Genome | Fitness |
|---|---|
| 0 | 1.000 |
| 1 | 0.400 |
| 2 | 0.600 |
| 3 | 1.000 |

The spread (0.4-1.0) confirms genuine exploration of the schema space — distinct hyperparameter combinations produce distinct outcomes. Without the fixes, the optimiser was searching an inert space (every genome decoded to the same broken extreme).

## Wall-time scaling

Pilot total: 27 min (4 seeds × 20 gens × 12 pop × ~35 episodes each). Per-seed:

| Seed | Wall | Notes |
|---|---|---|
| 42 | 4.6 min | "Normal" |
| 43 | 2.3 min | Fast — likely converged on small networks |
| 44 | 2.3 min | Fast |
| 45 | 18.0 min | Slow — sampled deeper-network + small-lr regions |

**Implication for future pilots**: per-genome wall time varies 5-10× depending on hyperparams sampled (deeper networks + smaller lr → more train iterations to converge). With parallel=4 worker pool, the slowest genome dominates each generation. For wall-time-sensitive campaigns, consider:

1. Tighter bounds on `actor_hidden_dim` / `num_hidden_layers` to reduce variance.
2. Higher `parallel_workers` (the M2 pilot was bottlenecked by 4 cores; 8+ workers would amortise the variance).
3. A wall-time budget on individual genome evals (kill stragglers; mark them as fitness=0). Out of scope here but worth considering for the LSTMPPO pilot.

## Reproducing the pilot

```bash
# 1. Calibration smoke
uv run python scripts/run_evolution.py \
    --config configs/evolution/hyperparam_mlpppo_pilot.yml \
    --fitness learned_performance \
    --generations 1 --population 4 --seed 42 \
    --output-dir tmp/calibration

# 2. Full 4-seed pilot
OUTPUT_ROOT=tmp/m2_pilot scripts/campaigns/phase5_m2_hyperparam_mlpppo.sh

# 3. Hand-tuned baseline for comparison
OUTPUT_ROOT=tmp/m2_baseline scripts/campaigns/phase5_m2_hyperparam_baseline.sh

# 4. Aggregate + plot
uv run python scripts/campaigns/aggregate_m2_pilot.py \
    --pilot-root tmp/m2_pilot \
    --baseline-root tmp/m2_baseline \
    --output-dir tmp/m2_summary
```

Total wall time: ~30 minutes on a 10-core macOS machine.
