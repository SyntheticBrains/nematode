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

All four seeds reach `best_fitness=1.000` from gen 1 onwards; CMA-ES's job here is finding the broad-but-mostly-saturated region rather than climbing a fitness gradient. The numbers below are from the **post-bug-fix re-run** that replaces PR #134's now-superseded data; the archived CSVs at `artifacts/logbooks/012/m2_hyperparam_pilot/seed-{42-45}/history.csv` match these tables exactly.

### Seed 42

| Gen | Best | Mean | Std |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 0.000 |
| 2 | 1.000 | 0.983 | 0.055 |
| 5 | 1.000 | 1.000 | 0.000 |
| 10 | 1.000 | 0.983 | 0.055 |
| 15 | 1.000 | 0.883 | 0.276 |
| 20 | 1.000 | 0.967 | 0.075 |

Mean fitness across 20 gens: **0.974**.

### Seed 43

| Gen | Best | Mean | Std |
|---|---|---|---|
| 1 | 1.000 | 0.983 | 0.055 |
| 2 | 1.000 | 0.967 | 0.111 |
| 5 | 1.000 | 0.983 | 0.055 |
| 10 | 1.000 | 1.000 | 0.000 |
| 15 | 1.000 | 0.750 | 0.433 |
| 20 | 1.000 | 0.917 | 0.128 |

Mean fitness across 20 gens: **0.921**. Wider variance than seed 42 — CMA-ES wandered briefly into a mixed-quality region (gens 13-15) before settling back near 1.000.

### Seed 44

| Gen | Best | Mean | Std |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 0.000 |
| 2 | 1.000 | 0.733 | 0.427 |
| 5 | 1.000 | 0.967 | 0.075 |
| 10 | 1.000 | 0.933 | 0.094 |
| 15 | 1.000 | 0.917 | 0.276 |
| 20 | 1.000 | 0.850 | 0.307 |

Mean fitness across 20 gens: **0.908**. Notable dip at gen 2 (mean 0.733, std 0.427) — CMA-ES sampled wider in the early generations before retracting around the high-fitness mean.

### Seed 45

| Gen | Best | Mean | Std |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 0.000 |
| 2 | 1.000 | 0.983 | 0.055 |
| 5 | 1.000 | 0.983 | 0.055 |
| 10 | 1.000 | 0.900 | 0.277 |
| 15 | 1.000 | 0.983 | 0.055 |
| 20 | 1.000 | 0.983 | 0.055 |

Mean fitness across 20 gens: **0.978**. Cleanest trajectory of the four seeds — most generations sit at or near 1.000 mean.

### Across-seed mean-fitness summary

| Seed | Mean across gens |
|---|---|
| 42 | 0.974 |
| 43 | 0.921 |
| 44 | 0.908 |
| 45 | 0.978 |

Across-seed mean: **0.95 ± 0.03**. Compare to PR #134's original (now-superseded) figures of 0.79-0.87 range — those measurements ran on body=6 (Bug 1) with sometimes-mismatched sensing (Bug 2 inactive for oracle MLPPPO but Bug 3 still in effect), so the population mean was depressed. Post-fix the population mean sits in the 0.91-0.98 band and the schema's broad viable region produces mostly-saturated genomes.

## Bug investigation traces (PR #134 — pre-pilot calibration)

The four bugs below were uncovered by PR #134's pre-pilot calibration probe and fixed before the original MLPPPO pilot ran. Bug numbers in this section refer to PR #134's investigation chronology. **The three additional framework bugs surfaced by the LSTMPPO arm are documented under "Part 2 → Bug-investigation chain" further down**, with their own (independent) numbering.

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

Post-bug-fix pilot total: **~10 min** (4 seeds × 20 gens × 12 pop × ~35 episodes each at parallel=4). Per-seed wall sits in the 2-3 minute band — much faster than PR #134's pre-fix 27-min run because Bug 1's fix means episodes 1+ run on body=2 (configured) rather than body=6 (the agent's default), and body=2 episodes have far fewer steps to traverse the 20×20 grid.

**Implication for future pilots**: per-genome wall time still varies based on hyperparams sampled (deeper networks + smaller lr take more train iterations to converge), but the variance is much smaller post-fix because the env is consistent across all episodes. For wall-time-sensitive campaigns, consider:

1. Tighter bounds on `actor_hidden_dim` / `num_hidden_layers` to reduce variance.
2. Higher `parallel_workers` (the M2 pilot was bottlenecked by 4 cores; 8+ workers would amortise the variance).
3. A wall-time budget on individual genome evals (kill stragglers; mark them as fitness=0). Out of scope here but worth considering for any pilot with a non-saturated fitness landscape.

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

______________________________________________________________________

## Part 2 (post bug-fix re-run): MLPPPO + LSTMPPO+klinotaxis

This Part 2 covers the bug-fix investigation that started during the LSTMPPO arm and the re-runs that followed. The "Part 1" tables above were refreshed to the post-bug-fix re-run data in commit `d33ffa9c` (matching the archived CSVs at `artifacts/logbooks/012/m2_hyperparam_pilot/seed-{42-45}/history.csv`). The MLPPPO arm's gen-20 best ceiling holds at 1.000 both pre- and post-fix because MLPPPO+oracle is easy enough that the bugs didn't change the headline decision — but the population-mean trajectories differ noticeably (post-fix means sit in 0.91-0.98 vs pre-fix 0.79-0.87), which is why the tables needed refreshing.

### Bug-investigation chain

The LSTMPPO arm's first run scored mean **0.140 vs baseline 0.925 = −78.5pp**. A drafted STOP was retracted on the calibration probe described below.

**Probe protocol**: pre-train baseline LSTMPPO+klinotaxis brain to checkpoints at 50, 100, 200, 500 episodes; frozen-eval each at L=25; check if the baseline (which `run_simulation.py` reports at 92-93%) reaches non-trivial scores.

**Probe v1** (single seed throughout K+L, original framework):

| Pre-train | Frozen-eval | Successes/L |
|---|---|---|
| 50 ep | 0.000 | 0/25 |
| 100 ep | 0.000 | 0/25 |
| 200 ep | 0.000 | 0/25 |
| 500 ep | 0.040 | 1/25 |

**Probe v2** (per-episode `set_global_seed` + env-seed patching matching `run_simulation.py`'s pattern, but the framework's brain-construction bugs still in place):

| Pre-train | Frozen-eval | Successes/L |
|---|---|---|
| 50 ep | 0.040 | 1/25 |
| 100 ep | 0.000 | 0/25 |
| 200 ep | 0.000 | 0/25 |
| 500 ep | 0.000 | 0/25 |

The trajectory is *worse* with more training — catastrophic forgetting, but only because the brain was learning on a degenerate task (body=6 from episode 1 onwards because of Bug 1).

**Probe v3** (Bug 1 fix only — pass `max_body_length` to agent constructor):

| Pre-train | Frozen-eval | Successes/L |
|---|---|---|
| 50 ep | 0.120 | 3/25 |
| 100 ep | 0.040 | 1/25 |
| 200 ep | 0.000 | 0/25 |
| 500 ep | 0.000 | 0/25 |

Improvement at ep=50 but trajectory still wrong. Bug 2 (sensing-mode translation) was still in effect — the brain was running on `food_chemotaxis` (oracle) modules while env was klinotaxis-mode. Feature-dim mismatch silently breaking learning.

**Probe v4** (Bug 1 + Bug 2 + per-episode reseed all in place):

| Pre-train | Frozen-eval | Successes/L |
|---|---|---|
| 50 ep | **1.000** | **25/25** |
| 100 ep | **1.000** | **25/25** |
| 200 ep | **1.000** | **25/25** |
| 500 ep | **1.000** | **25/25** |

Perfect frozen-eval at every snapshot, proving the framework is mechanically correct after the three fixes. Wall-time also dropped to ~23 sec for 50 episodes (vs ~37 sec pre-Bug-1-fix), confirming body=2 is genuinely cheaper to step than body=6.

### Bug 1 detail: `_build_agent` missing `max_body_length`

```python
# packages/quantum-nematode/quantumnematode/evolution/fitness.py — pre-fix
def _build_agent(brain, env, sim_config):
    return QuantumNematodeAgent(
        brain=brain,
        env=env,
        # ... satiety_config, sensing_config ...
        # max_body_length NOT PASSED — defaults to 6
    )
```

Defaulting to 6 had no effect on episode 0 (the env was constructed separately with the configured body length). But `agent.reset_environment()` between episodes ([agent.py:1122]) rebuilt the env using `self.max_body_length`:

```python
# packages/quantum-nematode/quantumnematode/agent/agent.py:1111-1135
def reset_environment(self) -> None:
    self.env = DynamicForagingEnvironment(
        # ...
        max_body_length=self.max_body_length,  # ← 6, not 2
        # ...
    )
```

So episode 0 was on body=2 (configured); episodes 1+ were on body=6. Worm length 6 in a 20×20 grid is much harder — fewer valid manoeuvres, longer paths to food.

**Fix**: one-line change to pass `max_body_length=sim_config.body_length`.

**Regression test**: `test_build_agent_threads_max_body_length` ([test_fitness.py]).

### Bug 2 detail: `apply_sensing_mode` not invoked in evolution brain factory

`run_simulation.py` translates sensory module names based on env sensing mode:

```python
# scripts/run_simulation.py:381 (canonical pattern)
translated = apply_sensing_mode(original_modules, sensing_config)
brain_config = brain_config.model_copy(update={"sensory_modules": translated_modules})
```

For a klinotaxis env, `food_chemotaxis` translates to `food_chemotaxis_klinotaxis` (different feature extractor — uses `food_concentration`, `food_lateral_gradient`, `food_dconcentration_dt` instead of `food_gradient_strength` / `food_gradient_direction`). `validate_sensing_config` also auto-enables STAM for klinotaxis/derivative modes.

`instantiate_brain_from_sim_config` did not do this translation. So a klinotaxis env + `food_chemotaxis` brain produced a brain that:

1. Computed features using the oracle gradient extractor (looking at `food_gradient_*` fields on `BrainParams`).
2. Received `BrainParams` populated from a klinotaxis env (which fills `food_concentration`, `food_lateral_gradient`, `food_dconcentration_dt` instead).
3. Got feature vectors silently mismatched between brain expectation and env input.

The brain still ran end-to-end (no crash) — `BrainParams` fields default to None, the oracle extractor returns 0.0 for missing fields, the brain saw constant zero inputs and never learned anything.

**Fix**: call `validate_sensing_config(sim_config.environment.sensing)` + `apply_sensing_mode(...)` in `instantiate_brain_from_sim_config` BEFORE `setup_brain_model`. ~15 lines mirroring `run_simulation.py`'s pattern. Also added a None-guard for configs without an `environment.sensing` block.

**Regression tests**: `test_instantiate_brain_translates_klinotaxis_modules` (klinotaxis config gets translated + STAM appended) and `test_instantiate_brain_oracle_modules_unchanged` (oracle config stays as-is) — both in [test_brain_factory.py].

**Side effect**: this fix exposed an existing test (`test_lstmppo_encoder_roundtrip`) that depended on the bug — pre-fix, the LSTMPPO klinotaxis test was secretly running on oracle modules with feature_dim=10. Post-fix, the brain expects klinotaxis feature_dim=10 (with STAM auto-appended) but the test's hand-crafted `BrainParams` only populated oracle-style fields. Updated `_make_seeded_brain_params` to populate both oracle and klinotaxis fields plus a properly-sized `stam_state`, so the round-trip test exercises the new (correct) brain configuration.

### Bug 3 detail: Single seed across K+L episodes

`run_simulation.py`'s per-run loop reseeds globally:

```python
# scripts/run_simulation.py:610-625 (canonical pattern)
for run in range(total_runs_done, runs):
    run_seed = derive_run_seed(simulation_seed, run)
    set_global_seed(run_seed)
    # ... run the episode ...
    if run_num < runs:
        next_run_seed = derive_run_seed(simulation_seed, run_num)
        agent.env.seed = next_run_seed
        agent.env.rng = get_rng(next_run_seed)
        agent.reset_environment()
```

`LearnedPerformanceFitness.evaluate` and `EpisodicSuccessRate.evaluate` did neither. They called `agent.reset_environment()` between episodes (the M2 part-1 fix from commit `7795c6b2`), but `reset_environment` rebuilds the env from `self.env.seed` — *the original* env's seed. Every reset rebuilt the same layout. Brain saw zero env diversity across the K train episodes. Fitness was measuring "policy quality on this one specific layout", not "policy quality on a population of layouts" — which is what the simulation's 92-93% baseline measures.

**Fix**: per-episode `set_global_seed(derive_run_seed(seed, ep_idx))` plus `agent.env.seed = run_seed; agent.env.rng = get_rng(run_seed)` BEFORE `agent.reset_environment()`, in both train and eval loops. Eval phase uses an offset `seed + K` so eval layouts don't replay the last K train layouts.

**Regression test**: none dedicated; this is a behaviour fix that's hard to assert without re-running a multi-episode trajectory. The full evolution test suite (118 tests) catches it implicitly via integration tests.

### Re-run: per-seed history (post bug-fix)

The MLPPPO arm's re-run data is documented in the **Per-seed history** section above (Part 1) — the artefacts at `artifacts/logbooks/012/m2_hyperparam_pilot/` are the post-fix data, and the tables there reflect the re-run. This Part 2 section covers only the LSTMPPO+klinotaxis arm, which is new in this PR.

#### LSTMPPO+klinotaxis arm — Seed 42

| Gen | Best | Mean | Std |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 0.000 |
| 5 | 1.000 | 1.000 | 0.000 |
| 10 | 1.000 | 1.000 | 0.000 |
| 15 | 1.000 | 1.000 | 0.000 |
| 20 | 1.000 | 1.000 | 0.000 |

Mean fitness across 20 gens: **1.000**. Every single genome CMA-ES sampled across all 20 generations × 12 population = 240 evaluations produced a policy that scored exactly 25/25 on frozen eval. The schema is *trivially* saturated for this brain at K=50.

#### LSTMPPO+klinotaxis arm — Seed 43

| Gen | Best | Mean | Std |
|---|---|---|---|
| 1 | 1.000 | 0.997 | 0.011 |
| 5 | 1.000 | 1.000 | 0.000 |
| 10 | 1.000 | 1.000 | 0.000 |
| 15 | 1.000 | 0.997 | 0.011 |
| 20 | 1.000 | 1.000 | 0.000 |

Mean fitness across 20 gens: **0.999**. Two minor dips at gens 1 and 15 (one genome scored 24/25 instead of 25/25), but otherwise saturated.

### Wall-time

| Arm | Re-run total | PR #134 / pre-fix | Δ |
|---|---|---|---|
| MLPPPO pilot | ~10 min | ~27 min | ~3× faster |
| LSTMPPO+klinotaxis pilot | ~80 min | ~100 min | ~1.25× faster |

Both speedups are explained by Bug 1's body=2 vs body=6 fix (fewer steps to traverse the grid). The LSTMPPO arm's smaller speedup (relative to MLPPPO's) is because LSTMPPO's per-step cost dominates; MLPPPO's per-step is cheap so step-count savings show up more.

### Reproducing the re-run

```bash
# 1. MLPPPO baseline (4 seeds × 100 episodes via run_simulation.py)
scripts/campaigns/phase5_m2_hyperparam_baseline.sh

# 2. MLPPPO pilot (4 seeds × 20 gens × pop 12)
scripts/campaigns/phase5_m2_hyperparam_mlpppo.sh

# 3. LSTMPPO+klinotaxis baseline (2 seeds × 100 episodes)
scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_baseline.sh

# 4. LSTMPPO+klinotaxis pilot (2 seeds × 20 gens × pop 12)
scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis.sh

# 5. Aggregate each arm
uv run python scripts/campaigns/aggregate_m2_pilot.py \
    --pilot-root evolution_results/m2_hyperparam_mlpppo \
    --baseline-root evolution_results/m2_hyperparam_baseline \
    --output-dir evolution_results/m2_hyperparam_mlpppo_summary

uv run python scripts/campaigns/aggregate_m2_pilot.py \
    --pilot-root evolution_results/m2_hyperparam_lstmppo_klinotaxis \
    --baseline-root evolution_results/m2_hyperparam_lstmppo_klinotaxis_baseline \
    --output-dir evolution_results/m2_hyperparam_lstmppo_klinotaxis_summary \
    --seeds 42 43
```

Total wall-time: ~90 minutes for both arms on a 10-core macOS machine (was ~130 minutes pre-fix).
