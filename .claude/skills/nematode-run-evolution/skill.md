---
name: nematode-run-evolution
description: Launch evolutionary optimisation runs via scripts/run_evolution.py. Use when the user wants to evolve brain weights with CMA-ES or GA, evolve hyperparameters with CMA-ES or Optuna/TPE, or run evolution-framework smoke/pilot configs. Covers the timing pitfalls that crashed earlier sessions.
metadata:
  author: nematode
  version: '1.0'
---

Run brain-agnostic evolutionary optimisation via [scripts/run_evolution.py](../../scripts/run_evolution.py).

**Input**: A YAML config (typically under `configs/evolution/`) specifying `brain` and `evolution` blocks. Optionally specify `--generations`, `--population`, `--episodes`, `--seed`, `--parallel`, `--algorithm` (cmaes|ga|tpe), `--resume`, or `--output-dir` to override the YAML. TPE (Optuna's Tree-structured Parzen Estimator) is for hyperparameter evolution only — it requires the encoder to expose `genome_bounds`, which `HyperparameterEncoder` does but the weight encoders do not.

## Critical timing pitfalls (read first)

The Bash tool defaults to a 120 s timeout per command. Most evolution invocations are fast enough today, but two things can still blow past 120 s:

1. **`cma_diagonal: false` (the default) on a brain-weight genome**. CMA-ES `tell()` is O(n²); at LSTMPPO weight scale (~47k params) a single `tell()` takes ~3 minutes per generation, regardless of episode count. Symptom: the generation appears to hang after the last episode finishes. **Fix: set `cma_diagonal: true` in the YAML's `evolution:` block.** See the "When to enable cma_diagonal" section below for the rule.

2. **Long real campaigns**: even at sub-second per generation, 50 gens × pop 16 × 4 seeds × multiple brains is hours. Always background long campaigns.

Per-episode cost (post-perf fixes, both brains):

- MLPPPO + oracle/temporal: ~50 ms/episode at `max_steps: 500`.
- LSTMPPO + klinotaxis: ~100 ms/episode at `max_steps: 1000`.

Multiplier rule of thumb for total wall time: `generations × population × episodes × per-episode-cost / parallel_workers + per-generation optimiser overhead`. With `cma_diagonal: true`, the per-generation optimiser overhead is negligible. Without it, on weight-evolution genomes, the optimiser overhead dominates everything else.

## When to enable `cma_diagonal`

`cma_diagonal: true` switches CMA-ES to diagonal-only covariance (sep-CMA-ES). It's an `EvolutionConfig` field set in the YAML's `evolution:` block.

| Genome dim | Recommended `cma_diagonal` | Why |
|---|---|---|
| n < ~100 | **false** (default) | Full-cov is cheap and adapts to off-diagonal dependencies. Use this for hyperparameter evolution (e.g. small `HyperparameterEncoder` runs at n\<20). |
| n in ~100-1000 | **true** | Full-cov is borderline; diagonal is safer wall-clock and convergence is comparable for most fitness landscapes. |
| n > ~1000 | **true** (mandatory) | Full-cov is intractable: the n×n covariance matrix doesn't fit in memory and `tell()` takes minutes-to-hours per generation. Includes all neural-network weight evolution (MLPPPO ~9k, LSTMPPO ~47k). |

**Trade-off**: diagonal mode gives up off-diagonal covariance adaptation, so per-generation convergence is slower — typically 2-10× more generations to reach the same fitness on non-separable problems. But at large n, full-cov isn't a competing option (you can't run it long enough to find out), so net wall-clock to convergence is dramatically faster with diagonal anyway.

The shipped `configs/evolution/lstmppo_foraging_small_klinotaxis.yml` already sets `cma_diagonal: true`. The shipped `mlpppo_foraging_small.yml` does not (n≈9k would benefit from diagonal, but `false` is preserved for back-compat with M0 expectations). If you create a new pilot config that evolves brain weights at scale, set `cma_diagonal: true` in its `evolution:` block.

## Don't kill background processes by pattern matching

Earlier sessions ran `pgrep -f run_evolution | xargs kill` to clean up zombies and accidentally killed their own actively-running background invocation (because the PID it was filtering against hadn't started yet). Use the **task ID** the Bash tool returns when `run_in_background=true` to manage background runs, not pgrep.

## Steps

1. **Pick the config**

   - Smoke tests / framework verification: `configs/evolution/mlpppo_foraging_small.yml` (fast, small MLPPPO).
   - Headline biological brain: `configs/evolution/lstmppo_foraging_small_klinotaxis.yml` (also fast post-perf-fixes thanks to `cma_diagonal: true`).
   - For new pilot configs, copy from one of the above and edit only the brain hyperparams + the `evolution:` block. Set `cma_diagonal: true` if your genome dim is >~1000 (see the table above).

2. **Compute expected runtime**

   Use the multiplier from the timing-pitfalls section: `generations × population × episodes × per-episode-cost / parallel_workers`. Per-episode is ~50 ms (MLPPPO) or ~100 ms (LSTMPPO). Per-generation optimiser overhead is negligible with `cma_diagonal: true`.

   Example: LSTMPPO, 10 gen × pop 8 × 3 episodes, parallel 1 = 240 episodes × 0.1 s ≈ 24 s. With `--parallel 4` ≈ 6 s.

   Tell the user the estimate before launching anything that's expected to take more than a couple minutes.

3. **Launch (foreground or background)**

   Smoke runs (any brain):

   ```bash
   uv run python scripts/run_evolution.py \
     --config configs/evolution/mlpppo_foraging_small.yml \
     --generations 1 --population 4 --episodes 2 --seed 42 \
     --output-dir /tmp/smoke_$(date +%s)
   ```

   Real campaigns (always background since they may take hours regardless of brain):

   ```bash
   # Bash tool call: run_in_background=true
   uv run python scripts/run_evolution.py \
     --config configs/evolution/lstmppo_foraging_small_klinotaxis.yml \
     --generations 50 --population 16 --episodes 5 --parallel 4 --seed 42 \
     --output-dir evolution_results
   ```

   Resume a crashed/killed run:

   ```bash
   uv run python scripts/run_evolution.py \
     --config <same-config> \
     --generations <higher-number> \
     --resume evolution_results/<session>/checkpoint.pkl
   ```

4. **Verify artefacts**

   On completion, the per-session output dir contains four files:

   - `best_params.json` — top genome weights (round-trips back to a working brain via the encoder)
   - `history.csv` — per-generation fitness summary
   - `lineage.csv` — every fitness evaluation, with `parent_ids` listing every member of the previous generation
   - `checkpoint.pkl` — resume point (written every `checkpoint_every` generations)

   Quick sanity: `wc -l evolution_results/<session>/lineage.csv` should equal `1 + generations × population_size`.

5. **Look for warnings in the log**

   The script writes a per-session log under `logs/<session_id>.log`. Grep for `ERROR` or `WARNING`. Worker SIGINT-ignore is normal; `Traceback` is not.

## Constraints

- **Frozen weights only**: `EpisodicSuccessRate` runs episodes via `FrozenEvalRunner` which neuters `brain.learn` and `brain.update_memory`. The framework deliberately does not ship a learn-then-evaluate fitness in this version.
- **Only `mlpppo` and `lstmppo` brains** are registered in `ENCODER_REGISTRY`. Running the script against a quantum-brain config (e.g. `qvarcircuit`) will fail with a clear error listing the registered names.
- **`--parallel N` uses `multiprocessing.Pool`**: workers fork the parent process. Don't run with `--parallel > os.cpu_count()`. Workers ignore SIGINT — Ctrl-C the parent to stop everything cleanly.

## Tips

- For seeded reproducibility, always pass `--seed`; the per-evaluation seed is derived from this. Two runs with the same seed produce byte-identical lineage CSVs and best_params.
- The MLPPPO config uses `feature_gating: False`. There is no shipped fixture for `feature_gating: true` evolution.
- The smoke test [test_run_evolution_smoke_mlpppo](../../packages/quantum-nematode/tests/quantumnematode_tests/test_smoke.py) is the canonical "did I break the framework" check. Runs in ~4 s.
