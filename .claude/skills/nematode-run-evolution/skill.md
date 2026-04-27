---
name: nematode-run-evolution
description: Launch evolutionary optimisation runs via scripts/run_evolution.py. Use when the user wants to evolve brain weights with CMA-ES or GA, or when running evolution-framework smoke/pilot configs. Covers the timing pitfalls that crashed earlier sessions.
metadata:
  author: nematode
  version: '1.0'
---

Run brain-agnostic evolutionary optimisation via [scripts/run_evolution.py](../../scripts/run_evolution.py).

**Input**: A YAML config (typically under `configs/evolution/`) specifying `brain` and `evolution` blocks. Optionally specify `--generations`, `--population`, `--episodes`, `--seed`, `--parallel`, `--algorithm` (cmaes|ga), `--resume`, or `--output-dir` to override the YAML.

## Critical timing pitfalls (read first)

The Bash tool defaults to a 120 s timeout per command. Several evolution invocations exceed that and get SIGKILL'd (exit code 137):

- **MLPPPO + oracle/temporal sensing**: episodes are fast (~50 ms each). A `--generations 1 --population 4 --episodes 2` smoke completes in ~4–10 s. Safe to run in foreground with the default timeout.

- **LSTMPPO + klinotaxis** (the headline biological brain): each episode runs up to `max_steps: 1000` with a GRU forward pass per step, so a single episode is ~30–60 s. Even the *minimum* smoke (`--generations 1 --population 2 --episodes 1`) takes 1–3 minutes and WILL get SIGKILL'd at 120 s. Use one of:

  1. `run_in_background=true` on the Bash tool call (preferred — you get a notification on completion and can do other work).
  2. Explicit `timeout: 600000` (up to 10 min) on the Bash call.

- **Any non-trivial campaign** (e.g. `--generations 10 --population 8 --episodes 3` for either brain) takes minutes to hours — always background it.

Multiplier rule of thumb for total runtime: `generations × population × episodes × per-episode-cost / parallel_workers`.

## Don't kill background processes by pattern matching

Earlier sessions ran `pgrep -f run_evolution | xargs kill` to clean up zombies and accidentally killed their own actively-running background invocation (because the PID it was filtering against hadn't started yet). Use the **task ID** the Bash tool returns when `run_in_background=true` to manage background runs, not pgrep.

## Steps

1. **Pick the config**

   - Smoke tests / framework verification: `configs/evolution/mlpppo_foraging_small.yml` (fast, MLPPPO).
   - Headline biological brain: `configs/evolution/lstmppo_foraging_small_klinotaxis.yml` (slow, plan accordingly).
   - For new pilot configs, copy from one of the above and edit only the brain hyperparams + the `evolution:` block.

2. **Compute expected runtime**

   Example: LSTMPPO, 10 gen × pop 8 × 3 episodes, no parallel = 240 episodes × ~45 s = ~3 hours. With `--parallel 4` ≈ 45 min. Tell the user the estimate before launching.

3. **Launch (foreground or background)**

   Fast (MLPPPO smoke):

   ```bash
   uv run python scripts/run_evolution.py \
     --config configs/evolution/mlpppo_foraging_small.yml \
     --generations 1 --population 4 --episodes 2 --seed 42 \
     --output-dir /tmp/m0_smoke_$(date +%s)
   ```

   Slow (LSTMPPO or any real campaign) — always background:

   ```bash
   # Bash tool call: run_in_background=true
   uv run python scripts/run_evolution.py \
     --config configs/evolution/lstmppo_foraging_small_klinotaxis.yml \
     --generations 10 --population 8 --episodes 3 --parallel 4 --seed 42 \
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
   - `lineage.csv` — every fitness evaluation, with `parent_ids` per Decision 5a
   - `checkpoint.pkl` — resume point (written every `checkpoint_every` generations)

   Quick sanity: `wc -l evolution_results/<session>/lineage.csv` should equal `1 + generations × population_size`.

5. **Look for warnings in the log**

   The script writes a per-session log under `logs/<session_id>.log`. Grep for `ERROR` or `WARNING`. Worker SIGINT-ignore is normal; `Traceback` is not.

## Constraints

- **Frozen weights only** in M0: `EpisodicSuccessRate` runs episodes via `FrozenEvalRunner` which neuters `brain.learn` and `brain.update_memory`. Do NOT add `LearnedPerformanceFitness` or any learning-during-evaluation behaviour without checking that's the milestone you're on (M2 is where learn-then-eval lands).
- **Only `mlpppo` and `lstmppo` brains** are registered in `ENCODER_REGISTRY` as of M0. Quantum brain support (e.g. `qvarcircuit`) is deferred to a future Phase 6 re-evaluation — running the script against a quantum-brain config will fail with a clear error listing the registered names.
- **`--parallel N` uses `multiprocessing.Pool`**: workers fork the parent process. Don't run with `--parallel > os.cpu_count()`. Workers ignore SIGINT — Ctrl-C the parent to stop everything cleanly.

## Tips

- For seeded reproducibility, always pass `--seed`; the per-evaluation seed is derived from this. Two runs with the same seed produce byte-identical lineage CSVs and best_params.
- The MLPPPO config uses `feature_gating: False`; if you need to evolve a feature-gated MLPPPO, check if M2 has landed first (M0 doesn't ship a feature-gated fixture).
- The smoke test [test_run_evolution_smoke_mlpppo](../../packages/quantum-nematode/tests/quantumnematode_tests/test_smoke.py) is the canonical "did I break the framework" check. Runs in ~4 s.
