---
name: nematode-run-experiments
description: Run parallel experiment groups with multiple seeds. Use when the user wants to launch evaluation experiments across different configurations.
metadata:
  author: nematode
  version: '1.0'
---

Run parallel experiment groups with multiple seeds for evaluation.

**Input**: Specify experiment configs (file paths or descriptions of what to test). Optionally specify number of runs/episodes (controlled by `--runs`, default varies by task), number of seeds (default 4), and any other parameters.

**Constraints**

- **16 concurrent sessions** (4 groups × 4 seeds) is the throughput sweet spot — measured, not assumed. Going wider still finishes more work, it just costs wall-clock per session: 16 sessions run at ~1.8 sessions/s, 24 at ~1.77/s, 32 at ~1.68/s. So **24 is fine** when a round genuinely needs six groups or six seeds; past 32 the returns flatten.
- **CPU cores are the binding constraint, not memory.** A session peaks around 0.5 GB, so even 32 at once is a small fraction of what this machine has — never reduce a matrix out of memory worry. Reduce it because sessions are competing for cores.
- **Pin threads when running many sessions**: prefix with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Each session otherwise spawns a full BLAS thread pool and they oversubscribe each other. Worth ~6% throughput at 24 sessions, more the wider you go.
- **4 experiment groups** per round by default — the limit is interpretive, not machine capacity: more arms make a comparison harder to read, and the machine will happily run six
- **4 seeds per group** (default: 42, 43, 44, 45) — sufficient for variance estimation
- **Temporary configs** go in `/tmp/` — permanent configs stay in `configs/scenarios/`

**Steps**

1. **Design the experiment matrix**

   Present a clear table to the user before launching:

   ```markdown
   | Exp | Key Variable | Episodes | Purpose |
   |-----|--------------|----------|---------|
   | A   | ...          | ...      | ...     |
   | B   | ...          | ...      | ...     |
   ```

   Confirm with user before proceeding.

2. **Create temporary configs**

   Write experiment YAML configs to `/tmp/` (or a temp directory).

   - Copy the base config and modify only the experimental variable
   - Name clearly: `expA_descriptive_name.yml`
   - Environment sections should be identical between experiments (only brain/hyperparams differ) unless the experiment specifically tests environment changes

3. **Launch all sessions in parallel**

   Use a single background bash command with `&` for parallelism:

   ```bash
   for cfg in expA expB expC expD; do
     for seed in 42 43 44 45; do
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       uv run ./scripts/run_simulation.py --log-level INFO --show-last-frame-only \
         --runs {NUM_RUNS} --config /tmp/{cfg}.yml \
         --theme headless --track-experiment --seed $seed 2>&1 | tail -25 &
     done
   done
   echo "All sessions launched. Waiting..."
   wait
   echo "ALL COMPLETE"
   ```

   Launch as a background task so we can do other work while waiting.

4. **Analyse results when complete**

   Use the nematode-evaluate skill pattern to extract and compare results.
   Group by experiment, show per-seed and mean metrics.

5. **Update configs if improvements found**

   If an experiment outperforms the current best:

   - Update the permanent config in `configs/scenarios/` with the winning parameters
   - Update performance comments in the config header

**Tips for effective experiments**

- **One variable per experiment** where possible — isolates cause and effect
- **Always include a control/baseline** — either the current best config or oracle
- **Oracle baselines need fewer episodes** (1000 is usually sufficient) since MLP PPO converges fast
- **Temporal experiments need more episodes** than derivative (typically 2x or more)
- **Klinotaxis experiments** need similar episode counts to derivative — the lateral gradient signal accelerates learning compared to temporal-only
- **Large environment experiments** (100×100) take significantly longer per episode than small (20×20)
- **Check episode counts against LR schedule** — ensure lr_decay_episodes covers the training duration
