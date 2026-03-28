---
name: nematode-run-experiments
description: Run parallel experiment groups with multiple seeds. Use when the user wants to launch evaluation experiments across different configurations.
metadata:
  author: nematode
  version: "1.0"
---

Run parallel experiment groups with multiple seeds for evaluation.

**Input**: Specify experiment configs (file paths or descriptions of what to test). Optionally specify number of seeds (default 4), episodes per run, and any other parameters.

**Constraints**

- **Max 16 concurrent sessions** (4 groups × 4 seeds) — machine handles this without degradation
- **Max 4 experiment groups** per round — keep comparisons manageable
- **4 seeds per group** (default: 42, 43, 44, 45) — sufficient for variance estimation
- **Temporary configs** go in `/tmp/` — permanent configs stay in `configs/examples/`

**Steps**

1. **Design the experiment matrix**

   Present a clear table to the user before launching:
   ```
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
       uv run ./scripts/run_simulation.py --log-level INFO --show-last-frame-only \
         --runs {EPISODES} --config /tmp/{cfg}.yml \
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
   - Update the permanent config in `configs/examples/` with the winning parameters
   - Update performance comments in the config header

**Tips for effective experiments**

- **One variable per experiment** where possible — isolates cause and effect
- **Always include a control/baseline** — either the current best config or oracle
- **Oracle baselines need fewer episodes** (1000 is usually sufficient) since MLP PPO converges fast
- **Temporal experiments need more episodes** than derivative (typically 2x or more)
- **Large environment experiments** (100×100) take significantly longer per episode than small (20×20)
- **Check episode counts against LR schedule** — ensure lr_decay_episodes covers the training duration
