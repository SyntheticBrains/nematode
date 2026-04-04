---
name: nematode-evaluate
description: Analyse experiment results from simulation sessions. Use when the user wants to evaluate how experiments performed, diagnose issues, or decide next steps after running experiments.
metadata:
  author: nematode
  version: '1.0'
---

Analyse experiment results from completed simulation sessions.

**Input**: Optionally specify experiment session IDs, config names, or a description of which experiments to analyse. If omitted, find the most recent experiments.

**Data Locations**

- Experiment metadata: `experiments/{SESSION_ID}/{SESSION_ID}.json` + `experiments/{SESSION_ID}/*.yml`
- Per-run CSV data: `exports/{SESSION_ID}/session/data/simulation_results.csv`
- Trained weights: `exports/{SESSION_ID}/weights/final.pt`
- Logs: `logs/simulation_{TIMESTAMP}_{HASH}.log`

**Analysis Steps**

1. **Identify experiments to analyse**

   If session IDs or config names are provided, use them. Otherwise, find recent experiments:

   ```bash
   ls -t exports/ | head -20
   ```

   Group sessions by config name. Show the user what was found and confirm scope.

2. **Extract summary metrics per session**

   For each session, read the CSV and compute:

   - Total episodes, success rate, avg foods collected, avg steps
   - Death rate (health_depleted), starvation rate
   - Evasion rate (if predator_encounters column exists)
   - L100 (last 100 episodes success rate — primary convergence metric)
   - L500/L1000 if enough episodes exist

3. **Group and compare**

   Display results grouped by config name, showing all seeds and mean:

   ```text
   config_name (4 seeds, N episodes):
     Seed 1: succ=X% food=Y L100=Z%
     Seed 2: ...
     MEAN:   ...
   ```

4. **Learning curve analysis**

   For the best and worst seeds, show 500-1000 episode windows to identify:

   - Is performance still improving? (warrants longer training)
   - Did a collapse/regression occur? (LR/entropy schedule issue)
   - Is there a breakthrough transition? (plateau then sudden jump)
   - High variance between seeds? (suggests training instability)

5. **Diagnose issues** (assess each as applicable to the environment)

   - **Foraging**: Is the agent finding food? Check avg foods, starvation rate, distance efficiency
   - **Predator evasion**: Check death rate, evasion rate, survival steps. Is the agent dying early (before learning) or late (after partial learning)?
   - **Thermotaxis**: Check for temperature-related deaths, comfort zone navigation
   - **Aerotaxis**: Check for oxygen-related deaths, comfort zone (5-12% O2) navigation
   - **Training dynamics**: Is entropy collapsing too fast? Is LR decaying before convergence? Check the schedule against the learning curve inflection point.

6. **Provide recommendations**

   Based on the analysis, recommend the most promising next steps:

   - Parameter changes (LR schedule, entropy, chunk length, buffer size)
   - Architectural changes (if the current approach hits a structural ceiling)
   - More training (if curves are still rising)
   - Environment tuning (gradient steepness, HP/damage balance)
   - Prioritise recommendations by expected impact and effort

**Output Format**

Show a summary table first, then detailed analysis, then recommendations. Be concise — lead with numbers, not narrative. Use tables for comparisons.

**Guardrails**

- Always compute L100 for consistency with prior logbooks (007, 008, 009)
- When comparing to oracle, state the oracle numbers explicitly
- Flag any seeds that failed to converge or collapsed
- If the experiment CSV doesn't have predator/thermotaxis/aerotaxis columns, skip those analyses — don't assume all experiments have all objectives
