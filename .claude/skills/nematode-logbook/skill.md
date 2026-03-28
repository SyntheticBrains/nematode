---
name: nematode-logbook
description: Create or update an experiment logbook with artifacts, supporting data, and roadmap updates. Use when the user wants to document evaluation results permanently.
metadata:
  author: nematode
  version: "1.0"
---

Create or update an experiment logbook with artifacts and documentation.

**Input**: Specify whether creating a new logbook or updating an existing one. Optionally provide a logbook number, title, or angle/hypothesis.

**Steps**

1. **Determine logbook number and scope**

   If creating new:

   - Check existing logbooks: `ls docs/experiments/logbooks/*.md`
   - Assign the next number (e.g., if 009 exists, use 010)
   - Discuss angle/hypothesis with user if not provided

   If updating existing:

   - Identify which logbook to update
   - Determine what new data to add

2. **Collect experiment artifacts**

   For each experiment config to include:

   a. **Find all sessions** matching the config in `exports/`:

   ```python
   # Group by config name, find all 4 seeds
   ```

   b. **Create artifact directory**: `artifacts/logbooks/{NNN}/{brain}_{environment}/`

   c. **Copy per session**:

   - Experiment JSON from `experiments/{SESSION_ID}/{SESSION_ID}.json`
   - Config YAML from `experiments/{SESSION_ID}/*.yml`

   d. **Copy best seed weights**:

   - Find the session with highest success rate
   - Copy `exports/{SESSION_ID}/weights/final.pt` to `artifacts/logbooks/{NNN}/{dir}/weights/final.pt`

   Each artifact directory should contain: N session JSONs + 1 config YAML + 1 weights/final.pt

3. **Create/update the main logbook**

   File: `docs/experiments/logbooks/{NNN}-{title}.md`

   Follow the template structure from existing logbooks (see `docs/experiments/templates/experiment.md`):

   - **Objective**: What question is being answered
   - **Background**: Context and prior work
   - **Hypothesis**: Testable predictions with expected ranges
   - **Method**: Architecture, environments, key configuration, code changes
   - **Results**: Summary tables (L100 as primary metric for consistency with logbooks 007-009), overall success rates, key findings
   - **Analysis**: Hypothesis outcomes, root cause explanations, comparisons
   - **Conclusions**: Numbered key takeaways
   - **Next Steps**: Actionable items
   - **Data References**: Artifact locations, config files, link to supporting doc

4. **Create/update the supporting appendix**

   File: `docs/experiments/logbooks/supporting/{NNN}/{title}-details.md`

   Include:

   - Per-seed results tables (overall and L100)
   - Learning curve analysis
   - Ablation comparisons
   - Hyperparameter tables
   - Any data too detailed for the main logbook

5. **Update the experiment index**

   Add entry to `docs/experiments/README.md` in the Active Experiments table.

6. **Update the roadmap** (if findings affect project direction)

   Check `docs/roadmap.md` for:

   - Phase status updates
   - Exit criteria that can be checked off
   - Quantum checkpoint assessments
   - Go/no-go decisions

7. **Verify**

   - All artifact directories have sessions + config + weights
   - All markdown links resolve
   - Numbers match between logbook summary and appendix detail
   - L100 metrics are included for consistency with prior logbooks

**Guardrails**

- Always include L100 (last 100 episodes) as the primary convergence metric
- Always copy best-seed weights alongside session data
- Artifact configs are historical records — don't modify them after copying
- Keep the main logbook concise — detailed per-seed data goes in the appendix
- Reference existing logbooks (007, 008) for style consistency
