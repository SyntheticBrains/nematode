---
name: nematode-scratchpad
description: Update the evaluation scratchpad with latest experiment results. Use when the user wants to record findings from an evaluation round to the working scratchpad document.
metadata:
  author: nematode
  version: '1.0'
---

Update the evaluation scratchpad with latest experiment results.

**Input**: Specify which evaluation round/results to record. Optionally provide a scratchpad path (default: the active study scratchpad in `tmp/evaluations/`).

**Steps**

1. **Identify the scratchpad**

   Find the active scratchpad:

   ```bash
   ls tmp/evaluations/*/
   ```

   If multiple evaluations exist, ask which one. The scratchpad is typically named `*_scratchpad.md`.

2. **Read the current scratchpad end**

   Read the last section to understand:

   - What round number we're on (increment for new section)
   - What format/style the prior sections use
   - What the last recommendations were

3. **Collect the new results**

   Use the nematode-evaluate analysis pattern to extract metrics from the latest experiments. Compute:

   - Per-seed overall success rates
   - L100/L500/L1000 convergence metrics
   - Evasion rates (if applicable)
   - Death/starvation rates
   - Key learning curve observations

4. **Write the new section**

   Append a new section following the established format:

   ```markdown
   ---

   ## {Title} R{N}: {Description}

   ### Experiment Groups
   [Table of configs and key variables]

   ### Results
   [Summary table with per-seed and mean results]

   ### Key Findings
   [Numbered list of insights]

   ### Updated Configs
   [Which configs were updated based on results]
   ```

5. **Update configs if needed**

   If results warrant config changes:

   - Update the permanent config YAML
   - Update the performance comment in the config header
   - Note what changed in the scratchpad section

**Guardrails**

- Always include L100 for consistency with logbooks
- Number the round sections sequentially (R1, R2, R3...)
- Include per-seed detail — don't just report means
- Note if curves are still rising (suggests more training needed)
- Reference oracle baselines explicitly for comparison
- Keep each section self-contained — a reader should understand the round without reading prior rounds
