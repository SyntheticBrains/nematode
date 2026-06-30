# 030 supporting — bit-memory positive control forensics

Supporting analysis for [Logbook 030](../../030-bit-memory-positive-control.md). Committed
artefacts only (analysis outputs); the 40 raw per-run training `.out` files are uncommitted
evaluation forensics under `tmp/evaluations/bit-memory-prep/` (gitignored, per the 025/029
precedent).

## Files

- `separation.json` — the `bit_memory_separation.py` output: per-arm plateau-tail cue-match,
  the pairwise paired-seed Wilcoxon + 80% bootstrap CI + BH-FDR table, and the verdict.
- `cue-match-per-seed.csv` — `arch, seed, cue_match` (40 rows, 5 arms × 8 seeds).

## Calibration (§6.1 / §6.2)

- **Learnability pre-check (§6.1):** LSTM on an easy setting (delay 2, span 5), 1500 episodes.
  Cue-match rose 0.50 → **0.92** (final-quarter plateau), confirming the task + reward plumbing
  (cue injection, `sign(turn)` scoring, the `run_brain` reward timing, the per-run
  `reset_environment` env-recreation carry) genuinely teach working memory before the full panel.
- **Full setting (§6.2):** delay 8 (span 11, kept < the Transformer window 16 so the cue stays
  attendable), 20 trials/episode, 1500 episodes. All three memory arms plateau well within budget;
  the pre-registered 0.80 threshold is cleared by all three.

## n=4 → n=8 (a false null caught)

The n=4 read already showed the full separation in the means (memory arms 0.92–1.00 vs
MLP/connectome 0.50, deltas ≈ +0.5 with tight CIs), but the harness returned NULL: with 4 paired
seeds the one-sided Wilcoxon signed-rank **floor** p-value is 1/2⁴ = 0.0625, so the BH-FDR q can
never drop below 0.05 regardless of effect size (every pair came back q = 0.069). Scaling to n=8
(floor 1/2⁸ ≈ 0.004) made the overwhelming effect significant (q = 0.006). Lesson reused from the
025/029 protocol: n ≥ 8 is required for the paired-seed Wilcoxon to reach significance.

## Per-seed notes

- **LSTM seed-fragility:** seeds 1 and 7 plateau at 0.771 / 0.844 vs the other six at 0.96–1.00
  (mean 0.939). Consistent with the 029 continuous tanh-Gaussian seed-fragility; reported, not
  tuned away. It still beats the MLP at q = 0.006.
- **CfC / Transformer:** tight across seeds (0.982–0.999 / 0.973–0.981) — robust solvers.
- **MLP / connectome:** both pinned at 0.49–0.51 across all 8 seeds; their pairwise delta is
  +0.003 (q = 0.174, ns) — indistinguishable at chance.

## Reproduce

```
# 5 arms × 8 seeds × 1500 ep (parallel, OMP_NUM_THREADS=1), then the harness:
#   configs/scenarios/bit_memory/{arm}_small_bit_memory.yml  (delay 8, [cue, go_signal])
uv run python scripts/analysis/bit_memory_separation.py \
    --manifest <run-dir>/_manifest.txt --num-responses 20 --out separation.json
```
