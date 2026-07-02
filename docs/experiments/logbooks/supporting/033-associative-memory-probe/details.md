# 033 supporting — associative-memory probe forensics

Supporting analysis for [Logbook 033](../../033-associative-memory-probe.md). Committed
artefacts only (analysis outputs); the 48 raw per-run training `.out` files are uncommitted
evaluation forensics under `tmp/evaluations/t7-associative-memory/` (gitignored, per the
025/029/030 precedent).

## Files

- `separation.json` — the `associative_memory_separation.py` output: per-arm plateau-tail
  response accuracy, the reversal / non-reversal split, the pairwise paired-seed Wilcoxon + 80%
  bootstrap CI + BH-FDR table, and the verdict.
- `accuracy-per-seed.csv` — `arch, seed, overall, reversal, non_reversal` (48 rows, 6 arms × 8
  seeds; each value the final-quarter plateau mean).

## Calibration (§6.1 / §6.2)

- **Learnability pre-check (§6.1):** Transformer (the reliable detector) on two settings, 1500
  episodes each. On an **easy** config (no reversal, delay 4 — pure hold) response accuracy rose
  0.51 → **0.93**; on the **default** config (reversal_prob 0.5, delay 8 — the update demand) it
  rose 0.51 → **0.99** with the reversal-split at **0.98** (genuine update, not hold-only). This
  confirmed the task + reward plumbing (cue/outcome injection, `sign(turn)` scoring, the
  probabilistic reversal, the env-recreation carry) teach working-memory *update* before the full
  panel.
- **Full setting (§6.2):** the pre-check validated the defaults with no retune — 20 trials/episode,
  `cond_steps_per_cue` 1, `reversal_prob` 0.5, delay 8 (worst-case span 13, kept < the Transformer
  window 16), `response_steps` 1, per-arm entropy carried from the `bit_memory` configs. The
  pre-registered 0.80 update threshold is the harness `_SEPARATION_THRESHOLD`.

## The reversal split is what earns the "update vs hold" reading

At `reversal_prob = 0.5` a *hold-only* policy (retains the initial association, never overwrites) is
at chance overall, because ~half the trials flip the rewarded cue. So overall accuracy above chance
requires genuine working-memory **update**. The per-arm split makes the breakdown visible:

- **Genuine updaters** (reversal ≈ non-reversal): Transformer 0.984 / 0.995, CfC 0.920 / 0.908,
  LSTM 0.787 / 0.835.
- **Hold-biased** (non-reversal ≫ reversal): minGRU 0.732 / 0.989, minLSTM 0.711 / 0.994 — near
  perfect at *retaining* the initial association, markedly weaker at *overwriting* it.

The hold-bias is a **converged plateau**, not undertraining: windowing reversal accuracy over the
1500 runs, minGRU is flat at 0.73 across the final two eighths (0.73 → 0.73) and minLSTM ~0.72
(0.70 → 0.72, negligible residual), while CfC and the Transformer climb to 0.92 / 0.98. Plausibly
the memory-friendly **retention-gate init** the minimal-RNN arms carry ([Logbook
031](../../031-minimal-rnn-candidates.md)) — initialised to hold, they settle into a hold-heavy
basin — but this attribution is a hypothesis, not causally tested here (a minGRU-without-retention-init
ablation would confirm it; recorded as a follow-up).

## Per-seed notes

- **MLP** pinned at chance on *both* reversal and non-reversal (0.485–0.500 across all 8 seeds): with
  no cue in the observation at response time it can neither hold nor update — the structural
  memoryless anchor.
- **LSTM seed-fragility:** the widest spread (overall 0.616–0.926); seed 2 even reads *below* chance
  on reversal (0.488) while holding non-reversal (0.752). Consistent with the recurrent-PPO
  instability documented at 029/032; reported, not tuned away. It still clears 0.80 (0.811) and
  beats the MLP at q = 0.007.
- **CfC** strong (0.914) with one weak seed (0.657) pulling the mean; the other seven are 0.81–0.99.
- **Transformer** tightest across seeds (0.983–0.994) — the reliable solver.

## Verdict

**SEPARATION.** All five memory arms clear the 0.80 update threshold and beat the memoryless MLP at
BH-FDR q = 0.007. The Transformer beats every other arm (q = 0.007); the other four
(CfC / minGRU / minLSTM / LSTM) are a statistical cluster (pairwise ns, q 0.15–0.37).

## Reproduce

```shell
# 6 arms × 8 seeds × 1500 ep (parallel, OMP_NUM_THREADS=1), then the harness:
#   configs/scenarios/associative_memory/{arm}_small_associative_memory.yml
#     (reversal_prob 0.5, delay 8, channels [cue, outcome, go_signal])
uv run python scripts/analysis/associative_memory_separation.py \
    --manifest <run-dir>/_manifest.txt --num-responses 20 --out separation.json
```
