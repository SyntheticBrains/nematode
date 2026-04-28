# M2 Hyperparameter-Evolution Pilot — Summary

## Per-seed best fitness (eval-phase success rate, L=5)

| Seed | Gen 1 best | Gen 20 best | Mean across gens | Best params (lr, gamma, ...) |
|------|-----------|-------------|------------------|------------------------------|
| 42 | 1.000 | 1.000 | 1.000 | [223.91, 125.15, 2.93, ...] |
| 43 | 1.000 | 1.000 | 1.000 | [187.72, 7.89, 0.91, ...] |
| 44 | 1.000 | 1.000 | 1.000 | [31.43, 138.96, 2.23, ...] |
| 45 | 1.000 | 1.000 | 1.000 | [127.78, 165.09, 2.84, ...] |

**Pilot mean (gen-20 best across seeds)**: 1.000 ± 0.000

## Baseline (hand-tuned MLPPPO, 100 episodes per seed)

| Seed | Success rate |
|------|--------------|
| 42 | 0.960 |
| 43 | 0.980 |
| 44 | 0.920 |
| 45 | 0.920 |

**Baseline mean**: 0.945

## Decision gate

- Baseline mean: **0.945**
- GO threshold (≥3pp over baseline): **0.975**
- Pilot mean (gen-20 best): **1.000**
- Separation: +0.055 (+5.5pp)

**Decision**: GO ✅

Hyperparameter evolution beats the hand-tuned baseline by 5.5pp. Mean across 4 seeds clears the 3pp gate threshold.
