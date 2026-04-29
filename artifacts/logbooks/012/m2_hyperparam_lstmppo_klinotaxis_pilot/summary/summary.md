# M2 Hyperparameter-Evolution Pilot — Summary

## Per-seed best fitness (frozen-eval success rate)

| Seed | Gen 1 best | Gen 20 best | Mean across gens | Best params (lr, gamma, ...) |
|------|-----------|-------------|------------------|------------------------------|
| 42 | 1.000 | 1.000 | 1.000 | [0.50, 71.92, -7.06, ...] |
| 43 | 1.000 | 1.000 | 1.000 | [1.26, 21.67, -11.71, ...] |

**Pilot mean (gen-20 best across seeds)**: 1.000 ± 0.000

## Baseline (hand-tuned brain, 100 episodes per seed)

| Seed | Success rate |
|------|--------------|
| 42 | 0.930 |
| 43 | 0.920 |

**Baseline mean**: 0.925

## Decision gate

- Baseline mean: **0.925**
- GO threshold (≥3pp over baseline): **0.955**
- Pilot mean (gen-20 best): **1.000**
- Separation: +0.075 (+7.5pp)

**Decision**: GO ✅

Hyperparameter evolution beats the hand-tuned baseline by 7.5pp. Mean across 2 seeds clears the 3pp gate threshold.
