# M2 Hyperparameter-Evolution Pilot — Summary

## Per-seed best fitness (frozen-eval success rate)

| Seed | Gen 1 best | Gen 20 best | Mean across gens | Best params (lr, gamma, ...) |
|------|-----------|-------------|------------------|------------------------------|
| 42 | 0.480 | 0.720 | 0.752 | [-0.76, 95.04, -7.36, ...] |
| 43 | 0.000 | 0.000 | 0.000 | [1.26, 21.67, -11.71, ...] |
| 44 | 0.000 | 0.920 | 0.506 | [-5.13, 88.09, -1.76, ...] |
| 45 | 0.520 | 0.920 | 0.724 | [-1.28, 48.65, -6.07, ...] |

**Pilot mean (gen-20 best across seeds)**: 0.640 ± 0.378

## Baseline (hand-tuned brain, 100 episodes per seed)

| Seed | Success rate |
|------|--------------|
| 42 | 0.150 |
| 43 | 0.160 |
| 44 | 0.150 |
| 45 | 0.220 |

**Baseline mean**: 0.170

## Decision gate

- Baseline mean: **0.170**
- GO threshold (≥3pp over baseline): **0.200**
- Pilot mean (gen-20 best): **0.640**
- Separation: +0.470 (+47.0pp)

**Decision**: GO ✅

Hyperparameter evolution beats the hand-tuned baseline by 47.0pp. Mean across 4 seeds clears the 3pp gate threshold.
