# M2 Hyperparameter-Evolution Pilot — Summary

## Per-seed best fitness (frozen-eval success rate)

| Seed | Gen 1 best | Gen 20 best | Mean across gens | Best params (lr, gamma, ...) |
|------|-----------|-------------|------------------|------------------------------|
| 42 | 0.760 | 1.000 | 0.882 | [1.09, 35.41, -7.11, ...] |
| 43 | 0.720 | 1.000 | 0.914 | [0.59, 78.20, -7.25, ...] |
| 44 | 0.520 | 0.920 | 0.874 | [0.66, 73.42, -6.91, ...] |
| 45 | 0.680 | 0.920 | 0.882 | [1.21, 35.55, -7.38, ...] |

**Pilot mean (gen-20 best across seeds)**: 0.960 ± 0.040

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
- Pilot mean (gen-20 best): **0.960**
- Separation: +0.790 (+79.0pp)

**Decision**: GO ✅

Hyperparameter evolution beats the hand-tuned baseline by 79.0pp.  Mean across 4 seeds clears the 3pp gate threshold.
