## Baldwin Inheritance Pilot — Summary

**Seeds**: [42, 43, 44, 45]
**Hand-tuned baseline mean**: 0.170

### Decision Gates

- **Speed gate** (mean_gen_baldwin + 2 <= mean_gen_control): FAIL
  - Baldwin mean gen-to-0.92: 8.50
  - Control mean gen-to-0.92: 8.50
  - Margin: +0.00 (need >= 2)

- **Genetic-assimilation gate** (mean_f1_baldwin > mean_baseline + 0.1): FAIL
  - Baldwin F1 innate-only mean: 0.000
  - Hand-tuned baseline mean: 0.170
  - Margin: -0.170 (need > 0.1)

- **Comparative gate** (mean_gen_baldwin <= mean_gen_lamarckian + 4): PASS
  - Baldwin mean gen-to-0.92: 8.50
  - Lamarckian mean gen-to-0.92: 4.50
  - Margin: +0.00 (need >= 0)

**Decision**: STOP ❌

The speed gate failed: Baldwin does not accelerate convergence over the from-scratch control by the required margin.  Either the richer learnability schema offers no exploitable signal, or the fitness landscape is the wrong testbed.  Re-evaluate before committing to follow-up scope.

### Per-seed convergence speed (generations to first reach best_fitness >= 0.92) + F1 innate-only success rate

| Seed | Baldwin | Lamarckian | Control | F1 innate-only |
|------|---------|------------|---------|----------------|
| 42 | — | 3 | — | 0.000 |
| 43 | 8 | 4 | 5 | 0.000 |
| 44 | 7 | 4 | 5 | 0.000 |
| 45 | 3 | 7 | 8 | 0.000 |
