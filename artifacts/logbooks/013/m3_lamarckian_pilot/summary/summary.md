## M3 Lamarckian Inheritance Pilot — Summary

**Seeds**: [42, 43, 44, 45]
**Hand-tuned baseline mean**: 0.170

### Decision Gate

- **Speed gate** (mean_gen_lamarckian + 4 \<= mean_gen_control): PASS

  - Lamarckian mean gen-to-0.92: 4.5
  - Control mean gen-to-0.92: 9.8
  - Margin: +5.2 (need >= 4)

- **Floor gate** (mean_gen1_lamarckian >= mean_gen3_control): FAIL

  - Lamarckian gen-1 mean: 0.720
  - Control gen-3 mean: 0.840
  - Margin: -0.120

**Decision**: PIVOT ⚠️

Only the speed gate passed. Inheritance helps in one direction but not the other; treat M3 as inconclusive and review the per-seed trajectories before committing M4 scope.

### Per-seed convergence speed (generations to first reach best_fitness >= 0.92)

| Seed | Lamarckian | Control |
|------|------------|---------|
| 42 | 3 | — |
| 43 | 4 | 5 |
| 44 | 4 | 5 |
| 45 | 7 | 8 |
