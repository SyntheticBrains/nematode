## Baldwin Retry Pilot — Summary (M4.5)

> **NOTE**: This is the literal aggregator output. The verdict text below
> reflects pre-registered Decision-6 STOP semantics. Logbook
> [015](../../../../docs/experiments/logbooks/015-baldwin-iterative-evaluation.md)
> reinterprets this run as **iteration step 1 of N** with a structural
> finding (Baldwin and Control evolved bit-identical genome populations
> because the framework's `inherited_from` is metadata-only). M4.6 is the
> planned follow-up; M5/M6 dependencies stay deferred. Read the logbook
> for the authoritative interpretation.

**Seeds**: [42, 43, 44, 45, 46, 47, 48, 49] (n=8)
**Hand-tuned baseline mean**: 0.170 (n=4 seeds 42-45 from M2.11)
**F1 K' (training budget)**: 25

### Schema-equalisation pre-flight check (audit A1 closure)

| Arm | First-gen mean best_fitness |
|-----|------------------------------|
| Baldwin | 0.7150 |
| Control | 0.7150 |
| **Abs delta** | **0.0000** (tolerance: 0.05) |
| **Status** | **PASS** |

### Decision Gates

- **Speed gate** (mean_gen_baldwin + 2 \<= mean_gen_control): FAIL

  - Baldwin mean gen-to-0.92: 6.38
  - Control mean gen-to-0.92: 6.38
  - Margin: +0.00 (need >= 2)

- **F1 learning-acceleration gate** (mean elite - mean baseline > 0.05, K' = 25): PASS

  - Baldwin elite mean (K'=25, L=25): 0.110
  - Schema-prior baseline mean (K'=25, L=25): 0.000
  - Signal delta mean: +0.110 (need > 0.05)

- **Comparative gate** (mean_gen_baldwin \<= mean_gen_lamarckian + 4): PASS

  - Baldwin mean gen-to-0.92: 6.38
  - Lamarckian mean gen-to-0.92: 2.75
  - Margin: +0.38 (need >= 0)

**Decision**: STOP ❌

The speed gate failed: Baldwin does not accelerate convergence over the from-scratch control by the required margin. Per the pre-registered STOP semantic (Decision 6): the Baldwin Effect is NOT exhibited on this testbed. M5 (co-evolution) proceeds without Baldwin in its substrate; M6 (transgenerational memory) uses Lamarckian. No further Baldwin pilot in this Phase.

### Per-seed convergence speed (generations to first reach best_fitness >= 0.92) + F1 learning-acceleration

| Seed | Baldwin | Lamarckian | Control | F1 elite | F1 baseline | F1 signal |
|------|---------|------------|---------|----------|-------------|-----------|
| 42 | 2 | 2 | 2 | 0.280 | 0.000 | +0.280 |
| 43 | 4 | 3 | 4 | 0.000 | 0.000 | +0.000 |
| 44 | 3 | 3 | 3 | 0.000 | 0.000 | +0.000 |
| 45 | 9 | 6 | 9 | 0.000 | 0.000 | +0.000 |
| 46 | 9 | 2 | 9 | 0.520 | 0.000 | +0.520 |
| 47 | 13 | 2 | 13 | 0.040 | 0.000 | +0.040 |
| 48 | 8 | 2 | 8 | 0.040 | 0.000 | +0.040 |
| 49 | 3 | 2 | 3 | 0.000 | 0.000 | +0.000 |
