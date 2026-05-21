# TEI-as-prior-on-Lamarckian — aggregator summary

## Per-arm cross-seed verdicts

| arm | verdict |
|---|---|
| `tei_weights` | **STOP** |
| `weights_only` | **STOP** |
| `control` | **STOP** |

## Cross-arm primary verdict

**Verdict: STOP**

Per-check breakdown (GO requires ALL four):

- Per-arm gate (tei_weights ≥ 2/4 seeds pass): **False** (tei_weights arm verdict: STOP)
- Wilcoxon p < 0.1: **False** (p = 1.0000)
- Mean delta ≥ 5pp: **False** (mean = -9.33pp)
- 80% bootstrap CI excludes zero: **False** (CI = [-9.33pp, -9.33pp])

## Cross-arm pairwise statistics

| arm_a | arm_b | n | mean Δ | Wilcoxon p | 80% CI lo | 80% CI hi |
|---|---|--:|--:|--:|--:|--:|
| `tei_weights` | `weights_only` | 1 | -9.33pp | 1.0000 | -9.33pp | -9.33pp |
| `weights_only` | `control` | 1 | +4.00pp | 0.5000 | +4.00pp | +4.00pp |
| `tei_weights` | `control` | 1 | -5.33pp | 1.0000 | -5.33pp | -5.33pp |
