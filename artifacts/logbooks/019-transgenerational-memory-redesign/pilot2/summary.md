# M6.9+ TEI re-evaluation — aggregator summary

## Per-arm cross-seed verdicts

| arm | verdict |
|---|---|
| `tei_on` | **STOP** |
| `weights_only` | **PIVOT** |
| `control` | **STOP** |

## Cross-arm primary verdict

**Verdict: STOP**

Per-check breakdown (GO requires ALL four):

- Per-arm gate (tei_on ≥ 2/4 seeds pass): **False** (tei_on arm verdict: STOP)
- Wilcoxon p < 0.1: **False** (p = 1.0000)
- Mean delta ≥ 5pp: **False** (mean = -49.17pp)
- 80% bootstrap CI excludes zero: **False** (CI = [-49.17pp, -49.17pp])

## Cross-arm pairwise statistics

| arm_a | arm_b | n | mean Δ | Wilcoxon p | 80% CI lo | 80% CI hi |
|---|---|--:|--:|--:|--:|--:|
| `tei_on` | `control` | 1 | -49.17pp | 1.0000 | -49.17pp | -49.17pp |
| `weights_only` | `control` | 1 | +17.50pp | 0.5000 | +17.50pp | +17.50pp |
| `tei_on` | `weights_only` | 1 | -66.67pp | 1.0000 | -66.67pp | -66.67pp |
