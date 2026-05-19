# M6.9+ TEI re-evaluation — aggregator summary

> **Pilot mode caveat (n=1).** This summary was produced by
> `aggregate_m69_pilot.py --mode pilot` against a single-seed pilot.
> The cross-arm Wilcoxon p-value and bootstrap CIs below are
> mathematically **degenerate at n=1** (the cross-arm pairwise table
> shows `n=1` with `p=1.0` and CI lo==hi) and MUST NOT be used as
> definitive cross-arm decision evidence. The per-arm gate
> (`tei_on ≥ 2/4 seeds pass`) also cannot be satisfied at n=1.
>
> The load-bearing pilot artefact is `pilot_pivot_decision.md` in
> the same directory — it classifies the observed pattern against
> design.md § D6's six pre-declared pivots. The pilot's STOP-signal
> for the `tei_on` arm is driven by the per-arm gate (F1+ collapse
> below threshold within the single seed), not by the degenerate
> cross-arm statistics.

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
- Mean delta ≥ 5pp: **False** (mean = -49.00pp)
- 80% bootstrap CI excludes zero: **False** (CI = [-49.00pp, -49.00pp])

## Cross-arm pairwise statistics

| arm_a | arm_b | n | mean Δ | Wilcoxon p | 80% CI lo | 80% CI hi |
|---|---|--:|--:|--:|--:|--:|
| `tei_on` | `control` | 1 | -49.00pp | 1.0000 | -49.00pp | -49.00pp |
| `weights_only` | `control` | 1 | +17.50pp | 0.5000 | +17.50pp | +17.50pp |
| `tei_on` | `weights_only` | 1 | -66.50pp | 1.0000 | -66.50pp | -66.50pp |
