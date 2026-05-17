# Transgenerational pilot aggregator — survival-rate decision-gate summary

Companion to [`summary.md`](summary.md) (which records the choice-index gate). This file records the survival-rate gate with F0 training-time fitness override applied — the metric configuration documented in logbook 018 § "Survival-rate gate with F0 training-time override".

## Arm: `tei_off` — verdict: **STOP**

| seed | F0 (trained) | F1 | F2 | F3 | F1≥40%xF0 | F2≥25%xF0 | F3≥15%xF0 | monotone | overall |
|------|-------------:|---:|---:|---:|:---------:|:---------:|:---------:|:--------:|:-------:|
| 42 | 0.462 | 0.120 | 0.040 | 0.000 | ✗ | ✗ | ✗ | ✓ | FAIL |
| 43 | 0.358 | 0.040 | 0.080 | 0.000 | ✗ | ✗ | ✗ | ✗ | FAIL |
| 44 | 0.291 | 0.040 | 0.000 | 0.000 | ✗ | ✗ | ✗ | ✓ | FAIL |
| 45 | 0.173 | 0.040 | 0.080 | 0.040 | ✗ | ✓ | ✓ | ✗ | FAIL |

0 of 4 seeds pass → STOP.

## Arm: `tei_on` — verdict: **PIVOT**

| seed | F0 (trained) | F1 | F2 | F3 | F1≥40%xF0 | F2≥25%xF0 | F3≥15%xF0 | monotone | overall |
|------|-------------:|---:|---:|---:|:---------:|:---------:|:---------:|:--------:|:-------:|
| 42 | 0.462 | 0.120 | 0.080 | 0.000 | ✗ | ✗ | ✗ | ✓ | FAIL |
| 43 | 0.358 | 0.000 | 0.080 | 0.040 | ✗ | ✗ | ✗ | ✗ | FAIL |
| 44 | 0.291 | 0.240 | 0.240 | 0.120 | ✓ | ✓ | ✓ | ✓ | **PASS** |
| 45 | 0.173 | 0.160 | 0.240 | 0.040 | ✓ | ✓ | ✓ | ✗ | FAIL |

1 of 4 seeds pass → PIVOT (per the M6 cross-seed verdict spec: GO ≥ 2, PIVOT = 1, STOP = 0).

## TEI-on vs TEI-off paired-arm survival retention

Mean survival_rate per generation (averaged across seeds):

| arm | F0 (trained) | F1 | F2 | F3 |
|------|-------------:|---:|---:|---:|
| tei_off | 0.321 | 0.060 | 0.050 | 0.010 |
| tei_on | 0.321 | 0.130 | 0.160 | 0.050 |

Cross-arm delta (TEI-on − TEI-off) at F1+:

| Gen | delta |
|----:|------:|
| F1 | +0.070 |
| F2 | +0.110 |
| F3 | +0.040 |

TEI-on outperforms TEI-off at every F1+ generation. See logbook 018 § Audit D for why this comparison answers a different question than the spec posed (non-symmetric F1+ compute envelope: TEI-on F1+ is fresh-random brain + logit_bias with 25 eval episodes; TEI-off F1+ is K=1000 fresh-trained brain with no inheritance).

## Notes

- **F0 column** is the training-time `LearnedPerformanceFitness` composite (`success_rate × (1 − death_rate)`) read from each arm/seed's `per_gen_elites.jsonl` via the aggregator's `--campaign-root` override flag. Without the override, the post-hoc evaluator decodes an UNTRAINED brain at F0 (since the F0 weights are GC'd by the substrate-extraction pipeline) and F0 reads ~0.080 across all seeds, trivially failing the monotone check. The override is the M6 commit-8 fix.
- **F1+ columns** are the post-hoc survival_rate = 1 − HEALTH_DEPLETED rate from `transgenerational_per_gen_eval.py`'s termination_reason tracking.
- **Source CSVs**: [`survival_decision_gate.csv`](survival_decision_gate.csv) (per-seed gate evaluation) and [`survival_retention_table.csv`](survival_retention_table.csv) (per-generation means).
