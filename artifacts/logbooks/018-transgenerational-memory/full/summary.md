# Transgenerational pilot aggregator — decision-gate summary

## Arm: `tei_off` — verdict: **STOP**

| seed | F0 | F1 | F2 | F3 | F1≥40%xF0 | F2≥25%xF0 | F3≥15%xF0 | monotone | overall |
|------|----|----|----|----|-----------|-----------|-----------|----------|---------|
| 42 | 0.933 | 0.929 | 0.908 | 0.927 | ✓ | ✓ | ✓ | ✗ | FAIL |
| 43 | 0.944 | 0.934 | 0.943 | 0.934 | ✓ | ✓ | ✓ | ✗ | FAIL |
| 44 | 0.927 | 0.911 | 0.904 | 0.912 | ✓ | ✓ | ✓ | ✗ | FAIL |
| 45 | 0.903 | 0.929 | 0.949 | 0.954 | ✓ | ✓ | ✓ | ✗ | FAIL |

## Arm: `tei_on` — verdict: **STOP**

| seed | F0 | F1 | F2 | F3 | F1≥40%xF0 | F2≥25%xF0 | F3≥15%xF0 | monotone | overall |
|------|----|----|----|----|-----------|-----------|-----------|----------|---------|
| 42 | 0.933 | 0.955 | 0.959 | 0.933 | ✓ | ✓ | ✓ | ✗ | FAIL |
| 43 | 0.944 | 0.909 | 0.924 | 0.941 | ✓ | ✓ | ✓ | ✗ | FAIL |
| 44 | 0.927 | 0.948 | 0.977 | 0.957 | ✓ | ✓ | ✓ | ✗ | FAIL |
| 45 | 0.903 | 0.973 | 0.950 | 0.945 | ✓ | ✓ | ✓ | ✗ | FAIL |

## TEI-on vs TEI-off paired-arm retention

Mean choice index per generation (averaged across seeds):

| arm | F0 | F1 | F2 | F3 |
|-----|----|----|----|----|
| tei_on | 0.927 | 0.946 | 0.952 | 0.944 |
| tei_off | 0.927 | 0.926 | 0.926 | 0.932 |

Substrate is the primary cross-arm difference identified — the pairing validator enforces `enabled=true ⇔ inheritance=transgenerational` and `enabled=false ⇔ inheritance=none` — though asymmetries in the F1+ compute envelope between TEI-on and TEI-off (see logbook 018 § Audit D — TEI-on F1+ is fresh-random brain + logit_bias with 25 eval episodes; TEI-off F1+ short-circuits the schedule and runs K=1000 fresh training every generation) mean other factors cannot be excluded. F1+ retention observed in `tei_on` versus `tei_off` is therefore consistent with substrate influence but not definitively attributable to the substrate alone.

## Gate thresholds

- F1 ≥ 40% x F0

- F2 ≥ 25% x F0

- F3 ≥ 15% x F0

- Monotone non-increasing: F0 ≥ F1 ≥ F2 ≥ F3

- **GO** iff ≥2 seeds pass; **PIVOT** iff exactly 1; **STOP** otherwise.
