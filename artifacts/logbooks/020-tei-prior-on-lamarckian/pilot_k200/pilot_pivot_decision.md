# Pilot pivot decision

Per design.md § D6, the pilot's outcome is classified against six pre-declared pivots. Branch order is by specificity — catastrophic failure modes (rows 5, 6) match before positive- or null-delta variants.

**Pilot signal: clean GO at K_test (D6 row 2).** `tei_weights > weights_only` by ≥ 5pp — substrate prior accelerates Lamarckian retraining at K_test. The substrate-accelerates-retraining hypothesis is supported.
Pivot: NONE. Proceed to full campaign at K_test only (no K sweep — the test-point selection was made at calibration time).

Observed metrics:

- `tei_weights` per-arm verdict: **STOP**
- Cross-arm mean delta (tei_weights - weights_only): +5.33pp
- Per-arm gate pass (tei_weights): False
- Wilcoxon p-threshold met: False

The full pivot table lives in design.md § D6.
