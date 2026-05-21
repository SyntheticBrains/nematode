# Pilot pivot decision

Per design.md § D6, the pilot's outcome is classified against six pre-declared pivots. Branch order is by specificity — catastrophic failure modes (rows 5, 6) match before positive- or null-delta variants.

**Pilot signal: substrate INTERFERES with Lamarckian (D6 row 4).** `tei_weights < weights_only` by > 2pp — the substrate prior actively HURTS Lamarckian retraining rather than accelerating it. Likely the prior misleads early F1+ exploration before inherited weights take over.
Pivot: STOP. Recommends substrate-policy alignment as a future-work direction (the prior must be calibrated for the warm-start child's policy, not just the F0 elite's).

Observed metrics:

- `tei_weights` per-arm verdict: **STOP**
- Cross-arm mean delta (tei_weights - weights_only): -9.33pp
- Per-arm gate pass (tei_weights): False
- Wilcoxon p-threshold met: False

The full pivot table lives in design.md § D6.
