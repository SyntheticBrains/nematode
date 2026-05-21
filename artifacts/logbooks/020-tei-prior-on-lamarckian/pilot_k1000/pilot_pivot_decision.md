# Pilot pivot decision

Per design.md § D6, the pilot's outcome is classified against six pre-declared pivots. Branch order is by specificity — catastrophic failure modes (rows 5, 6) match before positive- or null-delta variants.

**Pilot signal: substrate inert (D6 row 1).** `tei_weights ≈ weights_only` (|Δ| < 2pp) — the substrate prior carries no measurable signal under retraining. The composed mode collapses to pure Lamarckian; the substrate-accelerates-retraining hypothesis is falsified.
Pivot: STOP. Logbook 020 documents the null finding.

Observed metrics:

- `tei_weights` per-arm verdict: **PIVOT**
- Cross-arm mean delta (tei_weights - weights_only): +0.00pp
- Per-arm gate pass (tei_weights): False
- Wilcoxon p-threshold met: False

The full pivot table lives in design.md § D6.
