# Pilot pivot decision

Per design.md § D6, the pilot's outcome is classified against six pre-declared pivots. Branch order is by specificity — catastrophic failure modes (rows 5, 6) match before positive- or null-delta variants.

**Pilot signal: K-sensitivity pivot (D6 row 3).** `tei_weights > weights_only` by 2-5pp at K_test — the substrate signal exists but is K-dependent. Mapping the dose-response curve is required to decide GO/STOP.
Pivot: rerun pilot at K=500 AND K=1500 to map the dose-response. Decide GO/STOP at the K where Δ is largest. +6 wall-h cap per design.md § D6.

Observed metrics:

- `tei_weights` per-arm verdict: **STOP**
- Cross-arm mean delta (tei_weights - weights_only): +4.00pp
- Per-arm gate pass (tei_weights): False
- Wilcoxon p-threshold met: False

The full pivot table lives in design.md § D6.
