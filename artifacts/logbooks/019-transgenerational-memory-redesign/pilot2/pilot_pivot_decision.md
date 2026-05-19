# Pilot pivot decision

Per design.md § D6, the pilot's outcome is classified against six pre-declared pivots:

**Pilot signal: substrate likely inert (D6 row 2).** `tei_on ≈ control` at F1+ — substrate carries no measurable signal.
Pivot: widen `bias_network.hidden_dim` 8→16 OR add features to `input_features` (e.g. `stam_state_mean`). Re-run pilot.

Observed metrics:

- `tei_on` per-arm verdict: **STOP**
- Cross-arm mean delta (tei_on - control): -49.17pp
- Per-arm gate pass (tei_on): False
- Wilcoxon p-threshold met: False

The full pivot table lives in design.md § D6.
