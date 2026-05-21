# M6.13 pilot artefacts — reading guide

These are verbatim captures of `aggregate_m613_pilot.py --mode pilot` outputs across the four M6.13 pilot runs. Files are **forensic snapshots, not narrative documents** — they reflect what the aggregator reported at the moment of each run, before any cross-pilot reconciliation.

## Directory layout

- `smoke/` — calibration smoke (1 seed × pop 6 × 2 gens × weights_only × K=1000) — established K_test=1000 and verified T1'/T2'/T3'/T4' tripwires before pilot dispatch.
- `pilot_k1000/` — initial pilot at K=1000 (all three arms aligned at K_test). Aggregator verdict: STOP (substrate inert).
- `pilot_k500/`, `pilot_k200/` — K-sensitivity sweep (tei_weights F0 held at K=2000, weights_only/control F0 at K=K_test). Aggregator pivot-row classification appeared to support a clean GO at K=200 (`+5.33pp`).
- `pilot_k200_f0matched/` — F0-confound disambiguation pilot (tei_weights F0 ALSO at K=200). Aggregator verdict: STOP, substrate INTERFERES (-9.33pp). **This is the load-bearing pilot for the M6.13 STOP verdict.**

## How to read `pilot_pivot_decision.md`

Each pivot-decision file contains **two independent reporting channels** that can disagree at low n:

1. **Pivot-row classification** (the headline "Pilot signal: ..." line) — purely a function of the cross-arm mean Δ against the design.md § D6 thresholds. At n=1 it ignores Wilcoxon/per-arm-gate signals because they're structurally uninformative.
2. **Observed metrics list** — the full set of GO checks (per-arm gate, Wilcoxon p, mean Δ ≥ 5pp, bootstrap CI excludes zero). All four MUST pass for a GO verdict; under n=1 at least Wilcoxon p=0.500 (uninformative) will always fail.

The K=500 and K=200 pivot files look like GO outcomes on channel 1 but show STOP signals on channel 2. **This is a feature, not a bug** — the aggregator faithfully reports both channels so the operator (or downstream reviewer) can see exactly which signal a pivot decision is keyed to. The user-review pause at task 6.5 is where the two channels get reconciled by the operator looking at the full pilot context.

For M6.13's actual verdict, the F0-confound disambiguation pilot (`pilot_k200_f0matched/`) supersedes the K-sweep pivot-row classifications: the K-sweep's apparent `+5.33pp` signal at K=200 inverted to `-9.33pp` under fair F0, classifying as **D6 row 4 (substrate INTERFERES)**. See [logbook 020](../../../docs/experiments/logbooks/020-tei-prior-on-lamarckian.md) § Pilot 4 + § Audit for the full reconciliation.

## Why these files weren't edited post-hoc

Forensic captures preserve the aggregator's reasoning at each step. Editing them to retroactively reflect the final STOP verdict would lose the audit trail showing how the F0 confound was discovered (K-sweep produced apparent GO → disambiguation flipped sign). The logbook 020 narrative reconciles the channels; these files preserve the raw signals.
