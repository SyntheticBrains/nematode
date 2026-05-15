# R4 — Per-generation re-aggregation

Re-runs the M5 verdict gate (cycling + escalation) on the per-generation
mean-fitness series from each session's `prey/lineage.csv` and
`predator/lineage.csv`, instead of the production aggregator's
`champion_history.json` K-block-elite series. The hypothesis is that
the gate didn't fire on pilot data because the K-block-elite series
(only 3 points/side at `generation_pairs=3`) was too short for the
metric's lag/window thresholds, AND because prey K-block-elites
saturate at the population's success-rate ceiling. Per-gen mean
is ~10x more samples and never saturates at 1.0 (population mean
stays slightly below the elite max).

Cycling lag range: 3-15 gens
Escalation gen window: 5-30 gens

## tmp/evaluations/coevolution/pr6_overnight_20260511T121704Z/x4/run/20260511_142906_75351f54

- Prey per-gen series: n=20, min=0.003, max=0.923, std=0.277
- Predator per-gen series: n=20, min=1.432, max=5.854, std=1.036
- Prey K-block-elite series: [0.84, 1.0]
- Predator K-block-elite series: [7.25, 6.666666666666667]

### R4 (per-generation series)

- Prey cycling_detected=False period=8 p=0.9940
- Predator cycling_detected=False period=14 p=0.9721
- Prey escalation_detected=True slope=+0.0389 sign=1 p=0.0000
- Predator escalation_detected=True slope=-0.1646 sign=-1 p=0.0000
- **Gate fires (per-gen): True**

### Production gate (K-block-elite series, for comparison)

- Prey cycling_detected=False period=None p=NaN
- Predator cycling_detected=False period=None p=NaN
- Prey escalation_detected=False slope=NaN sign=0 p=NaN
- Predator escalation_detected=False slope=NaN sign=0 p=NaN
- **Gate fires (K-block-elite): False**

## Summary

- Sessions analysed: 1
- Per-gen gate fires: 1/1
- K-block-elite gate fires: 0/1

**R4 hypothesis supported**: per-generation series fires more often than K-block-elite series — the production gate was bottlenecked by series length, not signal absence.
