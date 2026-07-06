# Real-worm behavioural-thermotaxis validation

## Why

The chemotaxis validation ([Logbook 035](../../../docs/experiments/logbooks/035-realworm-chemotaxis-validation.md),
archived `add-realworm-chemotaxis-validation`) anchored the continuous-2D substrate's **chemotaxis**
fidelity to real *C. elegans* data (klinokinesis + weathervane, both REPRODUCED). But it validated
**one modality only**. The substrate has three behaviours — foraging (chemotaxis), predator-evasion,
and **thermotaxis** — and thermotaxis is a MUST behaviour in the T7 ranking. Before Phase 6a closes
(and Phase 7, which is about publication / plasticity / transfer, won't naturally backfill it), a
**second real-worm behavioural-validation arm** broadens the claim from "the substrate reproduces
*C. elegans* chemotaxis" to "…across multiple gradient-navigation modalities" (tracker
`T7.validation.thermotaxis`).

This is a **non-gating enrichment**: Gate 3 G3.d is already satisfied by chemotaxis. It extends the
existing `realworm-behavioural-validation` capability rather than adding a new one.

Thermotaxis differs from chemotaxis in one important way that makes it a genuine test rather than a
copy: it is **homeostatic**. Real *C. elegans* navigates toward its cultivation temperature `Tc`
(migrates up-gradient when too cold, down-gradient when too hot) and biases turning/curving toward
`Tc` (Hedgecock & Russell 1975; Ryu & Samuel 2002; thermal klinotaxis — Clark et al. 2007; Luo et al.
2014). So the "drive" is a **setpoint error** toward `Tc`, not a monotonic gradient to climb.

## What Changes

- **Setpoint-drive behavioural capture (a new capture modality).** Add
  `SensingConfig.capture_behaviour_modality: food | thermotaxis` (default `food` → byte-identical to
  today). In `thermotaxis` mode the captured `BehaviourStep` drive/derivative/gradient fields carry
  the **homeostatic thermal drive** `−|T − Tc|` (0 at the setpoint, more negative further from
  comfort), its one-step derivative, and the **toward-comfort direction** (up the thermal gradient
  when too cold, opposite when too hot) sampled live from `env.get_temperature` /
  `env.get_temperature_gradient` / the thermotaxis cultivation temperature. This setpoint adjustment
  is exactly what lets the **existing** klinokinesis/weathervane bias-curve metrics + agreement
  grading + aggregation harness apply **unchanged** — the metrics see a "drive" and its derivative,
  agnostic to modality.
- **Thermotaxis literature reference set.** A `data/thermotaxis/behavioural_bias_signatures.json`
  (same four statistic keys as chemotaxis) encoding the documented thermal bias **directions** +
  citations (Ryu & Samuel 2002; thermal klinotaxis). Because thermal bias magnitudes are not cleanly
  comparable to our rad/mm units, all four are **sign-only** references (as the chemotaxis weathervane
  already is). `load_bias_signatures(modality=...)` selects the set; a caller-supplied missing path
  still raises.
- **Harness `--modality food|thermotaxis`.** Grade against the modality's reference set; the metrics,
  grading, bootstrap CI, floor, θ-sweep, and figures are otherwise the committed 035 code.
- **A thermotaxis-dominant continuous cell + panels.** A thermotaxis-seeking continuous config (the
  thermal analogue of the food-only klinotaxis cell) with capture on; MLP primary + connectome
  companion, n ≥ 8, post-convergence; report REPRODUCED / PARTIAL / ABSENT with bootstrap CIs.
- **Logbook + tracker.** Logbook 036; tick `T7.validation.thermotaxis`; feed the T9a synthesis's
  biological-validation section alongside 035.

## Impact

- **Modified capability**: `realworm-behavioural-validation` (adds the thermotaxis capture modality +
  the thermal reference + the harness modality selector). No new capability.
- **Code**: `agent/agent.py` (a modality-branch capture helper), `utils/config_loader.py` (the
  modality field), `validation/datasets.py` (modality-aware reference loading + thermal fallback),
  `scripts/analysis/behavioural_chemotaxis_validation.py` (the `--modality` flag). New
  `data/thermotaxis/behavioural_bias_signatures.json` + a thermotaxis config + Logbook 036.
- **Invariant preserved**: `capture_behaviour_modality` defaults to `food`, so every existing run and
  the 035 pipeline are byte-identical; the change is additive.
- **Non-gating**: Gate 3 G3.d stands on chemotaxis; this strengthens the 6a biological-validation
  story, it does not change the gate.
