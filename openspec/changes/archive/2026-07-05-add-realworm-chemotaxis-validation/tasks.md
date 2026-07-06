# Tasks — real-worm behavioural-chemotaxis validation

## 1. Opt-in continuous behavioural-trajectory capture

- [x] 1.1 Add a config-gated `capture_behaviour: bool = False` and a
  `BehaviourStep(step, x, y, heading_rad, concentration, dc_dt, grad_dx, grad_dy)` record. Capture at
  the **agent step** (`agent.py`, `_create_brain_params` / `_build_temporal_result`) where position,
  heading, `env.get_food_concentration(pos)`, the step's `food_dconcentration_dt`, and the **live**
  `env.get_separated_gradients(pos)` direction are all in hand — NOT the runner loop (those sensing
  values aren't available there). Log the gradient direction live (the food field mutates). Ride the
  series on `SimulationResult` (optional `behaviour`), flushed per run.
- [x] 1.2 Byte-identical test: with capture off (default), the run result + `result.path` are identical
  to the pre-change behaviour (no behavioural record, no RNG/step-order perturbation).
- [x] 1.3 Capture test: with capture on, a short continuous-2D run yields one `BehaviourStep` per step
  with finite position/heading/concentration/dC-dt.

## 2. Bias-curve metrics + binning helper

- [x] 2.1 New `validation/behavioural_curves.py` (pure functions over `list[BehaviourStep]`): a shared
  `rate_vs_binned_covariate(covariate, value, bins, *, kind)` (numpy `digitize`); the
  klinokinesis/klinotaxis split (sharp reorientation `|Δθ| > θ_sharp` vs gradual signed `Δθ`/mm);
  **curve A** turn-rate vs dC/dt; **curve B** mean signed curving-rate vs bearing-to-gradient (bearing
  = `wrap(atan2(grad_dy, grad_dx) - heading_rad)` from the logged per-step gradient direction).
- [x] 2.2 Tests: on synthetic trajectories (a hand-built down-gradient-turns worm; a curve-toward-gradient
  worm) the metrics recover the expected bias sign; wrap-around Δθ handled; empty/one-step safe.

## 3. Reference literature signatures

- [x] 3.1 Add a `BiasCurveReference` dataclass + `data/chemotaxis/behavioural_bias_signatures.json`:
  per strategy, the documented bias direction + a reported magnitude range + citation
  (Pierce-Shimomura 1999 down/up turn-rate ratio; Iino & Yoshida 2009 weathervane slope). Pin the
  numbers against the papers' reported summary stats; annotate any conservatively-set range.
- [x] 3.2 Tests: the reference loads, has both strategies with sign + range + citation, and the
  no-bias null is representable.

## 4. Agreement statistic + report + figures

- [x] 4.1 Reduce each model curve to its bias statistic (down/up turn-rate ratio; weathervane slope)
  with an 80% bootstrap CI across seeds; per-curve verdict REPRODUCED / PARTIAL / ABSENT vs the
  reference (§3). Write `behavioural_curves.json`. (Grading + serialisable `AgreementResult` in
  `validation/behavioural_agreement.py`; the harness §5 writes the JSON.)
- [x] 4.2 Add `plot_turn_rate_curve` + `plot_weathervane_curve` to `report/continuous_figures.py`
  (model curve + across-seed CI band + a null line + the reduced-statistic/verdict/citation
  annotation); headless-safe.
- [x] 4.3 Tests: verdict logic (REPRODUCED/PARTIAL/ABSENT) on synthetic statistics; figures write
  headless.

## 5. Aggregation harness

- [x] 5.1 `scripts/analysis/behavioural_chemotaxis_validation.py` — load a `<seed> <behaviour-file>`
  manifest (or the per-seed run captures), compute both curves + statistics + verdicts + the summary
  JSON, print the two-curve agreement table. Reuse the committed bootstrap layer. (Run persistence:
  `report/behaviour_export.py` writes `behaviour_capture.json` when `capture_behaviour` is on.)
- [x] 5.2 Tests: manifest parse + end-to-end verdict on synthetic captures (mirror the associative /
  connectome-controls harness tests).

## 6. Calibration / smoke (before the panel)

- [x] 6.1 Single-seed smoke: run a trained klinotaxis MLP forager with `capture_behaviour: true` on the
  food-only klinotaxis cell; calibrate `θ_sharp` from the `|Δθ|` histogram (natural bimodal cut);
  confirm the pipeline produces both curves and record which strategy the worm shows. *Seed 42, 300
  episodes → 99.3% success. Caught + fixed a capture bug (the food-gradient keys are popped under
  derivative sensing → zero weathervane bearing). Finding: the `|Δθ|` distribution SATURATES at the
  `max_turn_rad` bound (no natural bimodal cut) → thresholded statistics are `θ_sharp`-sensitive
  (weathervane slope flips sign across the range). Klinokinesis PRESENT, weathervane ABSENT.*
- [x] 6.2 **PAUSE for user review of the smoke** (which strategies appear + the calibrated threshold)
  before the full panel. *Reviewed; decision: "report both metrics" — added `θ_sharp`-independent
  threshold-free companions (`klinokinesis_magnitude_ratio`, `weathervane_slope_all`) + a
  per-strategy combined verdict reconciling the two families. Reconciled smoke: klinokinesis
  PRESENT_PARTIAL (both families agree), weathervane ABSENT (both slopes negative).*

## 7. Evaluation + verdict

- [x] 7.1 Panel: the trained klinotaxis forager (leading MLP arm), `capture_behaviour: true`, **n ≥ 8
  seeds**, post-convergence, headless. Plus the connectome companion (architecture-robustness) and a
  derivative-mode specificity control (spatial head-sweep removed). *All n=8, converged 91-100%.*
- [x] 7.2 Run the harness (§5); record both bias curves, the statistics + CIs, and the per-curve
  verdicts. Robustness checked across `θ_sharp`, tail-window, curving-rate floor fraction, and CI level
  (80%→95%). *Klinokinesis + weathervane both PRESENT (MLP + connectome); control shows the weathervane
  collapses 87-93% (thresholded slope non-significant at 95%) while klinokinesis persists/strengthens —
  a double dissociation. A creep-step curving-rate floor was required (pre-floor weathervane was a
  spurious ABSENT outlier artifact).*
- [x] 7.3 **PAUSE for user review of the evaluation + verdict before writing the logbook** (project
  convention). *Reviewed; framing agreed: weathervane direction-only (not magnitude); klinokinesis
  led by the θ-independent magnitude ratio; specificity by effect-size collapse (label is
  magnitude-blind); connectome = architecture-robust, not native biological dynamics; scope =
  chemotaxis-only on the calibrated substrate; dissociation-led narrative.*

## 8. Logbook + tracker

- [x] 8.1 Write the logbook (objective / method / results / analysis / limitations, incl. the
  behaviour-level reference caveat) + committed supporting artefacts (no `tmp/` references); it feeds
  the 6a synthesis (T9a) and Gate 3 G3.d. \*[Logbook 035](../../../docs/experiments/logbooks/035-realworm-chemotaxis-validation.md)
  - `supporting/035-realworm-chemotaxis-validation/` (3 per-arm JSONs + 4 figures).\*
- [x] 8.2 Add the logbook row to `docs/experiments/README.md`.
- [x] 8.3 Tick `T7.validation.1/2/3` in `openspec/changes/phase6-tracking/tasks.md` with the verdict;
  note the Gate-3 G3.d contribution.
- [x] 8.4 Document the `capture_behaviour` flag / the validation entry point in `AGENTS.md`.

## 9. Pre-merge gates

- [x] 9.1 Targeted `pre-commit` during iteration; full `pre-commit run -a` before push. *Full suite green.*
- [x] 9.2 `openspec validate add-realworm-chemotaxis-validation --strict`. *Valid.*
- [x] 9.3 Full `uv run pytest -m "not nightly"` green (byte-identical capture-off invariant holds).
  *4123 passed, 1 skipped, 2 xfailed.*
- [x] 9.4 Archive the change in-PR (`openspec archive add-realworm-chemotaxis-validation -y`). *Archived
  as `2026-07-05-add-realworm-chemotaxis-validation`; `realworm-behavioural-validation` spec created.*
