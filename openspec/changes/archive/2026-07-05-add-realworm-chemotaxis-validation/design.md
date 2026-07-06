# Design — real-worm behavioural-chemotaxis validation

## Context

The continuous-2D klinotaxis substrate computes, per step, everything the two bias curves need — but
none of it is persisted. `result.path` is `list[GridPosition]` (integer-snapped, `runners.py:771`);
`heading_rad` / `pos_continuous` are transient `AgentState` fields (`env/env.py:667`); local food
concentration (`env.get_food_concentration`, `env.py:2288`), the directional gradient
(`env.get_separated_gradients`, `env.py:2453`), the lateral head-sweep gradient
(`_continuous_lateral_offsets`, `agent.py:840`), and dC/dt (STAM `compute_temporal_derivative`,
`stam.py:336`, surfaced as `food_dconcentration_dt`, `agent.py:973`) are all computed and discarded.
The existing validation (`validation/chemotaxis.py`) computes only the scalar CI; there is no
turn-rate / curving-rate / bearing / binning code anywhere in the repo. The pure-function
`validation/adaptive_sensor.py` (step-input transient + Weber invariance) is the style template.

## Goals

1. Reproduce, from the trained RL worm's own trajectory, the two documented *C. elegans* strategies:
   **klinokinesis** (turn-rate vs dC/dt) and **klinotaxis** (curving-rate vs bearing-to-gradient).
2. Report quantitative agreement vs the published signatures with bootstrap confidence intervals
   (Gate 3 G3.d).
3. Zero behaviour change to existing runs (byte-identical when the logging flag is off).

## Decisions

### D1 — Opt-in continuous behavioural-trajectory capture

Add a config-gated `capture_behaviour: bool` (default false) + a per-step
`BehaviourStep(step, x, y, heading_rad, concentration, dc_dt, grad_dx, grad_dy)`. Capture at the
**agent step** (`agent.py`, in `_create_brain_params` / `_build_temporal_result`) — the one place
where the continuous state **and** the sensing all flow together: `pos_continuous` / `heading_rad`,
`env.get_food_concentration(pos)`, the step's `food_dconcentration_dt` (STAM), and the **live gradient
direction** from `env.get_separated_gradients(pos)`. Capturing in the runner step loop (near
`runners.py:771`) is rejected: `dc_dt`, the concentration, and the gradient are computed inside the
agent step and are not in hand there.

**The gradient direction MUST be captured live, not recomputed post-hoc.** `get_separated_gradients`
reads the env's food field, which *mutates* during a foraging run (food eaten + respawned), so the
gradient at a logged position under the *final* field would be wrong — hence `grad_dx`/`grad_dy` are
logged per step (curve B's bearing is `wrap(atan2(grad_dy, grad_dx) - heading_rad)`).

Off by default → nothing is captured and every existing run is byte-identical (guarded by a test);
capture is read-only (append only), so it perturbs neither the RNG nor step order even when on. The
series rides on `SimulationResult` (a new optional `behaviour: list[BehaviourStep] | None`), flushed
per run like the other histories.

Rationale: capturing raw per-step state (not pre-binned metrics) keeps the metric layer a pure
function of a trajectory — testable in isolation, re-analysable without re-running, and honest about
what the model produced.

### D2 — Klinokinesis / klinotaxis decomposition

Per step compute the heading change `Δθ = wrap(heading_rad[t] - heading_rad[t-1])`. Classify:

- **Reorientation (pirouette / sharp turn)** when `|Δθ| > θ_sharp` — the klinokinesis event feeding
  curve A's turn-rate. `θ_sharp` is a substrate-calibrated threshold (separating the sharp-turn mode
  from the gradual-curving mode; calibrated in §6, reported).
- **Gradual curving** = the signed `Δθ` on non-reorientation steps — the klinotaxis weathervane
  feeding curve B, normalised per unit path length (deg/mm) so it is speed-independent.

This is the standard biased-random-walk-vs-weathervane split (Pierce-Shimomura; Iino & Yoshida both
isolate their strategy this way).

### D3 — The two bias curves + a shared binning helper

New `validation/behavioural_curves.py` (pure functions over a `list[BehaviourStep]`), with one shared
`rate_vs_binned_covariate(covariate, value, bins, *, kind)` helper (numpy `digitize`; no such helper
exists to reuse):

- **Curve A — turn-rate vs dC/dt.** Covariate = `dc_dt` (or its sign: up- vs down-gradient); value =
  reorientation indicator; per-bin **turn-rate** = reorientations / total steps in bin. Expect
  turn-rate **higher at dC/dt < 0** (down-gradient) than dC/dt > 0.
- **Curve B — curving-rate vs bearing.** Bearing = `wrap(atan2(grad_dy, grad_dx) - heading_rad)` from
  the per-step **logged** gradient direction (D1 — logged live, not recomputed, because the food field
  mutates); value = signed gradual `Δθ`; per-bin = **mean signed curving-rate**. Expect curving
  **toward** the gradient (positive toward-gradient sign) at off-axis bearings, zero at 0°/180°.

Per-bin **80% bootstrap CIs across seeds** reuse the committed
`weight_search_architecture_ranking` bootstrap style (paired resampling over the per-seed curves).

### D4 — Reference literature signatures (behaviour-level)

Add a `BiasCurveReference` dataclass + a small `data/chemotaxis/behavioural_bias_signatures.json`
holding, per strategy, the **documented bias direction + a reported magnitude range** with the
citation — NOT a pixel-digitised curve (out of reach and unnecessary for a behaviour-level claim):

- **Klinokinesis** (Pierce-Shimomura 1999): turn-rate is suppressed heading up-gradient — encode the
  **down/up turn-rate ratio** the paper reports (a range), sign = ratio > 1.
- **Klinotaxis** (Iino & Yoshida 2009): curving is biased toward the gradient — encode the
  **weathervane curving-rate magnitude** (deg/mm at off-axis bearing) as a range, sign = toward.

The exact numbers are pinned during authoring against the papers' reported summary statistics; where a
precise figure is unavailable the range is set conservatively and the source annotated. This honesty
is the point (matches the Phase-7 claim-discipline habit): the claim is **behaviour-level** — correct
strategy, correct sign, magnitude within the reported range — with model CIs, not a figure re-plot.

### D5 — Agreement statistic + report

Per curve, reduce the model curve to a single **bias statistic**: curve A → the down/up-gradient
turn-rate ratio; curve B → the toward-gradient weathervane slope (deg/mm per unit off-axis bearing).
Each with an 80% bootstrap CI across seeds. Verdict per curve (pre-registered):

- **REPRODUCED** — the model statistic's sign matches the literature and its CI overlaps the literature
  range.
- **PARTIAL** — sign matches but magnitude is out of range (weaker/stronger bias).
- **ABSENT** — sign does not match / CI spans the no-bias null.

Output: a `validation/behavioural_curves.json` summary + two figures in `report/continuous_figures.py`
(`plot_turn_rate_curve`, `plot_weathervane_curve`) — the model curve + CI band overlaid on the
literature reference band. A `scripts/analysis/behavioural_chemotaxis_validation.py` aggregates the
per-seed captures, computes the curves + statistics + verdicts, and writes the JSON.

### D6 — Subject, cell, and budget

Validate a **trained continuous-2D klinotaxis forager** on the food-only klinotaxis foraging cell (the
substrate whose chemotaxis fidelity G3.d anchors) — the **leading MLP arm** as the platform's best
forager (the cleanest "does our worm navigate like a real worm?" subject). n ≥ 8 seeds for the CIs,
`capture_behaviour: true`, evaluated post-convergence (the plateau, where the learned strategy is
stable). Optionally also the **connectome arm** as the biological-fidelity companion (does the
connectome-constrained worm reproduce the strategies?) — recorded as a secondary run, not gating.

## Risks / open questions

- **Reference-data precision.** The primary risk is the literature magnitude ranges (D4). Mitigation:
  scope the claim to sign + reported-range (behaviour-level), state the digitisation limit explicitly,
  and lead with the qualitative-strategy-reproduced result. A pixel-exact figure match is a
  non-goal.
- **Turn-threshold sensitivity.** `θ_sharp` (D2) partitions turns vs curving; a bad value could blur
  the two strategies. Mitigation: calibrate on the substrate in §6 (a bimodal `|Δθ|` histogram gives a
  natural cut), report the value, and check the verdicts are stable across a small threshold sweep.
- **Strategy dominance.** If the RL worm relies almost entirely on one strategy (e.g. pure weathervane
  from the lateral head-sweep sensing), one curve may be flat. That is itself an honest, reportable
  finding about which strategy the learned policy uses — not a failure of the pipeline.
