# 036: Real-Worm Behavioural-Thermotaxis Validation — Weathervane Reproduced, Klinokinesis Absent (T7 / Phase 6a)

**Status**: completed — **PARTIAL / behavioural-difference finding**. Extending the chemotaxis
validation ([Logbook 035](035-realworm-chemotaxis-validation.md)) to a second modality, the
continuous-2D RL nematode reproduces the *C. elegans* thermotaxis **weathervane** (gradual curving
toward the cultivation temperature `Tc`) — **weakly but robustly** (θ-free slope **+0.020 \[+0.010,
+0.031\]** at 95% CI, all 4 seeds positive), and a derivative-sensing **specificity control** confirms
it is **sensor-driven** (the slope collapses to **+0.000 [−0.013, +0.013]** with the head-sweep
removed). It does **not** reproduce thermal **klinokinesis** in *either* sensing mode (turn-rate
ratio ~1.0, 95% CI spans the null). So the substrate reproduces the **spatial-steering** component of
thermotaxis but not the biased-random-walk — a genuine behavioural gap explained by (a) the
continuous-Gaussian action head (state-independent std → no stochastic random walk) and (b) the RL
worm **migrating-and-parking** at `Tc` rather than isothermal-tracking (temperature is not consumable,
so thermotaxis converges to a static endpoint). **Non-gating** (Gate 3 G3.d stands on chemotaxis);
chemotaxis (035) remains the strong, complete validation of record, and this difference motivates
Phase 7.

**Branch**: `feat/realworm-thermotaxis-validation`.

**Date**: 2026-07-06.

**OpenSpec change**: `add-realworm-thermotaxis-validation` (adds a `thermotaxis` capture modality +
thermal reference set + harness `--modality` to the archived `realworm-behavioural-validation`
capability; byte-identical when `capture_behaviour_modality: food`, the default).

## Objective

Broaden the behaviour-level validation from chemotaxis to a second MUST behaviour (thermotaxis), so
the "high-fidelity substrate is biologically validated" claim spans modalities rather than resting on
chemotaxis alone. Thermotaxis differs in one important way: it is **homeostatic** — real *C. elegans*
navigates toward its cultivation temperature `Tc` (up-gradient when too cold, down when too hot) and
biases turning/curving toward `Tc` (Hedgecock & Russell 1975; Ryu & Samuel 2002; thermal klinotaxis —
Clark et al. 2007; Luo et al. 2014). So the drive is a **setpoint error**, not a monotonic gradient.

## Method

### Setpoint-drive capture (reuses the 035 machinery unchanged)

The 035 bias-curve metrics operate on a per-step scalar drive, its derivative, and a gradient
direction — agnostic to modality. For thermotaxis the capture logs the **setpoint-adjusted** signals
so the same metrics apply: drive `= −|T − Tc|` (0 at `Tc`, more negative further from comfort), its
one-step derivative, and the **toward-comfort** direction (up the live thermal gradient when too
cold, opposite when too hot). The capture math was verified exact against the temperature field. The
thermal literature reference (`data/thermotaxis/behavioural_bias_signatures.json`) is **sign-only**
for all four statistics (Luo et al. 2014 primary; thermal bias magnitudes are not comparable to the
model's rad/mm units) — as 035 already treated the weathervane.

### Cell — a faithful linear-gradient `Tc`-seeking assay

A thermotaxis-dominant continuous cell: a **linear** thermal gradient (base 16 °C at the spawn centre,
`Tc` 20 °C, 0.6 °C/mm), a narrow comfort band (`comfort_delta` 2 °C), a strong comfort reward /
discomfort penalty so the worm's motion is thermally steered, no thermal HP damage (survivable). The
linear gradient is the geometry of every reference assay. **A radial "comfort-spot" geometry was
tried and rejected** — it gave a much stronger, rich-bearing signal, but a point-source attractor is
*chemotaxis* geometry, and switching to it to rescue a weak number is assay-shopping. Two arms, MLP,
**n = 4 seeds, 300 episodes, post-convergence tail**:

- **klinotaxis cell** (`thermotaxis_mode: klinotaxis`, head-sweep) — measures the weathervane.
- **derivative control** (`thermotaxis_mode: derivative`, temporal dT/dt only, no head-sweep) — the
  direct analogue of 035's food-derivative arm; the specificity control for the weathervane, and the
  regime where klinokinesis would appear if the worm did a biased random walk.

The worm reaches `Tc` in both arms (klinotaxis near-comfort 62–86% of tail steps; derivative more
variable — one seed, 44, failed to converge at 0%).

## Results

### Verdict: weathervane weakly reproduced + sensor-driven; klinokinesis absent (n=4, θ_sharp = 0.45)

| statistic (n=4) | klinotaxis cell (head-sweep) | derivative control (no head-sweep) |
|---|---|---|
| klinokinesis, turn-rate ratio | 1.015 [0.996, 1.034] | 0.959 [0.910, 1.007] |
| klinokinesis, magnitude ratio | 1.006 [0.999, 1.013] | 0.983 [0.962, 1.004] |
| weathervane slope *(thresholded)* | **+0.011 [+0.007, +0.015]** | +0.000 [−0.002, +0.003] |
| weathervane slope *(θ-free)* | **+0.020 [+0.013, +0.031]** | +0.000 [−0.013, +0.013] |
| **combined**: klinokinesis / weathervane | PARTIAL / **PRESENT** | ABSENT / *(collapsed)* |

*(80% CIs shown for the model statistics; the weathervane also excludes the null at **95%** — see
robustness.)*

**Weathervane — reproduced (weakly) and sensor-driven.** With the head-sweep the slope is small but
**significantly positive** — all four seeds positive [0.028, 0.004, 0.015, 0.033], the 95% CI
[+0.010, +0.031] excludes zero, and it is stable across θ_sharp (0.30–0.45), tail window (50/150),
and leave-one-out (every LOO mean positive — no single seed drives it). It is **~5× weaker than
chemotaxis's +0.09**. Removing the head-sweep (derivative control) collapses it to **+0.000**
[−0.013, +0.013], and leave-one-out confirms the collapse is *not* an artifact of the one failed seed
— so the specificity control establishes the thermal weathervane as a genuine, sensor-driven signal,
not a geometry confound.

**Klinokinesis — absent in both sensing modes.** Turn-rate and magnitude ratios sit at ~1.0 with 95%
CIs spanning the null in *both* arms. Unlike 035 — where the food-derivative control *elicited*
klinokinesis (1.56) — the thermal-derivative control does not (0.96, if anything below the null). The
RL worm does not reproduce the thermal biased-random-walk at all.

## Analysis

**The substrate reproduces the spatial-steering half of thermotaxis, not the biased-random-walk
half.** Two principled reasons, reported rather than engineered around:

1. **The continuous-Gaussian action head steers; it does not random-walk.** The MLP-PPO continuous
   head has a *state-independent* learnable log-std, so the worm cannot modulate its turn *randomness*
   by state — true stochastic klinokinesis (a biased random walk) is architecturally out of reach. The
   weathervane (deterministic mean-turn toward the sensed gradient) is exactly what this head *can*
   do, so it is the component that reproduces. (This also sharpens 035: its "klinokinesis" is a
   deterministic reorient-when-worse, not the stochastic mechanism — a shared limitation.)
2. **The RL worm migrates-and-parks; it does not isothermal-track.** Real *C. elegans* keeps moving
   along the `Tc` isotherm; our worm is rewarded for *being* in comfort, so once there it dwells
   (~94% of steps near-stationary at the target). Temperature is not consumable — unlike food, which
   drives the continuous chemotaxis that made 035 strong — so thermotaxis converges to a static
   endpoint and the fine-grained bias lives only in the brief migration phase, which is why the
   weathervane is weak and klinokinesis (which needs sustained off-`Tc` reorientation) never appears.

This is a behavioural **difference** between the RL worm and the animal, and a genuine finding: the
substrate + continuous-control policy capture *directed thermal steering* but not the stochastic
biased-random-walk or isothermal-tracking repertoire. It parallels the memory-axis finding (Logbook
032): the comparison resolves what the task and policy *demand*, and thermotaxis-as-implemented
demands directed steering, not klinokinesis.

## Limitations

- **Direction-only, and one modality-component only.** The thermal reference is sign-only (units not
  comparable); only the weathervane is reproduced, weakly. No quantitative slope match is claimed.
- **n = 4, one failed derivative seed (44).** A light confirmation (this is a non-gating
  behavioural-difference finding, not a gate), not the n≥8 two-architecture panel 035/029 use. The
  weathervane result is robust across seeds/θ/tail/CI regardless.
- **Migrate-and-park, not isothermal-tracking** — the RL worm does not reproduce the continuous
  isothermal-tracking behaviour; the signal is migration-phase.
- **Metric hardening surfaced here.** The curving-rate floor (`suggest_min_path_len`) collapsed when
  the worm parks (median stride → 0), letting the `dθ/ds` creep artifact through (an unfloored smoke
  read +12.8, ~99% artifact); fixed to fall back to the moving-stride scale, with 035 preserved
  byte-identical (its foraging worm moves continuously). This is why the audit-before-panel mattered.

## Conclusions

- **Thermotaxis weathervane reproduced (weakly) + sensor-driven; klinokinesis absent.** A partial,
  honest second-modality result — the substrate captures directed thermal steering, not the
  biased-random-walk or isothermal-tracking repertoire.
- **Chemotaxis (035) remains the strong, complete validation of record** (both strategies, strong,
  n=8 × 3 arms). Gate 3 G3.d is unchanged; this is enrichment.
- **Motivates Phase 7.** The behavioural gap (park-not-track; steer-not-random-walk, rooted in the
  state-independent-std head) is a concrete faithful-behaviour target for the L4 / faithful-dynamics
  work.

## Supporting artefacts

`supporting/036-realworm-thermotaxis-validation/`:

- `klinotaxis-curves.json`, `derivative-control-curves.json` — per-arm statistics + per-seed values +
  verdicts.
- `klinotaxis-turn-rate-vs-dcdt.png` — the (flat) klinokinesis curve; `klinotaxis-weathervane.png` —
  the weak positive toward-`Tc` weathervane.

Configs: `configs/scenarios/thermal_foraging/mlpppo_small_continuous2d_thermotaxis_seeking_{klinotaxis,derivative}.yml`.
Reproduce: run with `capture_behaviour_modality: thermotaxis`, then
`scripts/analysis/behavioural_chemotaxis_validation.py --manifest <seed file> --tail-runs 100 --modality thermotaxis --out <curves.json> --theta-sharp 0.45`.
