# 035: Real-Worm Behavioural-Chemotaxis Validation — Klinokinesis + Weathervane (T7 / Phase 6a Gate 3 G3.d)

**Status**: completed — **BOTH strategies reproduced**. The continuous-2D RL nematode reproduces both
documented *C. elegans* gradient-navigation strategies at the behaviour level: **klinokinesis**
(biased random walk — reorientations enriched heading down-gradient, Pierce-Shimomura et al. 1999) and
**klinotaxis weathervane** (trajectory curves toward the gradient, Iino & Yoshida 2009). Both are
PRESENT across n=8 seeds on the gating **MLP** arm and the **connectome** companion (architecture-
robust). A derivative-mode **specificity control** (spatial head-sweep removed) delivers a **double
dissociation**: klinokinesis persists (in fact *strengthens*) while the weathervane **collapses
87–93%** to a small geometric residual — so the weathervane is a genuine sensor-driven strategy, not a
foraging-geometry artifact. This is the behaviour-level real-worm validation Gate 3 G3.d requires.

**Branch**: `openspec/add-realworm-chemotaxis-validation`.

**Date**: 2026-07-06.

**OpenSpec change**: `add-realworm-chemotaxis-validation` (opt-in behavioural-trajectory capture +
bias-curve metrics + behaviour-level literature references + agreement grading + aggregation harness;
committed §1–§7, byte-identical when `capture_behaviour` is off).

## Objective

Gate 3 G3.d (a MUST for Phase-6 exit) asks for a **behaviour-level** validation of the RL worm's
chemotaxis against published *C. elegans* data: does the worm's own navigation output reproduce the
documented gradient-climbing strategies, using public literature and no neuron-identity mapping? Real
*C. elegans* climbs chemical gradients with two dissociable strategies:

- **Klinokinesis** (a biased random walk): the pirouette/large-reorientation rate is **elevated when
  the worm heads down-gradient** (dC/dt < 0) — Pierce-Shimomura, Morse & Lockery (1999).
- **Klinotaxis / weathervane**: between reorientations the worm **gradually curves its trajectory
  toward the gradient** — Iino & Yoshida (2009).

The question is whether the RL worm — a continuous-2D forager with klinotaxis head-sweep sensing, a
tanh-Gaussian (speed, turn) action head, PPO-trained — exhibits either or both, and how quantitatively.

## Background — the behaviour-level reference discipline

This is validated at the **behaviour level** (the worm's own kinematics), not by matching internal
neuron dynamics. The literature reference is therefore encoded as **documented bias direction + a
reported magnitude range + citation**, NOT a pixel-digitization of the original figures (a deliberate
non-goal — the published assays differ in substrate, units, and analysis from this simulator). Each
model bias statistic is graded REPRODUCED / PARTIAL / ABSENT against that reference with an 80%
bootstrap CI across seeds. This matches the project's claim-discipline (Phase-7 framing) and the
Beiran & Litwin-Kumar 2025 degeneracy stance used elsewhere in T7: state what the behaviour shows,
scope it honestly, do not over-fit a citable figure.

## Method

### Metrics (committed §2 / §4 / §6, pure functions over the captured trajectory)

Each captured step logs continuous position, heading, food concentration, its temporal derivative
dC/dt, and — sampled **live** from the environment (the food field mutates during foraging) — the
**true** food-gradient direction/strength as the ground-truth bearing reference. Per-transition
kinematics give a wrapped heading change `dtheta`, a bearing `wrap(grad_dir − heading)`, and a signed
curving-rate `dtheta / path_len`. Two strategies, each measured **two ways** for robustness:

- **Klinokinesis** — (thresholded) down/up-gradient **turn-rate ratio** (fraction of sharp
  reorientations `|dtheta| > θ_sharp` heading down- vs up-gradient); (threshold-free) down/up
  **turn-magnitude ratio** (mean `|dtheta|`, no threshold). > 1 = biased random walk.
- **Weathervane** — (thresholded) slope of gradual (non-turn) curving-rate vs bearing; (threshold-
  free) the same slope over **all** usable steps. > 0 = curves toward the gradient.

**Why two families per strategy.** The single-seed smoke found the `|dtheta|` distribution
**saturates** at the per-step turn bound (`max_turn_rad = 0.5`): ~55% of transitions pile up at the
ceiling, so the sharp/gradual split has **no natural bimodal cut** and the thresholded statistics'
*magnitude* is θ_sharp-sensitive (the thresholded weathervane slope even flips sign across the θ
range). The **threshold-free** companions are θ_sharp-independent and anchor the directional claim;
the per-strategy verdict reconciles the two families (a robust call where they agree). References for
the companions are **sign-only** (the magnitude units are not literature-comparable).

**Curving-rate floor (committed §7).** The connectome companion exposed that the curving-rate
`dtheta / path_len` blows up on near-stationary creep/dwell steps — a handful of high-leverage outliers
(one connectome seed's raw slope was **−33**) drove the pooled weathervane slope wildly negative and
produced a *spurious ABSENT*. A scale-free floor (drop steps below 0.25× the median stride — ~6% of
steps) removes the creep tail; the weathervane signal is then uniformly positive and stable across
floor fractions 0.05–0.5. **This floor was essential** — without it the weathervane reads as a
measurement artifact, not its true value.

### Behavioural capture (committed §1, byte-identical when off)

A config-gated `capture_behaviour` (default `false`; a byte-identical no-op when off) records the
per-step series on the agent, flushed to `behaviour_capture.json` per run. The gradient direction is
snapshotted **live** at the sensing step before the derivative-sensing pipeline pops those keys — a
capture bug the smoke caught (the weathervane bearing was logging zero under klinotaxis sensing).

### Panels

Three arms, each **n = 8 seeds (42–49), 300 episodes, headless, parallelised**, on the calibrated
food-only klinotaxis foraging cell (Fick gradient geometry + adaptive/biphasic chemosensor,
`capture_radius 2.0`, `ε 0.1`). Bias statistics computed on the **post-convergence tail** (last 100
episodes); θ_sharp fixed at **0.45** (0.9× the turn bound, a stable sub-ceiling cut).

| arm | brain | role | mean success (n=8) |
|-----|-------|------|-----|
| **MLP** | `mlpppo`, `chemotaxis_mode: klinotaxis` | **gating** | 99.3–100% |
| **connectome** | `connectomeppo`, `chemotaxis_mode: klinotaxis` | architecture-robustness companion | 90.7–100% |
| **control** | `mlpppo`, `chemotaxis_mode: derivative` (no spatial head-sweep) | weathervane specificity control | 98.7–100% |

## Results

### Verdict: BOTH strategies reproduced (MLP + connectome), weathervane specificity confirmed

Bias statistics (post-convergence tail, θ_sharp = 0.45, curving-rate floor 0.25× median stride;
80% bootstrap CI):

| statistic (n=8) | MLP (gating) | connectome (companion) | control (no head-sweep) |
|---|---|---|---|
| klinokinesis, turn-**magnitude** ratio *(θ-free, primary)* | **1.099** [1.087, 1.110] | **1.041** [1.036, 1.046] | **1.169** [1.156, 1.182] |
| klinokinesis, turn-**rate** ratio *(thresholded)* | 1.30 [1.26, 1.34] | 1.15 [1.13, 1.17] | 1.56 [1.51, 1.62] |
| weathervane slope *(thresholded)* | **+0.027** [+0.022, +0.031] | **+0.027** [+0.021, +0.033] | +0.002 [+0.000, +0.004] |
| weathervane slope *(θ-free)* | **+0.091** [+0.079, +0.103] | **+0.065** [+0.053, +0.078] | +0.012 [+0.010, +0.014] |
| **combined**: klinokinesis / weathervane | PRESENT_PARTIAL / **PRESENT** | PRESENT_PARTIAL / **PRESENT** | PRESENT / *(residual)* |

**Klinokinesis — reproduced (directional, robust).** The θ-independent magnitude ratio is
significantly > 1 on every arm and every seed (MLP 1.10, connectome 1.04) — reorientations are larger
heading down-gradient, as documented. The direction is the robust claim; it survives at 95% CI on all
arms. The thresholded turn-*rate* ratio corroborates but is threshold-sensitive in magnitude: at the
broad cut it is ~1.3 (MLP), rising to **2.78 [2.53, 3.03]** — squarely inside the Pierce-Shimomura
~2× (≈[1.5, 3.0]) range — for the sharpest reorientations near the turn ceiling. So the klinokinesis
*match to the literature magnitude* holds specifically for sharp reorientations; the broad-population
enrichment is directional.

**Weathervane — reproduced in direction (not magnitude).** Both slope families are significantly
positive on the MLP and connectome arms — the trajectory curves *toward* the gradient, as Iino &
Yoshida document. This is a **sign-only** reproduction: our rad/mm-per-bearing units are not comparable
to the paper's deg/mm-per-normalized-gradient parameterization, so **no quantitative slope match is
claimed** — only the documented direction. The positive slope is robust at 95% CI (both metrics, both
architectures) and stable across the θ, tail-window, and floor-fraction sweeps.

### The double dissociation (specificity control)

Removing the spatial klinotaxis head-sweep (`chemotaxis_mode: derivative` — the worm keeps dC/dt but
loses the left/right lateral-gradient signal) dissociates the two strategies cleanly:

- **Klinokinesis persists — in fact strengthens.** Magnitude ratio 1.10 → **1.17**, rate ratio 1.30 →
  **1.56**. With no spatial sensor the worm leans *harder* on the temporal-derivative strategy.
  Confirms klinokinesis is the dC/dt strategy, independent of the spatial channel.
- **Weathervane collapses 87–93%.** θ-free slope +0.091 → **+0.012**; thresholded +0.027 → **+0.002**
  (the thresholded control slope is **non-significant at 95% CI**, [−0.001, +0.004]). The small +0.012
  θ-free residual is an honest, minor *geometric* coupling (a klinokinesis worm's path curves slightly
  toward the gradient on average) — an order of magnitude below the sensor-driven signal.

So the weathervane in the sensing arms is **predominantly sensor-driven**, not a foraging-geometry
confound: remove the input the strategy needs and the behaviour largely vanishes, while the other
strategy is untouched. (Note: the control's weathervane *verdict label* is still nominally REPRODUCED
because the sign-only grader is magnitude-blind — the specificity evidence is the **effect-size
collapse** and the loss of 95% significance, not the label.)

### Robustness

The MLP verdict is unchanged across: θ_sharp 0.30–0.50 (directional klinokinesis + weathervane hold;
only the thresholded magnitudes move); tail window 50/100/150; curving-rate floor fraction 0.05–0.5;
and CI level 80% → 95% (the strong effects all exclude the null at 95%). The single-seed smoke was
directionally consistent but noisier (n=1 magnitudes off; one seed a weathervane outlier) — the n≥8
panel is the bar, as elsewhere in T7.

## Analysis

**The RL nematode reproduces both documented klinotaxis strategies of real *C. elegans*.** On the
gating MLP arm, klinokinesis (biased-random-walk turn enrichment down-gradient) and the klinotaxis
weathervane (curving toward the gradient) are both present, robust across n=8 seeds and across every
robustness axis. The connectome companion shows the **same** pattern, so the reproduction is
architecture-robust — it is a property of the task + sensing + reward substrate, not of one policy
class.

**The specificity control makes the weathervane claim a controlled result, not an assertion.** The
double dissociation — klinokinesis survives (strengthens under) sensor removal, the weathervane
collapses — is cleaner evidence than either curve alone: each strategy behaves exactly as its
mechanism predicts. The weathervane is genuinely driven by the spatial head-sweep the biology uses,
with only a small geometric baseline.

**What this does not claim.** (1) The weathervane is reproduced in **direction only** — no quantitative
slope match to Iino & Yoshida (incomparable units). (2) The klinokinesis literature-magnitude match
(~2×) holds for **sharp reorientations** at a specific threshold; the broad-population claim is
directional. (3) The **connectome** arm is connectome-*topology*-constrained with PPO-trained weights
— it evidences *architecture-robustness*, **not** that the biological connectome natively produces
these dynamics (consistent with the [034](034-connectome-structure-controls.md) degree-statistics
finding). (4) Scope: this validates **chemotaxis navigation strategies** on the **calibrated** food-
only substrate; it is not a validation of the full multi-objective behaviour, and a different sensor
calibration could shift the klinokinesis/weathervane balance.

## Conclusions

- **Gate 3 G3.d — satisfied.** The RL worm's own behavioural output reproduces both documented *C.
  elegans* gradient-navigation strategies (klinokinesis + weathervane), at the behaviour level, against
  public literature, with no neuron-identity mapping — quantitatively for klinokinesis (directional
  everywhere; within the Pierce-Shimomura range for sharp reorientations) and directionally for the
  weathervane, each with bootstrap CIs across n=8 seeds.
- **Double dissociation** via the derivative-mode specificity control establishes the weathervane as
  sensor-driven, not a geometry artifact; klinokinesis as the temporal-derivative strategy.
- **Architecture-robust**: the connectome companion reproduces the same pattern.
- **Methodological**: the `|dtheta|` turn-bound saturation (→ threshold-free companions + reconciled
  verdict) and the curving-rate creep-step floor (→ de-artifacted weathervane) are the two measurement
  points that a naive single-statistic reading would have gotten wrong.

## Supporting artefacts

`supporting/035-realworm-chemotaxis-validation/`:

- `mlp-curves.json`, `connectome-curves.json`, `control-derivative-curves.json` — per-arm statistics +
  per-seed values + verdicts.
- `mlp-turn-rate-vs-dcdt.png` — klinokinesis curve (turn-rate elevated at dC/dt < 0).
- `mlp-weathervane.png`, `connectome-weathervane.png`, `control-weathervane.png` — the weathervane
  curve across arms (positive toward-gradient slope in the sensing arms; flat in the control).

Reproduce: `capture_behaviour: true` on the klinotaxis foraging cell, then
`scripts/analysis/behavioural_chemotaxis_validation.py --manifest <seed file> --tail-runs 100 --out <behavioural_curves.json> --theta-sharp 0.45`.
