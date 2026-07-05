# Real-worm behavioural-chemotaxis validation

## Why

The continuous-2D substrate — float kinematics, Fick-diffusion food fields, klinotaxis head-sweep
sensing, the adaptive/biphasic chemosensor — was built to model real *C. elegans* chemotaxis. **Gate 3
G3.d** (MUST for Phase-6 exit) requires anchoring that fidelity to real-worm data. This change
delivers the external anchor: does our trained RL worm reproduce the two documented *C. elegans*
gradient-navigation strategies?

- **Klinokinesis (biased random walk)** — the reorientation/pirouette rate rises when the worm heads
  **down**-gradient and is suppressed heading **up**-gradient: **turn-rate vs dC/dt**
  (Pierce-Shimomura et al. 1999; Morse & Lockery 1999).
- **Klinotaxis (weathervane)** — the worm gradually curves **toward** the gradient: **curving-rate vs
  bearing-to-gradient** (Iino & Yoshida 2009).

These are the *lead validation target* per `T7.validation.1`: they are **exactly the RL worm's own
behavioural output**, use **public literature data**, and need **no neuron-identity mapping** (unlike
the Ca²⁺-matrix option, which also carries a category-mismatch caveat for a behavioural model).

The existing validation stack (`quantumnematode/validation/chemotaxis.py`, `datasets.py`, the
`--validate-chemotaxis` flow) computes only the **scalar chemotaxis index (CI)** — neither bias curve
exists, and the per-run trajectory (`result.path`) stores only **integer grid positions**, discarding
the continuous position / heading / concentration / dC-dt the curves need (those live only as
transient `AgentState` + sensing values). So this change adds the behavioural-curve layer on top of
the existing CI validation.

## What Changes

- **Opt-in continuous behavioural-trajectory logging.** Capture a per-step record — continuous
  position, heading, local food concentration, dC/dt, and the **live gradient direction** — at the
  agent step (where those values are already computed together), gated behind a flag so default runs
  are byte-identical. Reuse the live sensing (`env.get_food_concentration`,
  `env.get_separated_gradients`, STAM `compute_temporal_derivative`) rather than recomputing — and log
  the gradient direction live, since the food field mutates during a foraging run.
- **Two bias-curve metrics + a shared binning helper.** Decompose each step's heading change into a
  **sharp reorientation** (pirouette → turn-rate) vs a **gradual curving** (weathervane), then bin:
  turn-rate by dC/dt (curve A) and mean signed curving-rate by bearing-to-gradient (curve B), with
  per-bin bootstrap CIs across seeds.
- **Reference literature signatures.** Encode the documented bias direction + magnitude ranges for
  Pierce-Shimomura 1999 and Iino & Yoshida 2009 (the current `datasets.py` models only scalar CI;
  Pierce-Shimomura appears as a CI value, Iino & Yoshida is absent).
- **Quantitative agreement + report.** Per curve, a bias statistic (the up-vs-down-gradient turn-rate
  ratio; the weathervane slope) with a bootstrap CI, compared to the literature range — sign match +
  CI/range overlap. A validation summary JSON + two curve figures (model with CI band overlaid on the
  literature reference), extending `report/continuous_figures.py`.

## Capabilities

**Added**: `realworm-behavioural-validation` — the two-bias-curve validation of the RL worm's
klinotaxis against published *C. elegans* data (behavioural-trajectory capture contract, the two curve
metrics, and the literature-agreement report). Sibling to the existing scalar-CI chemotaxis validation,
not a replacement.

**Modified**: none of the existing capability specs. The behavioural-trajectory logging is opt-in and
byte-identical when off, so the continuous-2D environment / runner capabilities are unchanged in
behaviour.

## Impact

- **Code**: opt-in per-step behavioural logging at the agent step (`agent.py`, where position, heading,
  concentration, dC/dt, and the gradient direction are computed together), a new
  `validation/behavioural_curves.py` (pure-function metrics + binning), a
  reference-signature dataset, two figures in `report/continuous_figures.py`, and a
  `scripts/analysis/` aggregation harness. Default paths unchanged → byte-identical when the flag is
  off (guarded by a test).
- **Docs / tracking**: resolves `T7.validation.1/2/3` (Gate 3 G3.d); the result feeds the 6a synthesis
  (`T9a`). No behaviour change to any existing run.
- **Caveat (recorded up front)**: the reference is the *documented behavioural signature* (bias
  direction + reported magnitude range) of each strategy, not a pixel-exact digitisation of the
  original figures — so the agreement target is **behaviour-level** (correct sign + order-of-magnitude,
  with model CIs), consistent with `T7.validation`'s "behaviour-encoding, no neuron-mapping" framing
  and the Phase-7 claim-discipline habit.
