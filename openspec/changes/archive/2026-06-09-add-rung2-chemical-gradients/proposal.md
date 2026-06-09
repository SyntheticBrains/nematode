## Why

Phase 6 Tranche 5 delivered the continuous-2D substrate (Gate 2 GO, [logbook 027](../../../docs/experiments/logbooks/027-platform-refactor-continuous-2d.md)) but deliberately deferred *env fidelity* to T6: the continuous arena still uses Rung-0 gradient geometry (superposed static 1/r / exponential-decay terms), still places food/predator sources on the integer lattice, and still feeds chemosensory brains a raw concentration scalar. The Phase 6 Rung-2 commitment (`docs/roadmap.md` § Continuous environment + sensory physics) is the field's actual fidelity standard — and the 2026-06-04 checkpoint established **where that fidelity budget goes**: into the *sensor* (adaptive-threshold / biphasic LN dynamics — Kato et al. 2014 *Neuron*; Levy & Bargmann 2020 *Neuron*), of which plain log-concentration is an under-powered special case, **not** into a live time-evolving diffusion PDE (no published *C. elegans* chemotaxis model uses one — a single-point sensor cannot perceive global field dynamics on assay timescales).

Now, because T5 closed Gate 2 and **T7 depends on T6**: the T7 apples-to-apples architecture ranking and the behavioural-chemotaxis real-worm validation (turn-rate vs dC/dt; curving-rate vs bearing) both require a sensor + field that match the computational-chemotaxis literature, not the Rung-0 placeholder.

## What Changes

- **PRIMARY — adaptive-threshold / biphasic chemosensory sensor.** AWC/AWA/ASE-style sensors gain a background-tracking adaptive threshold or biphasic LN filter over the sensed concentration; **log-concentration is retained as a documented baseline special case**, not the headline. This is the load-bearing, iteration-prone deliverable.
- **Signal-type-specific static Fick-shaped gradient geometry.** Food vs pheromone vs CO₂ get distinct diffusion coefficients (D) setting **static** Fick-shaped field geometry (frozen analytic Fick solution at assay time — the field-standard form), replacing the Rung-0 superposition fields on the continuous-2D substrate.
- **Float source placement + Euclidean fields + continuous-field sampling.** Lifts the explicit T6 deferrals already recorded in the `continuous-2d-environment` spec (sources move off the integer lattice; remaining fields/sampling go Euclidean/continuous; the `self.foods: list[tuple[int, int]]` type ripple is resolved).
- **Continuous-substrate fidelity renderer (seed / non-gating).** Adapt the existing pygame `PIXEL` renderer to continuous coordinates with concentration-field heatmap, gradient-vector, sensor-zone, and adaptive-sensor-state overlays; reuse `scripts/export_screenshot.py` for frame export. Feeds the T7 validation figures. *(**Deferred** to a follow-up as the non-gating seed it is — the existing grid render was made float-safe so continuous runs work headless; see design.md § Deviations.)*
- **Analysis + logbook.** Quantify the env-fidelity gain; the **load-bearing check is the adaptation transient on a step-input test** vs the prior log-concentration baseline. Publish the T6 logbook (required reading before T7).
- **STRETCH (explicitly gated — NOT in this change's gating scope).** Dynamic diffusion (`∂C/∂t = D∇²C`) + source dynamics (depletion / replenishment / decay) is pursued *only* on a concrete behavioural need (the strongest being depletion-driven area-restricted search). Recorded here as bounded future scope, gated behind a written justification.
- **Pre-structured iteration sub-tranche** for the adaptive-sensor kinetics: if the sensor needs M4-style parameter sweeps (adaptation timescales / filter shape), T6 absorbs them as a bounded sub-tranche without re-opening Gate 2.

## Capabilities

### New Capabilities

- `chemical-gradient-fidelity`: Rung-2 environmental chemosensory fidelity on the continuous-2D substrate — signal-type-specific **static Fick-shaped gradient field geometry** (per-signal D values; frozen analytic solution) **paired with the adaptive-threshold / biphasic chemosensory sensor model** (log-concentration as a documented baseline special case). Owns the step-input adaptation-transient validation. The roadmap treats gradient geometry and sensor adaptation as the two co-designed halves of "Rung 2", and they share the T6 analysis gate, so they form one capability.

### Modified Capabilities

- `continuous-2d-environment`: float source placement (replacing the integer-lattice deferral), Euclidean fields, and continuous-field klinotaxis sampling — lifting the three explicit "T6 Rung-2 work" deferrals already written into this spec's *Source placement and Euclidean fields* and *Heading-aware klinotaxis sensing* requirements. The discrete grid environment remains unchanged and byte-stable.
- `klinotaxis-sensing`: the sensed concentration `strength` is passed through the adaptive / biphasic chemosensory transform (log-concentration as the baseline special case) rather than delivered as a raw scalar, on the continuous substrate. The head-sweep geometry and STAM temporal-derivative behaviour are unchanged.

## Impact

- **Code:** continuous-2D env field generation (Fick-shaped static fields, float sources, Euclidean distances); a chemosensory adaptive-sensor module (adaptation state per modality); `configs/scenarios/` parameters for per-signal D values + sensor kinetics; the pygame renderer + theme for continuous coordinates and fidelity overlays.
- **Brains:** interface unchanged — brains consume the sensor output; the T7 continuous arms read the new adaptive sensor. No per-architecture branches.
- **Dependencies:** none new (numpy for the analytic Fick solution; pygame already present).
- **Cross-tranche:** feeds T7 (klinotaxis *and* thermotaxis evaluations consume the T6 gradients + adaptive sensor; thermotaxis benefits from the continuous Euclidean fields even though the adaptive transform is chemosensory). Does **not** touch the discrete grid substrate (byte-stable) — the T4-vs-T7 env-upgrade delta (RQ5) stays interpretable. The renderer is non-gating for T6's analysis gate.
- **Out of scope:** dynamic-diffusion PDE / source dynamics (gated stretch); 3D physics; Sibernetic interop; aerotaxis / pheromone-behaviour restoration; multi-agent — all per Decision 5 + roadmap Future Directions.
