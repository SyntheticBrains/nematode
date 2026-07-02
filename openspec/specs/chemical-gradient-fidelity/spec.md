# chemical-gradient-fidelity Specification

## Purpose

Defines the Rung-2 chemical-gradient fidelity of the chemosensory system: signal-type-specific **static Fick-shaped gradient geometry** (a frozen analytic Gaussian/Fick kernel with per-signal diffusion coefficients on the continuous-2D substrate, with the legacy exponential-decay kernel retained as the default so existing and discrete-grid configs stay byte-stable), and an **adaptive/biphasic chemosensory sensor** that codes concentration relative to a slowly-tracked background (a leaky-integrator background feeding fold-change / contrast / log readouts; plain log-concentration is the documented under-powered baseline), modelling *C. elegans* chemosensory adaptation (Kato et al. 2014; Levy & Bargmann 2020 "hyper-Weber"). It also defines the load-bearing **step-input adaptation-transient validation gate** that distinguishes the adaptive sensor from the static log baseline. Time-evolving diffusion (a dynamic `∂C/∂t = D∇²C` solve) is an explicit out-of-scope stretch.

## Requirements

### Requirement: Signal-type-specific static Fick-shaped gradient geometry

The environment SHALL be able to compute **diffusing chemical** concentration fields using a **frozen analytic Fick (Gaussian) kernel** `C(r) = A · exp(−r² / (4·D·t_assay))` with **per-signal diffusion coefficients** `D` (e.g. food, predator sulfolipid, pheromone, CO₂), selectable as a per-field mode on the continuous-2D substrate. The pre-existing exponential-decay kernel SHALL remain the default field mode for every field so that legacy and discrete-grid configurations are unaffected and byte-stable. The Fick mode SHALL be selectable **independently per diffusing chemical field, each with its own `D`** — this includes the **food** field and the **predator sulfolipid** distal-chemo field (a diffusing chemical the worm senses at a distance; Liu et al. 2018). Fields that are **not point-source diffusion** are out of scope and retain their existing behaviour: **temperature** (an imposed thermal-conduction gradient) and the **oxygen base gradient** (an imposed/boundary aerotaxis gradient). The field **geometry** SHALL remain **static** — the analytic kernel and per-signal `D` are frozen at assay time, and time-evolving diffusion (a `∂C/∂t = D∇²C` solve) is explicitly out of scope (a gated stretch). The per-source **amplitude** `A` MAY vary *within* an episode when the config-gated source-depletion dynamic is enabled (see the `source-depletion-dynamics` capability): this scales only the per-source strength, leaving the frozen geometry and `D` unchanged, and is distinct from the out-of-scope diffusion PDE.

#### Scenario: Fick field mode produces the Gaussian kernel

- **WHEN** a configuration selects the Fick field mode for a diffusing chemical signal (food or predator sulfolipid)
- **THEN** the concentration at distance `r` from a source SHALL be evaluated as `A · exp(−r² / (4·D·t_assay))` using that signal's configured `D`, rather than the exponential-decay kernel

#### Scenario: Per-signal diffusion coefficients set distinct geometry

- **WHEN** two diffusing chemical signals (e.g. food and predator sulfolipid) are configured with different `D` values
- **THEN** their static field geometries SHALL differ in spatial spread accordingly (larger `D` → broader gradient), independently per field

#### Scenario: Predator sulfolipid field supports the Fick mode

- **WHEN** the predator field is configured with the Fick mode
- **THEN** the predator concentration / distal-chemo (sulfolipid) field SHALL use the Fick Gaussian kernel with the predator's own configured `D`, replacing the exponential kernel for that field

#### Scenario: Imposed/boundary fields are excluded

- **WHEN** temperature or the oxygen base gradient is computed
- **THEN** it SHALL retain its existing imposed/boundary-gradient behaviour and SHALL NOT be evaluated with the Fick chemical-diffusion kernel (these are not point-source diffusion)

#### Scenario: Legacy exponential kernel remains the default

- **WHEN** a configuration does not select the Fick field mode for a given field
- **THEN** the environment SHALL use the existing exponential-decay kernel for that field, and discrete-grid field values SHALL remain byte-stable

#### Scenario: Source depletion varies amplitude, not geometry

- **WHEN** the source-depletion dynamic is enabled and a source is fed upon
- **THEN** only that source's amplitude `A` SHALL decrease (its frozen Fick geometry and `D` are unchanged), and the time-evolving `∂C/∂t = D∇²C` diffusion PDE SHALL remain out of scope

### Requirement: Adaptive chemosensory sensor with background-tracking relative coding

When the adaptive chemosensory sensor is enabled for a chemical channel, the system SHALL transform the raw concentration through a **leaky-integrator background** `B_t = (1−α)·B_{t−1} + α·C_t` (adaptation rate `α`, timescale `τ`) and a **biphasic relative readout** (positive when the signal rises above background, negative when it falls below). The readout form SHALL be config-selectable among at least: (a) **derivative-channel fold-change** — the temporal derivative normalized by concentration, `(dC/dt)/(C + ε) ≈ d(log C)/dt` (the regularising `+ ε`, or normalisation by the tracked background `B + ε`, is REQUIRED — raw `(dC/dt)/C` is singular at the common `C ≈ 0` regime far from a source); (b) **instantaneous magnitude contrast** — `(C_t − B_t)/(C_t + B_t + ε)`; and (c) **log-concentration baseline** — a documented under-powered special case for ablation. The active readout mode and its **channel interaction** (see below) SHALL be explicit in configuration. The adaptive sensor applies to **chemosensory channels** (food chemotaxis, and pheromone / CO₂ where those signals are active); thermosensory and mechanosensory channels are out of scope. When the adaptive sensor is disabled, chemosensory behaviour SHALL match the current non-adaptive (tanh) pipeline.

#### Scenario: Adaptive readout codes relative to background

- **WHEN** the adaptive sensor is enabled and the agent experiences a sustained concentration above its tracked background
- **THEN** the sensed signal SHALL reflect the signal **relative to** the leaky-integrator background, not the absolute concentration, and SHALL relax toward neutral as the background catches up

#### Scenario: Channel-interaction mode is explicit

- **WHEN** the adaptive sensor is configured
- **THEN** the configuration SHALL state whether adaptation **reshapes the existing derivative/turning channel** (fold-change) or **adds a standalone contrast magnitude channel**, and the system SHALL apply exactly the configured interaction (these are distinct behavioural contracts, not interchangeable tunings); the default SHALL be derivative-channel fold-change

*(As-built: the channel interaction is stated by the single `adaptive_chemosensor_readout` enum — `fold_change` ⇒ derivative channel, `contrast` ⇒ magnitude channel, `log` ⇒ magnitude baseline. This tranche wires the **food** chemosensory channel; pheromone / CO₂-chemo channels are a follow-up since those behaviours are deferred.)*

#### Scenario: Log-concentration baseline mode available

- **WHEN** the log-concentration baseline mode is selected
- **THEN** the sensor SHALL apply `log(1 + C)` as a documented baseline transform, usable as the ablation comparator for the adaptive readout

#### Scenario: Disabled sensor preserves current behaviour

- **WHEN** the adaptive sensor is not enabled
- **THEN** the chemosensory pipeline SHALL behave as the current non-adaptive (tanh-normalised) pipeline with no change

### Requirement: Step-input adaptation-transient validation gate

The change SHALL provide a validation that drives a chemosensory **step input** to the sensor in isolation and measures the **adaptation transient** — the peak response on the step followed by relaxation toward baseline as the background tracker catches up — and quantifies **Weber-like invariance to background level**, reported **against the log-concentration baseline**. This step-input transient SHALL be the load-bearing acceptance check for the adaptive sensor (alongside an env-fidelity-geometry comparison) before T7.

#### Scenario: Step input produces a measurable adaptation transient

- **WHEN** a concentration step is presented to the adaptive sensor
- **THEN** the validation SHALL record a transient with a peak response and a relaxation toward baseline, and SHALL report its shape against the log-concentration baseline

#### Scenario: Background invariance is quantified

- **WHEN** the same relative step is presented at different absolute background levels
- **THEN** the validation SHALL quantify how invariant the peak response is to the background level (the Weber / hyper-Weber signature)
