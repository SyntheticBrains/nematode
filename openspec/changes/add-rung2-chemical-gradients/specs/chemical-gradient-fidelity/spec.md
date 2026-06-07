## ADDED Requirements

### Requirement: Signal-type-specific static Fick-shaped gradient geometry

The environment SHALL be able to compute chemical concentration fields using a **frozen analytic Fick (Gaussian) kernel** `C(r) = A · exp(−r² / (4·D·t_assay))` with **per-signal diffusion coefficients** `D` (e.g. food, pheromone, CO₂), selectable as a field mode on the continuous-2D substrate. The pre-existing exponential-decay kernel SHALL remain the default field mode so that legacy and discrete-grid configurations are unaffected and byte-stable. The field SHALL remain **static** (frozen at assay time); time-evolving diffusion is explicitly out of scope (gated stretch).

#### Scenario: Fick field mode produces the Gaussian kernel

- **WHEN** a configuration selects the Fick field mode for a chemical signal
- **THEN** the concentration at distance `r` from a source SHALL be evaluated as `A · exp(−r² / (4·D·t_assay))` using that signal's configured `D`, rather than the exponential-decay kernel

#### Scenario: Per-signal diffusion coefficients set distinct geometry

- **WHEN** two chemical signals are configured with different `D` values
- **THEN** their static field geometries SHALL differ in spatial spread accordingly (larger `D` → broader gradient)

#### Scenario: Legacy exponential kernel remains the default

- **WHEN** a configuration does not select the Fick field mode
- **THEN** the environment SHALL use the existing exponential-decay kernel, and discrete-grid field values SHALL remain byte-stable

### Requirement: Adaptive chemosensory sensor with background-tracking relative coding

When the adaptive chemosensory sensor is enabled for a chemical channel, the system SHALL transform the raw concentration through a **leaky-integrator background** `B_t = (1−α)·B_{t−1} + α·C_t` (adaptation rate `α`, timescale `τ`) and a **biphasic relative readout** (positive when the signal rises above background, negative when it falls below). The readout form SHALL be config-selectable among at least: (a) **derivative-channel fold-change** — the temporal derivative normalized by concentration, `(dC/dt)/C ≈ d(log C)/dt`; (b) **instantaneous magnitude contrast** — `(C_t − B_t)/(C_t + B_t + ε)`; and (c) **log-concentration baseline** — a documented under-powered special case for ablation. The active readout mode and its **channel interaction** (see below) SHALL be explicit in configuration. When the adaptive sensor is disabled, chemosensory behaviour SHALL match the current non-adaptive (tanh) pipeline.

#### Scenario: Adaptive readout codes relative to background

- **WHEN** the adaptive sensor is enabled and the agent experiences a sustained concentration above its tracked background
- **THEN** the sensed signal SHALL reflect the signal **relative to** the leaky-integrator background, not the absolute concentration, and SHALL relax toward neutral as the background catches up

#### Scenario: Channel-interaction mode is explicit

- **WHEN** the adaptive sensor is configured
- **THEN** the configuration SHALL state whether adaptation **reshapes the existing derivative/turning channel** (fold-change) or **adds a standalone contrast magnitude channel**, and the system SHALL apply exactly the configured interaction (these are distinct behavioural contracts, not interchangeable tunings); the default SHALL be derivative-channel fold-change

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
