# realworm-behavioural-validation Specification

## Purpose

Behaviour-level validation of the RL nematode's chemotaxis against published *C. elegans* data: from
an opt-in captured behavioural trajectory, compute the two documented klinotaxis bias curves
(klinokinesis turn-rate/magnitude vs dC/dt; klinotaxis weathervane curving-rate vs bearing) and grade
each against a behaviour-level literature reference (documented direction + reported range + citation,
not a figure digitization) with a bootstrap CI. Supports Gate 3 G3.d ([Logbook 035](../../../docs/experiments/logbooks/035-realworm-chemotaxis-validation.md)).

## Requirements

### Requirement: Opt-in continuous behavioural-trajectory capture

The simulation SHALL support config-gated capture of a per-step behavioural trajectory — continuous
position, heading, local food concentration, dC/dt, and the local gradient direction — for a
continuous-2D run, disabled by default. The captured values SHALL be read from the state and sensing
already computed each step (no new sensing computation), and the gradient direction SHALL be captured
**live** (not recomputed post-hoc, since the food field mutates during a foraging run). When disabled
(the default), no behavioural record SHALL be produced and the run SHALL be byte-identical to the
pre-change behaviour.

#### Scenario: Capture is off by default and byte-identical

- **WHEN** a run executes without the behavioural-capture flag set
- **THEN** no per-step behavioural trajectory SHALL be recorded and the run's results SHALL be identical to the pre-change behaviour

#### Scenario: Capture records the per-step behavioural series

- **WHEN** a continuous-2D run executes with behavioural capture enabled
- **THEN** each step SHALL append a record carrying the continuous position, heading, local food concentration, and dC/dt, and the series SHALL be available on the run result

### Requirement: Klinokinesis turn-rate-versus-dC/dt bias curve

The validation SHALL compute, from a captured behavioural trajectory, a **turn-rate versus dC/dt** bias
curve: each step SHALL be classified as a sharp reorientation (heading change beyond a calibrated
threshold) or gradual motion, and the reorientation rate SHALL be binned by dC/dt (or its sign). The
curve SHALL be computed as a pure function of the trajectory, with per-bin confidence intervals across
seeds.

#### Scenario: Turn-rate is higher heading down-gradient

- **WHEN** the klinokinesis curve is computed for a trained klinotaxis forager
- **THEN** the reorientation rate in the down-gradient (dC/dt < 0) bins SHALL be reported against the up-gradient (dC/dt > 0) bins as a down/up turn-rate ratio, with a bootstrap confidence interval

#### Scenario: The turn threshold is calibrated and reported

- **WHEN** the sharp-reorientation threshold is chosen
- **THEN** it SHALL be calibrated on the substrate (e.g. from the heading-change distribution) and recorded with the result, and the verdict's stability to the threshold SHALL be checkable

### Requirement: Klinotaxis curving-rate-versus-bearing bias curve

The validation SHALL compute a **curving-rate versus bearing-to-gradient** bias curve: the signed
gradual heading change per unit path length SHALL be binned by the bearing between the heading and the
**per-step logged** gradient direction, with per-bin confidence intervals across seeds. The curve SHALL
express whether the worm curves toward or away from the gradient.

#### Scenario: Curving is biased toward the gradient

- **WHEN** the klinotaxis curve is computed for a trained klinotaxis forager
- **THEN** the mean signed curving-rate at off-axis bearings SHALL be reported as a toward-gradient weathervane slope, with a bootstrap confidence interval

### Requirement: Behaviour-level agreement verdict against published signatures

The validation SHALL compare each bias curve's summary statistic (the down/up turn-rate ratio; the
weathervane slope) against a **documented literature signature** — a bias direction plus a reported
magnitude range with citation (Pierce-Shimomura 1999; Iino & Yoshida 2009) — and emit a per-curve
verdict: REPRODUCED (sign matches and the model CI overlaps the range), PARTIAL (sign matches,
magnitude out of range), or ABSENT (sign does not match or the CI spans the no-bias null). The
reference SHALL be a behaviour-level signature, and the report SHALL state that a pixel-exact
figure digitisation is a non-goal.

#### Scenario: A reproduced strategy passes

- **WHEN** the model's bias statistic matches the literature sign and its bootstrap CI overlaps the literature magnitude range
- **THEN** the curve's verdict SHALL be REPRODUCED

#### Scenario: An absent strategy is reported honestly

- **WHEN** the model's bias statistic does not match the literature sign, or its CI spans the no-bias null
- **THEN** the curve's verdict SHALL be ABSENT, reported as a finding (which strategy the learned policy does or does not use), not suppressed

#### Scenario: A validation report + figures are produced

- **WHEN** the validation runs over the per-seed captures
- **THEN** it SHALL write a summary JSON (per-curve statistics, CIs, verdicts) and two figures overlaying the model curve + CI band on the literature reference band
