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

### Requirement: Selectable behavioural-capture modality (chemotaxis or thermotaxis)

The opt-in behavioural capture SHALL support a configurable **modality** selecting which navigation
drive the per-step record captures: `food` (chemotaxis — the default, byte-identical to the existing
capture) or `thermotaxis`. The modality SHALL default to `food` so every existing run and the
chemotaxis pipeline remain byte-identical. When the modality is `thermotaxis`, the captured drive /
drive-derivative / gradient fields SHALL carry the homeostatic thermal signals (below) rather than the
food signals, so the existing bias-curve metrics apply unchanged.

#### Scenario: Default modality is byte-identical chemotaxis capture

- **WHEN** capture is enabled without an explicit modality
- **THEN** the captured record SHALL carry the food concentration, its derivative, and the live
  food-gradient direction — identical to the pre-change chemotaxis capture

#### Scenario: Thermotaxis modality captures the homeostatic thermal drive

- **WHEN** capture is enabled with the thermotaxis modality on a thermotaxis-enabled run
- **THEN** each record SHALL carry the setpoint drive toward the cultivation temperature, its
  one-step derivative, and the toward-comfort direction (from the live temperature field), and the
  existing klinokinesis / weathervane metrics SHALL be computable from it without modification

### Requirement: Homeostatic setpoint thermal drive

For the thermotaxis modality the captured drive SHALL be a **homeostatic setpoint error** toward the
cultivation temperature `Tc`: the drive SHALL be maximal (zero) at `Tc` and decrease as the worm moves
away from `Tc` (e.g. `−|T − Tc|`), and the captured gradient direction SHALL point **toward comfort**
(up the thermal gradient when the worm is colder than `Tc`, the opposite direction when warmer). The
thermal temperature and gradient SHALL be sampled **live** at the sensing step (not recomputed
post-hoc). This setpoint adjustment is what makes the reproduced signature "turning/curving biased
toward the cultivation temperature", the thermal analogue of climbing the food gradient.

#### Scenario: Drive decreases away from the cultivation temperature

- **WHEN** the worm moves so that `|T − Tc|` increases (away from comfort)
- **THEN** the captured drive derivative SHALL be negative, so the klinokinesis curve measures
  elevated turning when heading away from `Tc`

#### Scenario: Toward-comfort direction flips across the setpoint

- **WHEN** the worm is warmer than `Tc` rather than colder
- **THEN** the captured toward-comfort direction SHALL be the opposite of the increasing-temperature
  direction, so the weathervane curve measures curving toward `Tc` in both regimes

### Requirement: Modality-specific literature reference set

The validation SHALL provide a **thermotaxis** reference set (the same four statistic keys as
chemotaxis) encoding the documented thermal bias directions + citations. The primary reference is
Luo et al. 2014, which explicitly decomposes *C. elegans* thermotaxis into **klinokinesis** (biased
turning) and **klinotaxis** (weathervane curving), with Ryu & Samuel 2002 and Clark et al. 2007 as
supporting behaviour-level thermotaxis sources. Because thermal bias magnitudes are not comparable to
the model's units, the thermotaxis references SHALL be **sign-only** (direction, no magnitude range). Reference loading SHALL select the set by
modality; a caller-supplied path that does not exist SHALL raise rather than silently substitute
defaults.

#### Scenario: Thermotaxis references are sign-only and cited

- **WHEN** the thermotaxis reference set is loaded
- **THEN** it SHALL contain the four bias statistics, each with the documented sign, no magnitude
  range, and a thermotaxis citation (Luo et al. 2014 primary) distinct from the chemotaxis references

### Requirement: Modality-selectable aggregation harness

The aggregation harness SHALL accept a modality selector that grades the captured curves against the
matching reference set (chemotaxis or thermotaxis), reusing the same bias statistics, bootstrap CIs,
curving-rate floor, verdict grading, and figures. The summary JSON SHALL record which modality was
graded.

#### Scenario: The harness grades thermotaxis captures against the thermal references

- **WHEN** the harness runs with the thermotaxis modality over thermotaxis captures
- **THEN** it SHALL grade each bias statistic against the thermotaxis references and emit the same
  REPRODUCED / PARTIAL / ABSENT verdicts with bootstrap CIs, recording the modality in the summary
