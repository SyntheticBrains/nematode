# Spec: realworm-behavioural-validation

## ADDED Requirements

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
