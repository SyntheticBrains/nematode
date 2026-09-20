## ADDED Requirements

### Requirement: A wiring contrast under a learner that does not write the wiring states what it is about

Where a structure contrast is run under a learner that leaves the substrate's own weights fixed, the
record SHALL state that the substrate enters as **fixed features** and not as something the rule
adapts, and SHALL state which claim the result bears on. Such a result SHALL NOT be reported as
satisfying a deliverable whose condition is that the substrate's own weights are plastic.

#### Scenario: The fixed tensors are compared before and after the run

- **GIVEN** a contrast whose premise is that the substrate's own weights do not change
- **WHEN** the result is scored
- **THEN** those tensors SHALL be compared against a control in which nothing learned, the drift
  SHALL be recorded with the result, and the comparison SHALL cover every scored seed
- **AND** any non-zero drift, or drift evidence missing for any scored seed, SHALL return **void** --
  a substrate that moved is not a fixed substrate, and "it could not be checked" is not "it held"

#### Scenario: The learner's relationship to the substrate is recorded with the contrast

- **GIVEN** a wiring contrast run under a learner that does not write the substrate's weights
- **WHEN** the result is recorded
- **THEN** the record SHALL state which tensors the learner writes and which it leaves fixed
- **AND** it SHALL state that the contrast is about the substrate as fixed features

#### Scenario: A positive result is not read as the plastic-substrate deliverable

- **GIVEN** a deliverable whose condition is that the substrate's own weights are plastic
- **WHEN** a contrast under a learner that leaves them fixed returns positive
- **THEN** the record SHALL state that the deliverable's condition remains unmet
- **AND** the positive SHALL be reported as a claim about the substrate's fixed features

### Requirement: A campaign whose null carries a registered consequence states its power in advance

Where a null result would trigger a registered consequence — closing a phase, retiring a programme,
or standing as a claim about the substrate — the registration SHALL state, before the campaign runs,
what effect size the design can detect and against which comparator. Where a comparator's own effect
size is on the record, the power against it SHALL be computed and stated.

#### Scenario: The power arithmetic is registered before the run

- **GIVEN** a campaign whose null outcome carries a registered consequence
- **WHEN** its protocol is registered
- **THEN** the registration SHALL state the seed count, the detectable effect size, and the power
  against the comparator the result will be read beside
- **AND** where the design is underpowered against that comparator, the registration SHALL say so
  rather than leave it to be discovered in the result

#### Scenario: An underpowered null is not recorded as a clean null

- **GIVEN** a campaign underpowered against its comparator
- **WHEN** it returns a null
- **THEN** the record SHALL state the null as underpowered against that comparator
- **AND** it SHALL NOT be reported as evidence that the effect is absent

### Requirement: A control drawn from the same seeds as its runs states that coupling

Where a contrast's control condition is generated from the same seed that generates the run it is
compared against — a rewired graph, a shuffled label, a permuted input — the record SHALL state that
the control and the run are coupled, and SHALL state which of the two a later panel varies. Where two
results share their control conditions, the record SHALL NOT describe them as independent
replications of each other.

#### Scenario: The coupling is recorded with the result

- **GIVEN** a control generated from the run seed rather than from a seed of its own
- **WHEN** the result is recorded
- **THEN** the record SHALL state that the control and the run vary together
- **AND** it SHALL state what a later panel on fresh seeds would and would not separate

#### Scenario: Results sharing controls are not independent replications

- **GIVEN** two results whose control conditions are drawn from overlapping seed sets
- **WHEN** they are cited together
- **THEN** the record SHALL state the overlap
- **AND** neither SHALL be described as an independent replication of the other

### Requirement: A replication uses the original instrument unmodified

Where a result is re-run to test whether it holds, the replication SHALL be scored by the same
analysis the original was scored by, without modification. Where the original instrument cannot score
the replication, that SHALL be recorded as a limitation of the replication rather than resolved by
writing a new instrument.

#### Scenario: The instrument is not rewritten for the replication

- **GIVEN** a committed analysis that produced a result
- **WHEN** that result is replicated
- **THEN** the replication SHALL be scored by that analysis unmodified
- **AND** any new code SHALL be confined to preparing its inputs and reporting its outputs

#### Scenario: A failure to replicate is not attributed to the instrument

- **GIVEN** a replication scored by the original instrument
- **WHEN** it does not reproduce the original result
- **THEN** the record SHALL state that the reading is unchanged and the evidence differs
- **AND** the original result SHALL be withdrawn or qualified on the record rather than defended

### Requirement: A multi-panel replication fixes the disagreement case in advance

Where a replication covers more than one panel, the registration SHALL state before it runs how a
disagreement between panels is read. A split SHALL be reported as a split, and SHALL NOT be resolved
toward whichever panel supports the original claim.

#### Scenario: A split is reported as a split

- **GIVEN** a replication of two or more panels
- **WHEN** some panels replicate and others do not
- **THEN** each panel SHALL be reported against the registered branches on its own
- **AND** the pooled reading SHALL be withheld, the split recorded as evidence about the original
  claim's scope

### Requirement: A capacity manipulation crossed with a structure contrast is read as an interaction

Where an experiment changes a learner's capacity — the parameter count of a readout, a layer width,
the number of adapted tensors — in order to ask whether a structure effect was hidden by that
capacity, the record SHALL read the **interaction** between capacity and structure as the primary,
and SHALL NOT read a capacity main effect as evidence about the structure. A larger parameter set
learning faster is a fact about the parameter set.

#### Scenario: The capacity change is crossed rather than compared across campaigns

- **GIVEN** a structure contrast that returned null at one capacity
- **WHEN** the question is whether that capacity hid the structure effect
- **THEN** the design SHALL run both capacities against **both** levels of the structure contrast
- **AND** the primary contrast SHALL be the interaction, stated as such before the campaign runs
- **AND** a capacity main effect SHALL be reported beside the interaction and never in place of it

#### Scenario: The capacities are matched at initialisation

- **GIVEN** two capacities of the same learner compared on the same seeds
- **WHEN** the arms are constructed
- **THEN** the record SHALL state what differs between them and what does not, covering the random
  draws consumed, the initial policy, and any control arm claimed to be shared
- **AND** where an arm is claimed to be unaffected by the capacity change, that invariance SHALL be
  asserted by test rather than assumed, and a failure SHALL stop the campaign

#### Scenario: An analytic equivalence is not treated as run-level identity

- **GIVEN** two configurations shown to compute the same function in exact arithmetic
- **WHEN** that equivalence is used to justify running one of them in place of the other
- **THEN** the record SHALL establish that the two agree at the precision the runs use, not only
  analytically
- **AND** where they differ at that precision in a system whose trajectory depends on the difference,
  the configurations SHALL be run separately and the equivalence SHALL be reported as a statement
  about the computed function rather than about the runs

#### Scenario: A null interaction states the panel's sensitivity

- **GIVEN** an interaction contrast, whose per-seed variance exceeds that of either single contrast
- **WHEN** the interaction is not significant
- **THEN** the record SHALL report it as no interaction detected **at the panel's sensitivity**, with
  that sensitivity computed from observed per-seed spread and registered before the campaign ran
- **AND** it SHALL NOT be reported as excluding an interaction of unspecified size

### Requirement: A metric is chosen for the contrast it must support, and a departure is registered with its reason

Where a campaign departs from the metric a committed instrument used for a comparable contrast, the
record SHALL state the departure and its reason **before the campaign runs**, and SHALL report the
committed instrument's metric beside the chosen one.

#### Scenario: A censored metric is not used for a difference of differences

- **GIVEN** a metric that is right-censored at a horizon
- **AND** a contrast formed as a difference between two differences across cells of a design
- **WHEN** the censoring rate is not known to be equal across those cells
- **THEN** that metric SHALL NOT be the primary for that contrast
- **AND** where it is reported, its censoring SHALL be counted **per cell** rather than pooled

#### Scenario: The departed-from metric is reported beside the chosen one

- **GIVEN** a campaign that changed its primary metric from the one a committed instrument used
- **WHEN** the result is recorded
- **THEN** both metrics SHALL be reported
- **AND** where they disagree, the record SHALL state the disagreement rather than reporting only the
  primary

### Requirement: A feature ablation on a positive structure result registers a minimum effect as a decision rule

Where an experiment removes one feature of a substrate to ask whether a previously established
structure effect survives, the record SHALL register, before the campaign runs, the smallest
reduction in that effect that will be read as the feature carrying it — stated as a fraction of the
established effect — and SHALL NOT read a significant reduction below that minimum as the feature
carrying the effect.

#### Scenario: The minimum is a fraction of the effect being ablated

- **GIVEN** an established structure effect of a known size
- **WHEN** an ablation is registered against it
- **THEN** the minimum reduction SHALL be stated as a fraction of that size, with the power to detect
  it computed from observed spread and registered beside it

#### Scenario: Significant below the minimum is not carrying

- **GIVEN** an ablation whose interaction is significant but smaller than the registered minimum
- **WHEN** the reading is assigned
- **THEN** it SHALL be reported as inconclusive at the panel's sensitivity, with the observed size and
  the minimum both stated
- **AND** it SHALL NOT be reported as the feature carrying the effect

#### Scenario: A removal that moves the frozen substrate is qualified

- **GIVEN** an ablation whose frozen floors differ from the baseline's frozen floors, or whose
  learning arms' gains over those floors are smaller than the baseline's on every level of the
  structure contrast
- **WHEN** the ablation reads as carrying the effect
- **THEN** the reading SHALL state that the substrate's operating point or learnability moved and
  SHALL be reported as carrying-or-saturating, or carrying-or-unlearnable, rather than as carrying
- **AND** the floor comparison SHALL be registered before the campaign runs

#### Scenario: Ablations are read separately

- **GIVEN** more than one ablation of the same established effect in one campaign
- **WHEN** their readings are assigned
- **THEN** each SHALL be read on its own, and a difference between them SHALL be reported as the
  finding rather than averaged into one reading

### Requirement: A committed baseline is reused only under a parsed-field identity check

Where a campaign reuses committed runs from an earlier campaign as one cell of a contrast, the record
SHALL establish that a run produced now reproduces a committed run on every field the analysis reads,
and SHALL re-run the baseline in full where any field differs. *(Renamed 2026-09-19 from "a
byte-identity check": the check compares every **parsed field**, at one seed per reused arm, which is
not byte equality of logs, exports, weights or configuration. The obligation is unchanged; the name now
says what it verifies.)*

#### Scenario: Reuse is licensed by a re-run, not by argument

- **GIVEN** committed runs proposed as a baseline for a new contrast
- **WHEN** anything in the execution path has changed since they ran — code, flags, environment
- **THEN** at least one seed per reused arm SHALL be re-run under the new path and compared to the
  committed log on every parsed field
- **AND** the record SHALL state the comparison's result as the evidence for reuse

#### Scenario: There is no partial reuse

- **GIVEN** a parsed-field identity check in which any field differs for any reused arm
- **WHEN** the campaign is planned
- **THEN** the whole baseline SHALL be re-run under the new path
- **AND** no committed run SHALL be mixed with re-run ones in the same contrast

### Requirement: A positive result carrying an inherited learner setting is re-read at a calibrated operating point before a synthesis cites it

Where a positive structure result was obtained under a learner setting inherited from a different
capacity or substrate rather than calibrated for the arms it ran on, and a later campaign shows the
setting moves the outcome, the record SHALL re-read the result's **registered primary** at the
calibrated setting before the result enters a phase synthesis, and SHALL carry the outcome as a
condition beside the committed verdict rather than as a rewrite of it.

#### Scenario: The re-read is the registered primary at one setting, not a new question

- **GIVEN** a committed positive whose primary was a crossed interaction
- **WHEN** it is re-read at the calibrated setting
- **THEN** the primary SHALL be the same interaction at that setting, with the cells already measured
  there reused only under the committed baseline-reuse requirement — one seed per reused arm re-run on
  the current path and compared on **every field the analysis parses** — and only the missing cells run
- **AND** the record SHALL state what that check establishes and what it does not: it establishes that
  the current path reproduces the committed run on every quantity any analysis in the programme reads,
  at the seed checked; it is **parsed-field identity, not byte equality** of logs, exports, weights or
  configuration, and it does not extend to the seeds left unchecked
- **AND** where a parsed field cannot be compared because the committed side no longer holds the
  artefact it derives from, that field SHALL be named as uncompared rather than counted as matching
- **AND** a minimum effect SHALL be registered as a fraction of the committed effect, with the
  reading that a significant result below it receives named before the runs, **and registered for
  both directions** where the reading is two-sided

#### Scenario: A headline form that is already known not to hold is stated before the runs

- **GIVEN** a committed positive with a registered primary and a more striking form it happened to
  take (a sign flip, a lead at one level)
- **WHEN** committed data already shows that form absent at the calibrated setting
- **THEN** the design SHALL say so before the runs, and SHALL read the registered primary
- **AND** a positive SHALL be reported as the weaker claim it is, never as the headline form
  reproduced

#### Scenario: The verdict is conditioned, not rewritten

- **GIVEN** any reading of the re-read
- **WHEN** the record is written
- **THEN** the committed verdict SHALL stand as read at its own setting
- **AND** the tracker, the original logbook and the roadmap SHALL each carry the condition as a dated
  note in the same place the verdict is cited
- **AND** the synthesis SHALL state the condition in the same sentence as the claim

#### Scenario: The re-read does not decide the setting and does not ablate against it

- **GIVEN** a re-read showing both arms learn better at the calibrated setting
- **WHEN** the record draws consequences
- **THEN** it SHALL NOT declare either setting correct, and SHALL leave the choice to the calibration
  of the rung that next runs there
- **AND** no ablation SHALL be read against the calibrated baseline inside the same campaign that
  establishes it

### Requirement: A shipped result with an uncontrolled confound carries it as a standing condition

Where a result ships with a confound its own record registered but did not control, the confound SHALL
be carried as a **standing condition** stated in the same sentence as the claim at every citation
site, and the control SHALL be named in the successor phase's opening scope rather than left as an
open caveat.

#### Scenario: The condition travels with the claim

- **GIVEN** a shipped result whose record names an uncontrolled confound
- **WHEN** the result is cited in a synthesis, a tracker, a roadmap or a later record
- **THEN** the condition SHALL appear in the same sentence as the claim, not in a separate caveat
  section
- **AND** the citation SHALL NOT state the effect size without it

#### Scenario: An external precedent for the confound raises its standing

- **GIVEN** published work that applies the missing control to the same kind of claim
- **WHEN** the close assesses the confound
- **THEN** that work SHALL be named, and the confound SHALL be treated as load-bearing rather than
  residual
- **AND** the control SHALL be scheduled ahead of further results that would inherit the confound

#### Scenario: A control needing a design decision opens a phase rather than closing one

- **GIVEN** a missing control whose specification is itself ambiguous
- **WHEN** the close schedules it
- **THEN** the ambiguity SHALL be stated as the reason it is the successor phase's work
- **AND** the close SHALL NOT report the control as a small remaining task
